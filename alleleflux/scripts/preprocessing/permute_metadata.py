#!/usr/bin/env python3
# pylint: disable=logging-fstring-interpolation
"""Generate a permuted-label metadata sheet for AlleleFlux null/control runs.

Why this exists
---------------
AlleleFlux's divergence score is only interpretable against a **null**: the same
analysis run on samples whose group labels have been shuffled. This script writes
that shuffled metadata. Paired with the pipeline's cache-reuse toggle
(``input.reuse_from`` + ``--permuted_metadata``), a permuted run reuses the real
run's profiles / QC / allele-frequency cache and only recomputes the
group-dependent tail.

The model
---------
Two columns and a count::

  --unit COLUMN   The randomization unit (a mouse, a cage, a sample): the thing
                  that was assigned to a group and that the swap relabels.  Each
                  value must be single-group.  A unit's new group is broadcast to
                  ALL of its rows (every timepoint, every member).  Default:
                  ``subjectID``.

  --block COLUMN  Optional.  A matched-pair *block* (stratum) to swap WITHIN.

  --swaps N       How many unit-pairs to swap.  A "swap" exchanges the group
                  labels of two units in DIFFERENT groups; each swap moves one
                  unit each way, so per-group counts are always preserved.
                  Defaults to **half** the available pairs.

How a swap works depends only on whether there's a block:
  * **with --block**  — each block is a matched pair (two single-group units, one
    per group); a swap *flips one block* (its two units exchange groups).
    ``--swaps N`` flips N randomly chosen blocks; the rest are untouched.
  * **without --block** — N cross-group unit-pairs are formed (one unit from each
    group) and exchanged.  Needs exactly two groups in scope (use ``--groups``).

No "modes", no auto-detection — you name the unit/block and the swap count.

The four designs, one operation
-------------------------------
====================  =======================  =========================================
design                structure                command
====================  =======================  =========================================
DRIDO                 independent mice         ``--unit subjectID --groups 1D AL``
antibiotic            cages, no block          ``--unit replicate``   (cage *is* replicate)
diet study            cage-pairs in blocks     ``--unit cage --block replicate``
soil                  sample-pairs in plots    ``--unit subjectID --block replicate``
====================  =======================  =========================================

Worked examples
---------------
Global swap (no ``--block``) — antibiotic cages, ``--swaps 1``::

    before            after  (one control<->antibiotic pair exchanged)
    cage  group       cage  group
    A35   control     A35   antibiotic
    A70   antibiotic  A70   control
    A56   antibiotic  A56   antibiotic   (untouched)

Within-block swap (``--block replicate``) — diet-study cages, ``--swaps 1``::

    before                           after  (block 1 flipped, block 2 kept)
    replicate cage     group         replicate cage     group
    1         control1 control       1         control1 fat
    1         fat1     fat           1         fat1     control
    2         control2 control       2         control2 control
    2         fat2     fat           2         fat2     fat

Only the ``group`` column ever changes; ``sample_id`` / ``subjectID`` /
``bam_path`` stay tied to their original rows.

Per-comparison-pair null (``--groups``)
---------------------------------------
``--groups G1 G2`` restricts the swap to the subjects in ``{G1, G2}``, leaving
every other group untouched — the clean per-comparison null for multi-group
studies (DRIDO: call once per ``groups_combinations`` pair).  Without
``--groups`` the whole sheet is permuted (and a global swap then requires the
sheet to have exactly two groups).

Single-run note
---------------
The common workflow is **one** permuted run (re-running a full pipeline at scale
is infeasible), so pick ``--swaps`` to scramble the labels as much as you want
that single null to be — the default (half the pairs) is a solid starting point.

Examples
--------
    # DRIDO, one comparison pair, half the pairs swapped (default)
    alleleflux-permute-metadata --input md.tsv --output perm.tsv \\
        --unit subjectID --groups 1D AL --seed 7

    # antibiotic (cage IS the 'replicate' column): swap 5 cage pairs
    alleleflux-permute-metadata --input md.tsv --output perm.tsv \\
        --unit replicate --swaps 5

    # diet study: flip 4 matched cage-pair blocks
    alleleflux-permute-metadata --input md.tsv --output perm.tsv \\
        --unit cage --block replicate --swaps 4
"""

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

GROUP_COL = "group"


def permute_group_labels(
    df: pd.DataFrame,
    unit_col: str = "subjectID",
    block_col: Optional[str] = None,
    seed: int = 0,
    swaps: Optional[int] = None,
) -> pd.DataFrame:
    """Swap the group labels of ``swaps`` unit-pairs, preserving per-group counts.

    The single permutation primitive.  A "swap" exchanges the group labels of two
    ``unit_col`` units that are in different groups; only the ``group`` column
    changes, and a unit's new group is broadcast to ALL of its rows.

    * ``block_col is None`` → **global** swap: ``swaps`` cross-group unit-pairs
      are formed (one unit from each group) and exchanged.  Needs exactly two
      groups in scope (independent designs: DRIDO subjects, antibiotic cages).
    * ``block_col`` given → **within-block** swap: each block is a matched pair
      (two single-group units, one per group); ``swaps`` randomly chosen blocks
      are flipped (matched-pair designs: diet cages per replicate, soil samples
      per plot).

    Each swap moves one unit each way, so per-group unit counts are preserved by
    construction.  ``swaps`` defaults to **half** the available pairs (blocks, or
    the smaller group's size for a global swap).

    Worked example — ``block_col="replicate"``, ``unit_col="cage"`` (diet study).
    Block 1 holds cages {control1 (control), fat1 (fat)}; flipping it makes BOTH
    mice of control1 (all timepoints) 'fat' and both of fat1 'control'.  Other
    blocks are left alone unless also chosen.

    Parameters
    ----------
    df : pd.DataFrame
        Metadata with at least ``group`` and ``unit_col`` (and ``block_col`` when
        given).  The caller's frame is never mutated.
    unit_col : str
        Randomization-unit column.  Every value must be single-group.
    block_col : str, optional
        Matched-pair stratum to swap within.  Each unit must lie in one block,
        and each block must hold exactly two units (one per group).
    seed : int
        RNG seed (same seed → same swaps chosen).
    swaps : int, optional
        Number of unit-pairs to swap.  ``None`` → half the available pairs.

    Returns
    -------
    pd.DataFrame
        A new frame with permuted ``group`` labels; all other columns untouched.

    Raises
    ------
    ValueError
        If ``unit_col``/``block_col`` are missing, a unit spans >1 group or >1
        block, a global swap doesn't see exactly two groups, a block isn't a
        2-unit matched pair, or ``swaps`` is out of range.
    AssertionError
        If per-group unit counts change (a swap cannot do this; safety net).
    """
    if GROUP_COL not in df.columns:
        raise ValueError(f"Input metadata has no '{GROUP_COL}' column.")
    if unit_col not in df.columns:
        raise ValueError(
            f"--unit '{unit_col}' is not a column in the metadata "
            f"(present: {list(df.columns)})."
        )
    if block_col is not None and block_col not in df.columns:
        raise ValueError(
            f"--block '{block_col}' is not a column in the metadata "
            f"(present: {list(df.columns)})."
        )

    # A unit must be single-group, else "swap the unit's group" is ambiguous
    # (usually means the wrong --unit was chosen, e.g. a block column).
    units_groups = df.groupby(unit_col)[GROUP_COL].nunique()
    straddling = units_groups[units_groups > 1]
    if not straddling.empty:
        raise ValueError(
            f"--unit '{unit_col}' values spanning >1 group: "
            f"{straddling.index.tolist()}. Pick a single-group unit column "
            f"(e.g. the cage/sample/subject), not a block."
        )

    # A unit must live in exactly one block.
    if block_col is not None:
        units_blocks = df.groupby(unit_col)[block_col].nunique()
        spanning = units_blocks[units_blocks > 1]
        if not spanning.empty:
            raise ValueError(
                f"--unit '{unit_col}' values spanning >1 --block '{block_col}': "
                f"{spanning.index.tolist()}. Each unit must sit in one block."
            )

    # unit -> its single group (the starting assignment we mutate in place).
    unit_group = df.groupby(unit_col, sort=False)[GROUP_COL].first()
    new_group = unit_group.to_dict()
    rng = np.random.default_rng(seed)

    if block_col is not None:
        # Within-block: each block is a matched PAIR; a swap flips one block.
        units_per_block = df.groupby(block_col)[unit_col].nunique()
        if not (units_per_block == 2).all():
            offenders = units_per_block[units_per_block != 2].to_dict()
            raise ValueError(
                f"--block '{block_col}' must hold exactly 2 '{unit_col}' units per "
                f"block (a matched pair); offending blocks: {offenders}. "
                f"Try a different --unit, or drop --block for a global swap."
            )
        blocks = df[block_col].drop_duplicates().to_numpy()
        n_pairs = len(blocks)
        if swaps is None:
            swaps = n_pairs // 2
        if not 0 <= swaps <= n_pairs:
            raise ValueError(
                f"--swaps must be in [0, {n_pairs}] (number of '{block_col}' "
                f"blocks), got {swaps}."
            )
        for b in rng.choice(blocks, size=swaps, replace=False):
            u0, u1 = df.loc[df[block_col] == b, unit_col].drop_duplicates().tolist()
            new_group[u0], new_group[u1] = unit_group[u1], unit_group[u0]
        logger.info(
            f"Swapped {swaps} of {n_pairs} '{block_col}' matched-pair blocks "
            f"(unit='{unit_col}')."
        )
    else:
        # Global: form `swaps` cross-group unit-pairs and exchange them.
        groups_in_scope = unit_group.unique().tolist()
        if len(groups_in_scope) != 2:
            raise ValueError(
                f"A global swap needs exactly 2 groups, found "
                f"{sorted(groups_in_scope)}. Use --groups to pick a comparison "
                f"pair, or --block for a matched-pair design."
            )
        g0, g1 = groups_in_scope
        units_g0 = unit_group[unit_group == g0].index.to_numpy()
        units_g1 = unit_group[unit_group == g1].index.to_numpy()
        max_swaps = min(len(units_g0), len(units_g1))
        if swaps is None:
            swaps = max_swaps // 2
        if not 0 <= swaps <= max_swaps:
            raise ValueError(
                f"--swaps must be in [0, {max_swaps}] (size of the smaller of "
                f"'{g0}'/'{g1}'), got {swaps}."
            )
        for u0, u1 in zip(
            rng.choice(units_g0, size=swaps, replace=False),
            rng.choice(units_g1, size=swaps, replace=False),
        ):
            new_group[u0], new_group[u1] = g1, g0
        logger.info(
            f"Swapped {swaps} of max {max_swaps} '{g0}'<->'{g1}' unit pairs "
            f"(unit='{unit_col}')."
        )

    out = df.copy()
    out[GROUP_COL] = out[unit_col].map(new_group)

    # Safety net: a swap cannot change the per-group unit counts.
    before = unit_group.value_counts().sort_index()
    after = out.groupby(unit_col)[GROUP_COL].first().value_counts().sort_index()
    assert before.equals(after), "per-unit group counts changed — this is a bug!"
    return out


def permute_metadata(
    df: pd.DataFrame,
    unit: str = "subjectID",
    block: Optional[str] = None,
    seed: int = 0,
    groups: Optional[list] = None,
    swaps: Optional[int] = None,
) -> pd.DataFrame:
    """Permute group labels, optionally restricted to one comparison pair.

    Thin wrapper over :func:`permute_group_labels` that adds the per-comparison
    ``groups`` restriction.  When ``groups`` is given, only the rows whose
    ``group`` is in ``groups`` are permuted (among themselves); every other row
    is returned untouched, and the permuted ``group`` labels are written back
    into the original rows BY INDEX so ``sample_id`` row order is preserved
    (downstream relabeling joins on ``sample_id``).

    Example — DRIDO has 5 groups; build the null for "1D vs AL" by permuting only
    those subjects::

        permute_metadata(df, unit="subjectID", seed=7, groups=["1D", "AL"])
        # 2D/20/40 rows come back unchanged; 1D/AL labels swapped among the
        # 1D∪AL subjects (counts of 1D and AL preserved).

    Parameters mirror :func:`permute_group_labels`, plus ``groups`` (an optional
    2-element list restricting the permutation to that pair).
    """
    if not groups:
        return permute_group_labels(
            df, unit_col=unit, block_col=block, seed=seed, swaps=swaps,
        )

    if GROUP_COL not in df.columns:
        raise ValueError(f"Input metadata has no '{GROUP_COL}' column.")
    mask = df[GROUP_COL].isin(groups)
    n_subset = int(mask.sum())
    if n_subset == 0:
        raise ValueError(
            f"No samples found for groups {groups}. Present groups: "
            f"{sorted(df[GROUP_COL].unique())}."
        )
    logger.info(
        f"Restricting permutation to groups {groups}: {n_subset} of {len(df)} rows."
    )

    permuted_subset = permute_group_labels(
        df[mask].copy(), unit_col=unit, block_col=block, seed=seed, swaps=swaps,
    )
    result = df.copy()
    result.loc[mask, GROUP_COL] = permuted_subset[GROUP_COL]
    return result


def main():
    """Parse arguments and write a permuted metadata TSV."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Shuffle group labels in a metadata TSV to build an AlleleFlux "
            "null/control. One operation: swap --swaps unit-pairs (preserving "
            "counts), within each --block when a block is given."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input metadata TSV."
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the permuted metadata TSV."
    )
    parser.add_argument(
        "--unit",
        type=str,
        default="subjectID",
        metavar="COLUMN",
        help=(
            "Randomization-unit column — the thing assigned to a group, which the "
            "swap relabels. Each value must be single-group. 'subjectID' for "
            "independent subjects; the cage/plot column when subjects share a "
            "single-group unit (e.g. a cage of mice → --unit cage)."
        ),
    )
    parser.add_argument(
        "--block",
        type=str,
        default=None,
        metavar="COLUMN",
        help=(
            "Optional matched-pair stratum to swap WITHIN. Each block must hold "
            "exactly two units (one per group); a swap flips one block. Omit for "
            "a global swap (which needs exactly two groups in scope)."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs=2,
        metavar=("GROUP1", "GROUP2"),
        default=None,
        help=(
            "Restrict the permutation to these two groups (one comparison pair); "
            "other groups are left untouched. Omit to permute the whole sheet."
        ),
    )
    parser.add_argument(
        "--swaps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of unit-pairs to swap (one unit from each group per swap). "
            "With --block, the number of matched-pair blocks to flip. Defaults to "
            "half the available pairs (blocks, or the smaller group's size)."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    logger.info(f"Reading input metadata: {args.input}")
    df = pd.read_csv(args.input, sep="\t")

    permuted_df = permute_metadata(
        df,
        unit=args.unit,
        block=args.block,
        seed=args.seed,
        groups=args.groups,
        swaps=args.swaps,
    )

    logger.info(f"Writing permuted metadata to: {args.output}")
    permuted_df.to_csv(args.output, index=False, sep="\t")
    logger.info("Done.")


if __name__ == "__main__":
    main()
