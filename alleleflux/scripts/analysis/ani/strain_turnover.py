"""Per-mouse strain-background calls from the pairwise ANI table.

The pairwise ANI CLI answers "how similar are these two samples of a MAG?".
This module turns that into the question the DRiDO analysis actually asks:
"between the baseline (5mo) sample and a later sample of the SAME mouse, did the
MAG's population stay the same strain?"  Two metrics answer it on EQUAL footing,
each with its own column -- neither ranks above the other:

* ``strain_replacement``      popANI < ``pop_threshold`` (99.999 %).  popANI only
  counts positions where the two samples share NO credible allele, so falling
  below the line means fixed differences accumulated: the population was
  replaced (or a strain that was invisible at baseline took over).
* ``dominant_strain_change``  conANI < ``con_threshold`` (99.9 %).  conANI counts
  positions whose MAJORITY base differs, so falling below the line means the
  dominant strain swapped, even when the old strain is still detectably present.

Both thresholds come from the inStrain paper's same-strain conventions and were
confirmed with Andy (2026-09-01); ``min_compared`` (10 %, Sam's paper) is the
minimum fraction of the genome both samples must cover at ``min_cov`` before
either verdict is trusted -- below it, or when nothing was compared at all,
both flags stay blank and ``background`` reads "undetermined".

Worked mouse (MRGM_0841, mouse DO-1D-3034, samples DO_1D_3034_023w -> _044w,
transition 5mo -> 10mo, real values):
    compared 3,325,951 bases = 78 % of the genome  -> determined
    conANI 0.999799  <  0.999?   no  -> dominant_strain_change = False
    popANI 0.999833  <  0.99999? yes -> strain_replacement      = True
    background = "strain_replacement"; frequency_shift = 0.000034
i.e. the same majority strain is still in charge, but 555 positions became
fixed differences over five months.

The ``alleleflux-strain-turnover`` command (bottom of the file) runs this for
one MAG and writes the per-mouse table plus a per-group rollup; those feed the
enrichment filter (``replacement_classification``) and the significant-site
annotator.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.pairwise_ani import parse_transitions
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Columns ``call_transitions`` ADDS to the pair table, in output order.
CALL_COLUMNS = (
    "transition",
    "sample_t1",
    "sample_t2",
    "strain_replacement",
    "dominant_strain_change",
    "frequency_shift",
    "background",
)


def call_transitions(
    pair_table: pd.DataFrame,
    transitions: list[tuple[str, str]],
    min_compared: float,
    pop_threshold: float,
    con_threshold: float,
) -> pd.DataFrame:
    """One row per (mouse, transition) with both strain-background verdicts.

    Parameters
    ----------
    pair_table
        The ``*_pairwise_ani.tsv`` table: ``sample1 < sample2`` BY ID (not by
        time), with ``subjectID_1/2``, ``time_1/2``, ``conANI``, ``popANI`` and
        ``percent_genome_compared``.
    transitions
        ``[(earlier, later), ...]`` as parsed from ``--transitions 5mo:10mo``.
        Only same-mouse rows whose two timepoints are one of these survive.
    min_compared
        Overlap floor on ``percent_genome_compared`` (a fraction; 0.1 = 10 %).
    pop_threshold, con_threshold
        The same-strain lines for popANI and conANI.  Values >= the line are
        "same"; strictly below is a change.

    Returns
    -------
    The surviving rows plus ``CALL_COLUMNS``: ``transition`` ("5mo_10mo"),
    ``sample_t1``/``sample_t2`` oriented by TIME, the two nullable-boolean
    verdicts (pd.NA when undetermined), ``frequency_shift`` = popANI - conANI
    (how much of the divergence is the SAME alleles changing proportion rather
    than being replaced), and the ``background`` label.  An input with no
    matching rows returns an empty frame with the full schema.
    """
    for earlier, later in transitions:
        # "5mo:5mo" would pair a sample with itself -- a config typo, fail loud.
        if earlier == later:
            raise ValueError(f"Degenerate transition {earlier}:{later}")

    same_subject = pair_table["subjectID_1"] == pair_table["subjectID_2"]
    frames = []
    for earlier, later in transitions:
        # The engine sorts sample1 < sample2 by id, so a 5mo->10mo pair can sit
        # in the table either way round; accept both orientations here and fix
        # the direction below.
        forward = (pair_table["time_1"] == earlier) & (pair_table["time_2"] == later)
        backward = (pair_table["time_1"] == later) & (pair_table["time_2"] == earlier)
        rows = pair_table[same_subject & (forward | backward)].copy()
        if rows.empty:
            continue
        rows["transition"] = f"{earlier}_{later}"
        # Orient by TIME: sample_t1 is always the earlier sample.
        is_forward = rows["time_1"] == earlier
        rows["sample_t1"] = np.where(is_forward, rows["sample1"], rows["sample2"])
        rows["sample_t2"] = np.where(is_forward, rows["sample2"], rows["sample1"])
        frames.append(rows)

    if frames:
        called = pd.concat(frames, ignore_index=True)
        # A pair is ONE mouse at two times, so diet group and replicate must agree
        # on both sides; a mismatch is a metadata error (a mouse labelled with two
        # groups), and taking group_1 would silently file it under the wrong diet.
        for side in ("group", "replicate"):
            mismatch = called[f"{side}_1"] != called[f"{side}_2"]
            if mismatch.any():
                raise ValueError(
                    f"{int(mismatch.sum())} within-mouse pairs have {side}_1 != {side}_2 "
                    f"(e.g. {called.loc[mismatch, 'subjectID_1'].iloc[0]})"
                )
    else:
        # Keep the schema even when nothing matched so downstream joins and
        # header-only outputs keep working.
        called = pair_table.iloc[0:0].copy()
        for col in ("transition", "sample_t1", "sample_t2"):
            called[col] = pd.Series(dtype=object)

    # Evidence gate: too little shared genome means NEITHER verdict is
    # trustworthy -> both flags stay blank.  The second term matters only when a
    # user sets min_compared=0: the engine leaves BOTH ANIs NaN when zero bases
    # were compared, and a NaN must not slip through as a verdict.
    determined = (called["percent_genome_compared"] >= min_compared) & (
        called["compared_bases_count"] > 0
    )

    # Two INDEPENDENT verdicts.  The nullable "boolean" dtype lets
    # ``.where(determined)`` blank the undetermined rows to pd.NA instead of a
    # misleading False (plain bool has no missing value).
    replacement = (called["popANI"] < pop_threshold).astype("boolean").where(determined)
    dominant = (called["conANI"] < con_threshold).astype("boolean").where(determined)
    called["strain_replacement"] = replacement
    called["dominant_strain_change"] = dominant

    # popANI - conANI is >= 0 by construction (every population SNP is also a
    # consensus SNP): the gap is divergence explained by allele-FREQUENCY change
    # of alleles both samples still carry, not by replacement.
    called["frequency_shift"] = called["popANI"] - called["conANI"]

    # A human-readable label DERIVED from the two flags -- a groupby key, not a
    # ranking.  Order of the conditions only matters for the "both" case, which
    # must be tested before either single flag.
    rep = replacement.fillna(False).to_numpy(dtype=bool)
    dom = dominant.fillna(False).to_numpy(dtype=bool)
    called["background"] = np.select(
        [~determined.to_numpy(dtype=bool), rep & dom, rep, dom],
        [
            "undetermined",
            "strain_replacement+dominant_strain_change",
            "strain_replacement",
            "dominant_strain_change",
        ],
        default="stable",
    )

    logger.info(
        f"{len(called)} transition rows called: "
        f"{dict(called['background'].value_counts())}"
    )
    return called


# ---------------------------------------------------------------------------
# The ``alleleflux-strain-turnover`` command: one MAG -> two files
# ---------------------------------------------------------------------------

# {mag}_strain_turnover.tsv -- one row per (mouse, transition).
TURNOVER_COLUMNS = [
    "MAG_ID",
    "subjectID",
    "replicate",
    "group",
    "transition",
    "sample_t1",
    "sample_t2",
    "compared_bases_count",
    "percent_genome_compared",
    "conANI",
    "popANI",
    "frequency_shift",
    "strain_replacement",
    "dominant_strain_change",
    "background",
    "min_compared",
    "pop_threshold",
    "con_threshold",
    "min_cov",  # provenance, stamped per row
]

# {mag}_turnover_rollup.tsv -- one row per (group, transition); groups never pooled.
ROLLUP_COLUMNS = [
    "MAG_ID",
    "group",
    "transition",
    "n_pairs",
    "n_undetermined",
    "n_stable",
    "n_strain_replacement",
    "n_dominant_strain_change",
    "n_both",
    "median_percent_genome_compared",
]


def build_rollup(turnover: pd.DataFrame, mag_id: str) -> pd.DataFrame:
    """Per (group, transition) summary of the turnover table; groups never pooled.

    Parameters
    ----------
    turnover
        The finished per-mouse table (``TURNOVER_COLUMNS``).
    mag_id
        Stamped into every row.

    Returns
    -------
    ``ROLLUP_COLUMNS``: the five background buckets (mutually exclusive labels,
    so they sum to ``n_pairs``) and the median genome overlap over ALL the
    group's pairs, scanned or not.

    Example: group fat with mice {stable, stable, strain_replacement,
    undetermined} at overlaps {0.83, 1.0, 1.0, 0.02} -> n_pairs 4, n_stable 2,
    n_strain_replacement 1, n_undetermined 1, median 0.915.
    """
    if turnover.empty:
        return pd.DataFrame(columns=ROLLUP_COLUMNS)
    rows = []
    for (group, transition), sub in turnover.groupby(
        ["group", "transition"], sort=True
    ):
        label = sub["background"]
        rows.append(
            {
                "MAG_ID": mag_id,
                "group": group,
                "transition": transition,
                "n_pairs": len(sub),
                "n_undetermined": int((label == "undetermined").sum()),
                "n_stable": int((label == "stable").sum()),
                "n_strain_replacement": int((label == "strain_replacement").sum()),
                "n_dominant_strain_change": int(
                    (label == "dominant_strain_change").sum()
                ),
                "n_both": int(
                    (label == "strain_replacement+dominant_strain_change").sum()
                ),
                "median_percent_genome_compared": float(
                    sub["percent_genome_compared"].median()
                ),
            }
        )
    return pd.DataFrame(rows, columns=ROLLUP_COLUMNS)


def chase_the_strains(args: argparse.Namespace) -> int:
    """Orchestrator: pair table -> per-mouse calls -> two files.

    Each numbered step is one library call in data-flow order.  Returns the
    process exit code; contract violations raise.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    mag = args.mag

    # ---- 1. The pair table this MAG's ANI run produced (ids as str: DRiDO groups are numeric).
    str_cols = {
        c: str
        for c in (
            "sample1",
            "sample2",
            "subjectID_1",
            "subjectID_2",
            "group_1",
            "group_2",
            "time_1",
            "time_2",
            "replicate_1",
            "replicate_2",
        )
    }
    pairs = pd.read_csv(args.pair_table, sep="\t", dtype=str_cols)
    logger.info(f"MAG {mag}: {len(pairs)} pairs from {args.pair_table}")

    # ---- 2. Per-mouse verdicts.
    transitions = parse_transitions(args.transitions)
    called = call_transitions(
        pairs, transitions, args.min_compared, args.pop_threshold, args.con_threshold
    )

    # ---- 3. Assemble: plain names for the (identical) two sides + provenance.  min_cov
    # is the PAIR TABLE's stamped value -- this command never touches profiles, so
    # it has no min_cov of its own; it just carries the upstream one forward.
    turnover = called.rename(
        columns={
            "subjectID_1": "subjectID",
            "replicate_1": "replicate",
            "group_1": "group",
        }
    )
    turnover["min_compared"] = args.min_compared
    turnover["pop_threshold"] = args.pop_threshold
    turnover["con_threshold"] = args.con_threshold
    turnover["MAG_ID"] = mag
    turnover = (
        turnover[TURNOVER_COLUMNS]
        .sort_values(["transition", "group", "subjectID"])
        .reset_index(drop=True)
    )
    rollup = build_rollup(turnover, mag)

    # ---- 4. Write.  Header-only files are the legitimate "nothing to report" outcome.
    turnover.to_csv(
        os.path.join(args.output_dir, f"{mag}_strain_turnover.tsv"),
        sep="\t",
        index=False,
    )
    rollup.to_csv(
        os.path.join(args.output_dir, f"{mag}_turnover_rollup.tsv"),
        sep="\t",
        index=False,
    )
    logger.info(
        f"MAG {mag}: {len(turnover)} mouse-transitions, backgrounds "
        f"{turnover['background'].value_counts().to_dict()}"
    )
    return 0


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Per-mouse strain-background calls for one MAG from its pairwise ANI table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mag", required=True, help="MAG identifier")
    parser.add_argument(
        "--pair_table",
        required=True,
        help="This MAG's {mag}_pairwise_ani.tsv from alleleflux-pairwise-ani",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--transitions",
        nargs="+",
        required=True,
        metavar="EARLIER:LATER",
        help="Timepoint transitions to call, e.g. 5mo:10mo 5mo:16mo",
    )
    parser.add_argument(
        "--min_compared",
        type=float,
        default=0.1,
        help="Minimum fraction of the genome compared for a verdict",
    )
    parser.add_argument(
        "--pop_threshold",
        type=float,
        default=0.99999,
        help="popANI below this = strain_replacement",
    )
    parser.add_argument(
        "--con_threshold",
        type=float,
        default=0.999,
        help="conANI below this = dominant_strain_change",
    )
    args = parser.parse_args()
    return chase_the_strains(args)


if __name__ == "__main__":
    raise SystemExit(main())
