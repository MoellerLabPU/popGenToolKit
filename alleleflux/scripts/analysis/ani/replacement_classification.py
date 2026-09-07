"""Roll per-mouse strain-background calls up to one row per MAG.

``strain_turnover.call_transitions`` gives one verdict per (mouse, transition).
The hypergeometric enrichment test works per MAG, so it needs ONE answer per
(MAG, diet group, transition): "did most of the mice swap strains?"  If yes the
MAG's significant sites are more likely linked-haplotype noise from a strain
replacement than parallel evolution, and the MAG is excluded from enrichment.

Every output row carries the SAME roll-up twice, over two units:

* the **mouse block**     (``n_mice_with_call`` ... ``no_mouse_changed``): each
  mouse is one vote.
* the **replicate block** (``n_replicates_with_call`` ...): each replicate is one
  vote, and a replicate counts as changed if ANY of its mice changed -- Sam's
  diet-manipulation rule, where a replicate is a cage of several mice.  In DRiDO
  the metadata has no replicate column, ``mag_metadata.py`` fills it with the
  mouse id, and the two blocks are identical; no toggle needed, the columns
  simply agree.

Both blocks are computed for BOTH metrics -- ``strain_replacement`` (popANI) and
``dominant_strain_change`` (conANI) -- and the ``metric`` column names which one
a row is about, so no consumer can silently favour one.  Groups are NEVER pooled.

Worked example (MAG_A, group 40, 5mo -> 22mo, metric strain_replacement):
    cage c1 = {m1 True, m2 False, m3 False}, cage c2 = {m4 True}, cage c3 = {m5 undetermined}
    mice:       with_call 4, undetermined 1, changed 2 -> majority 2 > 4/2 False, any True, all False, none False
    replicates: with_call 2 (c3 has no called mouse), changed 2 (c1 via m1, c2 via m4)
                -> majority True, all True
Undetermined mice are reported but never in either denominator.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# The two per-mouse verdict columns produced by strain_turnover.call_transitions.
METRICS = ("strain_replacement", "dominant_strain_change")

KEYS = ["MAG_ID", "group", "transition"]

MOUSE_COLUMNS = (
    "n_mice_with_call",
    "n_mice_undetermined",
    "n_mice_changed",
    "majority_mice_changed",
    "any_mouse_changed",
    "all_mice_changed",
    "no_mouse_changed",
)
# any/none at replicate level would equal the mouse-level ones by construction
# (some mouse changed <=> some replicate changed), so only the counts, the
# majority and the all-changed verdicts are reported for replicates.
REPLICATE_COLUMNS = (
    "n_replicates_with_call",
    "n_replicates_changed",
    "majority_replicates_changed",
    "all_replicates_changed",
)


def _verdicts(called: pd.Series, changed: pd.Series) -> dict[str, pd.Series]:
    """Turn two counts per key into the four yes/no verdicts.

    Parameters
    ----------
    called
        Number of voters (mice, or replicates) that had a verdict, one value per
        (MAG_ID, group, transition) key.  E.g. ``4``.
    changed
        Number of those voters flagged as changed.  E.g. ``2``.

    Returns
    -------
    ``{"majority", "any", "all", "none"}`` -> boolean Series aligned with the
    inputs.  For called=4, changed=2: majority False (2 is not > 2), any True,
    all False, none False.  For called=0 (every mouse undetermined) all four are
    False: "no change" is a positive finding that needs evidence, not the
    absence of data.
    """
    return {
        # Strictly MORE than half of the called voters (1 of 2 is not a majority).
        "majority": changed > called / 2,
        "any": changed > 0,
        "all": (called > 0) & (changed == called),
        "none": (called > 0) & (changed == 0),
    }


def _classify_one_metric(turnover: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Roll one verdict column up to one row per (MAG_ID, group, transition).

    Parameters
    ----------
    turnover
        The ``strain_turnover.call_transitions`` output: one row per mouse per
        transition (970 rows for MRGM_0841 on DRiDO).  Columns used here:
        ``MAG_ID``, ``group_1``/``group_2``, ``replicate_1``/``replicate_2``,
        ``transition``, and the two nullable-boolean verdict columns
        ``strain_replacement`` and ``dominant_strain_change`` (True = changed,
        False = same strain, pd.NA = undetermined).  Everything else rides along
        unused.
    metric
        WHICH verdict column to count: ``"strain_replacement"`` (the popANI
        verdict) or ``"dominant_strain_change"`` (the conANI verdict).  Nothing
        else in the row is read as evidence.

    Returns
    -------
    One row per (MAG_ID, group, transition) with the columns
    ``KEYS + ["metric"] + MOUSE_COLUMNS + REPLICATE_COLUMNS``.  Diet groups are
    separate keys and never pooled.

    Worked example, metric="strain_replacement", MAG_A / group fat / pre_end,
    five mice in three cages::

        mouse  cage  strain_replacement
        m1     c1    True
        m2     c1    False
        m3     c1    False
        m4     c2    True
        m5     c3    <NA>

    mouse block:     n_mice_with_call 4, n_mice_undetermined 1, n_mice_changed 2
                     -> majority False, any True, all False, none False
    replicate block: c1 changed (via m1), c2 changed, c3 has no called mouse
                     -> n_replicates_with_call 2, n_replicates_changed 2
                     -> majority True, all True
    """
    # A pair is always within one mouse, so its two sides must share group and
    # replicate; anything else is a corrupt table, not something to average over.
    for side in ("group", "replicate"):
        mismatch = turnover[f"{side}_1"] != turnover[f"{side}_2"]
        if mismatch.any():
            raise ValueError(f"{int(mismatch.sum())} rows have {side}_1 != {side}_2")

    df = turnover.rename(columns={"group_1": "group", "replicate_1": "replicate"})
    # ``metric`` is the NAME of the verdict column, e.g. "strain_replacement";
    # ``flag`` is that column's values (True / False / <NA>), one per mouse.
    flag = df[metric].astype("boolean")
    # Per-mouse 0/1 helper columns that sum cleanly under groupby.
    df = df.assign(
        _called=flag.notna().astype(int),
        _undetermined=flag.isna().astype(int),
        _changed=flag.fillna(False).astype(int),
    )

    # --- mouse block: every row is one mouse, so sums are mouse counts.
    mice = df.groupby(KEYS, sort=True, observed=True)[["_called", "_undetermined", "_changed"]].sum()
    mouse_verdicts = _verdicts(mice["_called"], mice["_changed"])
    mice = mice.rename(columns={
        "_called": "n_mice_with_call",
        "_undetermined": "n_mice_undetermined",
        "_changed": "n_mice_changed",
    })
    mice["majority_mice_changed"] = mouse_verdicts["majority"]
    mice["any_mouse_changed"] = mouse_verdicts["any"]
    mice["all_mice_changed"] = mouse_verdicts["all"]
    mice["no_mouse_changed"] = mouse_verdicts["none"]

    # --- replicate block: first collapse mice to replicates (a replicate has a
    # call if ANY mouse has one, changed if ANY mouse changed), then count the
    # replicates exactly like mice above.
    per_replicate = (
        df.groupby(KEYS + ["replicate"], sort=True, observed=True)[["_called", "_changed"]]
        .max()
    )
    reps = per_replicate.groupby(KEYS, sort=True, observed=True).sum()
    rep_verdicts = _verdicts(reps["_called"], reps["_changed"])
    reps = reps.rename(columns={
        "_called": "n_replicates_with_call",
        "_changed": "n_replicates_changed",
    })
    reps["majority_replicates_changed"] = rep_verdicts["majority"]
    reps["all_replicates_changed"] = rep_verdicts["all"]

    out = mice.join(reps).reset_index()
    out.insert(len(KEYS), "metric", metric)
    return out[KEYS + ["metric"] + list(MOUSE_COLUMNS) + list(REPLICATE_COLUMNS)]


def classify_mags(turnover: pd.DataFrame) -> pd.DataFrame:
    """The enrichment filter's input: both metrics rolled up per MAG, stacked.

    Parameters
    ----------
    turnover
        The ``call_transitions`` output, one row per mouse per transition (see
        ``_classify_one_metric`` for the columns read).

    Returns
    -------
    ``_classify_one_metric`` run for each of ``METRICS`` and concatenated, so
    every (MAG_ID, group, transition) key appears TWICE: once with
    ``metric == "strain_replacement"`` and once with
    ``metric == "dominant_strain_change"``.  Consumers filter on ``metric``;
    there is deliberately no way to compute only one.

    Real output for MRGM_0841, group 2D, 5mo_10mo (12 called, 12 undetermined)::

        metric                  n_mice_changed  majority_mice_changed  all_mice_changed
        strain_replacement      12              True                   True
        dominant_strain_change   3              False                  False
    """
    stacked = pd.concat([_classify_one_metric(turnover, metric) for metric in METRICS], ignore_index=True)
    logger.info(f"MAG-level classification: {len(stacked) // len(METRICS)} keys x {len(METRICS)} metrics")
    return stacked
