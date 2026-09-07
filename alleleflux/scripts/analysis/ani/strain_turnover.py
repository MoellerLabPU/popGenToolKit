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
confirmed with Andy (2026-09-01); ``min_compared`` (10 %) is the
minimum fraction of the genome both samples must cover at ``min_cov`` before
either verdict is trusted -- below it, or when popANI is NaN because nothing was
compared, both flags stay blank and ``background`` reads "undetermined".

Worked mouse (MRGM_0841, mouse DO-1D-3034, samples DO_1D_3034_023w -> _044w,
transition 5mo -> 10mo, real values):
    compared 3,325,951 bases = 78 % of the genome  -> determined
    conANI 0.999799  <  0.999?   no  -> dominant_strain_change = False
    popANI 0.999833  <  0.99999? yes -> strain_replacement      = True
    background = "strain_replacement"; frequency_shift = 0.000034
i.e. the same majority strain is still in charge, but 555 positions became
fixed differences over five months.
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from alleleflux.scripts.analysis.ani.classify import (
    BASES,
    classify_positions,
    consensus_mask,
    credible_alleles,
    presence_matrix,
)

logger = logging.getLogger(__name__)

# Columns this module ADDS to the pair table, in output order.
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
    determined = (
        (called["percent_genome_compared"] >= min_compared)
        & (called["compared_bases_count"] > 0)
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
# De novo scanner: "which alleles at t2 were not there at t1?"
# ---------------------------------------------------------------------------

# Keys of the dict ``scan_for_new_alleles`` returns, in output order.  Every
# value is an array with one entry per candidate (rows for the two count arrays).
SCAN_KEYS = (
    "position", "base",
    "t1_reads", "t1_threshold", "t1_evidence", "freq_t1",
    "t2_reads", "t2_threshold", "freq_t2", "t2_consensus",
    "fully_replaced",
    "counts_t1_rows", "coverage_t1", "counts_t2_rows", "coverage_t2",
)

# ``t1_evidence`` tiers.  ABSENT is the headline de novo count; BELOW_DETECTION
# rows are reported and labelled but NOT counted as de novo (user decision
# 2026-09-03: keep the marginal cases visible without over-reporting).
T1_ABSENT = "absent"
T1_BELOW_DETECTION = "below_detection"


def _empty_scan() -> dict[str, np.ndarray]:
    """The zero-candidate result: every key present, every array length 0."""
    return {
        "position": np.array([], np.int64),
        "base": np.array([], dtype=object),
        "t1_reads": np.array([], np.int64),
        "t1_threshold": np.array([], np.int64),
        "t1_evidence": np.array([], dtype=object),
        "freq_t1": np.array([], np.float64),
        "t2_reads": np.array([], np.int64),
        "t2_threshold": np.array([], np.int64),
        "freq_t2": np.array([], np.float64),
        "t2_consensus": np.array([], bool),
        "fully_replaced": np.array([], bool),
        "counts_t1_rows": np.empty((0, 4), np.int64),
        "coverage_t1": np.array([], np.int64),
        "counts_t2_rows": np.empty((0, 4), np.int64),
        "coverage_t2": np.array([], np.int64),
    }


def scan_for_new_alleles(
    counts_t1: npt.NDArray,
    counts_t2: npt.NDArray,
    model: npt.NDArray[np.int32],
    min_freq: float,
    min_cov: int,
) -> dict[str, np.ndarray]:
    """Find alleles present at t2 that baseline (t1) gave no evidence for.

    Parameters
    ----------
    counts_t1, counts_t2
        Dense ``(contig_length, 4)`` A/C/G/T read counts for ONE mouse's
        baseline and later sample on ONE contig (``profile_io.dense_contig_counts``
        output; uint16, zero rows where uncovered).
    model
        ``null_model.build_error_model`` array: ``model[coverage]`` = minimum reads
        for an allele to be more than sequencing error at that depth (3 for
        coverage 5..99 at Q30 / FDR 1e-6).
    min_freq
        Frequency floor for presence (0.05 = 5 %).
    min_cov
        Only positions with >= this many reads in BOTH samples are scanned;
        the same gate the ANI engine uses (5 by default).

    Returns
    -------
    A dict of aligned arrays (``SCAN_KEYS``), one entry per candidate
    (position, base):
      position, base ("A"/"C"/"G"/"T");
      t1 side: t1_reads, t1_threshold (the bar t1 had to clear), t1_evidence
      ("absent" = 0 reads at t1, "below_detection" = some reads but under the
      bar or under min_freq), freq_t1;
      t2 side: t2_reads, t2_threshold (the bar it DID clear -- the margin is
      t2_reads - t2_threshold), freq_t2, t2_consensus (the new allele is now
      the majority, old allele possibly still around);
      fully_replaced (population SNP AND t2_consensus: swept in, old allele
      gone); plus the raw count rows and coverages of both samples.
    Being in this table = candidate (present at t2, not credible at t1).  No
    single column says "de novo": that verdict is made downstream by combining
    t1_evidence with the mouse's strain background (Task 6 / the annotator).

    The rule -- a base is a candidate if PRESENT at t2 and NOT CREDIBLE at t1
    (credible = consensus set OR present).  Credible rather than bare presence
    is deliberate: a thin t1 majority such as [2,1,1,1] at 5x (A has 2 reads
    against a bar of 3) fails presence but IS A's position at t1; calling it
    absent would report A at t2 as newly arisen.  The flip side is labelled, not
    hidden: a t1 minor one read below the bar IS "not credible", and its row
    carries t1_reads and t1_threshold so a reader sees "1 read short".

    Worked examples (min_freq 0.05, min_cov 5, bar 3 at 30x):
      t1 [30,0,0,0] -> t2 [24,6,0,0]: C candidate, t1_reads 0, "absent",
          t2_reads 6 vs threshold 3, freq_t2 0.20, t2_consensus False,
          fully_replaced False (A is still consensus).
      t1 [30,0,0,0] -> t2 [5,25,0,0]: C candidate, t2_consensus True but
          fully_replaced False (A still present at 17 %).
      t1 [30,0,0,0] -> t2 [0,28,0,0]: C candidate, "absent", fully_replaced True.
      t1 [29,1,0,0] -> t2 [24,6,0,0]: C candidate, t1_reads 1, threshold 3,
          "below_detection".
      t1 [20,5,0,0] -> t2 [18,9,0,0]: nothing (C was present at t1).
      t1 [2,1,1,1]  -> t2 [30,0,0,0]: nothing (A was consensus at t1).
    """
    # Coverage = A+C+G+T per position (int64 so 4 x uint16 can't overflow).
    cov1 = counts_t1.sum(axis=1, dtype=np.int64)
    cov2 = counts_t2.sum(axis=1, dtype=np.int64)
    # Same depth gate as the ANI engine: both samples must clear min_cov.
    compared = np.flatnonzero((cov1 >= min_cov) & (cov2 >= min_cov))
    if not len(compared):
        return _empty_scan()

    # Slice down to the compared rows once; everything below is aligned to them.
    sub1 = counts_t1[compared].astype(np.int64)
    sub2 = counts_t2[compared].astype(np.int64)
    c1, c2 = cov1[compared], cov2[compared]

    # THE RULE: present at t2 AND not credible at t1  (see docstring).
    new_allele = presence_matrix(sub2, c2, model, min_freq) & ~credible_alleles(sub1, c1, model, min_freq)
    rows, bases = np.nonzero(new_allele)      # rows index into `compared`; bases 0..3 = A,C,G,T
    if not len(rows):
        return _empty_scan()

    # fully_replaced needs the pair-level population-SNP call (no shared credible
    # allele) AND the new base being t2's consensus: swept in, old allele gone.
    _, population_snp = classify_positions(sub1, c1, sub2, c2, model, min_freq)
    in_t2_consensus = consensus_mask(sub2)[rows, bases]

    t1_reads = sub1[rows, bases]
    t2_reads = sub2[rows, bases]
    # The bar t1 had to clear at its depth; clip like presence_matrix does so
    # ultra-deep positions reuse the deepest tabulated threshold.
    t1_threshold = model[np.clip(c1[rows], 0, len(model) - 1)].astype(np.int64)
    t2_threshold = model[np.clip(c2[rows], 0, len(model) - 1)].astype(np.int64)
    # Tier: zero reads is the clean de novo signal; any reads at all means the
    # allele was there but under the bar / under min_freq.
    t1_evidence = np.where(t1_reads == 0, T1_ABSENT, T1_BELOW_DETECTION).astype(object)

    return {
        "position": compared[rows],
        # Column index -> letter, so consumers never need the A,C,G,T convention.
        "base": np.asarray(BASES, dtype=object)[bases],
        "t1_reads": t1_reads,
        "t1_threshold": t1_threshold,
        "t1_evidence": t1_evidence,
        "freq_t1": t1_reads / c1[rows],
        "t2_reads": t2_reads,
        "t2_threshold": t2_threshold,
        "freq_t2": t2_reads / c2[rows],
        "t2_consensus": in_t2_consensus,
        "fully_replaced": population_snp[rows] & in_t2_consensus,
        "counts_t1_rows": sub1[rows],
        "coverage_t1": c1[rows],
        "counts_t2_rows": sub2[rows],
        "coverage_t2": c2[rows],
    }


def alleles_present_at(
    counts: npt.NDArray,
    positions: npt.NDArray[np.int64],
    model: npt.NDArray[np.int32],
    min_freq: float,
) -> np.ndarray:
    """Bare PRESENCE of each base at the named positions of one sample.

    Parameters
    ----------
    counts
        Dense ``(contig_length, 4)`` counts for one sample on one contig.
    positions
        0-based positions to look up (e.g. the BH-significant sites of a contig).
    model, min_freq
        As in ``scan_for_new_alleles``.

    Returns
    -------
    Boolean ``(len(positions), 4)``: ``[i, b]`` is True when base ``b`` clears
    the null-model bar AND ``min_freq`` at ``positions[i]``.  A zero-coverage
    position gives an all-False row (nothing present, no crash).  Used by the
    site annotator to ask "was the rising base present at baseline?".

    Example: counts row [24, 6, 0, 0] at 30x -> [True, True, False, False].
    """
    sub = counts[positions].astype(np.int64)
    cov = sub.sum(axis=1)                 # zero rows stay zero; presence_matrix guards the divide
    return presence_matrix(sub, cov, model, min_freq)

