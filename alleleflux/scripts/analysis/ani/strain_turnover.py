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

import argparse
import logging
import multiprocessing
import os

import numpy as np
import numpy.typing as npt
import pandas as pd

from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.ani.pairwise_ani import parse_transitions
from alleleflux.scripts.analysis.ani.profile_io import (
    contig_lengths_for_mag,
    dense_contig_counts,
    load_profile,
    profile_path,
)
from alleleflux.scripts.utilities.logging_config import setup_logging
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
# De novo scanner: "which alleles at t2 were not there at t1?"
# ---------------------------------------------------------------------------

# Keys of the dict ``scan_for_new_alleles`` returns, in output order.  Every
# value is an array with one entry per candidate (rows for the two count arrays).
SCAN_KEYS = (
    "position",
    "base",
    "t1_reads",
    "t1_threshold",
    "t1_evidence",
    "freq_t1",
    "t2_reads",
    "t2_threshold",
    "freq_t2",
    "t2_consensus",
    "fully_replaced",
    "counts_t1_rows",
    "coverage_t1",
    "counts_t2_rows",
    "coverage_t2",
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
    new_allele = presence_matrix(sub2, c2, model, min_freq) & ~credible_alleles(
        sub1, c1, model, min_freq
    )
    rows, bases = np.nonzero(
        new_allele
    )  # rows index into `compared`; bases 0..3 = A,C,G,T
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
    cov = sub.sum(axis=1)  # zero rows stay zero; presence_matrix guards the divide
    return presence_matrix(sub, cov, model, min_freq)


# ---------------------------------------------------------------------------
# The ``alleleflux-strain-turnover`` command: one MAG -> three files
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
    "n_absent",
    "n_below_detection",
    "n_de_novo",  # blank unless the mouse was scanned
    "min_compared",
    "pop_threshold",
    "con_threshold",
    "min_cov",  # provenance, stamped per row
]

# {mag}_de_novo_candidates.tsv.gz -- one row per (mouse, transition, position, base).
CANDIDATE_COLUMNS = [
    "MAG_ID",
    "subjectID",
    "replicate",
    "group",
    "transition",
    "sample_t1",
    "sample_t2",
    "contig",
    "position",
    "gene_id",
    "base",
    "t1_reads",
    "t1_threshold",
    "t1_evidence",
    "freq_t1",
    "t2_reads",
    "t2_threshold",
    "freq_t2",
    "t2_consensus",
    "fully_replaced",
    "A_t1",
    "C_t1",
    "G_t1",
    "T_t1",
    "coverage_t1",
    "A_t2",
    "C_t2",
    "G_t2",
    "T_t2",
    "coverage_t2",
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
    "n_absent",
    "n_below_detection",
    "n_de_novo",
    "de_novo_per_mb",
]

# Which mice get scanned: only a stable strain background makes "absent at
# baseline" mean "possibly new" rather than "brought in by the incoming strain"
# (measured on MRGM_0841: a replaced mouse yields ~15x more candidates per Mb).
SCANNED_BACKGROUNDS = ("stable",)


def _scan_one_pair(job: tuple) -> tuple[str, str, str, dict, pd.DataFrame]:
    """Worker: load one mouse's two profiles, scan every contig, return its candidates.

    Module-level so it pickles into ``multiprocessing.Pool``.  ``job`` =
    (subject, transition, sample_t1, sample_t2, path_t1, path_t2, contig_lengths,
    model, min_freq, min_cov).  Returns ``(subject, transition, sample_t2, counts,
    candidates)`` where ``counts`` = {n_absent, n_below_detection, n_de_novo} and
    ``candidates`` has the per-position columns of ``CANDIDATE_COLUMNS`` (the
    mouse-level columns are attached by the caller).

    Example: t1 = 20 A at positions 0..5, t2 = same except position 2 is 14 A +
    6 C -> one candidate row (c1, 2, "C", absent, freq_t2 0.30), counts
    {1, 0, 1}.
    """
    (
        subject,
        transition,
        sample_t1,
        sample_t2,
        path_t1,
        path_t2,
        contig_lengths,
        model,
        min_freq,
        min_cov,
    ) = job
    # gene_id is needed for the candidate rows, so opt in on both loads.
    profile_t1 = load_profile(path_t1, include_gene_id=True)
    profile_t2 = load_profile(path_t2, include_gene_id=True)
    dense_t1 = dense_contig_counts(profile_t1, contig_lengths)
    dense_t2 = dense_contig_counts(profile_t2, contig_lengths)

    frames = []
    for contig in contig_lengths:  # .fai order -> deterministic output
        hit = scan_for_new_alleles(
            dense_t1[contig], dense_t2[contig], model, min_freq, min_cov
        )
        if not len(hit["position"]):
            continue
        frame = pd.DataFrame(
            {
                "contig": contig,
                "position": hit["position"],
                "base": hit["base"],
                "t1_reads": hit["t1_reads"],
                "t1_threshold": hit["t1_threshold"],
                "t1_evidence": hit["t1_evidence"],
                "freq_t1": hit["freq_t1"],
                "t2_reads": hit["t2_reads"],
                "t2_threshold": hit["t2_threshold"],
                "freq_t2": hit["freq_t2"],
                "t2_consensus": hit["t2_consensus"],
                "fully_replaced": hit["fully_replaced"],
            }
        )
        # Unpack the raw count rows into named columns for both sides.
        for side, rows, cov in (
            ("t1", hit["counts_t1_rows"], hit["coverage_t1"]),
            ("t2", hit["counts_t2_rows"], hit["coverage_t2"]),
        ):
            for idx, base in enumerate(BASES):
                frame[f"{base}_{side}"] = rows[:, idx]
            frame[f"coverage_{side}"] = cov
        frames.append(frame)

    if frames:
        candidates = pd.concat(frames, ignore_index=True)
        # gene_id from the t2 profile: a candidate is PRESENT at t2, so t2 covers
        # the position and carries its gene annotation ("" when intergenic).
        genes = profile_t2[["contig", "position", "gene_id"]]
        candidates = candidates.merge(genes, on=["contig", "position"], how="left")
        candidates["gene_id"] = candidates["gene_id"].fillna("")
    else:
        candidates = pd.DataFrame(
            columns=[
                c
                for c in CANDIDATE_COLUMNS
                if c
                not in (
                    "MAG_ID",
                    "subjectID",
                    "replicate",
                    "group",
                    "transition",
                    "sample_t1",
                    "sample_t2",
                )
            ]
        )

    n_absent = (
        int((candidates["t1_evidence"] == T1_ABSENT).sum()) if len(candidates) else 0
    )
    n_below = (
        int((candidates["t1_evidence"] == T1_BELOW_DETECTION).sum())
        if len(candidates)
        else 0
    )
    counts = {
        "n_absent": n_absent,
        "n_below_detection": n_below,
        "n_de_novo": n_absent + n_below,
    }
    return subject, transition, sample_t2, counts, candidates


def build_rollup(turnover: pd.DataFrame, mag_id: str) -> pd.DataFrame:
    """Per (group, transition) summary of the turnover table; groups never pooled.

    Parameters
    ----------
    turnover
        The finished per-mouse table (``TURNOVER_COLUMNS``), candidate counts
        already attached (NaN for unscanned mice).
    mag_id
        Stamped into every row.

    Returns
    -------
    ``ROLLUP_COLUMNS``: the five background buckets (they sum to ``n_pairs``),
    the median overlap, the candidate tier totals over the scanned mice, and
    ``de_novo_per_mb`` = n_de_novo / (compared bases of the scanned mice) x 1e6
    (NaN when no mouse in the group was scanned).

    Example: group fat, one stable mouse with 5 compared bases and 1 absent
    candidate -> n_pairs 1, n_stable 1, n_de_novo 1, de_novo_per_mb 200,000.
    """
    if turnover.empty:
        return pd.DataFrame(columns=ROLLUP_COLUMNS)
    rows = []
    for (group, transition), sub in turnover.groupby(
        ["group", "transition"], sort=True
    ):
        label = sub["background"]
        scanned = sub[label.isin(SCANNED_BACKGROUNDS)]
        compared_scanned = float(scanned["compared_bases_count"].sum())
        n_de_novo = float(scanned["n_de_novo"].sum())
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
                "n_absent": int(scanned["n_absent"].sum()),
                "n_below_detection": int(scanned["n_below_detection"].sum()),
                "n_de_novo": int(n_de_novo),
                # Rate over the genome actually scanned, not the reference length.
                "de_novo_per_mb": (
                    n_de_novo / compared_scanned * 1e6 if compared_scanned else np.nan
                ),
            }
        )
    return pd.DataFrame(rows, columns=ROLLUP_COLUMNS)


def chase_the_strains(args: argparse.Namespace) -> int:
    """Orchestrator: pair table -> per-mouse calls -> scan stable mice -> three files.

    Each numbered step is one library call in data-flow order.  Returns the
    process exit code; contract violations raise.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    mag = args.mag

    # ---- 1. The pair table this MAG's ANI run produced (ids as str: DRiDO groups are numeric).
    pair_path = args.pair_table
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
    pairs = pd.read_csv(pair_path, sep="\t", dtype=str_cols)
    logger.info(f"MAG {mag}: {len(pairs)} pairs from {pair_path}")

    # ---- 2. Guard: the scan's depth gate must be the one the ANI was computed with,
    # otherwise "compared" means different things in the two tables.
    if len(pairs):
        stamped = pairs["min_cov"].unique()
        if len(stamped) != 1 or int(stamped[0]) != args.min_cov:
            raise ValueError(
                f"--min_cov {args.min_cov} does not match the pair table's min_cov {stamped.tolist()}"
            )

    # ---- 3. Per-mouse verdicts (Task 3).
    transitions = parse_transitions(args.transitions)
    called = call_transitions(
        pairs, transitions, args.min_compared, args.pop_threshold, args.con_threshold
    )

    # ---- 4. Scan the stable mice, one worker per pair (each loads its own two profiles).
    contig_lengths = contig_lengths_for_mag(args.fasta, args.mag_mapping, mag)
    model = build_error_model(min_base_quality=args.min_base_quality, fdr=args.fdr)
    to_scan = called[called["background"].isin(SCANNED_BACKGROUNDS)]
    jobs = [
        (
            row.subjectID_1,
            row.transition,
            row.sample_t1,
            row.sample_t2,
            profile_path(args.profiles_dir, row.sample_t1, mag),
            profile_path(args.profiles_dir, row.sample_t2, mag),
            contig_lengths,
            model,
            args.min_freq,
            args.min_cov,
        )
        for row in to_scan.itertuples(index=False)
    ]
    logger.info(
        f"scanning {len(jobs)} stable mouse-transitions of {len(called)} with {args.cpus} workers"
    )
    counts_by_key: dict[tuple[str, str], dict] = {}
    candidate_frames = []
    if jobs:
        with multiprocessing.Pool(processes=args.cpus) as pool:
            for (
                subject,
                transition,
                sample_t2,
                counts,
                candidates,
            ) in pool.imap_unordered(_scan_one_pair, jobs):
                counts_by_key[(subject, transition)] = counts
                if len(candidates):
                    # Attach the mouse-level context the worker didn't have.
                    meta = called[
                        (called["subjectID_1"] == subject)
                        & (called["transition"] == transition)
                    ].iloc[0]
                    for col, val in (
                        ("MAG_ID", mag),
                        ("subjectID", subject),
                        ("replicate", meta.replicate_1),
                        ("group", meta.group_1),
                        ("transition", transition),
                        ("sample_t1", meta.sample_t1),
                        ("sample_t2", meta.sample_t2),
                    ):
                        candidates[col] = val
                    candidate_frames.append(candidates)

    # ---- 5. Assemble the per-mouse table: Task 3 columns + counts + provenance.
    turnover = called.rename(
        columns={
            "subjectID_1": "subjectID",
            "replicate_1": "replicate",
            "group_1": "group",
        }
    )
    for col in ("n_absent", "n_below_detection", "n_de_novo"):
        turnover[col] = [
            counts_by_key.get((s, t), {}).get(col, np.nan)  # NaN = not scanned
            for s, t in zip(turnover["subjectID"], turnover["transition"])
        ]
    turnover["min_compared"] = args.min_compared
    turnover["pop_threshold"] = args.pop_threshold
    turnover["con_threshold"] = args.con_threshold
    turnover["min_cov"] = args.min_cov
    turnover["MAG_ID"] = mag
    turnover = (
        turnover[TURNOVER_COLUMNS]
        .sort_values(["transition", "group", "subjectID"])
        .reset_index(drop=True)
    )

    candidates = (
        pd.concat(candidate_frames, ignore_index=True)[CANDIDATE_COLUMNS].sort_values(
            ["transition", "subjectID", "contig", "position", "base"]
        )
        if candidate_frames
        else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    )
    rollup = build_rollup(turnover, mag)

    # ---- 6. Write.  Header-only files are the legitimate "nothing to report" outcome.
    turnover.to_csv(
        os.path.join(args.output_dir, f"{mag}_strain_turnover.tsv"),
        sep="\t",
        index=False,
    )
    candidates.to_csv(
        os.path.join(args.output_dir, f"{mag}_de_novo_candidates.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip",
    )
    rollup.to_csv(
        os.path.join(args.output_dir, f"{mag}_turnover_rollup.tsv"),
        sep="\t",
        index=False,
    )
    logger.info(
        f"MAG {mag}: {len(turnover)} mouse-transitions, backgrounds "
        f"{turnover['background'].value_counts().to_dict()}, {len(candidates)} candidate rows"
    )
    return 0


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Per-mouse strain-background calls and de novo allele scan for one MAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mag", required=True, help="MAG identifier")
    parser.add_argument(
        "--pair_table",
        required=True,
        help="This MAG's {mag}_pairwise_ani.tsv from alleleflux-pairwise-ani",
    )
    parser.add_argument(
        "--profiles_dir",
        required=True,
        help="Profiles root: {sample}/{sample}_{mag}_profiled.tsv.gz",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Reference FASTA (its .fai is used for contig lengths)",
    )
    parser.add_argument(
        "--mag_mapping", required=True, help="contig -> MAG mapping TSV"
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
    parser.add_argument(
        "--min_cov",
        type=int,
        default=5,
        help="Depth gate; must equal the pair table's min_cov",
    )
    parser.add_argument(
        "--min_freq", type=float, default=0.05, help="Presence frequency floor"
    )
    parser.add_argument(
        "--fdr", type=float, default=1e-6, help="Null-model false-discovery rate"
    )
    parser.add_argument(
        "--min_base_quality",
        type=int,
        default=30,
        help="Base quality assumed by the null model",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Worker processes for the scan",
    )
    args = parser.parse_args()
    return chase_the_strains(args)


if __name__ == "__main__":
    raise SystemExit(main())
