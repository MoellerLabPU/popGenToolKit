"""Was the significant allele already there at baseline?  (``alleleflux-baseline-presence``)

AlleleFlux's tests say *that* an allele won at a position -- its frequency
differs between groups at the later timepoint (divergence, the two-sample
tests) or changed within a group (parallelism, the single-sample test).  They
cannot say where the allele came from.  Two stories fit the same p-value:

* it was **standing variation** -- present in the mice before treatment and
  favoured by it (selection on existing diversity), or
* it **arose after** baseline, by mutation or by a different strain arriving.

This command tells them apart from the raw reads.  For every significant site
it takes the allele(s) the test flagged and, in EVERY sample of the comparison
(both groups, both timepoints), counts that allele's reads and applies the same
presence rule the ANI work uses: at least ``min_cov`` reads at the position to
say anything at all, then at least the null-model bar AND ``min_freq`` of reads
to call the allele present.  Each sample gets one of four verdicts::

    present          covered, allele clears the bar and min_freq
    below_detection  covered, 1+ reads of the allele but under the bar / min_freq
    absent           covered, zero reads of the allele
    not_covered      fewer than min_cov reads: no verdict, out of every denominator

Two outputs per (comparison, test):

* ``{comparison}_{family}_{statistic}_baseline_presence.tsv.gz`` -- one row per site x allele
  x sample: allele reads, total reads, detection bar, status, the mouse / replicate / group /
  timepoint.
* ``..._summary.tsv`` -- one row per site x allele: the any-mouse verdict with its
  counts (present in how many covered baseline samples, in how many replicates),
  the same-mouse counts (standing vs de novo per mouse), and the pooled reads per
  timepoint (total reads at the position over every sample, reads carrying the
  allele, their ratio) -- UNFILTERED, so "1,000 reads at baseline and the allele
  never seen" bounds its baseline frequency below 1/1,000.  Which framing is the
  paper's headline is a design question: littermates sharing a colony justify
  "any mouse"; outbred, half-cross-sectional DRiDO mice justify
  "same mouse" plus a stable strain background.

Column names and origin labels are spelled with the comparison's OWN timepoint
names (``pre`` / ``end`` for ``pre_end-fat_control``, ``5mo`` / ``22mo`` for
DRiDO), never a fixed t0/t1 vocabulary.  In this file "t0" and "t1" appear only
as placeholders in templates and as shorthand in comments for "the earlier /
the later timepoint of the comparison".

The ANI/strain work is OPTIONAL: ``--turnover_dir`` adds a ``strain_background``
column; without it the command runs on any AlleleFlux output.

Worked site (two replicates, two groups; comparison pre_end):
allele G at c1:10, pre reads m1 5/30, m2 0/30, m3 1/30, m4 0/2 -> present, absent,
below_detection, not_covered.  Summary: n_pre_samples_covered 3,
n_pre_samples_allele_present 1, origin_any_mouse standing_variation; per-mouse:
n_mice_standing_variation 1, n_mice_de_novo_candidate 1,
n_mice_de_novo_candidate_below_detection_at_pre 1; reads: total_reads_pre 92
(30+30+30+2, the thin m4 included), allele_reads_pre 6, allele_frequency_pre 0.065.
"""

import argparse
import glob
import logging
import multiprocessing
import os
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from alleleflux.scripts.analysis.ani.classify import BASES
from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.ani.profile_io import (
    contig_lengths_for_mag,
    dense_contig_counts,
    load_profile,
    profile_path,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

# The four per-sample verdicts (see module docstring).
EVIDENCE_PRESENT = "present"
EVIDENCE_BELOW_DETECTION = "below_detection"
EVIDENCE_ABSENT = "absent"
EVIDENCE_NOT_COVERED = "not_covered"



class Timepoints(NamedTuple):
    """The two timepoints of a comparison, and the one place their names get spelled.

    ``earlier`` / ``later`` are the labels exactly as the metadata and the
    comparison directory write them (``pre`` / ``end``, ``5mo`` / ``22mo``).
    Column names and origin labels are TEMPLATES holding ``{t0}`` / ``{t1}``;
    ``name()`` fills them, so every output name is derived from the comparison
    and no timepoint name is ever hard-coded (house rule).

    Example: ``Timepoints("pre", "end").name("n_{t0}_samples_covered")`` ->
    ``"n_pre_samples_covered"``; ``.name(ORIGIN_T1_NOT_COVERED)`` -> ``"end_not_covered"``.
    """

    earlier: str
    later: str

    def name(self, template: str) -> str:
        return template.format(t0=self.earlier, t1=self.later)


# Per-mouse origin verdict on later-timepoint rows (``label_origin_in_own_mouse``):
# what the allele's presence at t1 looks like against the SAME mouse's t0 sample.
# Templates: ``{t0}`` / ``{t1}`` are filled with the comparison's real timepoint
# names by ``Timepoints.name`` (``de_novo_candidate_below_detection_at_pre``).
ORIGIN_STANDING = "standing_variation"  # t1 present, t0 present
ORIGIN_DE_NOVO = "de_novo_candidate"  # t1 present, t0 covered and absent
ORIGIN_DE_NOVO_BELOW_DETECTION = "de_novo_candidate_below_detection_at_{t0}"  # t1 present, t0 had a few reads under the bar
ORIGIN_T0_NOT_COVERED = "{t0}_not_covered"  # t1 present, t0 too thin to say
ORIGIN_NO_T0_SAMPLE = "no_{t0}_sample"  # t1 present, mouse has no t0 sample
ORIGIN_BELOW_DETECTION_AT_T1 = (
    "allele_below_detection_at_{t1}"  # t1 covered, a few reads under the bar
)
ORIGIN_ABSENT_AT_T1 = "allele_absent_at_{t1}"  # t1 covered, zero reads of the allele
ORIGIN_T1_NOT_COVERED = "{t1}_not_covered"  # t1 too thin: nothing to explain
# Site-level only (origin_any_mouse): no covered t1 sample shows the allele at all.
ORIGIN_NOT_SEEN_AT_T1 = "allele_not_present_at_{t1}"

# ``--summary`` families -> the stem of the per-base p-value column in that family's
# source files.  A summary family fixes the file (``p_value_summary_{family}_*.tsv``),
# the source directory (``significance_tests/{family}_{comparison}/``) and the
# column convention; ``--test_type`` then picks rows within it exactly as the summary
# spells them (two_sample_paired_Wilcoxon, LMM_abs, ...).  CMH is not offered: it reports ONE p per site,
# never per base, so it cannot name the significant allele.
SUMMARY_FAMILIES = {
    # family: (column stem, group name appended to the column?)
    # single_sample files hold one block of columns PER GROUP (A_..._tTest_fat);
    # lmm_across_time files are one FILE per group with unsuffixed columns.
    "two_sample_paired": ("_frequency_p_value_", False),
    "two_sample_unpaired": ("_frequency_p_value_", False),
    "single_sample": ("_frequency_p_value_", True),
    "lmm": ("_p_value_", False),
    "lmm_across_time": ("_p_value_", False),
}

# Site identity shared by both outputs.
SITE_KEYS = [
    "mag_id",
    "contig",
    "position",
    "gene_id",
    "test_type",
    "group_analyzed",
    "allele",
]

LONG_COLUMNS = [
    "mag_id",
    "contig",
    "position",
    "gene_id",
    "test_type",
    "group_analyzed",
    "min_p_value",
    "q_value",
    "n_alleles_tied_at_min_p",
    "allele",
    "sample_id",
    "subjectID",
    "replicate",
    "group",
    "time",
    "allele_reads",
    "total_reads",
    "detection_threshold_reads",
    "allele_frequency",
    "allele_status",
    "allele_present",
    "origin_in_own_mouse",
    "strain_background",
    "min_cov",
    "min_freq",
]


# ---------------------------------------------------------------------------
# Pure steps
# ---------------------------------------------------------------------------


def parse_comparison(label: str) -> tuple[str, str, str, str]:
    """Split a comparison label into (earlier, later, group_a, group_b).

    AlleleFlux names comparison directories ``{tp1}_{tp2}-{g1}_{g2}``, e.g.
    ``pre_end-fat_control`` or ``5mo_22mo-40_AL``.  Exactly one ``-`` and one
    ``_`` on each side; anything else raises rather than guessing.
    """
    parts = label.split("-")
    if len(parts) != 2 or any(p.count("_") != 1 for p in parts):
        raise ValueError(f"comparison must look like tp1_tp2-g1_g2, got {label!r}")
    earlier, later = parts[0].split("_")
    group_a, group_b = parts[1].split("_")
    return earlier, later, group_a, group_b


def find_summary_file(summary_dir: str, family: str, period: str) -> str:
    """The one ``p_value_summary_{family}_{period}*.tsv`` in a comparison directory.

    Older runs name it ``..._{period}.tsv``, newer ones
    ``..._{period}-{groups}.tsv`` (DRiDO); the trailing ``*`` covers both.  The
    period is part of the pattern so that ``lmm`` never also matches
    ``lmm_across_time`` (same prefix).  Zero or two matches raise: no silent
    fallback to another family.

    Example: ("…/pre_end-fat_control", "lmm", "pre_end") ->
    ".../p_value_summary_lmm_pre_end.tsv" only.
    """
    pattern = os.path.join(summary_dir, f"p_value_summary_{family}_{period}*.tsv")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one file matching {pattern}, found {hits}; choose another --summary?"
        )
    return hits[0]


def allele_p_column(family: str, test_type: str):
    """Build the per-base p-value column name for one (family, test_type).

    Parameters
    ----------
    family
        A ``SUMMARY_FAMILIES`` key, e.g. ``"two_sample_paired"``.
    test_type
        The summary's ``test_type`` string, e.g. ``"two_sample_paired_Wilcoxon"``
        or ``"LMM_abs"``.  The family prefix (if any) is stripped to leave the
        statistic that names the column: ``Wilcoxon``, ``tTest_abs``, ``LMM``.

    Returns
    -------
    ``column(base, group) -> str``.  ``group`` is appended only for families
    whose files hold one column block per group (single_sample); families with
    one FILE per group (lmm_across_time) have plain column names.

    Examples: ("two_sample_unpaired", "two_sample_unpaired_MannWhitney_abs")
    -> "G_frequency_p_value_MannWhitney_abs"; ("single_sample",
    "single_sample_tTest") + group "fat" -> "A_frequency_p_value_tTest_fat";
    ("lmm", "LMM_abs") -> "T_p_value_LMM_abs"; ("lmm_across_time", "LMM") +
    group "control" -> "C_p_value_LMM" (group is in the file name).
    """
    if family not in SUMMARY_FAMILIES:
        raise ValueError(
            f"unknown summary family {family!r}; choose from {list(SUMMARY_FAMILIES)}"
        )
    stem, group_in_column = SUMMARY_FAMILIES[family]
    # The statistic that names the column: strip the family prefix when present
    # ("two_sample_paired_Wilcoxon" -> "Wilcoxon"); LMM types have no prefix.
    prefix = f"{family}_"
    stat = test_type[len(prefix) :] if test_type.startswith(prefix) else test_type

    def column(base: str, group: str) -> str:
        # Group goes into the column name only for families that keep one column
        # block per group; one-file-per-group families use plain names.
        suffix = f"_{group}" if (group_in_column and group) else ""
        return f"{base}{stem}{stat}{suffix}"

    return column


def output_label(family: str, test_type: str) -> str:
    """``{family}_{statistic}`` for output file names.

    ``lmm`` and ``lmm_across_time`` both report test_type ``LMM``, so the
    test_type alone would make their outputs overwrite each other; the family
    is always included and the family prefix is not repeated.

    Examples: ("two_sample_paired", "two_sample_paired_tTest") ->
    "two_sample_paired_tTest"; ("lmm", "LMM_abs") -> "lmm_LMM_abs";
    ("lmm_across_time", "LMM") -> "lmm_across_time_LMM".
    """
    prefix = f"{family}_"
    stat = test_type[len(prefix) :] if test_type.startswith(prefix) else test_type
    return f"{family}_{stat}"


def load_significant_sites(
    path: str, test_type: str, threshold_column: str, threshold: float
) -> pd.DataFrame:
    """Significant sites of ONE test from a p_value_summary file.

    Parameters
    ----------
    path
        The summary TSV (``period, mag_id, contig, position, gene_id, test_type,
        [group_analyzed,] min_p_value, source_file, q_value``).
    test_type
        The exact ``test_type`` string to keep, as the summary spells it
        (``two_sample_paired_Wilcoxon``, ``LMM_abs``, ...).  Required; a name
        the file does not contain raises, listing the ones it does.
    threshold_column, threshold
        ``"q_value"`` (BH-corrected, the default) or ``"min_p_value"`` (raw),
        kept where ``<= threshold``.

    Returns
    -------
    One row per site with ``gene_id`` stripped (real files carry a trailing
    space) and ``group_analyzed`` present ("" for two-sample tests, which have
    no such column).  Raises if the test_type never occurs in the file.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "mag_id": str,
            "contig": str,
            "gene_id": str,
            "group_analyzed": str,
            "source_file": str,
        },
    )
    present = df["test_type"].unique().tolist()
    if not test_type or test_type not in present:
        raise ValueError(
            f"no {test_type} rows in {path}; test_types present: {present}"
        )
    df = df[df["test_type"] == test_type]
    df = df[df[threshold_column] <= threshold].copy()
    df["gene_id"] = df["gene_id"].fillna("").str.strip()
    if "group_analyzed" not in df.columns:
        df["group_analyzed"] = ""
    df["group_analyzed"] = df["group_analyzed"].fillna("")
    df["position"] = df["position"].astype(np.int64)
    logger.info(
        f"{len(df):,} {test_type} sites at {threshold_column} <= {threshold} across {df['mag_id'].nunique()} MAGs"
    )
    return df.reset_index(drop=True)


def candidate_alleles(
    sites: pd.DataFrame, source: pd.DataFrame, column_of
) -> pd.DataFrame:
    """Which base(s) carried the significant p-value at each site.

    The summary's ``min_p_value`` is the minimum over the four per-base tests
    (``p_value_summary.py``); the per-MAG source file still has all four.  A
    base is a candidate when its p equals that minimum.  At a biallelic site
    the two allele frequencies are complementary (A = 1 - G), so their tests
    are the same test and BOTH bases tie -- that is why this returns a set,
    reported with ``n_alleles_tied_at_min_p`` so ties are visible.

    Parameters
    ----------
    sites
        Rows of ``load_significant_sites`` for ONE MAG (``contig, position,
        min_p_value, group_analyzed``).
    source
        That MAG's test file: ``contig, position`` plus
        ``{B}_frequency_p_value_tTest`` (two-sample) or
        ``{B}_frequency_p_value_tTest_{group}`` (single-sample, per group).
    column_of
        ``allele_p_column(family, test_type)``: maps (base, group_analyzed) to
        the source column holding that base's p-value.

    Returns
    -------
    ``sites`` expanded to one row per (site, allele), plus ``allele`` and
    ``n_alleles_tied_at_min_p``.  Sites are matched to the source on contig,
    position AND gene_id (so a gene-annotation disagreement is caught too); any
    site without a match raises, listing the first few.

    Example: A and G both at 8.1e-5, C and T at 1.0, min 8.1e-5 -> two rows,
    alleles A and G, n_alleles_tied_at_min_p 2.
    """
    # Normalise the source's gene_id the way load_significant_sites normalised the
    # summary's: both files write a trailing space and NaN for intergenic positions.
    source = source.assign(gene_id=source["gene_id"].fillna("").astype(str).str.strip())
    keys = ["contig", "position", "gene_id"]
    # Inner merge on all three keys: every summary row came from this source file,
    # so every site MUST match; a shortfall is a broken run and is reported below.
    merged = sites.merge(source, on=keys, how="inner")
    if len(merged) != len(sites):
        missing = sites.merge(source[keys], on=keys, how="left", indicator=True)
        missing = missing[missing["_merge"] == "left_only"]
        raise ValueError(
            f"{len(missing)} of {len(sites)} sites have no matching row in the source file, e.g. "
            f"{missing[keys].head(3).to_dict('records')}"
        )
    rows = []
    for row in merged.itertuples(index=False):
        p_values = {}
        for base in BASES:
            col = column_of(base, row.group_analyzed)
            if col not in merged.columns:
                raise ValueError(f"source file lacks column {col}")
            p_values[base] = getattr(row, col)
        winners = [
            b
            for b, p in p_values.items()
            if np.isclose(p, row.min_p_value, rtol=1e-9, atol=0.0)
        ]
        if not winners:
            raise ValueError(
                f"site {row.contig}:{row.position}: no per-base p equals min_p_value {row.min_p_value}"
            )
        for base in winners:
            rows.append(
                {
                    **{c: getattr(row, c) for c in sites.columns},
                    "allele": base,
                    "n_alleles_tied_at_min_p": len(winners),
                }
            )
    return pd.DataFrame(
        rows, columns=list(sites.columns) + ["allele", "n_alleles_tied_at_min_p"]
    )


def assess_allele(
    counts: npt.NDArray,
    positions: npt.NDArray[np.int64],
    base_idx: int,
    model: npt.NDArray[np.int32],
    min_freq: float,
    min_cov: int,
) -> dict[str, np.ndarray]:
    """Per-position verdict for ONE base in ONE sample's dense counts.

    Parameters
    ----------
    counts
        Dense ``(contig_length, 4)`` A/C/G/T counts of one sample on one contig.
    positions
        0-based positions to assess (a contig's significant sites).
    base_idx
        Column of the allele under test (0..3 = A, C, G, T).
    model, min_freq, min_cov
        The presence rule: ``model[coverage]`` reads AND ``min_freq`` of reads;
        ``min_cov`` reads at the position before any verdict (1 = off).

    Returns
    -------
    Aligned arrays: ``allele_reads``, ``total_reads``, ``detection_threshold_reads``
    (the bar at that depth), ``allele_frequency`` (NaN when not covered),
    ``allele_status`` (one of the four tiers), ``allele_present`` (bool).

    Example (bar 3 at 30x): counts row [25,0,5,0], G -> reads 5, present;
    [29,0,1,0] -> below_detection; [30,0,0,0] -> absent; [2,0,0,0] at
    min_cov 5 -> not_covered.
    """
    sub = counts[positions].astype(np.int64)
    coverage = sub.sum(axis=1)
    reads = sub[:, base_idx]
    covered = coverage >= min_cov
    # Same clip as presence_matrix: ultra-deep positions reuse the deepest bar.
    threshold = model[np.clip(coverage, 0, len(model) - 1)].astype(np.int64)
    freq = np.divide(
        reads, coverage, out=np.full(len(reads), np.nan), where=coverage > 0
    )
    present = covered & (reads >= threshold) & (freq >= min_freq)
    evidence = np.select(
        [~covered, present, reads == 0],
        [EVIDENCE_NOT_COVERED, EVIDENCE_PRESENT, EVIDENCE_ABSENT],
        default=EVIDENCE_BELOW_DETECTION,
    ).astype(object)
    # Frequency only means something on covered positions.
    freq = np.where(covered, freq, np.nan)
    return {
        "allele_reads": reads,
        "total_reads": coverage,
        "detection_threshold_reads": threshold,
        "allele_frequency": freq,
        "allele_status": evidence,
        "allele_present": present,
    }


def label_origin_in_own_mouse(long: pd.DataFrame, tps: Timepoints) -> pd.DataFrame:
    """Add ``origin_in_own_mouse`` to the long table: the per-mouse verdict.

    Parameters
    ----------
    long
        One row per site x allele x sample with ``time`` (the timepoint label as
        the metadata spells it), ``subjectID`` and ``allele_status``.
    tps
        Which label is the earlier and which the later timepoint; also spells
        the labels (``de_novo_candidate_below_detection_at_pre``).

    Returns
    -------
    The same table with ``origin_in_own_mouse`` filled on later-timepoint rows
    and NaN on earlier-timepoint rows.  The verdict reads BOTH timepoints of the
    same mouse (shown for pre/end)::

        end status                own pre status     label
        present                   present            standing_variation
        present                   below_detection    de_novo_candidate_below_detection_at_pre
        present                   absent             de_novo_candidate
        present                   not_covered        pre_not_covered
        present                   (no pre sample)    no_pre_sample
        below_detection           anything           allele_below_detection_at_end
        absent                    anything           allele_absent_at_end
        not_covered               anything           end_not_covered

    A later-timepoint sample that does not show the allele has nothing to
    explain, so its earlier sample is not consulted; that is why the later
    status is tested first.

    Example: mouse 533, G at 13509: pre present (5/19), end present -> standing_variation.
    """
    keys = SITE_KEYS + ["subjectID"]
    is_t0 = long["time"] == tps.earlier
    is_t1 = long["time"] == tps.later
    # The same mouse's earlier-timepoint status, joined onto every row of that
    # mouse at the site; a mouse with no earlier sample gets NaN.
    t0_status = long[is_t0][keys + ["allele_status"]].rename(
        columns={"allele_status": "_own_t0_status"}
    )
    out = long.merge(t0_status, on=keys, how="left")
    is_t1 = (out["time"] == tps.later).to_numpy()
    t1 = out["allele_status"]
    t0 = out["_own_t0_status"]
    out["origin_in_own_mouse"] = np.select(
        [
            ~is_t1,  # earlier-timepoint rows: no verdict
            t1 == EVIDENCE_NOT_COVERED,
            t1 == EVIDENCE_BELOW_DETECTION,
            t1 == EVIDENCE_ABSENT,
            t0.isna(),
            t0 == EVIDENCE_PRESENT,
            t0 == EVIDENCE_BELOW_DETECTION,
            t0 == EVIDENCE_ABSENT,
        ],
        [
            None,
            tps.name(ORIGIN_T1_NOT_COVERED),
            tps.name(ORIGIN_BELOW_DETECTION_AT_T1),
            tps.name(ORIGIN_ABSENT_AT_T1),
            tps.name(ORIGIN_NO_T0_SAMPLE),
            ORIGIN_STANDING,
            tps.name(ORIGIN_DE_NOVO_BELOW_DETECTION),
            ORIGIN_DE_NOVO,
        ],
        default=tps.name(ORIGIN_T0_NOT_COVERED),
    )
    out["origin_in_own_mouse"] = (
        out["origin_in_own_mouse"].astype(object).where(is_t1, np.nan)
    )
    return out.drop(columns="_own_t0_status")


# Summary column TEMPLATES (``{t0}`` / ``{t1}`` -> the comparison's timepoint names via
# ``summary_columns``).  Shown here for pre_end.
SUMMARY_COLUMN_TEMPLATES = [
    # site + allele identity
    "mag_id",
    "contig",
    "position",
    "gene_id",
    "group_analyzed",
    "allele",
    "n_alleles_tied_at_min_p",
    "q_value",
    # the verdict under the loosest framing (any baseline mouse)
    "origin_any_mouse",
    # any-mouse numbers, FILTERED by the presence rule (min_cov / bar / min_freq):
    # "present in 12 of the 14 pre samples we could judge, in 6 replicates"
    "n_{t0}_samples_allele_present",
    "n_{t0}_samples_covered",
    "n_replicates_with_allele_at_{t0}",
    "{t0}_mice_allele_present",
    # same-mouse numbers: the strict framing, one count per de-novo/standing verdict
    "n_mice_standing_variation",
    "n_mice_de_novo_candidate",
    "n_mice_de_novo_candidate_below_detection_at_{t0}",
    # pooled reads per timepoint, UNFILTERED: every read at the position from every
    # sample, thin ones included -- the numbers behind "never seen in 1,000 reads".
    "total_reads_{t0}",
    "allele_reads_{t0}",
    "allele_frequency_{t0}",
    "total_reads_{t1}",
    "allele_reads_{t1}",
    "allele_frequency_{t1}",
]


def summary_columns(tps: Timepoints) -> list[str]:
    """``SUMMARY_COLUMN_TEMPLATES`` with the comparison's timepoint names filled in.

    Example: ``summary_columns(Timepoints("pre", "end"))`` holds
    ``n_pre_samples_covered``, ``total_reads_end``, ...
    """
    return [tps.name(c) for c in SUMMARY_COLUMN_TEMPLATES]


def summarise_sites(long: pd.DataFrame, tps: Timepoints) -> pd.DataFrame:
    """One row per site x allele: the few numbers the baseline question asks for.

    Deliberately lean (decided 2026-09-10): everything else is in the long
    table and can be added back on request.  The read columns were added on
    request 2026-09-11.

    Parameters
    ----------
    long
        The long table after ``label_origin_in_own_mouse``: one row per site x
        allele x sample with ``time``, ``allele_status``, ``allele_present``,
        ``allele_reads``, ``total_reads``, ``subjectID``, ``replicate``,
        ``origin_in_own_mouse``.
    tps
        Earlier / later timepoint labels; spells the column names.

    Returns
    -------
    ``summary_columns(tps)`` (shown for pre/end):
      origin_any_mouse -- allele_not_present_at_end if no covered end sample
        shows the allele; else standing_variation if any covered pre sample has
        it present; else de_novo_candidate_below_detection_at_pre if any pre
        has it below detection; else de_novo_candidate if any covered pre is
        absent; else pre_not_covered.
      n_pre_samples_allele_present / n_pre_samples_covered -- present pre samples
        over pre samples deep enough to judge; n_replicates_with_allele_at_pre --
        distinct replicates among the present ones; pre_mice_allele_present --
        their subjectIDs, comma-joined.  (Presence rule applied.)
      n_mice_* -- end rows carrying that origin_in_own_mouse verdict (the
        same-mouse framing).
      total_reads_pre / allele_reads_pre / allele_frequency_pre (and _end) --
        reads at the position summed over EVERY sample at that timepoint, reads
        of this allele among them, and their ratio (NaN when no reads at all).
        No min_cov, no bar, no min_freq: a sample with 2 reads contributes its
        2 reads, a sample with no profile contributes 0.  This is what makes
        "never seen in N reads" a frequency bound of 1/N.

    Example (G at 13509, MAG bin.012, pre_end): origin standing_variation; present in
    12 of 14 covered pre samples, 6 replicates; same-mouse 6 standing, 1 de novo,
    1 de novo below detection; total_reads_pre 168, allele_reads_pre 125 (0.744),
    total_reads_end 245, allele_reads_end 179 (0.731).
    """
    rows = []
    for key, sub in long.groupby(SITE_KEYS, sort=True, dropna=False):
        t0 = sub[sub["time"] == tps.earlier]
        t1 = sub[sub["time"] == tps.later]
        t0_cov = t0[t0["allele_status"] != EVIDENCE_NOT_COVERED]
        t1_cov = t1[t1["allele_status"] != EVIDENCE_NOT_COVERED]
        # Which mice / replicates had the allele at t0 (covered + present).
        mice_t0_present = set(t0_cov.loc[t0_cov["allele_present"], "subjectID"])
        reps_t0_present = set(t0_cov.loc[t0_cov["allele_present"], "replicate"])
        record = dict(zip(SITE_KEYS, key))
        for col in ("q_value", "n_alleles_tied_at_min_p"):
            record[col] = sub[col].iloc[0] if col in sub.columns else np.nan
        # A site verdict needs the allele to be SEEN at t1 in at least one covered
        # sample; otherwise there is nothing whose origin to explain.
        if not t1_cov["allele_present"].any():
            origin = tps.name(ORIGIN_NOT_SEEN_AT_T1)
        elif mice_t0_present:
            origin = ORIGIN_STANDING
        elif (t0_cov["allele_status"] == EVIDENCE_BELOW_DETECTION).any():
            origin = tps.name(ORIGIN_DE_NOVO_BELOW_DETECTION)
        elif (t0_cov["allele_status"] == EVIDENCE_ABSENT).any():
            origin = ORIGIN_DE_NOVO
        else:
            origin = tps.name(ORIGIN_T0_NOT_COVERED)
        record.update(
            {
                "origin_any_mouse": origin,
                tps.name("n_{t0}_samples_allele_present"): int(t0_cov["allele_present"].sum()),
                tps.name("n_{t0}_samples_covered"): len(t0_cov),
                tps.name("n_replicates_with_allele_at_{t0}"): len(reps_t0_present),
                tps.name("{t0}_mice_allele_present"): ",".join(sorted(mice_t0_present)),
            }
        )
        for label in (ORIGIN_STANDING, ORIGIN_DE_NOVO, ORIGIN_DE_NOVO_BELOW_DETECTION):
            record[tps.name(f"n_mice_{label}")] = int(
                (t1["origin_in_own_mouse"] == tps.name(label)).sum()
            )
        # Pooled reads, unfiltered: t0 / t1 here are ALL rows at the timepoint,
        # not the covered subset, so thin and profile-less samples count too.
        for role, part in (("{t0}", t0), ("{t1}", t1)):
            total = int(part["total_reads"].sum())
            allele = int(part["allele_reads"].sum())
            record[tps.name(f"total_reads_{role}")] = total
            record[tps.name(f"allele_reads_{role}")] = allele
            record[tps.name(f"allele_frequency_{role}")] = allele / total if total else np.nan
        rows.append(record)
    return pd.DataFrame(rows, columns=summary_columns(tps))


def _assess_one_sample(job: tuple) -> pd.DataFrame:
    """Worker: one (MAG, sample) -> verdict rows for every candidate (site, allele) of that MAG.

    ``job`` = (mag_id, sample_id, profile_path, contig_lengths, candidates,
    model, min_freq, min_cov) where ``candidates`` has ``contig, position,
    allele``.  Loads the profile once and assesses every site on every contig.
    Module-level so it pickles into ``multiprocessing.Pool``.
    """
    mag_id, sample_id, path, contig_lengths, candidates, model, min_freq, min_cov = job
    dense = dense_contig_counts(load_profile(path), contig_lengths)
    frames = []
    for (contig, allele), sub in candidates.groupby(["contig", "allele"], sort=True):
        positions = sub["position"].to_numpy(dtype=np.int64)
        got = assess_allele(
            dense[contig], positions, BASES.index(allele), model, min_freq, min_cov
        )
        frames.append(
            pd.DataFrame(
                {
                    "mag_id": mag_id,
                    "sample_id": sample_id,
                    "contig": contig,
                    "position": positions,
                    "allele": allele,
                    **got,
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "mag_id",
                "sample_id",
                "contig",
                "position",
                "allele",
                "allele_reads",
                "total_reads",
                "detection_threshold_reads",
                "allele_frequency",
                "allele_status",
                "allele_present",
            ]
        )
    )


def chase_the_ancestors(args: argparse.Namespace) -> int:
    """Orchestrator: summary -> candidate alleles -> every sample's verdict -> two files."""
    os.makedirs(args.output_dir, exist_ok=True)
    earlier, later, group_a, group_b = parse_comparison(args.comparison)
    tps = Timepoints(earlier, later)  # spells every timepoint-bearing name from here on
    groups = (group_a, group_b)

    # ---- 1. The samples of this comparison: both groups, both timepoints.
    meta = pd.read_csv(args.metadata, sep="\t", dtype=str)
    if "replicate" not in meta.columns:
        meta["replicate"] = meta["subjectID"]  # same default as mag_metadata.py
    meta = meta[meta["group"].isin(groups) & meta["time"].isin((earlier, later))].copy()
    if meta.empty:
        raise ValueError(
            f"no samples in {args.metadata} for groups {groups} at {earlier}/{later}"
        )
    logger.info(
        f"{len(meta)} samples: "
        + ", ".join(
            f"{g}/{t}={int(((meta.group == g) & (meta.time == t)).sum())}"
            for g in groups
            for t in (earlier, later)
        )
    )

    # ---- 2. Significant sites of the chosen test, then their candidate alleles per MAG.
    summary_dir = os.path.join(args.run_dir, "p_value_summary", args.comparison)
    sites = load_significant_sites(
        find_summary_file(summary_dir, args.summary, f"{earlier}_{later}"),
        args.test_type,
        args.threshold_column,
        args.threshold,
    )
    test_type = args.test_type  # names the output files and the p-value columns
    column_of = allele_p_column(args.summary, test_type)
    if args.mags:
        sites = sites[sites["mag_id"].isin(args.mags)]
    tests_dir = os.path.join(
        args.run_dir, "significance_tests", f"{args.summary}_{args.comparison}"
    )
    candidates_by_mag = {}
    for mag_id, mag_sites in sites.groupby("mag_id", sort=True):
        frames = []
        for source_file, chunk in mag_sites.groupby(
            "source_file"
        ):  # single-sample: one file per group
            source = pd.read_csv(
                os.path.join(tests_dir, source_file),
                sep="\t",
                dtype={"contig": str, "gene_id": str},
            )
            frames.append(
                candidate_alleles(chunk.drop(columns="source_file"), source, column_of)
            )
        candidates_by_mag[mag_id] = pd.concat(frames, ignore_index=True)
    n_cand = sum(len(c) for c in candidates_by_mag.values())
    logger.info(
        f"{n_cand:,} (site, allele) candidates across {len(candidates_by_mag)} MAGs"
    )

    # ---- 3. One job per (MAG, sample); every job loads its own profile once.
    model = build_error_model(min_base_quality=args.min_base_quality, fdr=args.fdr)
    jobs = []
    no_profile = (
        []
    )  # (mag_id, sample_id) with no profile: the MAG has no reads in that sample
    for mag_id, cand in candidates_by_mag.items():
        lengths = contig_lengths_for_mag(args.fasta, args.mag_mapping, mag_id)
        slim = cand[["contig", "position", "allele"]].drop_duplicates()
        for sample_id in meta["sample_id"]:
            path = profile_path(args.profiles_dir, sample_id, mag_id)
            if not os.path.exists(path):
                # profile_mags writes no file when a MAG has zero mapped reads in a
                # sample, and the pipeline's own QC lists such samples as absent for
                # that MAG.  Zero reads = not_covered at every site, so the sample
                # is kept, with that status, rather than crashing the run.
                no_profile.append((mag_id, sample_id))
                continue
            jobs.append(
                (
                    mag_id,
                    sample_id,
                    path,
                    lengths,
                    slim,
                    model,
                    args.min_freq,
                    args.min_cov,
                )
            )
    logger.info(
        f"assessing {len(jobs)} (MAG, sample) profiles with {args.cpus} workers"
    )
    if no_profile:
        per_mag = pd.Series([m for m, _ in no_profile]).value_counts()
        for mag_id, n in per_mag.items():
            logger.warning(
                f"MAG {mag_id}: no profile for {n} of {len(meta)} samples (no reads there); "
                f"reported as not_covered"
            )
    if len(no_profile) == len(jobs) + len(no_profile) and no_profile:
        raise FileNotFoundError(
            f"no profile found for ANY (MAG, sample) under {args.profiles_dir}; wrong --profiles_dir?"
        )
    if not jobs:
        # No significant site for this test / threshold / MAG selection: a legitimate
        # outcome (Wilcoxon at n=8 cannot reach q<0.05, for instance).  Write the
        # two files with headers only so a pipeline over many comparisons never trips.
        stem = os.path.join(
            args.output_dir,
            f"{args.comparison}_{output_label(args.summary, test_type)}_baseline_presence",
        )
        pd.DataFrame(columns=LONG_COLUMNS).to_csv(
            f"{stem}.tsv.gz", sep="\t", index=False, compression="gzip"
        )
        pd.DataFrame(columns=summary_columns(tps)).to_csv(
            f"{stem}_summary.tsv", sep="\t", index=False
        )
        logger.info(f"no sites: wrote header-only outputs to {stem}*")
        return 0
    with multiprocessing.Pool(processes=args.cpus) as pool:
        verdicts = pd.concat(
            pool.imap_unordered(_assess_one_sample, jobs, chunksize=4),
            ignore_index=True,
        )
    if no_profile:
        # One not_covered row per candidate (site, allele) for each missing pair;
        # the bar at 0 reads is model[0] so the row still carries a threshold.
        frames = []
        for mag_id, sample_id in no_profile:
            slim = candidates_by_mag[mag_id][
                ["contig", "position", "allele"]
            ].drop_duplicates()
            frames.append(
                pd.DataFrame(
                    {
                        "mag_id": mag_id,
                        "sample_id": sample_id,
                        "contig": slim["contig"].to_numpy(),
                        "position": slim["position"].to_numpy(),
                        "allele": slim["allele"].to_numpy(),
                        "allele_reads": 0,
                        "total_reads": 0,
                        "detection_threshold_reads": int(model[0]),
                        "allele_frequency": np.nan,
                        "allele_status": EVIDENCE_NOT_COVERED,
                        "allele_present": False,
                    }
                )
            )
        verdicts = pd.concat([verdicts, *frames], ignore_index=True)

    # ---- 4. Long table: verdicts + site context + sample context (+ optional joins).
    site_ctx = pd.concat(candidates_by_mag.values(), ignore_index=True)
    long = verdicts.merge(
        site_ctx, on=["mag_id", "contig", "position", "allele"], how="left"
    )
    long = long.merge(
        meta[["sample_id", "subjectID", "replicate", "group", "time"]],
        on="sample_id",
        how="left",
    )
    long["strain_background"] = np.nan
    if args.turnover_dir:
        # Per (MAG, mouse) background for THIS period, from alleleflux-strain-turnover.
        files = glob.glob(os.path.join(args.turnover_dir, "*_strain_turnover.tsv"))
        turn = pd.concat(
            [pd.read_csv(f, sep="\t", dtype=str) for f in files], ignore_index=True
        )
        turn = turn[turn["transition"] == f"{earlier}_{later}"][
            ["MAG_ID", "subjectID", "background"]
        ]
        long = long.drop(columns="strain_background").merge(
            turn.rename(
                columns={"MAG_ID": "mag_id", "background": "strain_background"}
            ),
            on=["mag_id", "subjectID"],
            how="left",
        )
    long = label_origin_in_own_mouse(long, tps)
    long["min_cov"] = args.min_cov
    long["min_freq"] = args.min_freq
    # Earlier timepoint before later within a site: an ordered categorical sorts
    # by comparison order, not alphabetically ("end" < "pre" would flip them).
    long["time"] = pd.Categorical(long["time"], categories=[earlier, later], ordered=True)
    long = long[LONG_COLUMNS].sort_values(
        ["mag_id", "contig", "position", "allele", "time", "group", "sample_id"]
    )
    long["time"] = long["time"].astype(str)

    # ---- 5. Summary, then write both.
    summary = summarise_sites(long, tps)
    stem = os.path.join(
        args.output_dir,
        f"{args.comparison}_{output_label(args.summary, test_type)}_baseline_presence",
    )
    long.to_csv(f"{stem}.tsv.gz", sep="\t", index=False, compression="gzip")
    summary.to_csv(f"{stem}_summary.tsv", sep="\t", index=False)
    logger.info(
        f"wrote {len(long):,} long rows and {len(summary):,} summary rows to {stem}*"
    )
    if len(summary):
        logger.info(
            f"origin_any_mouse: {summary['origin_any_mouse'].value_counts().to_dict()}"
        )
    return 0


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="For each significant site: was the allele already present at the baseline timepoint?",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="AlleleFlux run root holding p_value_summary/ and significance_tests/",
    )
    parser.add_argument(
        "--comparison", required=True, help="Comparison label, e.g. pre_end-fat_control"
    )
    parser.add_argument(
        "--summary",
        choices=list(SUMMARY_FAMILIES),
        default="two_sample_paired",
        help="Which p_value_summary family (divergence: two_sample_*; parallelism: single_sample; lmm*)",
    )
    parser.add_argument(
        "--test_type",
        required=True,
        help="Row filter within the summary, spelled exactly as the file does "
        "(two_sample_paired_tTest, two_sample_paired_Wilcoxon, LMM_abs, ...)",
    )
    parser.add_argument(
        "--threshold_column", choices=("q_value", "min_p_value"), default="q_value"
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--profiles_dir",
        required=True,
        help="Profiles root: {sample}/{sample}_{mag}_profiled.tsv.gz",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Sample metadata TSV (sample_id, subjectID, group, time[, replicate])",
    )
    parser.add_argument(
        "--fasta", required=True, help="Reference FASTA (its .fai gives contig lengths)"
    )
    parser.add_argument(
        "--mag_mapping", required=True, help="contig -> MAG mapping TSV"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--turnover_dir",
        default=None,
        help="alleleflux-strain-turnover outputs, for a strain_background column",
    )
    parser.add_argument(
        "--mags", nargs="*", default=None, help="Restrict to these MAG ids"
    )
    parser.add_argument(
        "--min_cov",
        type=int,
        default=5,
        help="Reads at a position before any verdict (1 = off)",
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
    parser.add_argument("--cpus", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    return chase_the_ancestors(args)


if __name__ == "__main__":
    raise SystemExit(main())
