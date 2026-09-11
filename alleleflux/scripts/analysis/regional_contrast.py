#!/usr/bin/env python3
"""
Detect genes / sliding windows with differential allele-frequency evolution
between treatment and control groups across paired hosts.

For each genomic site the Total-Variation distance between allele-frequency
vectors at T0 and T1 is computed.  Sites are then aggregated into regions
(genes and/or sliding windows) per host × group, ranked into genome-wide
percentiles, and contrasted between treatment and control.  Finally, the
contrast is tested across hosts with a two-sided Wilcoxon signed-rank or
t-test, followed by Benjamini–Hochberg FDR correction.

Workflow integration
--------------------
Runs after ``allele_freq.py`` in the AlleleFlux pipeline.  Preferred input:
``*_allele_frequency_changes_mean.tsv.gz`` produced by ``allele_freq.py``
(columns: replicate, group, contig, position, gene_id, and the four
``*_frequency_diff_mean`` columns).

This file is computed from the allele changes produced by ``allele_freq.py``.
By default, the upstream zero-diff filter
(``quality_control.disable_zero_diff_filtering: False``) removes positions
where the net sum of per-sample frequency differences is zero across **all**
samples and groups before computing the mean.  This filter is safe to use
here because a genuine treatment-vs-control signal would not net to zero
across all samples combined.

When ``disable_zero_diff_filtering: True`` is set, those zero-net-sum sites
are retained in the input.  Sites with all-zero allele-frequency differences
receive ``site_score = 0`` and are counted towards ``n_informative_sites``
(since ``pandas.count`` counts non-null rows, not non-zero values).  The
statistical tests are unaffected: a region where every site has stasis still
yields a valid contrast of zero, and the Wilcoxon / t-test handle this
correctly (returning p = 1.0 for perfect null cases).  In practice the
results will be nearly identical whether or not the filter is applied,
because zero-net-sum sites carry no directional signal.

Do **not** use the statistically preprocessed file
(``*_allele_frequency_changes_mean_preprocessed.tsv.gz``).  That file
is filtered by a site-level paired test (``preprocess_between_groups.py``)
which pre-selects sites individually significant for treatment vs control.
Using it introduces selection bias: the region-level test would then
operate on sites already individually significant, inflating regional
signal and invalidating the FDR correction.

Typical placement::

    Step 1 → QC → allele_freq.py → **regional_contrast.py**

Example
-------
::

    alleleflux-regional-contrast \\
        --input MAG_001_allele_frequency_changes_mean.tsv.gz \\
        --output_dir results/ \\
        --treatment_group treatment \\
        --control_group control \\
        --mode both \\
        --window_size 1000 \\
        --agg_method median
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, trim_mean, ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Fixed column names matching allele_freq.py output
GROUP_COL = "group"
CONTIG_COL = "contig"
POSITION_COL = "position"
GENE_COL = "gene_id"
HOST_COL = "replicate"  # paired-unit column in AlleleFlux preprocessed tables

# Allele-frequency difference columns (mean across subjects within
# replicate × group, produced by allele_freq.py / preprocess_between_groups.py)
DIFF_COLS = [
    "A_frequency_diff_mean",
    "T_frequency_diff_mean",
    "G_frequency_diff_mean",
    "C_frequency_diff_mean",
]

# Output column names and method identifiers
SITE_SCORE_COL = "site_score"
REGION_SCORE_COL = "region_score"
PERCENTILE_COL = "percentile"
FDR_METHOD = "fdr_bh"
# Directional p-value / q-value column names.
# "greater" = treatment evolved more than control (contrast > 0).
# "less"    = control evolved more than treatment (contrast < 0).
PVAL_WILCOXON_GREATER_COL = "p_value_wilcoxon_greater"
QVAL_WILCOXON_GREATER_COL = "q_value_wilcoxon_greater"
PVAL_WILCOXON_LESS_COL = "p_value_wilcoxon_less"
QVAL_WILCOXON_LESS_COL = "q_value_wilcoxon_less"
PVAL_TTEST_GREATER_COL = "p_value_ttest_greater"
QVAL_TTEST_GREATER_COL = "q_value_ttest_greater"
PVAL_TTEST_LESS_COL = "p_value_ttest_less"
QVAL_TTEST_LESS_COL = "q_value_ttest_less"
FISHER_PVAL_TREATMENT_COL = "p_value_fisher_treatment"
FISHER_QVAL_TREATMENT_COL = "q_value_fisher_treatment"
FISHER_PVAL_CONTROL_COL = "p_value_fisher_control"
FISHER_QVAL_CONTROL_COL = "q_value_fisher_control"


# ── 1. Load & validate ──────────────────────────────────────────────────


def load_input_table(
    input_path: str | Path,
    required_cols: list[str],
    diff_cols: list[str],
) -> pd.DataFrame:
    """Load an allele-frequency-changes TSV and validate required columns.

    Parameters
    ----------
    input_path : str or Path
        Path to a tab-separated file (optionally gzip-compressed).
    required_cols : list[str]
        Non-diff columns that must be present.
    diff_cols : list[str]
        Allele-frequency-difference columns that must be present.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    logger.info(f"Loading input from {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if "gene_id" in df.columns:
            # astype("string"), NOT astype(str): the parquet column is a nullable
            # (pyarrow-backed) string, and astype(str) renders every null as the
            # *literal string* "<NA>".  That sentinel passes the notna() guard in
            # build_gene_regions, collapsing all unannotated sites into one
            # pseudo-gene per contig that shares the id "<NA>" across contigs.
            # StringDtype keeps nulls as pd.NA while still exposing .str.
            df["gene_id"] = df["gene_id"].astype("string")
    else:
        df = pd.read_csv(path, sep="\t", dtype={"gene_id": str})

    missing = set(required_cols + diff_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Input table is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    logger.info(f"Loaded {len(df):,} rows, {df.columns.size} columns")
    return df


# ── 2. Site-level scores ────────────────────────────────────────────────


def compute_site_scores(
    df: pd.DataFrame,
    diff_cols: list[str],
    score_col: str = "site_score",
) -> pd.DataFrame:
    """Compute the Total-Variation distance at each site.

    ``site_score = 0.5 * (|ΔA| + |ΔC| + |ΔG| + |ΔT|)``

    Parameters
    ----------
    df : pd.DataFrame
        Must contain every column listed in *diff_cols*.
    diff_cols : list[str]
        Columns holding per-nucleotide frequency changes (T1 − T0).
    score_col : str
        Name of the new column written to *df*.

    Returns
    -------
    pd.DataFrame
        Input frame with *score_col* appended.
    """
    abs_diffs = df[diff_cols].abs()
    df[score_col] = 0.5 * abs_diffs.sum(axis=1)
    n_zero = (df[score_col] == 0).sum()
    if n_zero:
        logger.info(f"{n_zero:,} sites with site_score == 0")
    return df


# ── 3. Region definitions ───────────────────────────────────────────────


# String spellings of "no annotation".  Compared case-insensitively against the
# whitespace-stripped gene id.  Upstream tools (and any accidental astype(str) on
# a nullable column) can serialize a missing gene as one of these instead of a
# real null; treating them as gene names silently fabricates regions.
_NULL_GENE_SENTINELS = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "<na>", "."}
)


def _is_missing_gene(values: pd.Series) -> pd.Series:
    """Boolean mask: True where a gene id is absent or spells "missing"."""
    stripped = values.astype("string").str.strip()
    is_sentinel = stripped.str.lower().isin(_NULL_GENE_SENTINELS).fillna(False)
    return (stripped.isna() | is_sentinel).astype(bool)


def build_gene_regions(
    df: pd.DataFrame,
    gene_col: str,
    contig_col: str,
    position_col: str,
) -> pd.DataFrame:
    """Create a site → gene-region mapping table.

    Positions without a gene annotation (NaN / empty) are skipped.
    Comma-separated overlapping-gene identifiers are split so that each
    position contributes independently to every gene it overlaps.

    ``region_start`` / ``region_end`` reflect the span of observed
    variable sites within the gene (not annotated gene boundaries).
    ``region_length`` is the count of unique variable sites in the gene.

    Returns
    -------
    pd.DataFrame
        Columns: *contig_col*, *position_col*, ``region_id``,
        ``region_type``, ``region_start``, ``region_end``,
        ``region_length``.
    """
    # Keep only rows that carry a real gene annotation.  Nulls *and* string
    # spellings of "missing" are dropped — see _NULL_GENE_SENTINELS.
    valid = df[~_is_missing_gene(df[gene_col])].copy()
    if valid.empty:
        logger.warning("No gene annotations found; skipping gene mode.")
        return pd.DataFrame()

    # Unique (contig, position, gene) triples; split comma-separated gene IDs
    # so each position is assigned to every overlapping gene individually.
    mapping = (
        valid[[contig_col, position_col, gene_col]]
        .drop_duplicates()
        .rename(columns={gene_col: "region_id"})
    )
    mapping["region_id"] = mapping["region_id"].astype("string").str.split(",")
    mapping = mapping.explode("region_id")
    mapping["region_id"] = mapping["region_id"].str.strip()
    # A multi-gene annotation can still carry a missing component ("g1,<NA>"),
    # which the row-level filter above cannot see.
    mapping = mapping[~_is_missing_gene(mapping["region_id"])]
    if mapping.empty:
        logger.warning("No gene annotations found; skipping gene mode.")
        return pd.DataFrame()
    mapping = mapping.drop_duplicates()  # remove any duplicates from the explode
    mapping["region_type"] = "gene"

    # Gene-level bounds and variable-site count
    bounds = (
        mapping.groupby([contig_col, "region_id"])[position_col]
        .agg(region_start="min", region_end="max", region_length="nunique")
        .reset_index()
    )

    mapping = mapping.merge(bounds, on=[contig_col, "region_id"], how="left")

    logger.info(f"Built {bounds.shape[0]:,} gene regions")
    return mapping


def build_sliding_windows(
    df: pd.DataFrame,
    contig_col: str,
    position_col: str,
    window_size: int,
) -> pd.DataFrame:
    """Assign every observed site to a non-overlapping genomic tile.

    Tiles are 1-based, non-overlapping windows of *window_size* bp,
    aligned to multiples of *window_size*: [1, W], [W+1, 2W], ...
    Each position falls in exactly one tile.

    Returns
    -------
    pd.DataFrame
        Columns: *contig_col*, *position_col*, ``region_id``,
        ``region_type``, ``region_start``, ``region_end``,
        ``region_length``.
    """
    mapping = df[[contig_col, position_col]].drop_duplicates().copy()

    if mapping.empty:
        logger.warning("No sliding windows could be built.")
        return pd.DataFrame()

    # Vectorised tile assignment.
    # Convert 1-based positions to 0-based for integer-division tiling, then
    # convert the resulting tile boundaries back to 1-based inclusive coordinates
    # (standard in bioinformatics: store/output 1-based, compute 0-based).
    p0 = mapping[position_col] - 1  # 0-based coordinate
    tile_idx = p0 // window_size  # which tile [0,W), [W,2W), ... each position falls in
    mapping["region_start"] = (tile_idx * window_size + 1).values  # 1-based start
    mapping["region_end"] = (
        (tile_idx + 1) * window_size
    ).values  # 1-based inclusive end
    mapping["region_id"] = (
        mapping[contig_col].astype(str)
        + ":"
        + mapping["region_start"].astype(str)
        + "-"
        + mapping["region_end"].astype(str)
    )
    mapping["region_type"] = "window"
    mapping["region_length"] = window_size
    n_windows = mapping["region_id"].nunique()
    logger.info(
        f"Assigned {mapping[position_col].nunique():,} sites to "
        f"{n_windows:,} sliding windows"
    )
    return mapping


# ── 4. Region-level aggregation ─────────────────────────────────────────


def aggregate_region_scores(
    df: pd.DataFrame,
    region_mapping: pd.DataFrame,
    host_col: str,
    group_col: str,
    contig_col: str,
    position_col: str,
    score_col: str = "site_score",
    agg_method: str = "median",
    trim_fraction: float = 0.1,
    min_sites: int = 5,
    min_fraction: float = 0.0,
) -> pd.DataFrame:
    """Aggregate site scores into region scores per host × group.

    Parameters
    ----------
    df : pd.DataFrame
        Site-level data with *score_col*.
    region_mapping : pd.DataFrame
        Maps ``(contig, position)`` → region metadata.
    agg_method : {"median", "mean", "trimmed_mean"}
        Summary statistic for scores within a region.
    trim_fraction : float
        Proportion to cut from each tail (``trimmed_mean`` only).
    min_sites : int
        Minimum number of rows (sites) per region after the
        ``(contig, position)`` merge with the region mapping.  Counted by
        pandas ``"count"`` (non-null values), so **sites whose
        ``site_score`` is exactly 0 are counted just like any other site** —
        a region showing perfect evolutionary stasis still has full
        ``n_informative_sites``.  A genuinely low count only occurs when
        positions were physically absent from the input because the
        zero-diff filter in ``allele_freq.py`` removed them upstream.
        Defaults to 0 (disabled).
    min_fraction : float
        Minimum informative fraction of region length.  0.0 disables.

    Returns
    -------
    pd.DataFrame
        One row per (host, group, region) with ``region_score``,
        ``n_informative_sites``, and ``informative_fraction``.
    """
    merged = df.merge(region_mapping, on=[contig_col, position_col], how="inner")

    if merged.empty:
        logger.warning("No sites matched any regions after merge.")
        return pd.DataFrame()

    group_keys = [
        host_col,
        group_col,
        "region_id",
        "region_type",
        contig_col,
        "region_start",
        "region_end",
        "region_length",
    ]

    # For median and mean, delegate directly to pandas built-ins (handles NaN
    # by default via skipna=True). trimmed_mean has no pandas built-in so it
    # uses a custom function that falls back to mean when n < 3.
    if agg_method == "trimmed_mean":

        def _agg_fn(x: pd.Series) -> float:
            values = x.dropna()
            if len(values) < 3:
                return float(np.mean(values)) if len(values) > 0 else np.nan
            return float(trim_mean(values, proportiontocut=trim_fraction))

        score_agg = (score_col, _agg_fn)
    else:
        score_agg = (score_col, agg_method)  # "median" or "mean"

    agg_df = (
        # observed=True is required: `group` (and other keys) can be Categorical
        # when loaded from parquet.  With observed=False pandas reindexes the
        # result to the full Cartesian product of every grouping key's levels —
        # with the collinear region-metadata columns that exploded to a 23.7 PiB
        # allocation.  observed=True emits only groups that actually occur.
        merged.groupby(group_keys, dropna=False, observed=True)
        .agg(
            region_score=score_agg,
            # Count rows present after the (contig, position) merge — i.e. the
            # number of sites in the region that survived the upstream zero-diff
            # filter in allele_freq.py and therefore appear in the input file.
            # NOTE: pandas "count" counts non-null values, NOT non-zero values.
            # A site whose site_score == 0 (perfect evolutionary stasis) is
            # still counted here.  A low n_informative_sites therefore means
            # sparse data coverage, not zero evolution.
            n_informative_sites=(score_col, "count"),
        )
        .reset_index()
    )

    agg_df["informative_fraction"] = (
        agg_df["n_informative_sites"] / agg_df["region_length"]
    )

    before = len(agg_df)
    if min_sites > 0:
        agg_df = agg_df[agg_df["n_informative_sites"] >= min_sites].copy()
    if min_fraction > 0:
        agg_df = agg_df[agg_df["informative_fraction"] >= min_fraction].copy()
    after = len(agg_df)

    if before != after:
        logger.info(
            f"Region eligibility filter: {before:,} → {after:,} host×region "
            f"entries (min_sites={min_sites}, min_fraction={min_fraction})"
        )

    return agg_df


# ── 5. Percentiles ──────────────────────────────────────────────────────


def compute_percentiles(
    df: pd.DataFrame,
    host_col: str,
    group_col: str,
    score_col: str = "region_score",
    out_col: str = "percentile",
) -> pd.DataFrame:
    """Compute within-(host, group) genome-wide percentile ranks.

    For each (host, group, region_type) combination, ranks region scores on
    a scale of 0–100, where higher values indicate regions with higher scores
    relative to other regions of the **same type** in the same host × group.
    Partitioning by ``region_type`` ensures that gene scores and window scores
    are always ranked within their own pools and are never mixed.

    Percentile = (rank / n) × 100, where rank uses the "average" method
    to handle ties fairly.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *host_col*, *group_col*, and *score_col*.
    host_col : str
        Column name for paired-unit identifier (e.g., "replicate").
    group_col : str
        Column name for group label (e.g., "treatment" or "control").
    score_col : str
        Column name containing region scores to be ranked.
    out_col : str
        Name of the output percentile column.

    Returns
    -------
    pd.DataFrame
        Input frame with *out_col* appended (0–100 scale).
    """

    # rank(pct=True, method="average") is the pandas equivalent of
    # scipy.stats.rankdata(method="average") / n, scaled to (0, 1].
    # Multiply by 100 to obtain a 0–100 percentile scale.
    # Percentiles are computed within each (host, group, region_type) partition
    # so that gene scores and window scores are never mixed.
    df[out_col] = df.groupby([host_col, group_col, "region_type"])[score_col].transform(
        lambda x: x.rank(pct=True, method="average") * 100
    )
    return df


# ── 6. Reshape treatment / control ──────────────────────────────────────


def reshape_treatment_control(
    df: pd.DataFrame,
    host_col: str,
    group_col: str,
    contig_col: str,
    treatment_label: str,
    control_label: str,
) -> pd.DataFrame:
    """Pivot aggregated region scores so treatment and control are side-by-side.

    Transforms long-format data (one row per host × group × region) into a
    paired format where each row represents a single host × region with both
    treatment and control metrics in separate columns.  This enables direct
    contrast computation and statistical testing across hosts.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated region data from :func:`compute_percentiles`.
        Must contain *group_col* with at least two distinct values matching
        *treatment_label* and *control_label*.
    host_col : str
        Column name for the paired-unit identifier (e.g., "replicate").
    group_col : str
        Column name for the group label (e.g., "group").
    contig_col : str
        Column name for genomic contig identifier.
    treatment_label : str
        Value in *group_col* identifying the treatment group.
    control_label : str
        Value in *group_col* identifying the control group.

    Returns
    -------
    pd.DataFrame
        One row per (host, region_id, region_type) tuple with columns:
        - Metadata: ``region_id``, ``region_type``, *contig_col*,
          ``region_start``, ``region_end``, *host_col*
        - Treatment metrics (suffixed ``_treatment``): ``region_score_treatment``,
          ``percentile_treatment``, ``n_informative_sites_treatment``,
          ``informative_fraction_treatment``
        - Control metrics (suffixed ``_control``): ``region_score_control``,
          ``percentile_control``, ``n_informative_sites_control``,
          ``informative_fraction_control``
        - ``contrast``: treatment score − control score

    Example
    -------
    Input (long format, 4 rows):

    .. code-block:: text

        replicate  group  region_id  region_score  percentile
        host_A     treatment  gene_X  0.5          75
        host_A     control    gene_X  0.3          60
        host_B     treatment  gene_X  0.6          80
        host_B     control    gene_X  0.2          45

    Output (paired format, 2 rows):

    .. code-block:: text

        replicate  region_id  region_score_treatment  percentile_treatment  \\
        host_A     gene_X     0.5                    75                    \\
        host_B     gene_X     0.6                    80

        region_score_control  percentile_control  contrast
        0.3                   60                  0.2
        0.2                   45                  0.4
    """
    # Define which columns to merge on (uniquely identifies a region per host).
    # contig_col is part of the key, not just carried metadata: a region_id is
    # only unique *within* a contig, and any id shared across contigs would
    # otherwise cross-join every contig's treatment row against every contig's
    # control row (N_contigs**2 rows per host), fabricating replicates.
    merge_keys = [host_col, contig_col, "region_id", "region_type"]
    # Metadata columns carried through from treatment data (control data has same values)
    meta_cols = ["region_start", "region_end"]
    # Numeric columns to be suffixed with _treatment / _control
    value_cols = [
        "region_score",
        "percentile",
        "n_informative_sites",
        "informative_fraction",
    ]

    # Split into treatment and control subsets
    trt = df[df[group_col] == treatment_label].copy()
    ctrl = df[df[group_col] == control_label].copy()

    # Create suffix dictionaries for column renaming (e.g., region_score → region_score_treatment)
    trt_rename = {c: f"{c}_treatment" for c in value_cols}
    ctrl_rename = {c: f"{c}_control" for c in value_cols}

    # Select and rename treatment columns: merge keys + metadata + metrics with _treatment suffix
    trt_sub = trt[merge_keys + meta_cols + value_cols].rename(columns=trt_rename)
    # Select and rename control columns: merge keys + metrics with _control suffix
    # (metadata comes from treatment data via merge)
    ctrl_sub = ctrl[merge_keys + value_cols].rename(columns=ctrl_rename)

    # Inner join: keep only host × region pairs present in BOTH treatment and control
    # This ensures a matched-pairs design for statistical testing
    paired = trt_sub.merge(ctrl_sub, on=merge_keys, how="inner")

    n_hosts = paired[host_col].nunique()
    n_regions = paired["region_id"].nunique()
    logger.info(
        f"Paired reshaping: {n_hosts} hosts × {n_regions} regions "
        f"({len(paired):,} rows)"
    )

    if paired.empty:
        logger.error(
            f"No paired host × region entries found.  Ensure that "
            f"'{host_col}' values appear in both '{treatment_label}' "
            f"and '{control_label}' groups."
        )
        return pd.DataFrame()

    # Compute treatment − control contrast for subsequent statistical testing
    paired["contrast"] = (
        paired["region_score_treatment"] - paired["region_score_control"]
    )
    return paired


# ── 7. Across-host tests ────────────────────────────────────────────────


def _wilcoxon_test(contrasts: np.ndarray, alternative: str) -> float:
    """Wilcoxon signed-rank test with a specified direction.

    Parameters
    ----------
    alternative : {"greater", "less"}
        ``"greater"`` tests H₁: median contrast > 0 (treatment evolved more).
        ``"less"``    tests H₁: median contrast < 0 (control evolved more).

    Notes
    -----
    The full *contrasts* array (including zeros) is passed to
    ``scipy.stats.wilcoxon``, which uses ``zero_method="wilcox"`` by default
    and therefore discards zero differences internally — the same behaviour as
    ``single_sample.py``.  Manually filtering zeros before the call is not
    needed and would produce an inconsistent minimum-sample threshold between
    the Wilcoxon and t-test.  If too few non-zero differences remain after
    scipy's internal exclusion, a ``ValueError`` is raised and caught here,
    returning ``NaN``.

    The minimum-observations guard (formerly ``len(contrasts) < 3``) is now
    handled upstream by ``min_replicates`` in :func:`test_region_contrasts`,
    so no explicit length check is needed here.
    """
    if len(contrasts) == 0:
        return np.nan
    # Perfect null: all contrasts are zero → no signal in either direction, p = 1.0
    # (mirrors single_sample.py's var==0 & mean==0 guard; avoids returning
    # NaN when the answer is unambiguously "no effect")
    if np.var(contrasts) == 0 and np.mean(contrasts) == 0:
        return 1.0
    try:
        _, p = wilcoxon(contrasts, alternative=alternative)
        return float(p)
    except ValueError:
        return np.nan


def _ttest_test(contrasts: np.ndarray, alternative: str) -> float:
    """One-sample t-test with a specified direction.

    Parameters
    ----------
    alternative : {"greater", "less"}
        ``"greater"`` tests H₁: mean contrast > 0 (treatment evolved more).
        ``"less"``    tests H₁: mean contrast < 0 (control evolved more).

    Notes
    -----
    The minimum-observations guard is handled upstream by ``min_replicates``
    in :func:`test_region_contrasts`, so no explicit length check is needed.
    """
    if len(contrasts) == 0:
        return np.nan
    # Perfect null: all contrasts are zero → p = 1.0 (same guard as
    # single_sample.py; without this, ttest_1samp returns NaN because
    # t = mean/stderr = 0/0 when std == 0, which misrepresents the null
    # as "undefined" rather than "no effect")
    if np.var(contrasts) == 0 and np.mean(contrasts) == 0:
        return 1.0
    try:
        result = ttest_1samp(contrasts, popmean=0, alternative=alternative)
        return float(result.pvalue)
    except ValueError:
        return np.nan


def test_region_contrasts(
    paired_df: pd.DataFrame,
    host_col: str,
    contig_col: str,
    min_replicates: int = 4,
) -> pd.DataFrame:
    """Test each region for a consistently positive contrast across hosts.

    For each region, aggregates contrasts (treatment − control allele-frequency
    scores) across replicates and runs two independent one-sided tests:

    - **"greater"** (H₁: contrast > 0): treatment evolved more than control.
    - **"less"**    (H₁: contrast < 0): control evolved more than treatment.

    Running both directions as separate one-sided tests is more powerful than
    a single two-sided test — each direction uses its full α budget — and more
    interpretable: output p-value / q-value columns are labelled by direction so
    no post-hoc sign-checking of ``mean_contrast`` is required.

    Both a **Wilcoxon signed-rank test** (nonparametric) and a **t-test**
    (parametric) are run for each direction.  FDR correction is applied
    independently per direction and per region type downstream.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Output of :func:`reshape_treatment_control` with one row per
        (replicate × region) pair and a ``contrast`` column.
    host_col : str
        Column name for the paired-unit identifier (e.g., "replicate").
    contig_col : str
        Column name for genomic contig identifier. Optional metadata.
    min_replicates : int
        Minimum number of replicates required for a region to be included in
        the output.  Regions with fewer replicates are silently dropped.
        Defaults to 4.

    Returns
    -------
    pd.DataFrame
        One row per region (identified by ``region_id`` and ``region_type``)
        with the following columns:

        - ``region_id``: Region identifier (gene name or window position range)
        - ``region_type``: ``"gene"`` or ``"window"``
        - *contig_col*, ``region_start``, ``region_end``: Genomic location metadata
          (included if present in input)
        - ``n_replicates``: Number of replicates contributing to the test
        - ``mean_contrast``: Arithmetic mean of contrasts across replicates
        - ``median_contrast``: Median of contrasts across replicates
        - ``p_value_wilcoxon_greater``: Wilcoxon p-value for H₁: treatment > control
        - ``p_value_wilcoxon_less``:    Wilcoxon p-value for H₁: control > treatment
        - ``p_value_ttest_greater``:    t-test p-value for H₁: treatment > control
        - ``p_value_ttest_less``:       t-test p-value for H₁: control > treatment

        FDR correction is applied independently per direction and per region
        type downstream, yielding ``q_value_*`` columns that mirror the above.

    Example
    -------
    Input paired_df (paired format, 4 rows):

    .. code-block:: text

        replicate  region_id      contrast  region_type
        host_A     contig_1:1-1000   0.15   window
        host_B     contig_1:1-1000   0.22   window
        host_C     contig_1:1-1000  -0.05   window
        host_D     contig_1:1-1000   0.18   window

    Output (1 row):

    .. code-block:: text

        region_id          region_type  n_replicates  mean_contrast  median_contrast  p_value_wilcoxon_greater  p_value_wilcoxon_less
        contig_1:1-1000    window       4             0.1250         0.1650           0.0625                    0.9688
    """
    # Gather all region-identifying column names (always include region_id and region_type)
    # and optionally add genomic location metadata if present
    region_meta_cols = ["region_id", "region_type"]
    for col in [contig_col, "region_start", "region_end"]:
        if col in paired_df.columns:
            region_meta_cols.append(col)

    results: list[dict] = []
    n_dropped = 0
    # Iterate over each unique region (identified by region_meta_cols)
    for keys, grp in paired_df.groupby(region_meta_cols):
        # Extract all contrasts (treatment − control) for this region across replicates
        contrasts = grp["contrast"].dropna().values
        n_replicates = len(contrasts)

        # Skip regions that do not meet the minimum replicate threshold
        if n_replicates < min_replicates:
            n_dropped += 1
            continue

        # Map region identifiers back to a dict (groupby with 2+ columns always returns tuple)
        row = dict(zip(region_meta_cols, keys))

        # Store basic statistics across replicates for this region
        row["n_replicates"] = n_replicates
        row["mean_contrast"] = float(np.mean(contrasts))
        row["median_contrast"] = float(np.median(contrasts))

        # Two independent one-sided tests — each uses its full α budget for one direction:
        #   "greater": treatment evolved more than control (contrast > 0)
        #   "less":    control evolved more than treatment (contrast < 0)
        # FDR is applied separately to each direction's p-value pool downstream.
        # Wilcoxon: nonparametric signed-rank test (robust to non-normality)
        # t-test:   parametric one-sample test (assumes normality, sensitive to mean)
        row["p_value_wilcoxon_greater"] = _wilcoxon_test(contrasts, "greater")
        row["p_value_wilcoxon_less"] = _wilcoxon_test(contrasts, "less")
        row["p_value_ttest_greater"] = _ttest_test(contrasts, "greater")
        row["p_value_ttest_less"] = _ttest_test(contrasts, "less")

        results.append(row)

    if n_dropped:
        logger.info(
            f"Dropped {n_dropped:,} regions with fewer than {min_replicates} replicates"
        )

    if results:
        summary = pd.DataFrame(results)
    else:
        # Return an empty DataFrame with the correct column schema so that a
        # downstream outer-merge with fisher_df (and subsequent groupby in the
        # FDR step) does not fail with a KeyError on 'region_id' or similar.
        empty_cols = region_meta_cols + [
            "n_replicates",
            "mean_contrast",
            "median_contrast",
            PVAL_WILCOXON_GREATER_COL,
            PVAL_WILCOXON_LESS_COL,
            PVAL_TTEST_GREATER_COL,
            PVAL_TTEST_LESS_COL,
        ]
        summary = pd.DataFrame(columns=empty_cols)
    logger.info(f"Tested {len(summary):,} regions across replicates")
    return summary


# ── 8. Fisher combined p-values (secondary / exploratory) ───────────────


def fisher_combine_empirical_pvalues(
    paired_df: pd.DataFrame,
    contig_col: str,
    min_replicates: int = 4,
) -> pd.DataFrame:
    """Combine percentile-derived empirical p-like values via Fisher's method.

    **Secondary / exploratory analysis.**  The primary result comes from
    the contrast-based test in :func:`test_region_contrasts`.

    For each region, derives empirical p-like values from treatment and control
    percentiles separately (``p_empirical ≈ 1 − percentile / 100``) and combines
    them across hosts using Fisher's chi-squared statistic (``−2 Σ ln(pᵢ)``).
    Running both groups mirrors the directional philosophy of
    :func:`test_region_contrasts`: a high treatment Fisher p signals that the
    treatment group evolved more at this region; a high control Fisher p signals
    the same for the control group.

    Parameters
    ----------
    paired_df : pd.DataFrame
        Output of :func:`reshape_treatment_control` with one row per
        (host × region) pair and ``percentile_treatment`` /
        ``percentile_control`` columns.
    contig_col : str
        Column name for genomic contig identifier.  Optional metadata.

    Returns
    -------
    pd.DataFrame
        One row per region (identified by ``region_id`` and ``region_type``)
        with the following columns:

        - ``region_id``: Region identifier
        - ``region_type``: ``"gene"`` or ``"window"``
        - *contig_col*, ``region_start``, ``region_end``: Genomic location metadata
          (included if present in input)
        - ``n_replicates_treatment``: Number of non-null treatment percentile values
        - ``n_replicates_control``: Number of non-null control percentile values
        - ``p_value_fisher_treatment``: Combined Fisher p-value across hosts for
          the treatment group (high percentile → small p → treatment evolved a lot)
        - ``p_value_fisher_control``: Combined Fisher p-value across hosts for
          the control group

        A region is included if *either* group has ≥ *min_replicates* non-null
        percentile values; the other group's column is NaN for that row if it
        has fewer than *min_replicates* values.

    Example
    -------
    Input paired_df (3 hosts for a single region):

    .. code-block:: text

        region_id  percentile_treatment  percentile_control
        gene_X     80                    40
        gene_X     75                    35
        gene_X     85                    45

    Treatment empirical p-values: 0.20, 0.25, 0.15 → Fisher p ≈ 0.050
    Control empirical p-values:   0.60, 0.65, 0.55 → Fisher p ≈ 0.713

    Output (1 row):

    .. code-block:: text

        region_id  region_type  p_value_fisher_treatment  p_value_fisher_control
        gene_X     gene         0.0502                    0.7130
    """
    # Gather all region-identifying column names (always include region_id and region_type)
    # and optionally add genomic location metadata if present
    region_meta_cols = ["region_id", "region_type"]
    for col in [contig_col, "region_start", "region_end"]:
        if col in paired_df.columns:
            region_meta_cols.append(col)

    def _fisher_p(pct: np.ndarray) -> float:
        """Convert percentiles to empirical p-values and combine via Fisher's method.

        Returns NaN if fewer than *min_replicates* non-null values are available.
        """
        if len(pct) < min_replicates:
            return np.nan
        # Convert percentile ranks to empirical p-like values: p ≈ 1 − (percentile / 100)
        # Clamp to (1e-10, 1−1e-10) to avoid log(0) in Fisher's chi-squared statistic
        emp_p = np.clip(1.0 - pct / 100.0, 1e-10, 1.0 - 1e-10)
        # Fisher's method: combines independent p-values via −2 Σ ln(pᵢ) ~ χ²(2k)
        # Returns (statistic, p_value); we only use the p_value (second element)
        _, p = combine_pvalues(emp_p, method="fisher")
        return float(p)

    results: list[dict] = []
    # Iterate over each unique region (identified by region_meta_cols)
    for keys, grp in paired_df.groupby(region_meta_cols):
        # Extract non-null percentiles for treatment and control separately
        pct_trt = grp["percentile_treatment"].dropna().values
        pct_ctrl = grp["percentile_control"].dropna().values

        # Skip regions where neither group has enough data to combine
        if len(pct_trt) < min_replicates and len(pct_ctrl) < min_replicates:
            continue

        # Map region identifiers back to a dict (groupby with 2+ columns always returns tuple)
        row = dict(zip(region_meta_cols, keys))
        # Track the number of replicates per group (may differ if unpaired)
        row["n_replicates_treatment"] = len(pct_trt)
        row["n_replicates_control"] = len(pct_ctrl)
        # Compute Fisher p for each group independently; NaN if < min_replicates hosts available
        row["p_value_fisher_treatment"] = _fisher_p(pct_trt)
        row["p_value_fisher_control"] = _fisher_p(pct_ctrl)
        results.append(row)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def _fisher_merge_keys(
    fisher_df: pd.DataFrame, summary_df: pd.DataFrame
) -> list[str]:
    """Columns to join *fisher_df* onto *summary_df*.

    The two frames share only their region-identifying columns (region_id,
    region_type, and any genomic-location metadata).  Everything else is
    per-frame data: *summary_df* carries a single ``n_replicates`` plus the
    contrast statistics, while *fisher_df* carries per-group counts
    (``n_replicates_treatment`` / ``n_replicates_control``) and the Fisher
    p-values.  Joining on the shared columns keeps those data columns as
    payload instead of treating the fisher-only counts as keys — which would
    raise ``KeyError`` because they are absent from *summary_df*.
    """
    return [c for c in fisher_df.columns if c in summary_df.columns]


# ── 9. Multiple-testing correction ──────────────────────────────────────


def adjust_pvalues(
    df: pd.DataFrame,
    p_col: str = "p_value",
    q_col: str = "q_value",
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply FDR correction to a p-value column and write q-values.

    Wraps ``statsmodels.stats.multitest.multipletests``.  Only non-NaN
    p-values enter the correction; NaN rows receive NaN q-values and are
    excluded from the effective number of hypotheses, so they do not
    dilute or inflate the correction.

    This function is called once per (region_type × direction) combination
    in Step 9 so that each independent hypothesis pool has its own FDR
    budget.  Calling it on the full table across directions would pool
    ``_greater`` and ``_less`` tests together, making the correction
    unnecessarily conservative for each individual direction.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Modified in-place and returned.
    p_col : str
        Name of the column containing raw p-values to correct.
    q_col : str
        Name of the new column to write adjusted p-values (q-values) into.
    method : str
        Correction method accepted by ``multipletests`` (default: ``"fdr_bh"``
        for Benjamini–Hochberg).

    Returns
    -------
    pd.DataFrame
        Same frame as *df* with *q_col* appended.

    Example
    -------
    Given a ``"gene"`` region_type subset with 1 000 regions and
    ``p_col="p_value_wilcoxon_greater"``, the Benjamini–Hochberg procedure
    ranks those 1 000 p-values and assigns q-values such that calling all
    regions with q < 0.05 significant controls the false-discovery rate at
    5 % within that directional test pool.
    """
    # Mask rows with a valid (non-NaN) p-value; NaN rows are skipped entirely
    # so they do not count as hypotheses and do not affect the correction.
    mask = df[p_col].notna()
    if mask.sum() == 0:
        # No testable rows — write NaN and return early to avoid calling
        # multipletests on an empty array (which raises an error).
        df[q_col] = np.nan
        return df

    # multipletests returns (reject, p_corrected, alpha_sidak, alpha_bonf);
    # we only need p_corrected (index 1), which are the BH-adjusted q-values.
    _, q_vals, _, _ = multipletests(df.loc[mask, p_col].values, method=method)

    # Write q-values back at the correct row positions; NaN rows get NaN.
    df.loc[mask, q_col] = q_vals
    df.loc[~mask, q_col] = np.nan

    n_sig = (df[q_col] < 0.05).sum()
    logger.info(
        f"FDR correction ({method}) on '{p_col}': "
        f"{n_sig} regions with q < 0.05 out of {mask.sum()}"
    )
    return df


# ── 10. Output ──────────────────────────────────────────────────────────


def write_outputs(
    per_host_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
    region_types: list[str],
) -> None:
    """Serialize per-host and region-summary results to disk.

    Writes two output tables:
    - **Per-host table**: One row per (host × region) pair; indexed by replicate ID
      and region identifiers. Gzip-compressed (`.tsv.gz`) due to larger size.
    - **Region summary table**: One row per region (aggregated across hosts) with
      statistical test results; includes p-values, q-values, and effect sizes.
      Stored as plain text (`.tsv`) for easy viewing.

    Parameters
    ----------
    per_host_df : pd.DataFrame
        Per-host region results, typically the output of ``main()`` with one row
        per (host × region) pair. Contains phenotype data, percentiles, and scores.
    summary_df : pd.DataFrame
        Region-level summary statistics (one row per region) from
        :func:`test_region_contrasts` and downstream FDR correction, including
        ``n_hosts``, ``mean_contrast``, ``median_contrast``, and p/q-value columns.
    output_dir : str | Path
        Directory for output files. Created if it does not exist.
    prefix : str
        File-name prefix (e.g., ``"regional_contrast"``). Output files will be
        named ``{prefix}_per_host_region.tsv.gz`` and ``{prefix}_region_summary.tsv``.
    region_types : list[str]
        Region types the run is expected to produce (derived from ``--mode``:
        ``["gene"]``, ``["window"]``, or ``["gene", "window"]``).  A per-type
        summary file is written for **every** listed type, even when no region of
        that type survived filtering — in that case an empty file with the correct
        header is written so the Snakemake output contract is always satisfied.
    host_col : str
        Name of the host/replicate column in the input DataFrames (e.g., ``"replicate"``,
        ``"host_id"``). Passed through unchanged to output.
    contig_col : str
        Name of the contig column in the input DataFrames (e.g., ``"contig_id"``,
        ``"contig"``). Passed through unchanged to output.

    Returns
    -------
    None
        Results written to disk; no return value.

    Example
    -------
    Given per-host results with 1 000 (host × region) pairs and region summary
    with 100 regions (50 genes, 50 windows), this function writes:

    .. code-block:: text

        output_dir/
            regional_contrast_per_host_region.tsv.gz          (1000 rows, compressed)
            regional_contrast_region_summary.tsv               (100 rows, combined)
            regional_contrast_gene_region_summary.tsv          (50 rows)
            regional_contrast_window_region_summary.tsv        (50 rows)
    """
    # Ensure output directory exists
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write per-host table (larger: one row per host × region pair)
    # Gzip compression saves significant disk space for large datasets
    host_path = out / f"{prefix}_per_host_region.tsv.gz"
    per_host_df.to_csv(host_path, sep="\t", index=False, compression="gzip")
    logger.info(f"Per-host table → {host_path} ({len(per_host_df):,} rows)")

    # Write combined region summary (all region types; useful for manual inspection)
    summ_path = out / f"{prefix}_region_summary.tsv"
    summary_df.to_csv(summ_path, sep="\t", index=False)
    logger.info(f"Summary table  → {summ_path} ({len(summary_df):,} rows)")

    # Write per-region-type summary files so downstream scoring rules can consume
    # each type independently without subprocess filtering or temp files.
    #
    # Iterate the *expected* region_types (from --mode), not just the types
    # present in summary_df.  A MAG can yield zero surviving regions of a type
    # (e.g. no window region reaches min_replicates) — writing only the observed
    # types would omit a file that the Snakemake rule declares as a mandatory
    # output, failing the job with MissingOutputException.  An empty subset still
    # serializes its header row, satisfying the output contract.
    has_region_type = "region_type" in summary_df.columns
    for rt in region_types:
        if has_region_type:
            rt_df = summary_df[summary_df["region_type"] == rt]
        else:
            rt_df = summary_df.iloc[0:0]  # empty, but preserves the column schema
        rt_path = out / f"{prefix}_{rt}_region_summary.tsv"
        rt_df.to_csv(rt_path, sep="\t", index=False)
        logger.info(f"{rt} summary    → {rt_path} ({len(rt_df):,} rows)")


# ── main ────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the regional contrast analysis."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Detect genes / sliding windows with differential "
            "allele-frequency evolution between treatment and control "
            "groups across paired hosts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Input / Output ──
    io_grp = parser.add_argument_group("Input / Output")
    io_grp.add_argument(
        "--input",
        required=True,
        type=str,
        metavar="PATH",
        help=(
            "Path to mean allele-frequency-changes TSV(.gz) produced by "
            "allele_freq.py, e.g. "
            "*_allele_frequency_changes_mean.tsv.gz.  "
        ),
    )
    io_grp.add_argument(
        "--output_dir",
        required=True,
        type=str,
        metavar="DIR",
        help="Output directory.",
    )
    io_grp.add_argument(
        "--prefix",
        type=str,
        default="regional_contrast",
        help="Output file-name prefix.",
    )

    # ── Groups ──
    grp = parser.add_argument_group("Group labels")
    grp.add_argument(
        "--treatment_group",
        required=True,
        type=str,
        help="Value in the 'group' column identifying the treatment group.",
    )
    grp.add_argument(
        "--control_group",
        required=True,
        type=str,
        help="Value in the 'group' column identifying the control group.",
    )

    # ── Region definition ──
    reg = parser.add_argument_group("Region definition")
    reg.add_argument(
        "--mode",
        choices=["gene", "window", "both"],
        default="gene",
        help="Region type(s) to analyse.",
    )
    reg.add_argument(
        "--window_size",
        type=int,
        default=1000,
        help="Non-overlapping tile width (bp). Sites are assigned to tiles [1,W], [W+1,2W], ...",
    )

    # ── Aggregation ──
    agg = parser.add_argument_group("Aggregation")
    agg.add_argument(
        "--agg_method",
        choices=["median", "mean", "trimmed_mean"],
        default="median",
        help="How to summarise site scores within a region.",
    )
    agg.add_argument(
        "--trim_fraction",
        type=float,
        default=0.1,
        help="Fraction cut from each tail (trimmed_mean only).",
    )
    agg.add_argument(
        "--min_informative_sites",
        type=int,
        default=5,
        help=(
            "Minimum number of sites present in the input data per region. "
            "Counted as non-null rows after the inner merge with the region "
            "mapping — NOT as non-zero site scores.  A region where every "
            "site has site_score==0 (perfect evolutionary stasis) still has "
            "full n_informative_sites and passes this filter.  A low count "
            "only occurs when positions were physically removed upstream by "
            "the zero-diff filter in allele_freq.py.  Set 0 to disable."
        ),
    )
    agg.add_argument(
        "--min_informative_fraction",
        type=float,
        default=0.0,
        help="Minimum informative fraction of region length (0.0 disables).",
    )

    # ── Statistical tests ──
    tst = parser.add_argument_group("Statistical tests")
    tst.add_argument(
        "--min_replicates",
        type=int,
        default=4,
        help=(
            "Minimum number of replicates (hosts) required for a region to "
            "be included in the statistical test output.  Regions with fewer "
            "replicates are dropped before testing and FDR correction."
        ),
    )
    tst.add_argument(
        "--use_fisher",
        action="store_true",
        default=False,
        help=(
            "Also compute Fisher combined p-values from percentile-derived "
            "empirical p-like values (secondary / exploratory)."
        ),
    )

    args = parser.parse_args()

    # Required columns (exclude gene_col unless gene mode is active)
    required = [HOST_COL, GROUP_COL, CONTIG_COL, POSITION_COL]
    if args.mode in ("gene", "both"):
        required.append(GENE_COL)

    # 1. Load
    logger.info("── Step 1: Loading input table ──")
    df = load_input_table(args.input, required, DIFF_COLS)

    # Keep only treatment & control rows
    mask = df[GROUP_COL].isin([args.treatment_group, args.control_group])
    n_before = len(df)
    df = df[mask].copy()
    logger.info(
        f"Filtered to groups '{args.treatment_group}' / "
        f"'{args.control_group}': {n_before:,} → {len(df):,} rows"
    )
    if df.empty:
        raise ValueError(
            f"No rows remain after filtering to groups "
            f"'{args.treatment_group}' and '{args.control_group}'."
        )

    # 2. Site scores
    logger.info("── Step 2: Computing site scores ──")
    df = compute_site_scores(df, DIFF_COLS, score_col=SITE_SCORE_COL)

    # 3. Region mapping(s)
    logger.info("── Step 3: Building region definitions ──")
    region_maps: list[pd.DataFrame] = []

    if args.mode in ("gene", "both"):
        gene_map = build_gene_regions(
            df,
            GENE_COL,
            CONTIG_COL,
            POSITION_COL,
        )
        if not gene_map.empty:
            region_maps.append(gene_map)

    if args.mode in ("window", "both"):
        win_map = build_sliding_windows(
            df,
            CONTIG_COL,
            POSITION_COL,
            args.window_size,
        )
        if not win_map.empty:
            region_maps.append(win_map)

    if not region_maps:
        raise ValueError(
            "No regions could be built.  Check gene annotations or "
            "window parameters."
        )

    region_mapping = pd.concat(region_maps, ignore_index=True)

    # 4. Aggregate
    logger.info("── Step 4: Aggregating region scores ──")
    agg_df = aggregate_region_scores(
        df,
        region_mapping,
        host_col=HOST_COL,
        group_col=GROUP_COL,
        contig_col=CONTIG_COL,
        position_col=POSITION_COL,
        score_col=SITE_SCORE_COL,
        agg_method=args.agg_method,
        trim_fraction=args.trim_fraction,
        min_sites=args.min_informative_sites,
        min_fraction=args.min_informative_fraction,
    )
    if agg_df.empty:
        raise ValueError("No regions passed eligibility filters.")

    # 5. Percentiles
    logger.info("── Step 5: Computing percentiles ──")
    agg_df = compute_percentiles(
        agg_df,
        HOST_COL,
        GROUP_COL,
        score_col=REGION_SCORE_COL,
        out_col=PERCENTILE_COL,
    )

    # 6. Reshape
    logger.info("── Step 6: Reshaping treatment / control ──")
    paired_df = reshape_treatment_control(
        agg_df,
        host_col=HOST_COL,
        group_col=GROUP_COL,
        contig_col=CONTIG_COL,
        treatment_label=args.treatment_group,
        control_label=args.control_group,
    )
    if paired_df.empty:
        raise ValueError("No paired host × region entries after reshaping.")

    # 7. Test
    logger.info("── Step 7: Testing region contrasts ──")
    summary_df = test_region_contrasts(
        paired_df,
        host_col=HOST_COL,
        contig_col=CONTIG_COL,
        min_replicates=args.min_replicates,
    )

    # 8. Optional Fisher (secondary / exploratory)
    if args.use_fisher:
        logger.info("── Step 8: Fisher combined p-values (exploratory) ──")
        fisher_df = fisher_combine_empirical_pvalues(
            paired_df,
            contig_col=CONTIG_COL,
            min_replicates=args.min_replicates,
        )
        if not fisher_df.empty:
            fisher_keys = _fisher_merge_keys(fisher_df, summary_df)
            summary_df = summary_df.merge(
                fisher_df,
                on=fisher_keys,
                how="outer",
            )
            # FDR for Fisher p-values is applied in step 9 together with
            # all other corrections, per region_type.

    # 9. FDR — applied separately per region_type so that gene and window
    #    hypotheses are corrected within their own independent test sets.
    #    Also applied separately per direction ("greater" / "less") so that
    #    each directional hypothesis pool has its own FDR budget.
    logger.info("── Step 9: FDR correction ──")
    fdr_parts = []
    for _rt, grp in summary_df.groupby("region_type"):
        grp = grp.copy()
        grp = adjust_pvalues(
            grp,
            p_col=PVAL_WILCOXON_GREATER_COL,
            q_col=QVAL_WILCOXON_GREATER_COL,
            method=FDR_METHOD,
        )
        grp = adjust_pvalues(
            grp,
            p_col=PVAL_WILCOXON_LESS_COL,
            q_col=QVAL_WILCOXON_LESS_COL,
            method=FDR_METHOD,
        )
        grp = adjust_pvalues(
            grp,
            p_col=PVAL_TTEST_GREATER_COL,
            q_col=QVAL_TTEST_GREATER_COL,
            method=FDR_METHOD,
        )
        grp = adjust_pvalues(
            grp, p_col=PVAL_TTEST_LESS_COL, q_col=QVAL_TTEST_LESS_COL, method=FDR_METHOD
        )
        if FISHER_PVAL_TREATMENT_COL in grp.columns:
            grp = adjust_pvalues(
                grp,
                p_col=FISHER_PVAL_TREATMENT_COL,
                q_col=FISHER_QVAL_TREATMENT_COL,
                method=FDR_METHOD,
            )
        if FISHER_PVAL_CONTROL_COL in grp.columns:
            grp = adjust_pvalues(
                grp,
                p_col=FISHER_PVAL_CONTROL_COL,
                q_col=FISHER_QVAL_CONTROL_COL,
                method=FDR_METHOD,
            )
        fdr_parts.append(grp)
    summary_df = pd.concat(fdr_parts, ignore_index=True) if fdr_parts else summary_df

    # 10. Write
    logger.info("── Step 10: Writing outputs ──")
    # Per-type summary files are required for every type implied by --mode, even
    # when a type has no surviving regions (see write_outputs).
    mode_region_types = {
        "gene": ["gene"],
        "window": ["window"],
        "both": ["gene", "window"],
    }[args.mode]
    write_outputs(
        paired_df,
        summary_df,
        output_dir=args.output_dir,
        prefix=args.prefix,
        region_types=mode_region_types,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
