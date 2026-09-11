#!/usr/bin/env python3

"""
QC Metrics Module for AlleleFlux

This module provides reusable, pure functions for calculating quality control metrics
on metagenomic data. These functions are used by both the main QC workflow
(quality_control.py) and specialized QC scripts (e.g., positions_qc.py).

Function Categories:
- Data Loading: load_and_validate_profile
- Metric Calculation: calculate_breadth_metrics, calculate_coverage_metrics,
  calculate_median_and_std_metrics, calculate_length_weighted_coverage
- Threshold Validation: check_breadth_threshold, check_coverage_threshold
- Aggregation: aggregate_mag_stats

All functions are pure (no side effects) and include comprehensive type hints
and docstrings. They handle edge cases gracefully and provide clear error messages.
"""

import logging
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

logger = logging.getLogger(__name__)


def load_and_validate_profile(
    profile_path: str,
    mag_id: str,
    sample_id: str,
    contig_to_mag: dict,
    usecols: Optional[list] = None,
    mag_contigs: Optional[set] = None,
) -> pd.DataFrame:
    """
    Load profile TSV (optionally gzip-compressed) and filter to MAG's contigs only.

    Uses pyarrow for fast columnar reads — ~7–9× faster than pandas read_csv on
    gzipped TSVs.  When ``usecols`` is supplied only those columns are read off
    disk; when ``mag_contigs`` is supplied a cheap unique-contig pre-check avoids
    the per-row contig map on the common case where all rows already belong to the
    correct MAG.

    Parameters
    ----------
    profile_path : str
        Path to the profile TSV (or .tsv.gz / .gz) file.
    mag_id : str
        MAG identifier to filter for.
    sample_id : str
        Sample identifier (for error messages).
    contig_to_mag : dict
        Mapping of contig_id -> mag_id.  Used as fallback when *mag_contigs* is
        not provided.
    usecols : list of str, optional
        Columns to read.  Defaults to all columns.  Pass
        ``["contig", "total_coverage"]`` from quality_control.py to cut I/O by ~8×.
    mag_contigs : set, optional
        Pre-inverted set of contigs that belong to *mag_id*.  When provided the
        function skips the per-row ``map`` call on the common fast path (all rows
        match) and uses ``pc.is_in`` only when a foreign contig is actually found.

    Returns
    -------
    pd.DataFrame
        Validated profile containing only rows for the specified MAG.

    Raises
    ------
    ValueError
        If no rows remain after filtering to the MAG's contigs.
    """
    # Explicit types for columns that can be misidentified by the inference engine.
    # total_coverage is left to inference (int64 from real data, float64 from tests).
    # gene_id must be forced to string because values like "001" would otherwise
    # be inferred as the integer 1, silently losing the leading zero.
    _type_map: dict = {
        "contig": pa.string(),
        "gene_id": pa.string(),
    }
    convert_opts = pacsv.ConvertOptions(
        include_columns=usecols,  # None → read all columns
        column_types=_type_map,
    )

    # pa.input_stream handles .gz/.tsv.gz decompression robustly regardless of
    # the double extension — unlike pandas which requires explicit engine choice.
    with pa.input_stream(profile_path) as stream:
        tbl = pacsv.read_csv(
            stream,
            parse_options=pacsv.ParseOptions(delimiter="\t"),
            convert_options=convert_opts,
        )

    if tbl.num_rows == 0:
        msg = f"Empty profile for MAG {mag_id}. file={profile_path}"
        logger.error(msg)
        raise ValueError(msg)

    # Filter rows to the MAG's contigs only when a 'contig' column was read.
    if "contig" in tbl.schema.names:
        contig_col = tbl.column("contig")

        # Cheap pre-check: collect unique contig names and test membership.
        # On the normal path every contig already belongs to this MAG, so
        # the expensive pc.is_in row filter never fires.
        unique_contigs = pc.unique(contig_col).to_pylist()

        if mag_contigs is not None:
            foreign = [c for c in unique_contigs if c not in mag_contigs]
        else:
            foreign = [c for c in unique_contigs if contig_to_mag.get(c) != mag_id]

        if foreign:
            # At least one foreign contig — filter row-by-row via Arrow kernel.
            if mag_contigs is not None:
                allowed = pa.array(list(mag_contigs), type=pa.string())
            else:
                allowed = pa.array(
                    [c for c in unique_contigs if c not in set(foreign)],
                    type=pa.string(),
                )
            before = tbl.num_rows
            tbl = tbl.filter(pc.is_in(contig_col, value_set=allowed))
            dropped = before - tbl.num_rows
            logger.warning(
                f"Profile contains {dropped} rows not belonging to MAG {mag_id}; "
                f"filtering them out. file={profile_path} sample={sample_id}"
            )
            if tbl.num_rows == 0:
                msg = (
                    f"No rows remain after filtering profile to MAG {mag_id}. "
                    f"file={profile_path}"
                )
                logger.error(msg)
                raise ValueError(msg)

    return tbl.to_pandas()


def calculate_breadth_metrics(
    df: pd.DataFrame,
    denom: int,
    mag_size: int,
    using_filter: bool,
    positions_with_coverage_genome: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate breadth and optionally breadth_genome.

    Breadth is defined as the proportion of positions with at least 1x coverage.
    When using a position filter, two breadth metrics are calculated:
    - breadth: based on filtered positions (denominator depends on mode)
    - breadth_genome: based on entire genome (always uses genome size)

    Parameters
    ----------
    df : pd.DataFrame
        Profile dataframe with 'total_coverage' column
    denom : int
        Denominator for breadth calculation (positions count or genome size)
    mag_size : int
        Size of the MAG in base pairs (for breadth_genome calculation)
    using_filter : bool
        Whether a positions filter is active
    positions_with_coverage_genome : int, optional
        Number of positions with coverage >= 1 across entire genome
        (required when using_filter is True)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'breadth': Breadth based on filtered positions/denominator
        - 'breadth_genome': Breadth across entire genome (only when using_filter=True)

    Notes
    -----
    - Breadth is calculated as: (positions with coverage >= 1) / denominator
    - Returns np.nan if denominator is 0
    - breadth_genome is None when not using a filter
    """
    # Calculate breadth for filtered positions (or all positions if no filter)
    positions_with_coverage = df[df["total_coverage"] >= 1].shape[0]
    breadth = positions_with_coverage / denom if denom > 0 else np.nan

    result = {"breadth": breadth, "breadth_genome": None}

    # Calculate genome-wide breadth when using position filter
    if using_filter:
        if positions_with_coverage_genome is None:
            raise ValueError(
                "positions_with_coverage_genome required when using_filter=True"
            )
        breadth_genome = (
            positions_with_coverage_genome / mag_size if mag_size > 0 else np.nan
        )
        result["breadth_genome"] = breadth_genome

    return result


def calculate_coverage_metrics(
    df: pd.DataFrame,
    denom: int,
    mag_size: int,
    using_filter: bool,
    total_coverage_sum_genome: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    Calculate average_coverage and optionally average_coverage_genome.

    Average coverage is calculated as the sum of all coverage values divided by
    the denominator. When using a position filter, two metrics are calculated:
    - average_coverage: based on filtered positions (denominator depends on mode)
    - average_coverage_genome: based on entire genome (always uses genome size)

    Parameters
    ----------
    df : pd.DataFrame
        Profile dataframe with 'total_coverage' column
    denom : int
        Denominator for average coverage calculation
    mag_size : int
        Size of the MAG in base pairs (for average_coverage_genome)
    using_filter : bool
        Whether a positions filter is active
    total_coverage_sum_genome : float, optional
        Sum of coverage across entire genome before filtering
        (required when using_filter is True)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'average_coverage': Average coverage for filtered positions
        - 'average_coverage_genome': Average coverage across entire genome
          (only when using_filter=True)

    Notes
    -----
    - Average coverage = sum(total_coverage) / denominator
    - Returns np.nan if denominator is 0
    - average_coverage_genome is None when not using a filter
    """
    # Calculate average coverage for filtered positions
    total_coverage_sum = df["total_coverage"].sum()
    avg_cov = total_coverage_sum / denom if denom > 0 else np.nan

    result = {"average_coverage": avg_cov, "average_coverage_genome": None}

    # Calculate genome-wide average coverage when using position filter
    if using_filter:
        if total_coverage_sum_genome is None:
            raise ValueError(
                "total_coverage_sum_genome required when using_filter=True"
            )
        avg_cov_genome = (
            total_coverage_sum_genome / mag_size if mag_size > 0 else np.nan
        )
        result["average_coverage_genome"] = avg_cov_genome

    return result


def calculate_median_and_std_metrics(
    df: pd.DataFrame, denom: int, avg_cov: float
) -> Dict[str, float]:
    """
    Calculate median and standard deviation metrics.

    This function calculates four related metrics:
    1. median_coverage: Median among positions with coverage > 0
    2. median_coverage_including_zeros: Median including zeros for absent positions
    3. coverage_std: Population std dev among positions with coverage > 0
    4. coverage_std_including_zeros: Population std dev including zeros

    Parameters
    ----------
    df : pd.DataFrame
        Profile dataframe with 'total_coverage' column
    denom : int
        Total number of positions (for including zeros calculations)
    avg_cov : float
        Average coverage (pre-calculated, used for std calculation)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'median_coverage': Median of non-zero coverage values
        - 'median_coverage_including_zeros': Median including zeros for absent positions
        - 'coverage_std': Population std dev of non-zero coverage (ddof=0)
        - 'coverage_std_including_zeros': Population std dev including zeros

    Notes
    -----
    - All std calculations use population std (ddof=0)
    - Returns np.nan for metrics when no data is available
    - "Including zeros" means treating absent positions as having 0 coverage
    - coverage_std_including_zeros uses variance formula: Var(X) = E[X²] - E[X]²
    """
    result = {
        "median_coverage": np.nan,
        "median_coverage_including_zeros": np.nan,
        "coverage_std": np.nan,
        "coverage_std_including_zeros": np.nan,
    }

    # Get positions with coverage > 0
    pos_cov = df.loc[df["total_coverage"] > 0, "total_coverage"]

    # Median coverage among covered positions only
    if not pos_cov.empty:
        result["median_coverage"] = float(pos_cov.median())

        # Standard deviation among covered positions only (population std, ddof=0)
        result["coverage_std"] = float(pos_cov.std(ddof=0))

    # Median coverage including zeros for absent positions
    observed_positions = len(df)
    absent_positions = int(denom - observed_positions) if denom is not None else 0

    if absent_positions >= 0:
        # Create array with observed coverage values plus zeros for absent positions
        all_coverage = np.concatenate(
            [df["total_coverage"].values, np.zeros(absent_positions)]
        )
        result["median_coverage_including_zeros"] = float(np.median(all_coverage))
    else:
        # Handle case where profile has more positions than denominator
        logger.warning(
            f"Profile has more positions ({observed_positions}) than denominator ({denom})"
        )
        result["median_coverage_including_zeros"] = result["median_coverage"]

    # Population std including zeros across entire genome length
    # Uses formula: Var(X) = E[X²] - E[X]²
    if denom > 0 and np.isfinite(avg_cov):
        ex2 = float((df["total_coverage"] ** 2).sum()) / denom
        var = max(ex2 - (avg_cov * avg_cov), 0.0)
        result["coverage_std_including_zeros"] = float(np.sqrt(var))

    return result


def calculate_length_weighted_coverage(
    df: pd.DataFrame,
    contig_length_dict: dict,
    mag_id: str,
    sample_id: str,
    mag_size: int,
) -> Optional[float]:
    """
    Calculate contig-length-weighted mean coverage.

    This metric provides a per-contig mean coverage weighted by contig length,
    accounting for the fact that longer contigs contribute more to genome coverage.
    Uses MAG size as denominator to account for contigs with no mapped reads.
    Formula: Σ(mean_cov_contig * contig_length) / mag_size

    Parameters
    ----------
    df : pd.DataFrame
        Profile dataframe with 'contig' and 'total_coverage' columns
    contig_length_dict : dict
        Mapping of contig_id -> length in bp
    mag_id : str
        MAG identifier (for logging)
    sample_id : str
        Sample identifier (for logging)
    mag_size : int
        Total size of the MAG in bp (used as denominator)

    Returns
    -------
    float or None
        Length-weighted mean coverage, or None if calculation not possible

    Notes
    -----
    - Only meaningful for full genome analysis (not for sparse position sets)
    - Contigs without valid lengths are dropped with a warning
    - Per-contig mean is calculated as: sum(coverage) / contig_length
      (implicitly includes zeros for unobserved positions)
    - Uses mag_size as denominator to account for contigs with no reads mapped
    - Returns None if no valid contigs remain after filtering or mag_size is 0
    """
    if "contig" not in df.columns:
        logger.warning(f"No 'contig' column found for MAG {mag_id}, sample {sample_id}")
        return None

    if mag_size <= 0:
        logger.warning(
            f"Invalid MAG size ({mag_size}) for {mag_id}, sample {sample_id}"
        )
        return None

    # Aggregate coverage per contig
    contig_means = (
        df.groupby("contig", as_index=False)["total_coverage"]
        .sum()
        .rename(columns={"total_coverage": "coverage_sum"})
    )

    if contig_means.empty:
        return None

    # Map contig lengths
    contig_means["length"] = contig_means["contig"].map(contig_length_dict)

    # Filter out contigs without valid lengths
    before = contig_means.shape[0]
    contig_means = contig_means[
        pd.to_numeric(contig_means["length"], errors="coerce") > 0
    ]
    dropped = before - contig_means.shape[0]

    if dropped > 0:
        logger.warning(
            f"Dropped {dropped} contigs without valid lengths for {mag_id} (sample {sample_id})."
        )

    if contig_means.empty:
        return None

    # Compute per-contig mean coverage (includes implicit zeros for unobserved positions)
    contig_means["mean_coverage"] = (
        contig_means["coverage_sum"] / contig_means["length"]
    )

    # Calculate length-weighted mean using mag_size as denominator
    # This accounts for contigs with no reads mapped
    weighted_sum = (contig_means["mean_coverage"] * contig_means["length"]).sum()
    return weighted_sum / mag_size


def check_breadth_threshold(
    breadth: float,
    breadth_genome: Optional[float],
    threshold: float,
    using_filter: bool,
    sample_id: str,
    mag_id: str,
) -> Tuple[bool, str]:
    """
    Check if breadth passes threshold.

    The breadth metric checked depends on whether a positions filter is active:
    - No filter: checks breadth against threshold
    - With filter: ALWAYS checks breadth_genome against threshold

    Parameters
    ----------
    breadth : float
        Breadth metric (filtered or full, depending on context)
    breadth_genome : float or None
        Genome-wide breadth (only available when using filter)
    threshold : float
        Minimum breadth required to pass (typically 0.0-1.0)
    using_filter : bool
        Whether a positions filter is active
    sample_id : str
        Sample identifier (for logging)
    mag_id : str
        MAG identifier (for logging)

    Returns
    -------
    tuple
        (passed: bool, fail_reason: str)
        - passed: True if threshold met, False otherwise
        - fail_reason: Empty string if passed, explanation if failed

    Notes
    -----
    - When using_filter=True, ALWAYS checks breadth_genome
    - This ensures genome-wide coverage is adequate even when analyzing subset positions
    - Logs info message when threshold not met
    """
    # Determine which breadth metric to check
    if using_filter:
        # When using position filter, ALWAYS check genome-wide breadth
        breadth_to_check = breadth_genome
        metric_name = "Breadth (genome)"
    else:
        # No filter: check regular breadth
        breadth_to_check = breadth
        metric_name = "Breadth"

    # Check threshold
    if breadth_to_check < threshold:
        fail_reason = (
            f"{metric_name} {breadth_to_check:.2%} < threshold {threshold:.2%}"
        )
        logger.info(f"{fail_reason} - sample={sample_id}, mag={mag_id}")
        return False, fail_reason

    return True, ""


def check_coverage_threshold(
    avg_cov: float,
    avg_cov_genome: Optional[float],
    threshold: float,
    breadth_passed: bool,
    using_filter: bool,
    sample_id: str,
    mag_id: str,
) -> Tuple[bool, str]:
    """
    Check if coverage passes threshold (cascades from breadth).

    Coverage threshold checking has cascading logic:
    1. If breadth check failed, coverage check automatically fails
    2. If breadth check passed, then evaluate coverage threshold

    The coverage metric checked depends on whether a positions filter is active:
    - No filter: checks avg_cov against threshold
    - With filter: ALWAYS checks avg_cov_genome against threshold

    Parameters
    ----------
    avg_cov : float
        Average coverage value (filtered or full, depending on context)
    avg_cov_genome : float or None
        Genome-wide average coverage (only available when using filter)
    threshold : float
        Minimum coverage required to pass
    breadth_passed : bool
        Whether breadth threshold check passed
    using_filter : bool
        Whether a positions filter is active
    sample_id : str
        Sample identifier (for logging)
    mag_id : str
        MAG identifier (for logging)

    Returns
    -------
    tuple
        (passed: bool, fail_reason: str)
        - passed: True if threshold met, False otherwise
        - fail_reason: Empty string if passed, explanation if failed

    Notes
    -----
    - This implements cascading failure logic: breadth must pass first
    - If breadth failed, fail_reason is "Breadth check failed"
    - When using_filter=True, ALWAYS checks avg_cov_genome
    - Logs info message when coverage threshold not met
    """
    # Cascading failure: if breadth check failed, coverage check must also fail
    if not breadth_passed:
        return False, "Breadth check failed"

    # Determine which coverage metric to check
    if using_filter:
        # When using position filter, ALWAYS check genome-wide coverage
        cov_to_check = avg_cov_genome
        metric_name = "Average coverage (genome)"
    else:
        # No filter: check regular average coverage
        cov_to_check = avg_cov
        metric_name = "Average coverage"

    # Breadth passed, now check coverage threshold
    if cov_to_check < threshold:
        fail_reason = f"{metric_name} {cov_to_check:.2f}x < threshold {threshold:.2f}x"
        logger.info(f"{fail_reason} - sample={sample_id}, mag={mag_id}")
        return False, fail_reason

    return True, ""


def aggregate_mag_stats(
    df_results: pd.DataFrame,
    mag_id: str,
    group_cols: list,
    overall: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Aggregate QC metrics for a MAG across samples.

    This function aggregates sample-level QC results by calculating mean values
    for various metrics. It can produce either:
    - Per-group aggregations (e.g., by treatment group)
    - Overall aggregations (collapsed across all samples)

    Parameters
    ----------
    df_results : pd.DataFrame
        Sample-level QC results for a single MAG
    mag_id : str
        MAG identifier
    group_cols : list
        Column names to group by (e.g., ['group'] or ['group', 'time'])
        Empty list for overall aggregation
    overall : bool, default=False
        If True, produce single-row overall summary with *_mean suffix
        If False, produce per-group summary

    Returns
    -------
    pd.DataFrame or None
        Aggregated results with mean values for each metric
        Returns None if no valid data to aggregate

    Notes
    -----
    - When coverage_threshold_passed column exists, only includes passing samples
    - If no samples pass threshold, falls back to aggregating all samples
    - Adds _mean suffix to all metric columns
    - Includes num_samples column showing sample count per group
    - Returns None if df_results is empty or group_cols is empty (when overall=False)

    Metrics aggregated (when present in input):
    - subjects_per_group, replicates_per_group, paired_replicates_per_group
    - breadth, breadth_genome
    - average_coverage, average_coverage_genome
    - median_coverage, median_coverage_including_zeros
    - length_weighted_coverage
    - coverage_std, coverage_std_including_zeros
    """
    if df_results.empty:
        logger.warning(f"No results to summarize for {mag_id}")
        return None

    # Filter to passing samples if coverage threshold column exists
    if "coverage_threshold_passed" in df_results.columns:
        passing = df_results[df_results["coverage_threshold_passed"]]
    else:
        # No threshold column (e.g., positions mode) - use all samples
        passing = df_results.copy()

    if passing.empty:
        # Documented behaviour (see docstring): when no samples pass the
        # coverage threshold, fall back to aggregating across ALL samples so
        # the MAG still appears in the summary table — otherwise downstream
        # rules would silently drop MAGs that had borderline coverage in
        # every sample.  Warn so this fallback is visible in the log.
        logger.warning(
            f"No samples passed coverage threshold for {mag_id}; "
            "falling back to aggregating across all samples."
        )
        passing = df_results.copy()

    # Define metrics to aggregate (only those present in the data)
    # Order matches the consistent column grouping used in result dictionaries
    metric_cols = [
        c
        for c in [
            # Breadth metrics (grouped together)
            "breadth",
            "breadth_genome",
            # Coverage metrics (grouped together)
            "average_coverage",
            "average_coverage_genome",
            # Additional coverage statistics
            "median_coverage",
            "median_coverage_including_zeros",
            "coverage_std",
            "coverage_std_including_zeros",
            "length_weighted_coverage",
            # Subject/replicate counts
            "subjects_per_group",
            "replicates_per_group",
            "paired_replicates_per_group",
        ]
        if c in passing.columns
    ]

    if overall:
        # Overall summary: single row with mean of all metrics
        row = {"MAG_ID": mag_id, "num_samples": passing.shape[0]}
        for col in metric_cols:
            row[col + "_mean"] = passing[col].mean()
        out_df = pd.DataFrame([row])
    else:
        # Per-group summary
        if not group_cols:
            logger.error(
                "group_cols empty but overall flag False; nothing to aggregate."
            )
            return None

        # Aggregate metrics by group
        agg_dict = {c: "mean" for c in metric_cols}
        agg_dict["sample_id"] = "count"
        grouped = passing.groupby(group_cols, dropna=False).agg(agg_dict)
        out_df = grouped.rename(columns={"sample_id": "num_samples"}).reset_index()
        out_df.insert(0, "MAG_ID", mag_id)

        # Add _mean suffix to all metric columns for consistency
        rename_dict = {
            col: col + "_mean" for col in metric_cols if col in out_df.columns
        }
        out_df = out_df.rename(columns=rename_dict)

    return out_df
