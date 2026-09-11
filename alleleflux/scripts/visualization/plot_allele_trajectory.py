#!/usr/bin/env python3
"""
Plot allele frequency trajectories for genomic sites.

This module provides visualization tools for allele frequency data from metagenomic
analysis. It generates trajectory plots showing how allele frequencies change across
timepoints or conditions, with support for:

- **Combined plots**: Summarize all selected sites in a single visualization
  (line, box, or violin plots) with trajectories colored by group.
- **Per-replicate combined plots**: The combined line plot repeated once per replicate,
  using the same site selection but averaging only within each replicate.
- **Per-site plots**: Generate individual plots for each genomic site, showing
  trajectories for all samples or replicates.
- **Time binning**: Aggregate data into time bins for cleaner visualization of
  dense longitudinal data.
- **Replicate grouping**: Consolidate subject trajectories within replicates
  into averaged lines to reduce visual clutter.

Input data should be in long format with columns including:
- contig, position, anchor_allele: Site identifiers
- frequency: Allele frequency value (0-1)
- sample_id, subjectID, group: Sample/subject metadata
- time or day: Temporal information for x-axis
- Optional: replicate (for --group_by_replicate), mag_id, q_value/min_p_value

Usage:
    python plot_allele_trajectory.py --input_file data.tsv --per_site --plot_types line box

See --help for full CLI options.
"""
import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def validate_input_columns(
    df: pd.DataFrame, required_cols: List[str], context: str = ""
) -> None:
    """
    Validate that required columns exist in the DataFrame.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        context: Context information for error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} {context}")


def assign_bin(d: float, bin_width_days: int) -> Tuple[str, float, int]:
    """
    Assign a day value to a bin for time binning functionality.

    Args:
        d: Day value to bin
        bin_width_days: Width of each bin in days

    Returns:
        Tuple containing:
        - label: String label for the bin (e.g., "0-9", "10-19")
        - midpoint: Numeric midpoint of the bin for plotting
        - sort_key: Numeric key for sorting bins

    Examples:
        >>> assign_bin(5, 10)
        ('0-9', 5.0, 0)
        >>> assign_bin(15, 10)
        ('10-19', 15.0, 10)
        >>> assign_bin(0, 10)
        ('0-9', 5.0, 0)
        >>> assign_bin(-3, 10)
        ('Pre (<0)', -5.0, -1)
        >>> assign_bin(pd.NA, 10)
        ('NaN', -999, -999)
    """
    # Handle missing values with sentinel values for sorting
    if pd.isna(d):
        return "NaN", -999, -999

    if d < 0:
        # Group all negative days into a single "pre-baseline" bin
        label = "Pre (<0)"
        midpoint = -(
            bin_width_days / 2
        )  # Center point for plotting (e.g., -5 for width=10)
        sort_key = -1  # Ensures this bin sorts before all positive bins
    else:
        # Calculate which bin this day falls into using integer division
        # e.g., day 5 with width 10: 5 // 10 = 0 (first bin)
        # e.g., day 15 with width 10: 15 // 10 = 1 (second bin)
        bucket_idx = int(d // bin_width_days)

        # Calculate the lower bound of the bin (inclusive)
        # e.g., bucket 0 → lower=0, bucket 1 → lower=10
        lower = bucket_idx * bin_width_days

        # Calculate the upper bound (inclusive) by subtracting 1 from next bin's start
        # This creates non-overlapping labels: "0-9", "10-19", "20-29", etc.
        # e.g., bucket 0 with width 10: upper = (0+1)*10 - 1 = 9
        upper = ((bucket_idx + 1) * bin_width_days) - 1

        # Human-readable label showing the day range
        label = f"{lower}-{upper}"

        # Midpoint is used as x-coordinate for line plots (centers the point in the bin)
        # e.g., bin "0-9" → midpoint=5, bin "10-19" → midpoint=15
        midpoint = lower + (bin_width_days / 2)

        # Sort key ensures bins are ordered chronologically
        # Using lower bound as sort key: 0, 10, 20, ...
        sort_key = lower

    return label, midpoint, sort_key


def prepare_x_axis(
    df: pd.DataFrame, x_col: str, custom_order: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """
    Prepares the DataFrame for plotting by handling the x-axis column.

    This function performs two main tasks:
    1. Validates that the specified x-axis column exists.
    2. If a custom order is provided via `custom_order`:
       - Subsets the DataFrame to include only rows where the x-axis value is in `custom_order`.
       - Converts the x-axis column to a Categorical type with the specified order.
    3. If no custom order is provided and the column is 'day':
       - Converts the column to numeric values for proper sorting.

    Args:
        df: Input DataFrame containing the data.
        x_col: Name of the column to use for the X-axis (e.g., 'time', 'day').
        custom_order: Optional list of strings defining the specific order of x-axis values.
                      Also acts as a filter: only values in this list will be kept.

    Returns:
        Tuple containing:
        - modified_df: DataFrame with the x-axis column processed (filtered/sorted).
        - sorted_values: The list of values defining the order (if custom_order was used), else None.
    """
    validate_input_columns(df, [x_col], f"for {x_col} x-axis")

    if custom_order:
        # Subset data to only include values in custom_order
        df = df[df[x_col].astype(str).isin(custom_order)].copy()

        if df.empty:
            logger.warning(
                f"No data remaining after subsetting with custom order: {custom_order}"
            )
            return df, custom_order

        # Apply categorical ordering
        df[x_col] = pd.Categorical(df[x_col], categories=custom_order, ordered=True)
        return df, custom_order

    # Default behavior
    if x_col == "day":
        df[x_col] = pd.to_numeric(df[x_col], errors="raise")
        if df[x_col].isna().all():
            raise ValueError("No valid numeric day values found")
        return df, None

    return df, None


def compute_top_sites(df: pd.DataFrame, value_col: str, n) -> Tuple[pd.DataFrame, str]:
    """
    Identifies the top `n` sites based on the lowest values in `value_col`.

    This is typically used to select the most significant sites (e.g., lowest p-values or q-values).
    If `n` is 'all', all unique sites are returned.

    Args:
        df: Input DataFrame containing site information.
        value_col: The column name used for ranking sites (e.g., 'q_value', 'min_p_value').
                   Lower values are considered "better" or "top".
        n: The number of sites to select. Can be an integer or the string 'all'.

    Returns:
        Tuple containing:
        - top_sites_df: A DataFrame containing only the unique identifiers (contig, position, anchor_allele)
                        and the value column for the selected top sites.
        - n_label: A string label representing the number of sites (e.g., "100" or "ALL"), used for file naming.

    Raises:
        ValueError: If `n` is not a positive integer or 'all', or if required columns are missing.
    """
    validate_input_columns(df, ["contig", "position", "anchor_allele", value_col])

    if n == "all":
        n_label = "ALL"
        top_sites = df[
            ["contig", "position", "anchor_allele", value_col]
        ].drop_duplicates()

    else:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer or 'all'")
        n_label = str(n)
        top_sites = (
            df[["contig", "position", "anchor_allele", value_col]]
            .drop_duplicates()
            .nsmallest(n, value_col)
        )

    return top_sites, n_label


def compute_group_means(
    df: pd.DataFrame, x_col: str, custom_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregates allele frequencies to compute the mean per group at each timepoint/x-value.

    This function groups the data by x-axis value, group, contig, and position, and calculates
    the mean frequency. It ensures that the categorical ordering of the x-axis is preserved
    or reapplied after the aggregation.

    Note: The valid_count uses pandas .count() which automatically excludes NA/NaN values,
    ensuring accurate sample counts for min_samples_per_bin filtering.

    Args:
        df: Input DataFrame containing individual sample data.
        x_col: The column name used for the X-axis.
        custom_order: Optional list of values defining the order of the x-axis.
                      Used to restore Categorical ordering after groupby operations.

    Returns:
        pd.DataFrame: A new DataFrame with aggregated mean frequencies.
                      Columns include [x_col, 'group', 'contig', 'position', 'frequency', 'valid_count'].
    """
    validate_input_columns(
        df, ["group", "contig", "position", "frequency", "anchor_allele", x_col]
    )

    # Enhanced aggregation with valid count tracking
    # Note: .count() excludes NA/NaN values, so valid_count reflects non-missing samples only
    mean_df = df.groupby(
        [x_col, "group", "contig", "position", "anchor_allele"],
        as_index=False,
        observed=True,
    ).agg(frequency=("frequency", "mean"), valid_count=("frequency", "count"))

    # Reapply correct sorting
    if custom_order:
        if x_col in mean_df.columns:
            mean_df[x_col] = pd.Categorical(
                mean_df[x_col], categories=custom_order, ordered=True
            )
    elif x_col == "day":
        mean_df = mean_df.sort_values(x_col)

    return mean_df


def compute_replicate_means(
    df: pd.DataFrame, x_col: str, custom_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aggregates allele frequencies to compute the mean per replicate at each timepoint/x-value.

    This function consolidates all subject trajectories within the same replicate into a single
    average line per replicate, grouped by group, contig, position, and anchor_allele.

    Note: The valid_count uses pandas .count() which automatically excludes NA/NaN values.

    Args:
        df: Input DataFrame containing individual sample data. Must have a 'replicate' column.
        x_col: The column name used for the X-axis.
        custom_order: Optional list of values defining the order of the x-axis.
                      Used to restore Categorical ordering after groupby operations.

    Returns:
        pd.DataFrame: A new DataFrame with aggregated mean frequencies per replicate.
                      Columns include [x_col, 'group', 'replicate', 'contig', 'position',
                      'anchor_allele', 'frequency', 'valid_count'].
    """
    # Aggregation by replicate with valid count tracking
    # Note: .count() excludes NA/NaN values, so valid_count reflects non-missing samples only
    # Include anchor_allele in groupby to preserve site identity for downstream operations
    mean_df = df.groupby(
        [x_col, "group", "replicate", "contig", "position", "anchor_allele"],
        as_index=False,
        observed=True,
    ).agg(frequency=("frequency", "mean"), valid_count=("frequency", "count"))

    # Reapply correct sorting
    if custom_order:
        if x_col in mean_df.columns:
            mean_df[x_col] = pd.Categorical(
                mean_df[x_col], categories=custom_order, ordered=True
            )
    elif x_col == "day":
        mean_df = mean_df.sort_values(x_col)

    return mean_df


def filter_by_initial_frequency(
    df: pd.DataFrame,
    x_col: str,
    max_initial_freq: Optional[float] = None,
    min_initial_freq: Optional[float] = None,
    group_col: str = "group",
    filter_group: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter sites based on initial frequency in a specific group.

    Selects sites where the frequency at the first timepoint in the specified
    filter_group falls within the given bounds. Once a site is selected, data from
    ALL groups is kept for that site. This allows users to focus on alleles
    that start at specific frequencies in one group while still comparing
    trajectories across all groups.

    What is the "first timepoint"?
        - For numeric x_col (e.g., 'day'): The minimum value (e.g., day 0)
        - For categorical x_col (e.g., 'time' with T1, T2, T3): The first category (e.g., T1)

    Args:
        df: Input DataFrame containing allele frequency data.
        x_col: Column name used for the X-axis (e.g., 'time', 'day').
        max_initial_freq: Maximum allowed frequency at the first timepoint (inclusive).
                          Sites with initial frequency > this value are excluded.
                          If None, no upper bound is applied.
        min_initial_freq: Minimum allowed frequency at the first timepoint (inclusive).
                          Sites with initial frequency < this value are excluded.
                          If None, no lower bound is applied.
        group_col: Column name for grouping (default: 'group').
        filter_group: The group to use for filtering. Can be:
                      - A specific group name (e.g., 'fat'): Filter based on that group only
                      - 'both' or 'all': Require criteria to be met in ALL groups
                      - None: Defaults to 'both/all' mode (criteria must be met in all groups)

    Returns:
        pd.DataFrame: Filtered DataFrame containing data from ALL groups,
                      but only for sites that meet the initial frequency
                      criteria in the specified filter_group(s).

    Example:
        Suppose you have two groups ("fat" and "control") with data at days 0, 7, 14:

        | site   | group   | day | frequency |
        |--------|---------|-----|-----------|
        | Site_A | fat     | 0   | 0.05      |  <- fat=0.05, control=0.08
        | Site_A | control | 0   | 0.08      |
        | Site_B | fat     | 0   | 0.05      |  <- fat=0.05, control=0.50
        | Site_B | control | 0   | 0.50      |
        | Site_C | fat     | 0   | 0.40      |  <- fat=0.40, control=0.02
        | Site_C | control | 0   | 0.02      |

        Example 1: filter_group='fat', max_initial_freq=0.1
        - Site_A: fat=0.05 <= 0.1 -> KEEP (both groups plotted)
        - Site_B: fat=0.05 <= 0.1 -> KEEP (both groups plotted)
        - Site_C: fat=0.40 > 0.1 -> EXCLUDE

        Example 2: filter_group='both', max_initial_freq=0.1 (must pass in ALL groups)
        - Site_A: fat=0.05 <= 0.1 AND control=0.08 <= 0.1 -> KEEP
        - Site_B: fat=0.05 <= 0.1 BUT control=0.50 > 0.1 -> EXCLUDE
        - Site_C: fat=0.40 > 0.1 -> EXCLUDE

        >>> # Keep sites where fat starts below 10%
        >>> filtered = filter_by_initial_frequency(df, 'day', max_initial_freq=0.1, filter_group='fat')
        >>> # Keep sites where BOTH groups start below 10%
        >>> filtered = filter_by_initial_frequency(df, 'day', max_initial_freq=0.1, filter_group='both')
    """
    validate_input_columns(
        df, ["contig", "position", "anchor_allele", "frequency", x_col, group_col]
    )

    if df.empty:
        logger.warning("Empty DataFrame provided to filter_by_initial_frequency")
        return df

    # Validate that at least one bound is specified
    if max_initial_freq is None and min_initial_freq is None:
        logger.warning(
            "Neither max_initial_freq nor min_initial_freq specified. Returning original data."
        )
        return df

    # Determine which group(s) to use for filtering
    all_groups = df[group_col].unique().tolist()

    # Check for special keywords "both" or "all"
    filter_all_groups = filter_group is not None and filter_group.lower() in [
        "both",
        "all",
    ]

    if filter_all_groups:
        target_groups = all_groups
        logger.info(
            f"Using 'both/all' mode: criteria must be met in ALL groups: {all_groups}"
        )
    elif filter_group is not None:
        if filter_group not in all_groups:
            logger.warning(
                f"Specified filter group '{filter_group}' not found in data. "
                f"Available groups: {all_groups}. Returning original data unfiltered."
            )
            return df
        target_groups = [filter_group]
    else:
        # Default to requiring criteria in ALL groups when no filter_group specified
        target_groups = all_groups
        filter_all_groups = True
        logger.warning(
            f"No filter_group specified, defaulting to 'both/all' mode: "
            f"criteria must be met in ALL groups: {all_groups}"
        )

    # Build filter description for logging
    filter_parts = []
    if min_initial_freq is not None:
        filter_parts.append(f">= {min_initial_freq}")
    if max_initial_freq is not None:
        filter_parts.append(f"<= {max_initial_freq}")
    filter_desc = " AND ".join(filter_parts)

    group_desc = "all groups" if filter_all_groups else f"group '{target_groups[0]}'"
    logger.info(
        f"Selecting sites where initial frequency {filter_desc} "
        f"in {group_desc}. Data from all groups will be kept for matching sites."
    )

    # Helper function to get passing sites for a single group
    def get_passing_sites_for_group(group_name):
        group_df = df[df[group_col] == group_name]

        # Get the first (minimum) timepoint for this group.
        # .min() works for both numeric columns (day) and categorical columns (time).
        # For categoricals, .min() returns the first category in the ordering.
        first_tp = group_df[x_col].min()

        # Get data at the first timepoint
        first_tp_data = group_df[group_df[x_col] == first_tp]

        if first_tp_data.empty:
            logger.warning(
                f"No data found at first timepoint {first_tp} for group '{group_name}'"
            )
            return None

        # Calculate mean frequency per site at the first timepoint
        initial_freqs = first_tp_data.groupby(
            ["contig", "position", "anchor_allele"], observed=True
        )["frequency"].mean()

        # Apply frequency bounds to get sites that pass the filter
        if min_initial_freq is not None and max_initial_freq is not None:
            # Both bounds specified - use between for clarity
            mask = initial_freqs.between(min_initial_freq, max_initial_freq)
        elif min_initial_freq is not None:
            mask = initial_freqs >= min_initial_freq
        elif max_initial_freq is not None:
            mask = initial_freqs <= max_initial_freq
        else:
            mask = pd.Series(True, index=initial_freqs.index)

        passing = initial_freqs[mask].reset_index()
        return set(
            zip(passing["contig"], passing["position"], passing["anchor_allele"])
        )

    # Get passing sites for each target group
    passing_sites_per_group = []
    for group in target_groups:
        sites = get_passing_sites_for_group(group)
        if sites is None:
            if filter_all_groups:
                # If using "both/all" and a group has no data, no sites can pass
                logger.warning(
                    f"No data for group '{group}', no sites can pass 'both/all' filter"
                )
                return df.iloc[:0]
            continue
        passing_sites_per_group.append(sites)

    if not passing_sites_per_group:
        logger.warning(f"No sites passed the initial frequency filter ({filter_desc})")
        return df.iloc[:0]

    # Combine results: intersection of all sets (works for both single and multi-group)
    # For single group, this just returns that group's sites
    # For multiple groups, sites must pass in ALL groups
    final_sites = set.intersection(*passing_sites_per_group)

    if not final_sites:
        logger.warning(
            f"No sites passed the initial frequency filter "
            f"({filter_desc} in {group_desc})"
        )
        return df.iloc[:0]

    # Convert back to DataFrame for merging
    passing_sites_df = pd.DataFrame(
        list(final_sites), columns=["contig", "position", "anchor_allele"]
    )

    # Filter original data to keep ALL groups' data for the passing sites
    filtered_df = df.merge(
        passing_sites_df,
        on=["contig", "position", "anchor_allele"],
        how="inner",
    )

    initial_count = (
        df[["contig", "position", "anchor_allele"]].drop_duplicates().shape[0]
    )
    final_count = (
        filtered_df[["contig", "position", "anchor_allele"]].drop_duplicates().shape[0]
    )

    logger.info(
        f"Initial frequency filter (based on {group_desc}): "
        f"{initial_count} sites -> {final_count} sites "
        f"({len(df)} rows -> {len(filtered_df)} rows)"
    )

    return filtered_df


def apply_min_samples_filter(
    mean_df: pd.DataFrame,
    plot_x_col: str,
    min_samples_per_bin: int,
    plot_type: Optional[str] = None,
    context: str = "",
) -> pd.DataFrame:
    """
    Drop aggregated points that were computed from too few valid samples.

    Means built from one or two samples are unreliable, so this removes any row whose
    `valid_count` (produced by compute_group_means/compute_replicate_means, which count
    non-NaN frequencies only) falls below the threshold.

    Note that `valid_count` counts *samples*, not distinct subjects. A bin containing two
    samples from the same subject passes a threshold of 2, so this filter bounds how much
    data backs a point but does not guarantee how many subjects contributed.

    Args:
        mean_df: Aggregated DataFrame containing a 'valid_count' column.
        plot_x_col: The column used for the X-axis, for reporting remaining coverage.
        min_samples_per_bin: Minimum number of valid (non-NaN) samples required per point.
                             Values <= 1 disable the filter.
        plot_type: Type of plot the data feeds ('line', 'box', 'violin'). Only used to warn
                   when too few x-values survive for a line to be drawn.
        context: Optional description of the subset being filtered (e.g., "replicate 3"),
                 included in log messages to disambiguate repeated calls.

    Returns:
        pd.DataFrame: The filtered DataFrame (unchanged if the filter is disabled).
    """
    if min_samples_per_bin <= 1:
        return mean_df

    prefix = f"[{context}] " if context else ""

    initial_len = len(mean_df)
    logger.info(
        f"{prefix}Filtering points: requiring >= {min_samples_per_bin} valid (non-NaN) samples per point."
    )

    # Filter out points where mean was calculated from insufficient samples
    mean_df = mean_df[mean_df["valid_count"] >= min_samples_per_bin]

    dropped_len = initial_len - len(mean_df)
    if dropped_len > 0:
        logger.info(f"{prefix}Dropped {dropped_len} sparse data points (low valid N).")

    # Warn user if filtering removed too much data for meaningful plots
    if not mean_df.empty:
        remaining_x_values = mean_df[plot_x_col].nunique()
        logger.info(
            f"{prefix}After filtering: {len(mean_df)} points remain with "
            f"{remaining_x_values} unique {plot_x_col} values"
        )
        # Line plots need at least 2 points to draw a line
        if remaining_x_values < 2 and plot_type == "line":
            logger.warning(
                f"{prefix}Only {remaining_x_values} unique x-value(s) remain after filtering. "
                f"Line plots require >= 2 x-values to draw lines."
            )
    else:
        logger.warning(
            f"{prefix}All data points were dropped by min_samples_per_bin={min_samples_per_bin}. "
            f"Consider lowering this threshold."
        )

    return mean_df


def get_plotting_data(
    df: pd.DataFrame,
    value_col: str,
    n_val: int | str,
    x_col: str,
    sorted_vals: Optional[List[str]],
    bin_width_days: Optional[int],
    min_samples_per_bin: int,
    plot_type: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str, Optional[str]]:
    """
    Prepare data for a specific plot configuration.

    This function handles the data preparation pipeline for plotting:
    1. Site selection (top N by value_col)
    2. Data filtering to selected sites
    3. X-column selection (original vs binned)
    4. Group mean computation
    5. Sparse data point filtering

    Args:
        df: Input DataFrame containing the raw allele frequency data (already preprocessed
            with binning columns if bin_width_days was specified).
        value_col: Column name used to rank sites for filtering (e.g., 'q_value').
        n_val: Number of top sites to select ('all' or an integer).
        x_col: Original column to use for the X-axis (e.g., 'time', 'day').
        sorted_vals: Optional list of values defining the order of the original x_col.
        bin_width_days: If set, indicates binning is enabled and affects x-column selection.
        min_samples_per_bin: Minimum number of valid (non-NA) samples required per bin.
        plot_type: Type of plot being generated ('line', 'box', 'violin').
                   Affects x-column selection when binning is enabled.

    Returns:
        Tuple containing:
        - filtered_df: DataFrame filtered to selected sites (raw sample data), or None on failure.
        - mean_df: DataFrame with computed group means, or None on failure.
        - n_label: String label for the number of sites (e.g., "10" or "ALL").
        - plot_x_col: The x-column to use for plotting, or None on failure.
    """
    # Select top N sites based on lowest values in value_col (e.g., p-values)
    top_sites, n_label = compute_top_sites(df, value_col, n_val)
    if top_sites.empty:
        logger.warning(f"No sites selected for n={n_val}")
        return None, None, n_label, None

    # Filter original data to only include selected sites
    # Uses inner join on site identifiers + value_col
    filtered = df.merge(
        top_sites[["contig", "position", "anchor_allele"]],
        on=["contig", "position", "anchor_allele"],
        how="inner",
    )
    if filtered.empty:
        logger.warning(f"No data remains after filtering for n={n_val}")
        return None, None, n_label, None

    # === DETERMINE X-AXIS COLUMN FOR THIS PLOT TYPE ===
    # When binning is enabled, line plots use numeric midpoints (for proper spacing)
    # while box/violin plots use categorical labels (for discrete grouping)
    plot_x_col = x_col
    plot_custom_order = sorted_vals  # Custom order only applies to original x_col
    if bin_width_days is not None:
        if plot_type == "line":
            # Use numeric midpoints so lines connect at correct x-positions
            plot_x_col = "bin_midpoint"
            # bin_midpoint is numeric, don't apply categorical order
            plot_custom_order = None
            logger.info("Plotting Lines using Bin Midpoints.")
        else:
            # Use categorical labels for discrete box/violin groupings
            plot_x_col = "bin_label"
            # bin_label has its own categorical order already set
            plot_custom_order = None
            logger.info("Plotting Distributions using Bin Intervals.")

    # Compute mean frequency per group at each x-value
    # Also tracks valid_count for sparse data filtering
    mean_df = compute_group_means(filtered, plot_x_col, plot_custom_order)

    # === SPARSE DATA FILTERING ===
    # Remove data points computed from too few samples (unreliable means)
    mean_df = apply_min_samples_filter(
        mean_df, plot_x_col, min_samples_per_bin, plot_type
    )

    # Ensure line plots connect points in chronological order
    if bin_width_days is not None and plot_type == "line":
        mean_df = mean_df.sort_values(plot_x_col)

    return filtered, mean_df, n_label, plot_x_col


def plot_combined(
    df: pd.DataFrame,
    x_col: str,
    n_label: str,
    value_col: str,
    output_dir: str,
    output_format: str,
    plot_type: str,
    mag_id: str = "MAG",
    line_alpha: float = 0.8,
    label_suffix: str = "",
    title_suffix: str = "",
    group_order: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    filename_tag: str = "",
) -> None:
    """
    Generates a combined plot summarizing allele frequencies across all selected sites.

    Supports multiple plot types:
    - 'line': Plots the mean frequency trajectory for each site, colored by group.
    - 'box': Shows the distribution of frequencies at each timepoint for each group.
    - 'violin': Similar to box plot but shows the probability density of the data.

    Args:
        df: Input DataFrame containing aggregated mean frequencies (from compute_group_means).
        x_col: The column name used for the X-axis.
        n_label: String label indicating the number of sites included (for the title and filename).
        value_col: The column name used for ranking (for context in the title, though not directly plotted).
        output_dir: Directory where the plot file will be saved.
        output_format: File format for the plot (e.g., 'png', 'pdf', 'svg').
        plot_type: The type of plot to generate ('line', 'box', or 'violin').
        mag_id: MAG identifier to prefix the output filename.
        line_alpha: Transparency level for lines in line plots (0.0 to 1.0).
        label_suffix: Optional string appended to the output filename before the extension
                      (e.g., "_rep3"). Empty by default, which preserves the standard filename.
        title_suffix: Optional string appended to the plot title (e.g., " - Replicate 3").
        group_order: Optional explicit ordering of groups. Controls both hue order and the
                     color assignment, so callers plotting several subsets (e.g., one figure
                     per replicate) can guarantee a group keeps the same color in every figure.
                     Defaults to the sorted groups present in `df`.
        xlim: Optional (left, right) limits for the X-axis. Used to share a common x-range
              across a set of figures; ignored when None.
        filename_tag: Optional string inserted after the plot type and before label_suffix
                      (e.g., "_bin10d" when day binning is active), so outputs produced with
                      different settings can coexist in one directory.
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to plot_combined")
        return

    validate_input_columns(df, ["frequency", "group", x_col])

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = (
        outdir
        / f"{mag_id}_combined_n{n_label}_{x_col}_{plot_type}{filename_tag}{label_suffix}.{output_format}"
    )

    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(24, 12))
    # --- LINE PLOT ---
    if plot_type == "line":
        # Define groups and palette to ensure consistency across all lines and legend.
        # An explicit group_order keeps colors stable across separate figures that may
        # not each contain every group.
        groups = (
            list(group_order) if group_order is not None else sorted(df["group"].unique())
        )
        palette = sns.color_palette(n_colors=len(groups))

        # We loop manually to get the "Spaghetti" effect (one line per site)
        # sns.lineplot(units=...) can do this, but explicit looping is often safer
        # for 'hue' control on complex groupings.
        for (contig, pos, allele), subdf in df.groupby(
            ["contig", "position", "anchor_allele"]
        ):
            # Skip empty subsets
            if subdf.empty:
                continue

            sns.lineplot(
                data=subdf,
                x=x_col,
                y="frequency",
                hue="group",
                hue_order=groups,
                palette=palette,
                estimator=None,
                lw=2.5,
                alpha=line_alpha,
                legend=False,
            )

        # Create legend handles
        handles = [
            plt.Line2D([0], [0], color=palette[i], lw=4, label=groups[i])
            for i in range(len(groups))
        ]
        plt.legend(handles=handles, title="Group")

    # --- BOX PLOT ---
    elif plot_type == "box":
        sns.boxplot(data=df, x=x_col, y="frequency", hue="group")
        plt.legend(title="Group")

    # --- VIOLIN PLOT ---
    elif plot_type == "violin":
        sns.violinplot(data=df, x=x_col, y="frequency", hue="group", split=True)
        plt.legend(title="Group")

    else:
        raise ValueError("plot_type must be one of: 'line','box','violin'")

    plt.ylabel("Mean Allele Frequency")
    plt.xlabel(x_col)
    plt.ylim(-0.05, 1.05)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.title(f"Group Mean Frequencies — N={n_label} — {plot_type}{title_suffix}")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", format=output_format)
    # plt.show()
    plt.close()
    print(f"[saved] {outfile}")


def plot_combined_per_replicate(
    df: pd.DataFrame,
    x_col: str,
    n_label: str,
    value_col: str,
    output_dir: str,
    output_format: str,
    mag_id: str = "MAG",
    line_alpha: float = 0.8,
    min_samples_per_bin: int = 1,
    custom_order: Optional[List[str]] = None,
    share_x: bool = True,
    filename_tag: str = "",
) -> None:
    """
    Generates one combined line plot per replicate.

    This is the per-replicate counterpart to the pooled combined line plot. The pooled plot
    averages every subject in a group into a single trajectory per site; here the same set of
    sites is plotted separately for each replicate, with means computed only from that
    replicate's subjects. Site selection and any upstream filtering (top-N ranking, initial
    frequency filtering, time binning) are inherited from the caller, so every replicate's
    figure shows the identical set of sites and differences between figures reflect the data
    rather than differing site sets.

    Output files are written to an `by_replicate/` subdirectory, mirroring the
    `single_sites_by_replicate/` convention used by plot_per_site.

    Args:
        df: Sample-level DataFrame (not group means) restricted to the sites to plot.
            Must contain a 'replicate' column.
        x_col: The column name used for the X-axis.
        n_label: String label indicating the number of sites included.
        value_col: The column name used for ranking (for context in the title).
        output_dir: Base plot directory; figures are written to `<output_dir>/by_replicate`.
        output_format: File format for the plots (e.g., 'png', 'pdf', 'svg').
        mag_id: MAG identifier to prefix the output filenames.
        line_alpha: Transparency level for lines (0.0 to 1.0).
        min_samples_per_bin: Minimum valid (non-NaN) samples required per plotted point,
                             applied *within* each replicate. Because a replicate holds far
                             fewer subjects than the full cohort, this threshold is much more
                             selective here than in the pooled plot.
        custom_order: Optional list of values defining the order of the x-axis.
        share_x: If True (default) all figures use the same numeric X-axis limits, computed
                 across every replicate, so figures can be compared side by side. Has no
                 effect when the x-axis is categorical.
        filename_tag: Optional filename tag forwarded to plot_combined; it lands before the
                      "_rep<N>" suffix (e.g., "..._line_bin10d_rep3.png").
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to plot_combined_per_replicate")
        return

    validate_input_columns(
        df,
        ["frequency", "group", "replicate", "contig", "position", "anchor_allele", x_col],
        "for per-replicate combined plots",
    )

    outdir = Path(output_dir) / "by_replicate"
    outdir.mkdir(parents=True, exist_ok=True)

    # Fix the group ordering across every figure so a group keeps the same color even if a
    # replicate happens to be missing one of them.
    group_order = sorted(df["group"].unique())

    # Shared x-limits let the figures be compared directly; replicates that ended earlier
    # simply stop short instead of being rescaled to look like the longer ones.
    xlim = None
    if share_x and pd.api.types.is_numeric_dtype(df[x_col]):
        x_min = float(df[x_col].min())
        x_max = float(df[x_col].max())
        pad = (x_max - x_min) * 0.02 if x_max > x_min else 0.5
        xlim = (x_min - pad, x_max + pad)
        logger.info(f"Sharing X-axis across replicate figures: {xlim}")

    replicates = sorted(df["replicate"].dropna().unique())
    if not len(replicates):
        logger.warning("No non-null replicate values found; no per-replicate plots made")
        return

    logger.info(f"Creating per-replicate combined plots for {len(replicates)} replicates")

    for rep in replicates:
        sub = df[df["replicate"] == rep]
        if sub.empty:
            logger.warning(f"No data for replicate {rep}; skipping")
            continue

        # Means are computed from this replicate's samples only -- this is what makes the
        # figure replicate-specific rather than a subset of the pooled means.
        mean_df = compute_group_means(sub, x_col, custom_order)
        mean_df = apply_min_samples_filter(
            mean_df, x_col, min_samples_per_bin, "line", context=f"replicate {rep}"
        )

        if mean_df.empty:
            logger.warning(
                f"Replicate {rep}: no points survived filtering; no figure written"
            )
            continue

        if pd.api.types.is_numeric_dtype(mean_df[x_col]):
            mean_df = mean_df.sort_values(x_col)

        n_sites_plotted = mean_df[["contig", "position", "anchor_allele"]].drop_duplicates().shape[0]
        logger.info(
            f"Replicate {rep}: plotting {n_sites_plotted} sites, "
            f"{mean_df[x_col].nunique()} x-values, "
            f"{sub['subjectID'].nunique() if 'subjectID' in sub.columns else 'NA'} subjects"
        )

        plot_combined(
            mean_df,
            x_col,
            n_label,
            value_col,
            str(outdir),
            output_format,
            "line",
            mag_id,
            line_alpha=line_alpha,
            label_suffix=f"_rep{rep}",
            title_suffix=f" — Replicate {rep}",
            group_order=group_order,
            xlim=xlim,
            filename_tag=filename_tag,
        )


def plot_per_site(
    df: pd.DataFrame,
    x_col: str,
    n_label: str,
    value_col: str,
    output_dir: str,
    output_format: str,
    group_by_replicate: bool = False,
    filename_tag: str = "",
) -> None:
    """
    Generates individual trajectory plots for each genomic site.

    For each unique site (contig, position, anchor_allele), this function creates a separate plot
    showing the allele frequency trajectories for all samples, colored by group.
    The title includes the site information and its score (e.g., q-value).

    When group_by_replicate is True, subject trajectories within the same replicate are
    consolidated into a single average line per replicate, reducing visual clutter.

    Args:
        df: Input DataFrame containing individual sample data for the selected sites.
        x_col: The column name used for the X-axis.
        n_label: Label for number of sites (unused in this function but kept for consistency).
        value_col: The column name containing the score (e.g., 'q_value') to display in the title.
        output_dir: Directory where the individual plot files will be saved.
        output_format: File format for the plots (e.g., 'png', 'pdf', 'svg').
        group_by_replicate: If True, aggregate subject trajectories by replicate before plotting.
        filename_tag: Optional string appended before the extension (e.g., "_bin10d"),
                      so per-site plots from different bin widths can coexist.
    """
    if df.empty:
        logger.warning("Empty DataFrame provided to plot_per_site")
        return

    # Determine output subdirectory based on grouping mode
    subdir_name = "single_sites_by_replicate" if group_by_replicate else "single_sites"
    outdir = Path(output_dir) / subdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Determine required columns based on grouping mode
    required_cols = [
        "frequency",
        "group",
        x_col,
        "contig",
        "position",
        "anchor_allele",
    ]
    if group_by_replicate:
        required_cols.append("replicate")
    else:
        required_cols.append("subjectID")

    validate_input_columns(df, required_cols)

    # If grouping by replicate, compute replicate means first
    if group_by_replicate:
        logger.info("Aggregating subject trajectories by replicate...")
        # Preserve value_col for score display in plot titles
        # Get one value per site (they should all be the same within a site)
        site_scores = df[
            ["contig", "position", "anchor_allele", value_col]
        ].drop_duplicates()
        df = compute_replicate_means(df, x_col)
        # Merge back the value_col for score display
        df = df.merge(
            site_scores, on=["contig", "position", "anchor_allele"], how="left"
        )
        unit_col = "replicate"
    else:
        unit_col = "subjectID"

    sns.set_theme(style="whitegrid", context="talk")

    total_sites = len(df.groupby(["contig", "position", "anchor_allele"]))

    # Define groups and palette to ensure consistency across all lines and legend
    groups = sorted(df["group"].unique())
    palette = sns.color_palette(n_colors=len(groups))

    for (contig, pos, allele), subdf in tqdm(
        df.groupby(["contig", "position", "anchor_allele"]),
        total=total_sites,
        desc="Plotting per-site trajectories",
    ):
        if subdf.empty:
            logger.warning(f"Empty data for {contig}:{pos} ({allele})")
            continue

        plt.figure(figsize=(20, 10))

        sns.lineplot(
            data=subdf,
            x=x_col,
            y="frequency",
            hue="group",
            hue_order=groups,
            palette=palette,
            units=unit_col,
            estimator=None,
            lw=2.5,
            alpha=0.9,
        )

        # Get score value safely
        if value_col in subdf.columns:
            score = subdf[value_col].iloc[0]
            if pd.notna(score):
                score_text = f"{score:.2e}"
            else:
                score_text = "N/A"
        else:
            score_text = "N/A"

        # Update title to reflect grouping mode
        title_suffix = " (by replicate)" if group_by_replicate else ""
        plt.title(f"{contig}:{pos} ({allele}) — {value_col}={score_text}{title_suffix}")

        plt.ylabel(
            "Mean Allele Frequency" if group_by_replicate else "Allele Frequency"
        )
        plt.ylim(-0.05, 1.05)

        plt.xlabel(x_col)

        plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")

        # Include grouping mode in filename
        suffix = "_by_replicate" if group_by_replicate else ""
        filename = (
            f"{contig}_{pos}_{allele}_{value_col}_{x_col}{suffix}{filename_tag}.{output_format}"
        )
        # Sanitize filename: replace any character that is not a letter, number, underscore, hyphen, or dot with an underscore
        filename = re.sub(r"[^\w\-_.]", "_", filename)
        outpath = outdir / filename
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight", format=output_format)
        # plt.show()
        plt.close()

    print(f"[saved] plots to {outdir}")


def plot_group_distributions(
    df: pd.DataFrame,
    value_col: str = "q_value",
    n_line: int | str = 10,
    n_dist: int | str = "all",
    n_per_site: Optional[int | str] = None,
    x_col: str = "time",
    x_order: Optional[List[str]] = None,
    plot_types: List[str] = ["line"],
    per_site: bool = False,
    output_dir: str = "./plots",
    output_format: str = "png",
    bin_width_days: Optional[int] = None,
    min_samples_per_bin: int = 1,
    group_by_replicate: bool = False,
    line_alpha: float = 0.8,
    max_initial_freq: Optional[float] = None,
    min_initial_freq: Optional[float] = None,
    initial_freq_group: Optional[str] = None,
    combined_per_replicate: bool = False,
) -> None:
    """
    Orchestrates the plotting process: filters data, computes means, and generates requested plots.

    This is the main driver function that:
    1. Prepares the x-axis (sorting/filtering).
    2. Selects the top `n` sites based on `value_col`.
    3. Generates "Combined" plots (line, box, violin) summarizing all selected sites.
    4. Optionally generates "Per-site" plots for each individual site.

    Args:
        df: Input DataFrame containing the raw allele frequency data.
        value_col: Column name used to rank sites for filtering (e.g., 'q_value').
        n_line: Number of top sites to include in line plots ('all' or an integer).
        n_dist: Number of top sites to include in distribution plots (box/violin) ('all' or an integer).
        n_per_site: Number of top sites for per-site plots ('all' or an integer). If None, uses n_line.
        x_col: Column to use for the X-axis (e.g., 'time', 'day').
        x_order: Optional list defining the specific order of x-axis values.
        plot_types: List of plot types to generate for the combined view (e.g., ['line', 'box']).
        per_site: Boolean flag; if True, generates individual plots for each of the top `n` sites.
        output_dir: Directory to save the output plots.
        output_format: File format for the plots.
        bin_width_days: Optional integer for day binning (e.g., 10 for 10-day bins).
        min_samples_per_bin: Minimum number of valid (non-NA) samples required per bin (default: 1).
        group_by_replicate: If True, aggregate subject trajectories by replicate in per-site plots.
        line_alpha: Transparency level for lines in line plots (0.0 to 1.0).
        max_initial_freq: Optional maximum initial frequency threshold (inclusive). Sites with
                          initial frequency > this value in initial_freq_group are excluded.
        min_initial_freq: Optional minimum initial frequency threshold (inclusive). Sites with
                          initial frequency < this value in initial_freq_group are excluded.
        initial_freq_group: The group to use for initial frequency filtering. Sites are
                            selected based on their initial frequency in this group only.
                            Data from ALL groups is kept for matching sites.
        combined_per_replicate: If True, additionally emit one combined line plot per
                                replicate, using the same site selection as the pooled plot
                                but computing means within each replicate. Requires a
                                'replicate' column in the input data.
    """
    # === VALIDATION PHASE ===
    # Early exit if no data to process
    if df.empty:
        logger.error("Empty DataFrame provided to plot_group_distributions")
        return

    # Detect duplicate rows that would indicate data integrity issues
    # Each sample should have exactly one frequency value per site
    dup = df.duplicated(
        subset=["contig", "position", "anchor_allele", "sample_id", value_col]
    )
    if dup.any():
        dup_rows = df.loc[
            dup, ["contig", "position", "anchor_allele", "sample_id", value_col]
        ]
        logger.error(f"Found {dup.sum()} duplicate rows")
        raise ValueError(
            f"Duplicate entries found! This should not happen.\n{dup_rows}"
        )

    # Fail before any expensive filtering/aggregation if the requested per-replicate output
    # cannot possibly be produced.
    if combined_per_replicate and "replicate" not in df.columns:
        raise ValueError(
            "Cannot create per-replicate combined plots: 'replicate' column not found in "
            "input data"
        )

    logger.info(f"Processing {len(df):,} rows for plotting")

    # Extract MAG identifier for output filenames (defaults to "MAG" if not present)
    mag_id = str(df["mag_id"].iloc[0]) if "mag_id" in df.columns else "MAG"

    # Filename tag derived from the bin width, so plots produced with different
    # widths (e.g., a sensitivity sweep) can coexist in one directory. Derived
    # here — not passed by callers — so the name can never disagree with the
    # width that actually produced the figure.
    bin_tag = f"_bin{bin_width_days}d" if bin_width_days is not None else ""

    # === X-AXIS PREPARATION ===
    # Apply custom ordering/filtering if specified, or convert 'day' to numeric
    df, sorted_vals = prepare_x_axis(df, x_col, x_order)

    # === VALIDATE TIME COLUMN ORDERING ===
    # When using 'time' as x_col, require explicit --x_order to avoid lexicographic
    # sorting issues (e.g., 'T10' < 'T2' in string order)
    if x_col == "time" and x_order is None:
        unique_times = sorted(df[x_col].unique().tolist())
        raise ValueError(
            f"When using --x_col time, you must specify --x_order to define the "
            f"correct chronological order. Lexicographic sorting can cause issues "
            f"(e.g., 'T10' sorts before 'T2'). "
            f"Found time values: {unique_times}. "
            f"Example: --x_order T1 T2 T3 T10"
        )

    # === TIME BINNING (OPTIONAL) ===
    # Groups continuous day values into discrete bins for cleaner visualization
    if bin_width_days is not None:
        # Binning requires a valid 'day' column with numeric values
        if "day" not in df.columns or df["day"].isnull().all():
            raise ValueError(
                "bin_width_days provided, but 'day' column is missing or empty."
            )

        logger.info(f"Binning day column by {bin_width_days} day intervals...")

        # Apply assign_bin() to each day value to get bin metadata
        # - bin_label: Human-readable range (e.g., "0-9", "10-19")
        # - bin_midpoint: Numeric center for line plot x-coordinates
        # - bin_sort: Numeric key for chronological ordering
        labels, midpoints, sort_keys = [], [], []
        for d in df["day"]:
            label, midpoint, sort_key = assign_bin(d, bin_width_days)
            labels.append(label)
            midpoints.append(midpoint)
            sort_keys.append(sort_key)

        df["bin_label"] = labels
        df["bin_midpoint"] = midpoints
        df["bin_sort"] = sort_keys

        # Convert bin_label to ordered Categorical for correct plot axis ordering
        # Sort by bin_sort to ensure chronological order (e.g., "0-9" before "10-19")
        unique_bins = (
            df[["bin_label", "bin_sort"]].drop_duplicates().sort_values("bin_sort")
        )
        ordered_labels = unique_bins["bin_label"].tolist()
        df["bin_label"] = pd.Categorical(
            df["bin_label"], categories=ordered_labels, ordered=True
        )

    # === INITIAL FREQUENCY FILTERING (OPTIONAL) ===
    # Filters trajectories based on their frequency at the first timepoint
    # This is done AFTER binning so that "first timepoint" refers to the first bin
    if max_initial_freq is not None or min_initial_freq is not None:
        # Determine which x_col to use for finding the first timepoint
        # When binning is enabled, use bin_midpoint (numeric) for correct ordering
        filter_x_col = "bin_midpoint" if bin_width_days is not None else x_col

        df = filter_by_initial_frequency(
            df,
            filter_x_col,
            max_initial_freq=max_initial_freq,
            min_initial_freq=min_initial_freq,
            filter_group=initial_freq_group,
        )
        if df.empty:
            filter_desc = []
            if min_initial_freq is not None:
                filter_desc.append(f"min={min_initial_freq}")
            if max_initial_freq is not None:
                filter_desc.append(f"max={max_initial_freq}")
            logger.error(
                f"No data remaining after initial frequency filtering "
                f"({', '.join(filter_desc)})"
            )
            return

    # === DATA CACHING ===
    # Cache computed data to avoid redundant processing when n_line == n_dist
    # Cache key includes plot_type when binning (different x-columns for line vs box)
    data_cache = {}

    # === GENERATE COMBINED PLOTS ===
    # Creates summary plots showing all selected sites together
    for plot_type in plot_types:
        logger.info(f"Creating {plot_type} plot")

        # Line plots use n_line; box/violin use n_dist (allows different N for each)
        if plot_type == "line":
            n_val = n_line
        else:  # box, violin
            n_val = n_dist

        # Cache key differentiates by plot_type when binning is enabled
        # (line uses bin_midpoint, box/violin use bin_label)
        cache_key = (n_val, plot_type) if bin_width_days is not None else n_val

        # Compute data if not already cached
        if cache_key not in data_cache:
            data_cache[cache_key] = get_plotting_data(
                df=df,
                value_col=value_col,
                n_val=n_val,
                x_col=x_col,
                sorted_vals=sorted_vals,
                bin_width_days=bin_width_days,
                min_samples_per_bin=min_samples_per_bin,
                plot_type=plot_type,
            )

        filtered, mean_df, n_label, plot_x_col = data_cache[cache_key]

        # Generate the plot if we have valid data
        if mean_df is not None:
            plot_combined(
                mean_df,
                plot_x_col,
                n_label,
                value_col,
                output_dir,
                output_format,
                plot_type,
                mag_id,
                line_alpha=line_alpha,
                filename_tag=bin_tag,
            )

        # === PER-REPLICATE COMBINED PLOTS ===
        # Same sites as the pooled line plot above, but one figure per replicate. Driven from
        # `filtered` (sample-level rows) rather than `mean_df`, because the means have to be
        # recomputed within each replicate instead of subset from the pooled means.
        if combined_per_replicate and plot_type == "line" and filtered is not None:
            plot_combined_per_replicate(
                filtered,
                plot_x_col,
                n_label,
                value_col,
                output_dir,
                output_format,
                mag_id=mag_id,
                line_alpha=line_alpha,
                min_samples_per_bin=min_samples_per_bin,
                # Custom order only applies to the original x_col; binned axes carry their own
                # ordering, matching the choice made in get_plotting_data.
                custom_order=None if bin_width_days is not None else sorted_vals,
                filename_tag=bin_tag,
            )

    # === GENERATE PER-SITE PLOTS ===
    # Creates individual trajectory plots for each selected site
    if per_site:
        # Use n_per_site if specified, otherwise fall back to n_line
        n_per_site_val = n_per_site if n_per_site is not None else n_line

        # Skip per-site plots if n_per_site is explicitly set to 0
        if n_per_site_val == 0:
            logger.info("Skipping per-site plots (n_per_site=0)")
        else:
            logger.info("Creating per-site plots")

            # Per-site plots are trajectory plots, so always use "line" config
            cache_key = (
                (n_per_site_val, "line")
                if bin_width_days is not None
                else n_per_site_val
            )
            if cache_key not in data_cache:
                data_cache[cache_key] = get_plotting_data(
                    df=df,
                    value_col=value_col,
                    n_val=n_per_site_val,
                    x_col=x_col,
                    sorted_vals=sorted_vals,
                    bin_width_days=bin_width_days,
                    min_samples_per_bin=min_samples_per_bin,
                    plot_type="line",
                )

            # Use filtered (raw sample data), not mean_df (we compute means per-site)
            filtered, _, n_label, plot_x_col = data_cache[cache_key]

            if filtered is not None:
                # Validate that replicate column exists if replicate grouping requested
                if group_by_replicate and "replicate" not in filtered.columns:
                    logger.error(
                        "--group_by_replicate was specified but 'replicate' column is missing from input data"
                    )
                    raise ValueError(
                        "Cannot group by replicate: 'replicate' column not found in input data"
                    )

                plot_per_site(
                    filtered,
                    plot_x_col,
                    n_label,
                    value_col,
                    output_dir,
                    output_format,
                    group_by_replicate=group_by_replicate,
                    filename_tag=bin_tag,
                )


def main():
    """Main CLI function for plot_allele_trajectory."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Plot allele frequency trajectories for genomic sites",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Input TSV file with allele frequency data in long format.",
    )
    parser.add_argument(
        "--value_col",
        default="min_p_value",
        choices=["min_p_value", "q_value"],
        help="Column name used to rank sites. Sites with the lowest values in this column will be selected.",
    )
    parser.add_argument(
        "--n_sites_line",
        default=10,
        help="Number of top sites to plot for line plots ('all' or integer).",
    )
    parser.add_argument(
        "--n_sites_dist",
        default="all",
        help="Number of top sites to plot for box/violin plots ('all' or integer).",
    )
    parser.add_argument(
        "--n_sites_per_site",
        default=None,
        help="Number of top sites to plot for per-site plots ('all' or integer). "
        "If not specified, uses --n_sites_line value.",
    )
    parser.add_argument(
        "--x_col",
        default="time",
        choices=["time", "day"],
        help="Column to use for the X-axis. 'time' is typically categorical (e.g., T1, T2), while 'day' is numeric.",
    )
    parser.add_argument(
        "--x_order",
        nargs="+",
        required=False,
        help="Explicitly define the order of X-axis values (space-separated). "
        "This also acts as a filter: only data matching these values will be plotted. "
        "Example: --x_order T1 T2 T3",
    )
    parser.add_argument(
        "--plot_types",
        nargs="+",
        default=["line"],
        choices=["line", "box", "violin"],
        help="Types of combined plots to create. Can specify multiple (e.g., line box).",
    )
    parser.add_argument(
        "--per_site",
        action="store_true",
        help="If set, create individual plots where each line is a separate subject for each of the top N sites.",
    )
    parser.add_argument(
        "--output_dir", default="./plots", help="Output directory for plots"
    )
    parser.add_argument(
        "--output_format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format",
    )
    parser.add_argument(
        "--bin_width_days",
        type=int,
        nargs="+",
        required=False,
        help="Bin width(s) in days for time binning (e.g., 10, or 10 20 30 for a "
        "sensitivity sweep). A 'day' column must be present. With multiple widths, "
        "the full plotting pass runs once per width and each output filename "
        "carries a _bin<W>d tag.",
    )
    parser.add_argument(
        "--min_samples_per_bin",
        type=int,
        default=1,
        help="Minimum number of valid (non-NA) samples required per bin. "
        "NA/NaN frequency values are automatically excluded from the count.",
    )
    parser.add_argument(
        "--group_by_replicate",
        action="store_true",
        help="If set, consolidate subject trajectories within the same replicate into a single "
        "average line per replicate in per-site plots. Requires 'replicate' column in input data.",
    )
    parser.add_argument(
        "--combined_per_replicate",
        action="store_true",
        help="If set, additionally write one combined line plot per replicate to a "
        "'by_replicate/' subdirectory. Site selection and filtering are identical to the "
        "pooled combined plot, but frequencies are averaged within each replicate. "
        "Requires 'replicate' column in input data.",
    )
    parser.add_argument(
        "--line_alpha",
        type=float,
        default=0.8,
        help="Transparency level for lines in line plots (0.0 = fully transparent, 1.0 = fully opaque).",
    )
    parser.add_argument(
        "--max_initial_freq",
        type=float,
        default=None,
        help="Maximum initial frequency threshold (inclusive). Only sites where the frequency at the "
        "first timepoint (in --initial_freq_group) is <= this value will be kept. "
        "Can be used alone or with --min_initial_freq to specify a range. "
        "Example: --max_initial_freq 0.1 --initial_freq_group fat",
    )
    parser.add_argument(
        "--min_initial_freq",
        type=float,
        default=None,
        help="Minimum initial frequency threshold (inclusive). Only sites where the frequency at the "
        "first timepoint (in --initial_freq_group) is >= this value will be kept. "
        "Can be used alone or with --max_initial_freq to specify a range. "
        "Example: --min_initial_freq 0.05 --max_initial_freq 0.2 --initial_freq_group fat",
    )
    parser.add_argument(
        "--initial_freq_group",
        type=str,
        default=None,
        help="The group to use for initial frequency filtering. Can be a specific group name (e.g., 'fat'), "
        "or 'both'/'all' to require the criteria be met in ALL groups. "
        "Data from ALL groups is plotted for sites that pass the filter. "
        "Examples: --max_initial_freq 0.1 --initial_freq_group fat | --initial_freq_group both",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, sep="\t")
    logger.info(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

    # Convert n_sites arguments to appropriate type
    def parse_n_sites(n_val, name):
        if n_val == "all":
            return "all"
        try:
            return int(n_val)
        except ValueError:
            logger.error(f"Invalid {name} value: {n_val}")
            return None

    n_line = parse_n_sites(args.n_sites_line, "n_sites_line")
    n_dist = parse_n_sites(args.n_sites_dist, "n_sites_dist")
    n_per_site = (
        parse_n_sites(args.n_sites_per_site, "n_sites_per_site")
        if args.n_sites_per_site
        else None
    )

    if n_line is None or n_dist is None:
        return

    # Run the full plotting pass once per requested bin width (or once, unbinned).
    # Each pass gets a fresh copy of the loaded table: the orchestrator adds
    # binning columns AND row-filters via the initial-frequency criteria (which
    # are evaluated on the FIRST BIN, so they depend on the width) — reusing one
    # DataFrame across widths would let one width's filtering corrupt the next.
    bin_widths = args.bin_width_days if args.bin_width_days else [None]
    for bw in bin_widths:
        if bw is not None:
            logger.info(f"=== Plotting pass: bin_width_days={bw} ===")
        plot_group_distributions(
            df=df.copy(),
            value_col=args.value_col,
            n_line=n_line,
            n_dist=n_dist,
            n_per_site=n_per_site,
            x_col=args.x_col,
            x_order=args.x_order,
            plot_types=args.plot_types,
            per_site=args.per_site,
            output_dir=args.output_dir,
            output_format=args.output_format,
            bin_width_days=bw,
            min_samples_per_bin=args.min_samples_per_bin,
            group_by_replicate=args.group_by_replicate,
            line_alpha=args.line_alpha,
            max_initial_freq=args.max_initial_freq,
            min_initial_freq=args.min_initial_freq,
            initial_freq_group=args.initial_freq_group,
            combined_per_replicate=args.combined_per_replicate,
        )

    logger.info("Plotting completed successfully")


if __name__ == "__main__":
    main()
