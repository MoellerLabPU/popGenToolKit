#!/usr/bin/env python3

"""
Genome-wide Quality Control Script for AlleleFlux

This script performs comprehensive quality control analysis on metagenome-assembled genomes (MAGs).
It calculates coverage statistics, validates sample quality, and performs threshold-based filtering.

Key Features:
- Genome-wide breadth and coverage analysis
- Length-weighted coverage calculations
- Timepoint validation for longitudinal studies
- Subject and replicate counting per experimental group
- Parallel processing for multiple samples
- Comprehensive logging and error handling

The script processes coverage profile files for each MAG and determines whether samples pass
breadth and coverage thresholds. It supports both single timepoint and longitudinal study designs.

For position-specific QC analysis (analyzing coverage at specific genomic positions),
use the separate positions_qc.py script in alleleflux/scripts/accessory/.

Author: AlleleFlux Development Team
"""

import argparse
import logging
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.qc_metrics import (
    aggregate_mag_stats,
    calculate_breadth_metrics,
    calculate_coverage_metrics,
    calculate_length_weighted_coverage,
    calculate_median_and_std_metrics,
    check_breadth_threshold,
    check_coverage_threshold,
    load_and_validate_profile,
)
from alleleflux.scripts.utilities.utilities import (
    build_contig_length_index,
    calculate_mag_sizes,
    load_mag_mapping,
    load_mag_metadata_file,
)

logger = logging.getLogger(__name__)


def init_worker(
    metadata,
    mag_sizes,
    contig_lengths,
    contig_to_mag_map,
    mag_to_contigs_map,
):
    """Initialize worker process globals shared across all sample tasks."""
    global metadata_dict
    global mag_size_dict
    global contig_length_dict
    global contig_to_mag
    global mag_to_contigs
    metadata_dict = metadata
    mag_size_dict = mag_sizes
    contig_length_dict = contig_lengths
    contig_to_mag = contig_to_mag_map
    mag_to_contigs = mag_to_contigs_map


def process_mag_files(args):
    """
    Process a MAG (Metagenome-Assembled Genome) profile file and calculate coverage statistics.

    This function reads a coverage profile for a specific MAG in a sample and determines
    whether it passes the breadth of coverage threshold and coverage depth threshold.
    The function uses global dictionaries `metadata_dict` and `mag_size_dict` to access
    sample metadata and MAG sizes.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - sample_id (str): Identifier for the sample
        - profile_fPath (str): Path to the coverage profile file
        - mag_id (str): Identifier for the MAG
        - breadth_threshold (float): Minimum required breadth of coverage (0.0-1.0)
        - coverage_threshold (float): Minimum required average coverage depth

    Returns
    -------
    dict
        A dictionary containing coverage statistics and metadata:
        - sample_id: Sample identifier
        - MAG_ID: MAG identifier
        - file_path: Path to the analyzed profile
        - group: Sample group from metadata
        - subjectID: Subject identifier from metadata
        - replicate: Replicate information from metadata
        - time: Time point (if available in metadata)
        - genome_size: Size of the MAG in base pairs
        - breadth: Calculated breadth of coverage (proportion of genome covered)
        - breadth_threshold_passed: Boolean indicating if breadth threshold was met
        - breadth_fail_reason: Explanation if threshold not met
        - average_coverage: Average coverage depth across genome
        - coverage_threshold_passed: Boolean indicating if coverage threshold was met
        - coverage_fail_reason: Explanation if threshold not met
        - median_coverage: Median of per-position total_coverage
        - median_coverage_including_zeros: Median coverage including zeros for absent positions
        - coverage_std: Standard deviation of per-position total_coverage (population, ddof=0)
        - coverage_std_including_zeros: Population std over entire genome length (absent positions as zero)
        - length_weighted_coverage: Length-weighted mean: Σ(mean_cov_contig * contig_length)/Σ(contig_length)

    Raises
    ------
    ValueError
        If more than 2 unique groups are found in the valid samples.

    Notes
    -----
    Breadth of coverage is calculated as the number of positions with at least 1x coverage
    divided by the genome size.
    """
    sample_id, profile_fPath, mag_id, breadth_threshold, coverage_threshold = args

    # Note: QC metrics (breadth, coverage) are group-independent.
    # Group-pair specific eligibility is handled downstream by eligibility_table.py.

    # Build a dictionary to store coverage stats
    # Columns are organized by category for better readability
    result = {
        # Sample metadata
        "sample_id": sample_id,
        "MAG_ID": mag_id,
        "file_path": profile_fPath,
        "group": metadata_dict[sample_id]["group"],
        "subjectID": metadata_dict[sample_id]["subjectID"],
        "replicate": metadata_dict[sample_id]["replicate"],
        "genome_size": mag_size_dict.get(mag_id, 0),
        # Breadth metrics (grouped together)
        "breadth": None,
        "breadth_threshold_passed": False,
        "breadth_fail_reason": "",
        # Coverage metrics (grouped together)
        "average_coverage": None,
        "coverage_threshold_passed": False,
        "coverage_fail_reason": "",
        # Additional coverage statistics
        "median_coverage": None,
        "median_coverage_including_zeros": None,
        "coverage_std": None,
        "coverage_std_including_zeros": None,
        "length_weighted_coverage": None,
    }

    # Add time if available in metadata
    if "time" in metadata_dict[sample_id]:
        result["time"] = metadata_dict[sample_id]["time"]

    # Check if MAG size is known
    mag_size = mag_size_dict.get(mag_id)
    if mag_size is None or mag_size <= 0:
        msg = f"MAG size for {mag_id} not found or invalid."
        logger.error(f"{msg} sample={sample_id}, profile_fPath={profile_fPath}")
        result["breadth_fail_reason"] = msg
        return result

    # Load and validate profile (filter to MAG's contigs)
    try:
        df = load_and_validate_profile(
                profile_fPath, mag_id, sample_id, contig_to_mag,
                usecols=["contig", "total_coverage"],
                mag_contigs=mag_to_contigs.get(mag_id),
            )
    except ValueError as e:
        # Profile empty after filtering to MAG contigs
        result["breadth_fail_reason"] = str(e)
        return result

    # Check for zero coverage values in the profile (should not exist)
    zero_coverage_count = (df["total_coverage"] == 0).sum()
    if zero_coverage_count > 0:
        logger.warning(
            f"Profile contains {zero_coverage_count} positions with zero coverage for MAG {mag_id}. "
            f"These should typically be absent from the profile. file={profile_fPath} sample={sample_id}"
        )

    # Use genome size as denominator for all metrics
    denom = mag_size

    # Calculate breadth metrics (genome-wide only)
    breadth_metrics = calculate_breadth_metrics(
        df, denom, mag_size, using_filter=False, positions_with_coverage_genome=None
    )
    # Only update with non-None values (filters out breadth_genome)
    result.update({k: v for k, v in breadth_metrics.items() if v is not None})

    # Calculate coverage metrics (genome-wide only)
    coverage_metrics = calculate_coverage_metrics(
        df, denom, mag_size, using_filter=False, total_coverage_sum_genome=None
    )
    # Only update with non-None values (filters out average_coverage_genome)
    result.update({k: v for k, v in coverage_metrics.items() if v is not None})

    # Get average coverage for use in median/std calculations
    avg_cov = result["average_coverage"]

    # Calculate median and standard deviation metrics
    median_std_metrics = calculate_median_and_std_metrics(df, denom, avg_cov)
    result.update(median_std_metrics)

    # Calculate length-weighted coverage
    length_weighted_cov = calculate_length_weighted_coverage(
        df, contig_length_dict, mag_id, sample_id, mag_size
    )
    result["length_weighted_coverage"] = length_weighted_cov

    # Check breadth threshold
    breadth_passed, breadth_fail_reason = check_breadth_threshold(
        result["breadth"],
        breadth_genome=None,
        threshold=breadth_threshold,
        using_filter=False,
        sample_id=sample_id,
        mag_id=mag_id,
    )
    result["breadth_threshold_passed"] = breadth_passed
    result["breadth_fail_reason"] = breadth_fail_reason

    # Check coverage threshold (cascades from breadth)
    coverage_passed, coverage_fail_reason = check_coverage_threshold(
        avg_cov,
        avg_cov_genome=None,
        threshold=coverage_threshold,
        breadth_passed=breadth_passed,
        using_filter=False,
        sample_id=sample_id,
        mag_id=mag_id,
    )
    result["coverage_threshold_passed"] = coverage_passed
    result["coverage_fail_reason"] = coverage_fail_reason

    return result


def check_timepoints(df, data_type):
    """
    Check if MAGs (Metagenome-Assembled Genomes) are present in two timepoints for each subject.
    This function verifies that subjects have data from exactly two timepoints for longitudinal analysis.
    For single datatype, it skips the timepoint checks and sets two_timepoints_passed to match coverage_threshold_passed.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing sample data with 'subjectID', 'time', and 'coverage_threshold_passed' columns
    data_type : str
        Type of data analysis, either "single" or assumed to be longitudinal

    Returns
    -------
    pandas.DataFrame
        The input dataframe with an additional 'two_timepoints_passed' column indicating samples
        that come from subjects with exactly 2 timepoints

    Raises
    ------
    ValueError
        If more than 2 unique timepoints are found in the passed samples

    Notes
    -----
    - For longitudinal analysis, the function ensures there are exactly 2 unique timepoints globally
    - A sample is marked as passing if its subject has data at both timepoints and the sample has passed
      the coverage threshold
    - Logs information about subjects with valid timepoints
    """
    # For single datatype, skip timepoint checks
    if data_type == "single":
        # Set two_timepoints_passed to match coverage_threshold_passed for single data type
        df["two_timepoints_passed"] = df["coverage_threshold_passed"]
        return df

    # Initialize columns
    df["two_timepoints_passed"] = False

    # Get samples that passed coverage threshold
    passed_samples = df[df["coverage_threshold_passed"]]

    # Check that there are exactly 2 unique timepoints globally.
    unique_times = passed_samples["time"].unique()
    if len(unique_times) > 2:
        raise ValueError(f"More than 2 unique timepoints found: {unique_times}")

    if passed_samples.empty:
        logger.warning("No samples passed coverage threshold")
        return df

    # Find subjects with MAG in exactly 2 timepoints
    valid_subjects = (
        passed_samples.groupby("subjectID")["time"]
        .nunique()
        .where(lambda x: x == 2)
        .dropna()
        .index
    )

    # Mark passed samples from valid subjects
    valid_mask = passed_samples["subjectID"].isin(valid_subjects)
    df.loc[passed_samples[valid_mask].index, "two_timepoints_passed"] = True

    # Add debug info
    if not valid_subjects.empty:
        logger.debug(f"Subjects with 2 valid timepoints: {len(valid_subjects)}")
    else:
        logger.warning("No subjects have valid MAG in 2 timepoints")

    return df


def add_subject_count_per_group(df):
    """
    Add columns for subject count and replicate count per group to the DataFrame.

    This function calculates the number of unique subjects and replicates for each group
    in the dataset, but only considers samples that have passed timepoint validation
    (where 'two_timepoints_passed' is True). It then adds these counts as new columns
    to the original DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least the following columns:
        - 'two_timepoints_passed': Boolean indicating if the sample has passed timepoint validation
        - 'group': Group identifier
        - 'subjectID': Subject identifier
        - 'replicate': Replicate identifier

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with two new columns added:
        - 'subjects_per_group': Number of unique subjects in the group (NaN for invalid samples)
        - 'replicates_per_group': Number of unique replicates in the group (NaN for invalid samples)

    Notes
    -----
    If no valid samples are found, both new columns will be set to NaN for all rows.
    The function logs a summary of subject and replicate counts per group for valid samples.
    """
    # Get samples that passed timepoint validation
    valid_subjects = df[df["two_timepoints_passed"]]

    if valid_subjects.empty:
        logger.debug("No valid subjects found")
        # Initialize the columns with NaN values
        df["subjects_per_group"] = np.nan
        df["replicates_per_group"] = np.nan
        # Set 0 only for valid subjects that passed the timepoint check
        df.loc[df["two_timepoints_passed"], "subjects_per_group"] = 0
        df.loc[df["two_timepoints_passed"], "replicates_per_group"] = 0
        return df

    # Count unique subjects per group in valid samples
    group_counts = (
        valid_subjects.groupby("group")["subjectID"]
        .nunique()  # Count unique subjects per group
        .reset_index(name="subjects_per_group")
    )

    # Count replicates per group in valid samples
    replicate_counts = (
        valid_subjects.groupby("group")["replicate"]
        .nunique()
        .reset_index(name="replicates_per_group")
    )

    # Merge both counts into one counts DataFrame keyed by group.
    counts = group_counts.merge(replicate_counts, on="group", how="outer")

    # Merge the counts back into the original DataFrame based on group
    df = df.merge(counts, on="group", how="left")

    # For rows where two_timepoints_passed is False, set the counts to NaN.
    df.loc[
        ~df["two_timepoints_passed"], ["subjects_per_group", "replicates_per_group"]
    ] = np.nan

    logger.debug("Subject and replicate counts per group (valid samples):\n%s", counts)
    return df


def count_paired_replicates(df):
    """
    Count the number of paired replicates per group in the dataset.

    This function identifies valid subjects (those that passed both timepoints) and counts
    replicates that appear in both groups. It ensures exactly two groups exist in the valid
    samples dataset.

    A "paired replicate" means the same replicate has samples in both experimental groups.
    This helps identify balanced experimental designs where the same experimental units
    are measured across different conditions.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the following columns:
        - 'two_timepoints_passed': Boolean indicating if the sample has two valid timepoints
        - 'group': The experimental group identifier
        - 'replicate': The replicate identifier for the sample

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an additional column:
        - 'paired_replicates_per_group': Number of replicates per group that have samples in both groups.
          NaN for samples that failed timepoint check or whose replicate isn't paired.
          Zero (0) for samples that passed timepoint check but have no paired replicates.


    Notes
    -----
    - If there are no valid subjects at all, the function returns the DataFrame with
      'paired_replicates_per_group' column set to NaN for all rows.
    - If there are valid subjects but no paired replicates, the function sets
      'paired_replicates_per_group' to 0 for valid subjects (those that passed timepoint check).
    - For samples that failed the timepoint check, the column is always set to NaN.
    """
    valid_subjects = df[df["two_timepoints_passed"]]

    if valid_subjects.empty:
        logger.debug("No valid subjects found")
        # Initialize the column with NaN values
        df["paired_replicates_per_group"] = np.nan
        # Set 0 only for valid subjects that passed the timepoint check
        df.loc[df["two_timepoints_passed"], "paired_replicates_per_group"] = 0
        return df

    rep_group_count = (
        valid_subjects.groupby("replicate")["group"]
        .nunique()
        .reset_index(name="group_count")
    )
    paired_reps = rep_group_count[rep_group_count["group_count"] == 2]["replicate"]

    # Check if there are any paired replicates
    if paired_reps.empty:
        logger.warning("No paired replicates found across groups")
        # Initialize the column with NaN values
        df["paired_replicates_per_group"] = np.nan
        # Set 0 only for valid subjects that passed the timepoint check
        df.loc[df["two_timepoints_passed"], "paired_replicates_per_group"] = 0
        return df

    paired_valid_subjects = valid_subjects[
        valid_subjects["replicate"].isin(paired_reps)
    ]

    # Count the number of unique paired replicates per group
    paired_counts = (
        paired_valid_subjects.groupby("group")["replicate"]
        .nunique()
        .reset_index(name="paired_replicates_per_group")
    )

    # Merge the paired replicate counts back into the original DataFrame based on 'group'
    df = df.merge(paired_counts, on="group", how="left")

    # Set counts to NaN where:
    # - Samples failed timepoint check OR
    # - Samples passed timepoint check but replicate isn't paired
    valid_mask = df["two_timepoints_passed"]
    paired_mask = df["replicate"].isin(paired_reps)
    df["paired_replicates_per_group"] = np.where(
        valid_mask & paired_mask, df["paired_replicates_per_group"], np.nan
    )

    logger.debug("Replicates paired per group:\n%s", paired_counts)
    return df


def _aggregate_mag(mag_id, results_list, output_dir, data_type):
    """Per-MAG aggregation: timepoint checks, subject counts, summaries.

    Runs in the main process after all samples for this MAG have been
    processed by the pool.  Pure pandas operations on a small DataFrame.
    """
    df_results = pd.DataFrame(results_list)

    # Check that each subject has exactly 2 timepoints (for longitudinal data only)
    df_results = check_timepoints(df_results, data_type)

    # Count the number of unique subjects and replicates per group
    df_results = add_subject_count_per_group(df_results)

    # Count the number of paired replicates per group
    df_results = count_paired_replicates(df_results)

    out_file = Path(output_dir) / f"{mag_id}_QC.tsv"
    df_results.to_csv(out_file, sep="\t", index=False)
    logger.debug(f"Saved QC report for {mag_id} to {out_file}")

    if not df_results["coverage_threshold_passed"].any():
        logger.warning(f"MAG {mag_id}: no samples passed QC; skipping summaries.")
        return None, None, None

    # Prepare summaries
    group_summary = aggregate_mag_stats(
        df_results, mag_id, group_cols=["group"], overall=False
    )
    group_time_summary = None
    if "time" in df_results.columns:
        group_time_summary = aggregate_mag_stats(
            df_results, mag_id, group_cols=["group", "time"], overall=False
        )
    overall_summary = aggregate_mag_stats(
        df_results, mag_id, group_cols=[], overall=True
    )
    return group_summary, group_time_summary, overall_summary


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description=("Script to perform genome-wide quality control on MAGs."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rootDir",
        required=True,
        type=Path,
        help="Path to the root directory containing metadata files.",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        type=Path,
        help="FASTA file with contigs (for MAG size)",
    )
    parser.add_argument(
        "--breadth_threshold",
        type=float,
        default=0.1,
        help="Minimum breadth coverage required to pass.",
    )
    parser.add_argument(
        "--coverage_threshold",
        type=float,
        default=1,
        help="Minimum average coverage depth required to pass.",
    )
    parser.add_argument(
        "--cpus", type=int, default=cpu_count(), help="Number of processors to use."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory to write output table.",
    )

    parser.add_argument(
        "--data_type",
        help="Is the data from a single timepoint or from a time series (longitudinal)",
        type=str,
        choices=["single", "longitudinal"],
        default="longitudinal",
    )
    parser.add_argument(
        "--mag_mapping_file",
        help="Path to tab-separated file mapping contig names to MAG IDs. "
        "Must have columns 'contig_id' and 'mag_id'.",
        type=Path,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()

    # Note: For position-based QC analysis, use the separate positions_qc.py script
    # located in alleleflux/scripts/accessory/positions_qc.py

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using breadth threshold: {args.breadth_threshold}")
    logger.info(f"Using coverage threshold: {args.coverage_threshold}")

    mag_size_dict = calculate_mag_sizes(args.fasta, args.mag_mapping_file)
    # Build contig length index once (re-used by workers)
    contig_length_dict = build_contig_length_index(args.fasta, args.mag_mapping_file)
    # Load mapping for MAG membership validation/subsetting
    contig_to_mag_global = load_mag_mapping(args.mag_mapping_file)

    metadata_files = list(args.rootDir.glob("*_metadata.tsv"))
    if not metadata_files:
        raise ValueError("No *_metadata.tsv files found in input directory")

    # ── Build a flat list of (sample) tasks and a per-MAG metadata index ──
    # load_mag_metadata_file returns:
    #   metadata_dict: {sample_id -> {group, subjectID, replicate, [time]}}
    #   sample_tuples: [(sample_id, profile_path, mag_id, breadth, cov), ...]
    #
    # We merge ALL per-MAG metadata_dicts into one combined dict (sample_ids
    # are globally unique) and build a flat task list so the Pool dispatches
    # at the sample level, giving perfect work-stealing across MAGs of
    # different sizes.
    combined_metadata = {}  # sample_id -> metadata row
    mag_sample_ids = {}     # mag_id -> [sample_id, ...] (for post-pool grouping)
    all_sample_tasks = []   # flat list fed to pool.imap_unordered

    for metadata_file in metadata_files:
        mag_id = metadata_file.stem.split("_metadata")[0]
        metadata_dict, sample_tuples = load_mag_metadata_file(
            metadata_file, mag_id,
            args.breadth_threshold, args.coverage_threshold, args.data_type,
        )
        if not sample_tuples:
            raise ValueError(
                f"No samples found in the metadata file {metadata_file}. "
                f"For MAG: {mag_id}. Exiting."
            )
        combined_metadata.update(metadata_dict)
        mag_sample_ids[mag_id] = [t[0] for t in sample_tuples]
        all_sample_tasks.extend(sample_tuples)

    n_workers = min(args.cpus, len(all_sample_tasks))
    n_mags = len(mag_sample_ids)
    logger.info(
        f"Processing {len(all_sample_tasks)} samples across {n_mags} MAGs "
        f"with {n_workers} parallel workers."
    )

    # Invert contig_to_mag once so load_and_validate_profile can do an O(1)
    # membership test per unique contig instead of an O(all_contigs) map per row.
    from collections import defaultdict
    mag_to_contigs_global = defaultdict(set)
    for contig, mid in contig_to_mag_global.items():
        mag_to_contigs_global[mid].add(contig)

    # Silence verbose utility loggers before forking workers.
    logging.getLogger("alleleflux.scripts.utilities").setLevel(logging.WARNING)

    # ── Run sample-level pool ──
    # The large dicts (combined_metadata, mag_size_dict, contig_length_dict,
    # contig_to_mag_global, mag_to_contigs_global) are sent ONCE per worker via
    # the initializer, not once per task.
    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(
            combined_metadata,
            mag_size_dict,
            contig_length_dict,
            contig_to_mag_global,
            mag_to_contigs_global,
        ),
    ) as pool:
        with logging_redirect_tqdm():
            sample_results = list(tqdm(
                pool.imap_unordered(process_mag_files, all_sample_tasks, chunksize=4),
                total=len(all_sample_tasks),
                desc="Processing samples",
                unit="sample",
            ))

    # ── Group results by MAG and run per-MAG aggregation ──
    # This is pure pandas on small DataFrames — fast in the main process.
    results_by_mag = {mag_id: [] for mag_id in mag_sample_ids}
    for result in sample_results:
        results_by_mag[result["MAG_ID"]].append(result)

    all_group = []
    all_group_time = []
    all_overall = []

    for mag_id in tqdm(mag_sample_ids, desc="Aggregating MAGs", unit="MAG"):
        group_df, group_time_df, overall_df = _aggregate_mag(
            mag_id, results_by_mag[mag_id], args.output_dir, args.data_type,
        )
        if group_df is not None:
            all_group.append(group_df)
        if group_time_df is not None:
            all_group_time.append(group_time_df)
        if overall_df is not None:
            all_overall.append(overall_df)

    # Write combined summaries
    summaries = [
        (all_group, "ALL_MAGs_QC_group_summary.tsv"),
        (all_group_time, "ALL_MAGs_QC_group_time_summary.tsv"),
        (all_overall, "ALL_MAGs_QC_overall_summary.tsv"),
    ]
    for dfs, filename in summaries:
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(
                args.output_dir / filename, sep="\t", index=False
            )

    logger.info("QC analysis completed.")


if __name__ == "__main__":
    main()
