#!/usr/bin/env python3
"""
Compile and Filter Significant Sites from AlleleFlux Test Results
=================================================================

This script processes and consolidates the output from various AlleleFlux
significance tests. It automatically discovers result files within a given
directory, includes all sites in the analysis, and applies FDR-BH correction
to control for multiple testing.

The script intelligently handles different p-value column naming conventions
from various tests (e.g., LMM, CMH, t-tests) and applies FDR-BH correction
to control for multiple testing.

Key Features:
- Automatic discovery of test result files from a single input directory.
- Intelligent identification of p-value columns for different test types.
- Inclusion of all sites for comprehensive FDR correction analysis.
- Consolidation of results from multiple tests and MAGs into one table.
- FDR-BH correction applied to each test type's concatenated results.
- Optional grouping by MAG ID for more stringent FDR correction.
- Optional generation of a human-readable summary report.

Usage:
    python compile_test_results.py --input-dir /path/to/alleleflux_results
                                 --outdir /path/to/output
                                 --timepoints pre_post
                                 --test-types lmm cmh paired_tTest
                                 --group-by-mag-id
"""

import argparse
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import extract_relevant_columns

# Set up logger for this script
logger = logging.getLogger(__name__)


@contextmanager
def suppress_logger(logger_name, level=logging.WARNING):
    """Context manager to temporarily suppress a logger's output."""
    target_logger = logging.getLogger(logger_name)
    original_level = target_logger.level
    target_logger.setLevel(level)
    try:
        yield
    finally:
        target_logger.setLevel(original_level)


def find_test_files(
    input_dir: Path,
    test_types: list[str],
    timepoint_label: str,
    groups_label: str,
) -> dict[str, list[Path]]:
    """
    Find and group compressed TSV test result files by statistical test type.

    This function searches a given input directory for subdirectories matching a
    single comparison (pattern: f"{test_type}_{timepoint_label}-{groups_label}").
    Within each matching directory it collects all files ending with ".tsv.gz" and
    groups them by test type.

    Parameters
    ----------
    input_dir : Path
        Root directory under which test result subdirectories are located.
    test_types : list[str]
        List of test type identifiers (e.g., ["lmm", "two_sample_paired"]) to search for.
    timepoint_label : str
        Label identifying the timepoint segment of the directory name.
    groups_label : str
        The group-pair label (e.g. "40_AL") identifying the single comparison to
        summarize. Required: discovery is restricted to exactly this comparison's
        directory so a summary only ever sees its own pair. Without it, every group
        pair sharing the timepoint would be pooled into one summary and, for tests
        without a group column (e.g. two_sample_unpaired), conflated irrecoverably.

    Returns
    -------
    dict[str, list[Path]]
        A mapping from each test type to a (possibly empty) list of matching
        compressed TSV result file paths.

    Side Effects
    ------------
    Logs (via the module-level logger) informational messages about:
      * The root directory being searched.
      * How many files were found per test type and directory.

    Notes
    -----
    A test type key will always be present in the returned dictionary even if
    no corresponding files were found (value will be an empty list).

    Example
    -------
    >>> from pathlib import Path
    >>> test_files = find_test_files(
    ...     input_dir=Path("results"),
    ...     test_types=["lmm", "two_sample_paired"],
    ...     timepoint_label="tp1",
    ...     groups_label="40_AL",
    ... )
    >>> test_files["lmm"]  # list of Path objects (possibly empty)
    """

    test_files = {test_type: [] for test_type in test_types}
    logger.info(f"Searching for test results in: {input_dir}")

    for test_type in test_types:
        # Match exactly this comparison's directory, e.g.
        # 'two_sample_unpaired_5mo_10mo-40_AL' -- never other group pairs.
        for test_dir in input_dir.glob(f"{test_type}_{timepoint_label}-{groups_label}"):
            if test_dir.is_dir():
                found_files = list(test_dir.glob("*.tsv.gz"))

                if found_files:
                    test_files[test_type].extend(found_files)
                    logger.info(
                        f"  Found {len(found_files)} file(s) for test '{test_type}' in {test_dir.name}"
                    )

    return test_files


def process_results_file(
    filepath: Path, test_type: str, timepoint_label: str
) -> dict[str, pd.DataFrame]:
    """
    Process a results file for a given test type and timepoint, extracting and summarizing p-values.

    This function loads a gzipped, tab-separated CSV file, identifies p-value columns grouped by sub-test,
    validates or extracts the MAG ID, adds metadata columns, and processes each sub-test to calculate
    the minimum p-value. It sets appropriate test_type and group_analyzed values based on the input
    test_type and sub-test name. Finally, it consolidates the results into a single DataFrame and
    returns it in a dictionary keyed by a combination of test_type and timepoint_label.

    Parameters:
    -----------
    filepath : Path
        The path to the input CSV file (expected to be gzipped and tab-separated).
    test_type : str
        The type of test (e.g., 'lmm', 'lmm_across_time', 'cmh', 'cmh_across_time', 'single_sample').
        This influences how test_type and group_analyzed are set for each sub-test.
    timepoint_label : str
        A label for the timepoint or period associated with the data.

    Returns:
    --------
    dict[str, pd.DataFrame]
        A dictionary with a single key in the format '{test_type}_{timepoint_label}' and a value
        that is a pandas DataFrame containing the processed results. The DataFrame includes columns
        such as 'period', 'mag_id', 'contig', 'position', 'gene_id', 'test_type', 'group_analyzed',
        'min_p_value', and 'source_file' (only those that exist in the data). Returns an empty dict
        if the file is empty, no p-value columns are found, or no sites are processed.

    Raises:
    -------
    ValueError
        If multiple MAG IDs are found in the file when 'mag_id' column is present, or if an unexpected
        sub-test name is encountered for 'single_sample' test type.

    Notes:
    ------
    - Assumes the presence of certain columns like 'contig', 'position', 'gene_id' in the input file.
    - For 'single_sample' test type, sub-test names must start with 'tTest_' or 'Wilcoxon_' followed by the group.
    - Uses logging to warn about empty files, missing p-value columns, or no sites found.
    - Dependencies: pandas (pd), Path from pathlib, and custom functions like extract_relevant_columns,
      extract_mag_id_from_filepath, and a logger.
    """

    # Load the data
    df = pd.read_csv(filepath, sep="\t", compression="gzip", low_memory=False)
    if df.empty:
        logger.warning(f"File {filepath} for test '{test_type}' is empty. Skipping.")
        return {}
    # Identify all possible p-value columns, grouped by the sub-test
    p_value_cols_by_test = extract_relevant_columns(df, capture_str="p_value_")
    if not p_value_cols_by_test:
        logger.warning(f"No p-value columns found in {filepath}. Skipping.")
        return {}
    # Extract MAG ID once for the entire file
    if "mag_id" not in df.columns:
        mag_id = extract_mag_id_from_filepath(filepath, test_type)
    else:
        # Check if all rows have the same MAG ID
        unique_mag_ids = df["mag_id"].unique()
        if len(unique_mag_ids) != 1:
            raise ValueError(
                f"Multiple MAG IDs found in file {filepath}: {unique_mag_ids}. "
                f"Expected exactly one MAG ID per file."
            )
        mag_id = unique_mag_ids[0]
    # Add common metadata columns once
    df["mag_id"] = mag_id
    df["source_file"] = filepath.name
    df["period"] = timepoint_label
    all_subtest_dfs = []
    # Process each sub-test
    for sub_test_name, p_value_cols in p_value_cols_by_test.items():
        # Work on a copy to avoid modifying the original DataFrame in the loop
        sub_df = df.copy()
        # Calculate minimum p-value for this sub-test
        sub_df["min_p_value"] = sub_df[p_value_cols].min(axis=1)
        # Validate min_p_value; if NAs exist, warn and drop those rows instead of aborting
        na_rows = sub_df[sub_df["min_p_value"].isna()]
        if not na_rows.empty:
            logger.warning(
                f"Dropping {len(na_rows)} row(s) with NA min_p_value in file {filepath} (test_type={test_type}, sub_test={sub_test_name}). "
                f"Showing offending rows:\n{na_rows}"
            )
            sub_df = sub_df.dropna(subset=["min_p_value"])
            if sub_df.empty:
                logger.warning(
                    f"All rows dropped due to NA min_p_value for file {filepath} (test_type={test_type}, sub_test={sub_test_name}). Skipping this sub-test."
                )
                continue
        # Set test_type and group_analyzed based on test_type and sub_test_name
        new_test_type = f"{test_type}_{sub_test_name}"  # Default
        if test_type in ["lmm", "lmm_across_time"]:
            # Special handling for LMM absolute vs regular p-value columns to prevent duplication
            if test_type == "lmm" and "_abs" in sub_test_name:
                new_test_type = "LMM_abs"
            elif test_type == "lmm" or test_type == "lmm_across_time":
                new_test_type = "LMM"
            if test_type == "lmm_across_time" and "group_analyzed" in df.columns:
                sub_df["group_analyzed"] = df["group_analyzed"]
        elif test_type in ["cmh", "cmh_across_time"]:
            new_test_type = "CMH"
            # For cmh_across_time retain analyzed_group if provided.
            if test_type == "cmh_across_time" and "analyzed_group" in df.columns:
                sub_df["group_analyzed"] = df["analyzed_group"]
            # if a 'time' column exists (some CMH outputs) use it as group_analyzed
            # only if group_analyzed has not already been set.
            if "group_analyzed" not in sub_df.columns and "time" in df.columns:
                sub_df["group_analyzed"] = df["time"]
        elif test_type == "single_sample":
            # Handles "tTest_{group}" and "Wilcoxon_{group}"
            if sub_test_name.startswith("tTest_"):
                new_test_type = "single_sample_tTest"
                sub_df["group_analyzed"] = sub_test_name[len("tTest_") :]
            elif sub_test_name.startswith("Wilcoxon_"):
                new_test_type = "single_sample_Wilcoxon"
                sub_df["group_analyzed"] = sub_test_name[len("Wilcoxon_") :]
            else:
                raise ValueError(
                    f"Unexpected sub-test name '{sub_test_name}' for single_sample test type in {filepath}. "
                )

        sub_df["test_type"] = new_test_type
        # if "group_analyzed" not in sub_df:
        #     sub_df["group_analyzed"] = None

        # Select final columns
        output_cols = [
            "period",
            "mag_id",
            "contig",
            "position",
            "gene_id",
            "test_type",
            "group_analyzed",
            "min_p_value",
            "source_file",
        ]
        # Store results (only columns that exist) for this sub-test
        available_cols = [col for col in output_cols if col in sub_df.columns]
        all_subtest_dfs.append(sub_df[available_cols])

    if not all_subtest_dfs:
        logger.warning(f"No sites found in {filepath} for test '{test_type}'.")
        return {}

    # Create a filename-friendly key that groups results by test type and timepoint
    combined_key = f"{test_type}_{timepoint_label}"
    # Consolidate all sub-test data from this file into one DataFrame
    consolidated_df = pd.concat(all_subtest_dfs, ignore_index=True)

    return {combined_key: consolidated_df}


def extract_mag_id_from_filepath(filepath: Path, test_type: str) -> str:
    """
    Extract MAG ID from the filename using AlleleFlux naming conventions.

    Args:
        filepath: Path to the result file
        test_type: The test type being processed (e.g., 'lmm_across_time', 'two_sample_paired', 'cmh_across_time')

    Returns:
        MAG ID extracted from filename
    """
    filename = filepath.name

    # Look for the test_type pattern in the filename
    if f"_{test_type}" in filename:
        # Split on the test_type pattern and take the first part as MAG ID
        mag_id = filename.split(f"_{test_type}")[0]
        return mag_id


def main():
    """Main function to drive the script."""
    parser = argparse.ArgumentParser(
        description="Compile and filter significant sites from AlleleFlux test results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing subdirectories of AlleleFlux test results.",
    )
    parser.add_argument(
        "--p-value-threshold",
        type=float,
        default=0.05,
        help="P-value threshold for reference (currently unused as all sites are included for FDR correction).",
    )
    parser.add_argument(
        "--timepoints",
        type=str,
        required=True,
        help="A single timepoint combination label to include (e.g., 'pre_post' or 'pre').",
    )
    parser.add_argument(
        "--groups",
        type=str,
        required=True,
        help=(
            "The group-pair label (e.g. '40_AL') identifying the single comparison to "
            "summarize. Required: the summary is restricted to exactly this comparison "
            "so pairs are never pooled/conflated (which would also pool their FDR)."
        ),
    )
    parser.add_argument(
        "--test-types",
        nargs="+",
        default=[
            "two_sample_unpaired",
            "two_sample_paired",
            "lmm",
            "lmm_across_time",
            "cmh",
            "cmh_across_time",
            "single_sample",
        ],
        help="Space-separated list of test types to include (e.g., lmm cmh paired_tTest) (default: all).",
    )
    parser.add_argument(
        "--outdir", required=True, type=Path, help="Directory to save the output files."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="p_value_summary",
        help="Prefix for the output file names.",
    )
    parser.add_argument(
        "--fdr-group-by-mag-id",
        action="store_true",
        help="Apply FDR correction separately for each MAG ID in addition to test_type and group_analyzed.",
    )
    args = parser.parse_args()

    setup_logging()

    # --- Main Workflow ---
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # 1. Discover all relevant result files
    test_files = find_test_files(
        args.input_dir, args.test_types, args.timepoints, groups_label=args.groups
    )
    total_files = sum(len(files) for files in test_files.values())
    if total_files == 0:
        logger.error(
            "No result files found for the specified test types and timepoints. Exiting."
        )
        sys.exit(1)
    logger.info(f"Found a total of {total_files} files to process.")

    # 2. Process each file and collect significant sites by sub-test
    all_significant_sites_by_subtest = {}
    for test_type, file_paths in test_files.items():
        if not file_paths:
            logger.warning(f"No files found for test type '{test_type}'. Skipping.")
            continue

        logger.info(f"Processing {len(file_paths)} files for test type: '{test_type}'")
        for filepath in tqdm(
            file_paths,
            desc=f"Processing {test_type}",
            unit="file",
            total=len(file_paths),
        ):
            # Suppress utilities logger to avoid interrupting tqdm
            with suppress_logger("alleleflux.scripts.utilities.utilities"):
                significant_sites_by_subtest = process_results_file(
                    filepath, test_type, args.timepoints
                )

            # Organize results by sub-test with combined key
            for combined_key, significant_df in significant_sites_by_subtest.items():
                # Initialize the list for this combined key if it doesn't exist,
                # this prevents KeyError, when initializing the list for the first time for a combined key
                if combined_key not in all_significant_sites_by_subtest:
                    all_significant_sites_by_subtest[combined_key] = []
                all_significant_sites_by_subtest[combined_key].append(significant_df)

    # 3. Consolidate, sort, and save separate tables for each sub-test
    # Process each sub-test separately
    for combined_key, subtest_dataframes in all_significant_sites_by_subtest.items():
        logger.info(f"Consolidating significant sites for: {combined_key}")

        # Concatenate all dataframes for this sub-test
        if not subtest_dataframes:
            logger.warning(f"No dataframes for key '{combined_key}', skipping.")
            continue
        final_df = pd.concat(subtest_dataframes, ignore_index=True)

        # Define grouping columns for FDR correction. Correction is applied per group.
        grouping_cols = ["test_type"]
        if "group_analyzed" in final_df.columns:
            grouping_cols.append("group_analyzed")

        # Group by mag_id is needed.
        if args.fdr_group_by_mag_id:
            grouping_cols.append("mag_id")

        logger.info(
            f"Applying FDR-BH correction grouping cols: {grouping_cols} within {combined_key}..."
        )

        # Apply FDR-BH correction within each group using transform.
        final_df["q_value"] = final_df.groupby(grouping_cols)["min_p_value"].transform(
            lambda x: multipletests(x, method="fdr_bh", alpha=args.p_value_threshold)[1]
        )

        # Create output filename using the combined key
        # Include the group-pair label so each file is self-describing once moved
        # out of its comparison directory, e.g.
        # 'p_value_summary_two_sample_unpaired_5mo_10mo-40_AL.tsv'.
        subtest_output_file = (
            args.outdir / f"{args.prefix}_{combined_key}-{args.groups}.tsv"
        )

        args.outdir.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(subtest_output_file, sep="\t", index=False)
        logger.info(
            f"Successfully wrote {len(final_df):,} sites for {combined_key} to: {subtest_output_file}"
        )

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
