#!/usr/bin/env python3
import argparse
import glob
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

from alleleflux.scripts.preprocessing.quality_control import (
    add_subject_count_per_group,
    count_paired_replicates,
)
from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import relabel_groups_from_metadata

logger = logging.getLogger(__name__)

# Group-dependent QC count columns recomputed after a permutation relabel.
_GROUP_COUNT_COLS = (
    "subjects_per_group",
    "replicates_per_group",
    "paired_replicates_per_group",
)


def _relabel_and_recount(df, permuted_metadata):
    """Relabel a per-MAG QC frame's group for a permuted run, then recompute counts.

    QC's breadth/coverage **and** ``two_timepoints_passed`` are group-independent
    (the permutation moves only group labels — never ``subjectID``/``time``/
    coverage), so they are reused as-is.  Only the three group-dependent count
    columns (``subjects_per_group`` / ``replicates_per_group`` /
    ``paired_replicates_per_group``) change.  We relabel ``group`` from the
    permuted metadata (joined on ``sample_id``), then re-derive just those counts
    with the same functions ``quality_control.py`` used originally — so
    eligibility sees exactly what a from-scratch QC on the permuted labels would
    produce, with no profile reads.
    """
    df = relabel_groups_from_metadata(df, permuted_metadata)
    df = df.drop(columns=[c for c in _GROUP_COUNT_COLS if c in df.columns])
    df = add_subject_count_per_group(df)
    df = count_paired_replicates(df)
    return df


def process_data(df, unique_groups, mag_id, min_sample_num, data_type):
    """
    Process and summarize data eligibility based on replicate counts for each group.

    This function checks for consistency in replicate and paired replicate counts within each given group.
    It computes a summary pivot table containing the number of replicates per group and determines eligibility
    for unpaired and paired tests as well as single sample eligibility (dependent on the data type).

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data with at least the columns "group",
                               "replicates_per_group", and "paired_replicates_per_group".
        unique_groups (list): List of unique group identifiers present in the DataFrame.
        mag_id (Any): Identifier of the current MAG (or sample) being processed.
        min_sample_num (int): The minimum number of replicates required for eligibility.
        data_type (str): Type of the data; if set to "longitudinal", single sample eligibility is assessed
                         per group. Otherwise, the single sample eligibility values are set to NaN.

    Returns:
        dict: A dictionary containing the pivoted summary of replicates and eligibility flags. Keys include:
              - 'unpaired_test_eligible': A boolean indicating if both groups have replicates greater than or equal to min_sample_num.
              - 'paired_test_eligible': A boolean indicating if both groups have paired replicates greater than or equal to min_sample_num.
              - 'single_sample_eligible_<group>': For each group, a boolean (if data_type == "longitudinal") showing
                if the replicates per group are greater than or equal to min_sample_num, or NaN otherwise.

    Raises:
        ValueError: If inconsistent replicate counts are detected within any group, indicating an error for the given MAG.
    """
    # Verify that replicates_per_group and paired_replicates_per_group are consistent within each group
    for group in unique_groups:
        group_df = df[df["group"] == group]
        if (
            group_df["replicates_per_group"].nunique() != 1
            or group_df["paired_replicates_per_group"].nunique() != 1
        ):
            raise ValueError(
                f"MAG {mag_id}, group {group}: Inconsistent replicates detected. Skipping."
            )

    summary = df.groupby(["MAG_ID", "group"], as_index=False).agg(
        replicates_per_group=("replicates_per_group", "first"),
        paired_replicates_per_group=("paired_replicates_per_group", "first"),
    )

    # Pivot the table so that each row is a unique combination of MAG_ID and the groups become separate columns.
    pivot = summary.pivot(index=["MAG_ID"], columns="group")
    pivot.columns = [f"{stat}_{group}" for stat, group in pivot.columns]
    pivot = pivot.reset_index()

    g1, g2 = unique_groups

    # Check unpaired test eligibility: True if replicates_per_group for both groups > min_sample_num.
    pivot["unpaired_test_eligible"] = (
        pivot[f"replicates_per_group_{g1}"] >= min_sample_num
    ) & (pivot[f"replicates_per_group_{g2}"] >= min_sample_num)

    # Check paired test eligibility: True if paired_replicates_per_group for both groups >= min_sample_num.
    pivot["paired_test_eligible"] = (
        pivot[f"paired_replicates_per_group_{g1}"] >= min_sample_num
    ) & (pivot[f"paired_replicates_per_group_{g2}"] >= min_sample_num)

    # For single-sample test eligibility, compute per group based on data_type:
    if data_type == "longitudinal":
        for group in unique_groups:
            pivot[f"single_sample_eligible_{group}"] = (
                pivot[f"replicates_per_group_{group}"] >= min_sample_num
            )
    elif data_type == "single":
        # Set single sample eligibility columns to NaN for single timepoint data
        for group in unique_groups:
            pivot[f"single_sample_eligible_{group}"] = np.nan

    return pivot.to_dict(orient="records")[0]


def analyze_qc_files(
    qc_dir: str,
    min_sample_num: int,
    data_type: str,
    groups: Optional[List[str]] = None,
    permuted_metadata: Optional[str] = None,
) -> pd.DataFrame:
    """
    Analyzes quality control (QC) files in the specified directory and processes
    eligible MAGs based on provided criteria.

    The function performs the following steps:
        - Searches for files ending with "_QC.tsv" in the given qc_dir.
        - Reads each QC file as a DataFrame and skips it if empty.
        - Filters the DataFrame for records where "two_timepoints_passed" is True.
        - If ``groups`` is provided, restricts the data to those two groups only.
        - Checks that the filtered data contains exactly two unique groups.
        - Processes the eligible data using the process_data function.
        - Aggregates the results from each eligible QC file into a single DataFrame.

    Parameters:
        qc_dir (str): The directory path containing the QC files.
        min_sample_num (int): The minimum number of samples required for eligibility.
        data_type: The type of data used in processing.
        groups (list of str, optional): If provided, filter to exactly these two
            group names before eligibility checks.  Required when the QC directory
            covers more than two groups (2A refactor).

    Returns:
        pandas.DataFrame: A DataFrame containing results from all MAGs that pass
        the eligibility criteria.

    Raises:
        ValueError: If no QC files are found in the specified directory.
        ValueError: If no MAG passes the eligibility criteria.
    """
    if groups is not None and len(groups) != 2:
        raise ValueError(
            f"--groups must specify exactly 2 group names for eligibility filtering, "
            f"got {len(groups)}: {groups}"
        )

    qc_files = glob.glob(os.path.join(qc_dir, "*_QC.tsv"))
    if not qc_files:
        raise ValueError(f"No QC files found in directory {qc_dir}")

    results = []
    for qc_file in qc_files:
        logger.info(f"Reading {qc_file}")
        df = pd.read_csv(qc_file, sep="\t")
        mag_id = os.path.basename(qc_file).replace("_QC.tsv", "")

        if df.empty:
            logger.warning(f"QC file {qc_file} is empty. Skipping MAG {mag_id}.")
            continue

        # Permuted (null) run: relabel this reused QC frame and recompute its
        # group-dependent counts before any eligibility logic runs.
        if permuted_metadata:
            df = _relabel_and_recount(df, permuted_metadata)

        # Get the list of groups that passed the two timepoints check
        passed_df = df[df["two_timepoints_passed"]]

        # Filter to the requested group pair (2A: QC dir now covers all groups)
        if groups is not None:
            passed_df = passed_df[passed_df["group"].isin(groups)]

        unique_groups = passed_df["group"].unique()

        if len(unique_groups) != 2:
            logger.warning(
                f"MAG {mag_id}: Found {len(unique_groups)} groups after filtering "
                f"(expected exactly 2). Skipping."
            )
            continue

        logger.info(f"Processing MAG: {mag_id}")
        result = process_data(
            passed_df, unique_groups, mag_id, min_sample_num, data_type
        )

        if result:
            results.append(result)

    if not results:
        raise ValueError("No MAGs passed eligibility criteria.")

    return pd.DataFrame(results)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Generate test eligibility summary table from QC files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--qc_dir", required=True, help="Directory containing *_QC.tsv files"
    )
    parser.add_argument(
        "--min_sample_num", type=int, default=4, help="Minimum replicates required"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output file for summary table"
    )
    parser.add_argument(
        "--data_type",
        choices=["single", "longitudinal"],
        default="longitudinal",
        help="Type of data: single timepoint or longitudinal",
    )
    parser.add_argument(
        "--groups",
        nargs=2,
        default=None,
        metavar=("GROUP1", "GROUP2"),
        help="Filter QC results to exactly these two group names before eligibility "
             "checks. Required when the QC directory covers more than two groups "
             "(2A refactor: per-timepoint QC covers all groups).",
    )
    parser.add_argument(
        "--permuted_metadata",
        type=str,
        default=None,
        help=(
            "Optional path to a permuted metadata TSV (null/control run).  When "
            "given, each reused QC frame's group/subjectID/replicate labels are "
            "re-derived from this sheet (joined on sample_id) and the per-group "
            "replicate counts are recomputed before eligibility — no profile "
            "reads.  Omit for a normal run."
        ),
    )

    args = parser.parse_args()

    eligibility_table = analyze_qc_files(
        args.qc_dir,
        args.min_sample_num,
        args.data_type,
        groups=args.groups,
        permuted_metadata=args.permuted_metadata,
    )
    eligibility_table.to_csv(args.output_file, sep="\t", index=False)

    logger.info(f"Eligibility summary table written to {args.output_file}")


if __name__ == "__main__":
    main()
