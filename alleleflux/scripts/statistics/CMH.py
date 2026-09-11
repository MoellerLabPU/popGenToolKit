#!/usr/bin/env python
"""
CMH Test Runner for AlleleFlux
-----------------------------
Performs Cochran-Mantel-Haenszel (CMH) tests on allele count data for metagenome-assembled genomes (MAGs),
leveraging R's mantelhaen.test via rpy2. Supports multiprocessing and progress tracking.

Supports three analysis modes:
1. Single timepoint: Compare two groups within a single timepoint
2. Longitudinal: Compare two groups, once for each timepoint
3. Across time: Compare the same group across different timepoints

Usage:
    python cmh_test.py --input_df <input.tsv> --preprocessed_df <preprocessed.tsv> --mag_id <MAG> --output_dir <dir> --data_type <type>

    For across_time mode:
    python cmh_test.py --input_df <input.tsv> --preprocessed_df <preprocessed.tsv> --mag_id <MAG> --output_dir <dir> --data_type across_time --group <group_name>
"""
import argparse
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import default_converter, pandas2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import (
    load_allele_freq_inputs,
    load_and_filter_data,
    relabel_groups_from_metadata,
)

NUCLEOTIDES: List[str] = ["A", "T", "G", "C"]

logger = logging.getLogger(__name__)


def process_group(
    args: Tuple[Tuple[str, str, int], pd.DataFrame],
    min_sample_num: int,
    group_1_name: str,
    group_2_name: str,
    mag_id: str,
    timepoint: Optional[str] = None,
) -> dict:
    """
    Processes a group of allele count data for a specific gene position and performs the Cochran-Mantel-Haenszel (CMH) test.

    Parameters:
        args (Tuple[Tuple[str, str, int], pd.DataFrame]):
            A tuple containing:
                - name (Tuple[str, str, int]): Identifiers for the group (contig, gene_id, position).
                - df_group (pd.DataFrame): DataFrame containing allele counts for the group.
        min_sample_num (int): Minimum number of samples required for the CMH test.
        group_1_name (str): Name of the first comparison group.
        group_2_name (str): Name of the second comparison group.
        mag_id (str): Identifier for the metagenome-assembled genome (MAG).
        timepoint (Optional[str]): The timepoint identifier for longitudinal data.

    Returns:
        dict or None:
            - A dictionary with test results and metadata if the group passes filtering and the CMH test runs successfully.
            - Returns None if there is insufficient data for the test.
            - If an error occurs during the CMH test, returns a dictionary with NaN p-value and error notes.
    """
    name, df_group = args  # Unpack the args tuple
    contig, gene_id, position = name  # Extract identifiers

    # Prepare the input for CMH test
    pivoted_filtered = prepare_cmh_input(
        df_group, group_1_name, group_2_name, min_sample_num
    )
    if pivoted_filtered is None:
        return None  # Skip if not enough data

    # logging.info(f"Running CMH for {contig}:{position}")
    # Run the CMH test in R
    try:
        cmh_result = run_cmh_test_in_r(pivoted_filtered)
    except RRuntimeError as e:
        logger.error(
            f"Error running CMH test for {contig}:{position} in MAG {mag_id}: {e}"
        )
        # Return NaN p-value with error note
        result = {
            "mag_id": mag_id,
            "contig": contig,
            "gene_id": gene_id,
            "position": position,
            "num_pairs": len(pivoted_filtered),
            "p_value_CMH": np.nan,
            "notes": f"RRuntimeError: {str(e).rstrip('\n')}",
        }
        if timepoint is not None:
            result["time"] = timepoint
        return result

    result = {
        "mag_id": mag_id,
        "contig": contig,
        "gene_id": gene_id,
        "position": position,
        "num_pairs": len(pivoted_filtered),
        "p_value_CMH": cmh_result,
        "notes": "",
    }
    if timepoint is not None:
        result["time"] = timepoint
    return result


def get_informative_strata(
    pivoted_df: pd.DataFrame, group_1_name: str, group_2_name: str
) -> List[str]:
    """
    Identifies nucleotide bases that are informative (i.e., present in at least one of two groups) within a pivoted DataFrame.

    Args:
        pivoted_df (pd.DataFrame): A DataFrame where columns are named as '{group}_{base}' for each group and nucleotide base.
        group_1_name (str): The name of the first group.
        group_2_name (str): The name of the second group.

    Returns:
        List[str]: A list of nucleotide bases that are present (sum > 0) in at least one of the two groups.
    """
    informative: List[str] = []
    for base in NUCLEOTIDES:
        # Build column names for each group
        col1 = f"{group_1_name}_{base}"
        col2 = f"{group_2_name}_{base}"
        # Only keep bases present in both groups
        if col1 in pivoted_df.columns and col2 in pivoted_df.columns:
            if pivoted_df[col1].sum() > 0 or pivoted_df[col2].sum() > 0:
                informative.append(base)
    return informative


def prepare_cmh_input(
    df_group: pd.DataFrame,
    group_1_name: str,
    group_2_name: str,
    min_sample_num: int,
) -> Optional[pd.DataFrame]:
    """
    Prepares input data for the Cochran-Mantel-Haenszel (CMH) test by aggregating nucleotide counts
    for two specified groups and filtering for informative bases and shared replicates.

    The function performs the following steps:
    - Splits the input DataFrame into two groups based on the provided group names.
    - Aggregates nucleotide counts by contig, position, group, and replicate.
    - Identifies replicates shared between both groups and filters out those not present in both.
    - Pivots the data so that each row corresponds to a replicate, and columns represent nucleotide counts per group.
    - Identifies informative nucleotide bases (those with nonzero counts in at least one group).
    - Filters out replicates with total nucleotide count less than or equal to 1.
    - Ensures that the number of replicates meets the minimum sample requirement.

    Parameters
    ----------
    df_group : pd.DataFrame
        Input DataFrame containing columns: 'contig', 'position', 'group', 'replicate', and nucleotide counts.
    group_1_name : str
        Name of the first group to compare.
    group_2_name : str
        Name of the second group to compare.
    min_sample_num : int
        Minimum number of shared replicates required to proceed.

    Returns
    -------
    Optional[pd.DataFrame]
        A DataFrame formatted for CMH test input, containing only informative bases and shared replicates,
        or None if requirements are not met (e.g., insufficient replicates or informative bases).
    """
    # Split group into two based on group name
    group1_df = df_group[df_group["group"] == group_1_name]
    group2_df = df_group[df_group["group"] == group_2_name]

    if group1_df.empty or group2_df.empty:
        return None  # If either group is missing, cannot proceed

    # Combine both groups for aggregation
    combined_df = pd.concat([group1_df, group2_df])

    # Aggregate nucleotide counts by contig, position, group, and replicate
    # Get the sum of each nucleotide for each replicate
    aggregated_df = (
        combined_df.groupby(["contig", "position", "group", "replicate"], dropna=False)[
            NUCLEOTIDES
        ]
        .sum()
        .reset_index()
    )

    # Find unique replicates present in both groups
    reps_by_group = aggregated_df.groupby("group")["replicate"].unique()

    # Get shared replicates
    shared_reps = np.intersect1d(
        reps_by_group[group_1_name], reps_by_group[group_2_name]
    )

    # Skip if not enough replicates
    if len(shared_reps) < min_sample_num:
        return None

    # Keep only replicates present in both groups
    subset = aggregated_df[aggregated_df["replicate"].isin(shared_reps)]

    # Pivot so each row is a replicate, columns are group_nucleotide (e.g., control_A)
    pivoted = (
        subset.pivot(
            index=["contig", "position", "replicate"],
            columns="group",
            values=NUCLEOTIDES,
        )
        .fillna(0)
        .astype(int)
    )
    # Flatten MultiIndex columns: (nucleotide, group) -> 'group_nucleotide'
    pivoted.columns = [f"{group}_{nt}" for nt, group in pivoted.columns]
    pivoted = pivoted.reset_index()

    # Keeps the base if at least one of the groups have a nonzero sum.
    informative_bases = get_informative_strata(pivoted, group_1_name, group_2_name)
    # P-value is NaN when only 2 bases are present and one of them is 0
    if len(informative_bases) < 2:
        return None  # Need at least two informative bases for CMH

    # Build list of columns to keep: contig, position, replicate, and informative group_nucleotide columns
    keep_cols = ["contig", "position", "replicate"]
    for base in informative_bases:
        keep_cols += [f"{group_1_name}_{base}", f"{group_2_name}_{base}"]
    pivoted_subset = pivoted[keep_cols].copy()

    nucleotide_cols = [
        col
        for col in pivoted_subset.columns
        if any(col.endswith(f"_{nuc}") for nuc in NUCLEOTIDES)
    ]
    # Sum all nucleotide columns to get the total count for each replicate
    pivoted_subset["total"] = pivoted_subset[nucleotide_cols].sum(axis=1)

    # Filter out replicates with total <= 1, else we get the error:
    # sample size in each stratum must be > 1
    pivoted_filtered = (
        pivoted_subset[pivoted_subset["total"] > 1].copy().drop(columns=["total"])
    )

    # Skip if not enough replicates after filtering
    if len(pivoted_filtered) < min_sample_num:
        return None

    return pivoted_filtered


def run_cmh_test_in_r(pivoted_filtered: pd.DataFrame) -> Optional[float]:
    """
    Runs the Cochran-Mantel-Haenszel (CMH) test in R on a given pivoted and filtered pandas DataFrame.

    This function transfers the provided DataFrame to the R environment, reshapes it for analysis,
    constructs a 3D contingency table, and performs the CMH test using R's `mantelhaen.test`.
    The DataFrame is expected to have columns: 'contig', 'position', 'replicate', and additional columns
    with group and nucleotide information separated by an underscore (e.g., 'fat_A', 'control_A').

    Args:
        pivoted_filtered (pd.DataFrame): A pandas DataFrame containing the data to be analyzed,
            with specific columns as described above.

    Returns:
        Optional[float]: The p-value from the CMH test if successful, otherwise None.
    """
    # Transfer DataFrame to R (new rpy2 conversion API)
    with localconverter(default_converter + pandas2ri.converter):
        ro.globalenv["r_df"] = pivoted_filtered
    # R code is generic: group names are parsed from column names
    r(
        f"""
        library(tidyr)
        # Convert wide to long format for R
        melted <- pivot_longer(
            r_df,
            cols = -c(contig, position, replicate),
            names_to = c("group", "nucleotide"),
            names_sep = "_",
            values_to = "count"
        )
        melted$count <- as.integer(melted$count)
        # Build 3D contingency table
        # Last dimension needs to be the 'strata' ie. replicate. Firt 2 columns are interchangeable
        tab3d <- xtabs(count ~ group + nucleotide + replicate, data = melted)
        # Run CMH test
        cmh_result <- mantelhaen.test(tab3d, alternative = c("two.sided"), correct=FALSE)$p.value
        """
    )
    # cmh_result is an R vector of length 1
    cmh_result = ro.globalenv["cmh_result"]
    return cmh_result[0]


def run_cmh_tests(
    df: pd.DataFrame,
    mag_id: str,
    cpus: int,
    min_sample_num: int,
    timepoint: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Run CMH test workflow on the given DataFrame.

    Args:
        df: DataFrame containing the data to process
        mag_id: MAG identifier
        cpus: Number of CPUs to use
        min_sample_num: Minimum number of samples required
        timepoint: Timepoint identifier for longitudinal data

    Returns:
        Optional[pd.DataFrame]: DataFrame with results if successful, None otherwise
    """
    groups = df["group"].unique()
    if len(groups) != 2:
        raise ValueError(
            f"Expected exactly 2 groups, but found {len(groups)} groups: {groups}"
        )
    group_1, group_2 = groups
    logger.info(f"Grouping data by contig, gene_id and position")
    grouped = df.groupby(["contig", "gene_id", "position"], dropna=False)
    count = len(grouped)
    logger.info(
        f"Analyzing {count:,} positions for CMH tests between {group_1} and {group_2}{f' (timepoint: {timepoint})' if timepoint else ''} using {cpus} cores"
    )
    with Pool(processes=cpus) as pool:
        worker = partial(
            process_group,
            min_sample_num=min_sample_num,
            group_1_name=group_1,
            group_2_name=group_2,
            mag_id=mag_id,
            timepoint=timepoint,
        )
        results = [
            res
            for res in tqdm(
                pool.imap_unordered(worker, grouped),
                total=count,
                desc=f"Running CMH test{(f' {timepoint}' if timepoint else '')}",
            )
            if res is not None
        ]
    if not results:
        logger.warning(
            f"No results for {mag_id} {f' (timepoint: {timepoint})' if timepoint else ''}."
        )
        return None

    df_out = pd.DataFrame(results)
    return df_out


def run_cmh_tests_across_time(
    df: pd.DataFrame,
    mag_id: str,
    cpus: int,
    min_sample_num: int,
    group_to_analyze: str,
) -> Optional[pd.DataFrame]:
    """
    Run CMH test workflow comparing the same group across different timepoints.

    Args:
        df: DataFrame containing the data for both timepoints
        mag_id: MAG identifier
        cpus: Number of CPUs to use
        min_sample_num: Minimum number of samples required
        group_to_analyze: The group name to analyze across timepoints

    Returns:
        Optional[pd.DataFrame]: DataFrame with results if successful, None otherwise
    """
    # Filter data for the specified group only
    group_df = df[df["group"] == group_to_analyze].copy()

    # Check we have exactly two timepoints
    timepoints = group_df["time"].unique()
    if len(timepoints) != 2:
        raise ValueError(
            f"Expected exactly 2 timepoints for group '{group_to_analyze}', but found {len(timepoints)}: {timepoints}"
        )

    # Add a pseudo-group column using the timepoint as the group
    # This allows us to reuse the existing CMH test framework which compares groups
    group_df["original_group"] = group_df["group"]  # Store original group
    group_df["group"] = group_df["time"]  # Use timepoint as the group for comparison

    logger.info(f"Grouping data by contig, gene_id and position")
    grouped = group_df.groupby(["contig", "gene_id", "position"], dropna=False)
    count = len(grouped)
    logger.info(
        f"Analyzing {count:,} positions for CMH tests for group '{group_to_analyze}' across timepoints {timepoints[0]} and {timepoints[1]} using {cpus} cores"
    )

    with Pool(processes=cpus) as pool:
        worker = partial(
            process_group,
            min_sample_num=min_sample_num,
            group_1_name=timepoints[0],  # First timepoint as first "group"
            group_2_name=timepoints[1],  # Second timepoint as second "group"
            mag_id=mag_id,
            timepoint=None,  # No single timepoint as we're comparing across time
        )
        results = [
            res
            for res in tqdm(
                pool.imap_unordered(worker, grouped),
                total=count,
                desc=f"Running CMH test for group '{group_to_analyze}' across timepoints",
            )
            if res is not None
        ]

    if not results:
        logger.warning(
            f"No results for {mag_id} group '{group_to_analyze}' across timepoints."
        )
        return None

    df_out = pd.DataFrame(results)

    # Add a column to indicate which group was analyzed across time
    df_out["analyzed_group"] = group_to_analyze

    return df_out


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run Cochran-Mantel-Haenszel test for a MAG across different samples using R's mantelhaen.test via rpy2. Uses raw counts and filters by preprocessed positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_df",
        required=True,
        nargs="+",
        help=(
            "Path(s) to input allele frequency dataframe(s). "
            "Accepts one or more files; if multiple are given they are concatenated. "
            "Files ending in .parquet are read with pd.read_parquet, otherwise as TSV. "
            "For single timepoint data, preprocessed_df from preprocess_two_sample.py can be used for --input_df."
        ),
        type=str,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional list of group names to keep. When the input cache contains samples "
            "from groups not in the current analysis (e.g. multiple group_combinations sharing "
            "a timepoint cache), restrict to these groups before testing."
        ),
    )
    parser.add_argument(
        "--preprocessed_df",
        help="Path to the filtered dataframe from preprocess_two_sample.py, used to filter positions. Only required for longitudinal or across_time data. For single timepoint data, preprocessed_df can be used for --input_df.",
        type=str,
    )
    parser.add_argument(
        "--min_sample_num",
        type=int,
        default=4,
        help="Minimum number of strata (replicates) required",
    )
    parser.add_argument(
        "--mag_id",
        help="MAG ID to process",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_type",
        help="Type of data to analyze: 'longitudinal' (compare groups within each timepoint), 'single' (compare two groups in a single timepoint), or 'across_time' (compare the same group across timepoints)",
        type=str,
        choices=["longitudinal", "single", "across_time"],
        default="longitudinal",
    )
    parser.add_argument(
        "--group",
        help="For 'across_time' data_type only: name of the group to analyze across timepoints",
        type=str,
    )
    parser.add_argument(
        "--permuted_metadata",
        type=str,
        default=None,
        help=(
            "Optional path to a permuted metadata TSV (null/control run).  When "
            "given, group labels are re-derived from this sheet (joined on "
            "sample_id) right after the cache is loaded, before group filtering "
            "and testing.  Omit for a normal run."
        ),
    )
    parser.add_argument(
        "--cpus",
        help="Number of processors to use.",
        default=cpu_count(),
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    # Define data types for columns
    dtype_map = {
        "contig": str,
        "gene_id": str,
        "position": int,
        "group": "category",
        "replicate": "category",
        **{nuc: "int32" for nuc in NUCLEOTIDES},
    }
    # Permuted run: ensure the per-sample join key is read so the cache can be
    # relabeled (the usecols-based loaders below would otherwise drop it).  Every
    # CMH input is the raw allele-freq cache (single/longitudinal/across_time all
    # read ``_cache_paths_for_combination``), which always carries sample_id, so
    # this is unconditional — unlike the LMM loader, whose across_time input is
    # the wide allele_analysis parquet that lacks sample_id and must be probed.
    if args.permuted_metadata:
        dtype_map["sample_id"] = str

    # Load and possibly filter data
    if args.data_type == "single":
        logger.info(
            "Datatype set to single. Loading input dataframe without any filtering."
        )
        df = load_allele_freq_inputs(
            args.input_df,
            usecols=list(dtype_map.keys()),
            dtype=dtype_map,
        )
        if args.permuted_metadata:
            df = relabel_groups_from_metadata(df, args.permuted_metadata)
            if "sample_id" in df.columns:
                df = df.drop(columns=["sample_id"])
        if args.groups:
            df = df[df["group"].isin(args.groups)].copy()

        tp_results = run_cmh_tests(
            df=df,
            mag_id=args.mag_id,
            cpus=args.cpus,
            min_sample_num=args.min_sample_num,
        )
        all_results = [tp_results] if tp_results is not None else []
    elif args.data_type == "across_time":
        # Comparing same group across timepoints
        if not args.group:
            raise ValueError(
                "--group parameter is required when data_type is 'across_time'"
            )

        # Add time field to dtype map
        dtype_map["time"] = str

        if args.preprocessed_df:
            logger.info(
                "Datatype set to across_time. Loading input dataframe with filtering."
            )
            df = load_and_filter_data(
                args.input_df,
                args.preprocessed_df,
                args.mag_id,
                dtype_map,
                group_to_analyze=args.group,
            )
        else:
            logger.info(
                "Datatype set to across_time. No preprocessed dataframe provided. Using input_df directly."
            )
            df = load_allele_freq_inputs(
                args.input_df,
                usecols=list(dtype_map.keys()),
                dtype=dtype_map,
            )
        if args.permuted_metadata:
            df = relabel_groups_from_metadata(df, args.permuted_metadata)
            if "sample_id" in df.columns:
                df = df.drop(columns=["sample_id"])
        if args.groups:
            df = df[df["group"].isin(args.groups)].copy()

        # Process the group across timepoints
        group_results = run_cmh_tests_across_time(
            df=df,
            mag_id=args.mag_id,
            cpus=args.cpus,
            min_sample_num=args.min_sample_num,
            group_to_analyze=args.group,
        )

        all_results = [group_results] if group_results is not None else []
    else:
        # Longitudinal
        dtype_map["time"] = str
        if args.preprocessed_df:
            logger.info(
                "Datatype set to longitudinal. Loading input dataframe with filtering."
            )
            df = load_and_filter_data(
                args.input_df, args.preprocessed_df, args.mag_id, dtype_map
            )
        else:
            logger.info(
                "Datatype set to longitudinal. No preprocessed dataframe provided. Using input_df directly."
            )
            df = load_allele_freq_inputs(
                args.input_df,
                usecols=list(dtype_map.keys()),
                dtype=dtype_map,
            )
        if args.permuted_metadata:
            df = relabel_groups_from_metadata(df, args.permuted_metadata)
            if "sample_id" in df.columns:
                df = df.drop(columns=["sample_id"])
        if args.groups:
            df = df[df["group"].isin(args.groups)].copy()

        timepoints = df["time"].unique()
        if len(timepoints) != 2:
            raise ValueError(
                f"Expected exactly 2 unique timepoints, but found {len(timepoints)}: {timepoints}"
            )

        # Process each timepoint and collect results
        all_results = []
        for tp in timepoints:
            logger.info(f"Processing timepoint: {tp}")
            df_tp = df[df["time"] == tp].copy()

            # For individual timepoint files
            tp_results = run_cmh_tests(
                df=df_tp,
                mag_id=args.mag_id,
                cpus=args.cpus,
                min_sample_num=args.min_sample_num,
                timepoint=tp,
            )

            if tp_results is not None:
                all_results.append(tp_results)

    # Combine results from all timepoints into a single dataframe
    if all_results:
        os.makedirs(args.output_dir, exist_ok=True)
        combined_results = pd.concat(all_results, ignore_index=True)
        if args.data_type == "across_time":
            combined_out_file = os.path.join(
                args.output_dir, f"{args.mag_id}_cmh_across_time_{args.group}.tsv.gz"
            )
        else:
            combined_out_file = os.path.join(
                args.output_dir, f"{args.mag_id}_cmh.tsv.gz"
            )

        logger.info(f"Saving combined CMH results to {combined_out_file}")
        combined_results.to_csv(
            combined_out_file, sep="\t", index=False, compression="gzip"
        )
        logger.info(f"Saved combined CMH test results to {combined_out_file}")
    else:
        logger.warning(f"No results from any timepoint for {args.mag_id}.")


if __name__ == "__main__":
    main()
