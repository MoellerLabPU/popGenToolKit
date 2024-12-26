import argparse
import gc
import logging
import os
import sys
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]


def calculate_mag_sizes(fasta_file):
    """
    Calculate the sizes of Metagenome-Assembled Genomes (MAGs) from a FASTA file.

    This function parses a given FASTA file and calculates the total size of each MAG
    by summing the lengths of its contigs. The MAG ID is extracted from the contig ID,
    which is assumed to be the part of the contig ID before '.fa'.

    Parameters:
        fasta_file (str): Path to the input FASTA file containing contig sequences.

    Returns:
        dict: A dictionary where keys are MAG IDs and values are the total sizes of the MAGs.
    """
    logging.info("Parsing FASTA file to calculate MAG size.")
    mag_sizes = defaultdict(int)
    for record in SeqIO.parse(fasta_file, "fasta"):
        contig_id = record.id
        # Extract MAG ID from contig ID
        # The MAG ID is everything before '.fa' in the contig ID
        mag_id = contig_id.split(".fa")[0]
        # Accumulate the length of the contig to the MAG's total size
        mag_sizes[mag_id] += len(record.seq)
    return mag_sizes


def load_mag_metadata_file(mag_metadata_file, mag_id, breath_threshold):
    """
    Load MAG metadata from a file and process it.

    Parameters:
        mag_metadata_file (str): Path to the MAG metadata file (tab-separated values).
        mag_id (str): The MAG identifier to be associated with each sample.
        breath_threshold (float): The breath threshold value to be associated with each sample.

    Returns:
        tuple: A tuple containing:
            - metadata_dict (dict): A dictionary where keys are sample IDs and values are dictionaries with keys 'group' and 'subjectID'.
            - sample_files_with_mag_id (list): A list of tuples, each containing (sample_id, file_path, mag_id, breath_threshold).

    Raises:
        ValueError: If the MAG metadata file is missing required columns.
    """

    logging.info(f"Loading MAG metadata from file: {mag_metadata_file}")
    df = pd.read_csv(mag_metadata_file, sep="\t")

    # Ensure required columns are present
    required_columns = {
        "sample_id",
        "file_path",
        "subjectID",
        "group",
        "time",
        "replicate",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in mag metadata file: {missing_columns}")

    # Convert columns to string
    df = df.astype(
        {
            "sample_id": str,
            "file_path": str,
            "subjectID": str,
            "group": str,
            "time": str,
            "replicate": str,
        }
    )

    # Build metadata_dict
    metadata_dict = df.set_index("sample_id")[
        ["group", "time", "subjectID", "replicate"]
    ].to_dict(orient="index")

    # Build sample_files_with_mag_id: list of (sample_id, file_path, mag_id, breath_threshold)
    sample_files_with_mag_id = (
        df[["sample_id", "file_path"]]
        .apply(
            lambda row: (row["sample_id"], row["file_path"], mag_id, breath_threshold),
            axis=1,
        )
        .tolist()
    )

    return metadata_dict, sample_files_with_mag_id


def process_mag_files(args):
    """
    Processes a MAG (Metagenome-Assembled Genome) file and adds metadata information.

    Parameters:
        args (tuple): A tuple containing the following elements:
            sample_id (str): The sample identifier.
            filepath (str): The path to the MAG file.
            mag_id (str): The MAG identifier.
            breath_threshold (float): The threshold for breadth coverage.

    Returns:
        pd.DataFrame or None: A DataFrame with added metadata and calculated breadth if the breadth is above the threshold,
                              otherwise None if the breadth is below the threshold or if the MAG size is not found.

    Notes:
        - The function reads the MAG file from the given filepath.
        - Adds sample_id, group, and subjectID columns to the DataFrame.
        - Inserts the MAG_ID as the first column.
        - Calculates the breadth of coverage for the MAG.
        - If the breadth is below the given threshold, the function logs a message and returns None.
        - If the breadth is above the threshold, the function adds the breadth and genome size to the DataFrame and returns it.
    """
    sample_id, filepath, mag_id, breath_threshold = args
    df = pd.read_csv(filepath, sep="\t")
    # Add sample_id column
    df["sample_id"] = sample_id
    # Add metadata columns
    metadata_info = metadata_dict.get(sample_id)
    df["group"] = metadata_info["group"]
    df["subjectID"] = metadata_info["subjectID"]
    df["time"] = metadata_info["time"]
    df["replicate"] = metadata_info["replicate"]
    # This adds MAG_ID as the first column
    df.insert(0, "MAG_ID", mag_id)

    # Get MAG size (total number of positions in the MAG)
    mag_size = mag_size_dict.get(mag_id)
    if mag_size is None:
        logging.warning(f"Size for MAG {mag_id} not found in sample {sample_id}.")
        return None  # Skip this sample-MAG combination

    # Calculate the number of positions with total_coverage >= 1
    positions_with_coverage = df[df["total_coverage"] >= 1].shape[0]

    # Calculate breadth
    breadth = positions_with_coverage / mag_size

    if breadth < breath_threshold:
        logging.info(
            f"MAG {mag_id} in sample {sample_id} has breadth {breadth:.2%}, which is less than {breath_threshold:.2%}. Skipping this sample-MAG combination."
        )
        return None  # Skip this sample-MAG combination
    else:
        df["breadth"] = breadth
        df["genome_size"] = mag_size
        df = calculate_frequencies(df, mag_id)
        return df


def init_worker(metadata, mag_sizes):
    """
    Initialize worker process with metadata and MAG sizes.

    This function sets up global dictionaries for metadata and MAG sizes
    that can be accessed by worker processes.

    Parameters:
        metadata (dict): A dictionary containing metadata information.
        mag_sizes (dict): A dictionary containing sizes of MAGs (Metagenome-Assembled Genomes).

    Returns:
        None
    """
    global metadata_dict
    global mag_size_dict
    metadata_dict = metadata
    mag_size_dict = mag_sizes


def calculate_frequencies(df, mag_id):
    """
    Calculate nucleotide frequencies for a given DataFrame.

    This function takes a DataFrame containing nucleotide counts and total coverage,
    and calculates the frequency of each nucleotide (A, C, T, G) by dividing the count
    of each nucleotide by the total coverage. The resulting frequencies are added as new
    columns to the DataFrame.

    Parameters:
        df (pandas.DataFrame): A DataFrame with columns "A", "C", "T", "G", and "total_coverage".

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for nucleotide frequencies:
                          "A_frequency", "C_frequency", "T_frequency", and "G_frequency".
    """
    total_coverage = df["total_coverage"]
    # logging.info(f"Calculating nucleotide frequencies for MAG {mag_id}.")
    # Calculate frequencies directly
    df["A_frequency"] = df["A"] / total_coverage
    df["T_frequency"] = df["T"] / total_coverage
    df["G_frequency"] = df["G"] / total_coverage
    df["C_frequency"] = df["C"] / total_coverage

    return df


def save_nucleotide_frequencies(data_dict, output_dir, mag_id):
    mag_df = pd.concat(
        [df for subject_dict in data_dict.values() for df in subject_dict.values()],
        ignore_index=True,
    )

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving nucleotide frequencies for MAG {mag_id} to {output_dir}")
    mag_df.to_csv(
        os.path.join(output_dir, f"{mag_id}_nucleotide_frequencies.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )


def create_data_dict(data_list):
    """
    Organizes a list of DataFrames into a nested dictionary structure based on subjectID and timepoint.

    Parameters:
        data_list (list of pandas.DataFrame): A list of DataFrames, each containing columns 'subjectID', 'time', and 'sample_id'.

    Returns:
        dict: A nested dictionary where the first level keys are subjectIDs and the second level keys are timepoints.
              The values are the corresponding DataFrames.

    Raises:
        ValueError: If a DataFrame contains multiple unique subjectIDs or timepoints.
    """
    # Organize DataFrames by subjectID
    data_dict = {}
    for df in data_list:
        # Validate and extract subjectID
        subject_ids = df["subjectID"].unique()
        if len(subject_ids) != 1:
            raise ValueError(
                f"Multiple subjectIDs found in DataFrame for sample {df['sample_id'].iloc[0]}"
            )
        subjectID = subject_ids[0]
        # Validate and extract group
        timepoints = df["time"].unique()
        if len(timepoints) != 1:
            raise ValueError(
                f"Multiple timepoints found in DataFrame for sample {df['sample_id'].iloc[0]}"
            )
        timepoint = timepoints[0]

        # Store the DataFrame for each subjectID and timepoint
        if subjectID not in data_dict:
            data_dict[subjectID] = {}
        data_dict[subjectID][timepoint] = df
    return data_dict


def calculate_allele_frequency_changes(data_dict, output_dir, mag_id):
    logging.info("Identifying unique timepoints.")

    unique_timepoints = set()
    for subject_data in data_dict.values():
        unique_timepoints.update(subject_data.keys())
    if len(unique_timepoints) != 2:
        raise ValueError(
            f"Expected exactly 2 unique timepoints, found {len(unique_timepoints)}."
        )

    # Unpack the two timepoints
    timepoint_1, timepoint_2 = unique_timepoints

    logging.info(
        f"Calculating change in allele frequency between {timepoint_1} and {timepoint_2} for each position between the same subjectID."
    )
    # Get sets of subjectIDs in each timepoint
    subjectIDs_timepoint1 = {
        subjectID for subjectID in data_dict if timepoint_1 in data_dict[subjectID]
    }
    subjectIDs_timepoint2 = {
        subjectID for subjectID in data_dict if timepoint_2 in data_dict[subjectID]
    }

    # Find subjectIDs present only in one timepoint
    subjectIDs_only_in_timepoint1 = subjectIDs_timepoint1 - subjectIDs_timepoint2
    subjectIDs_only_in_timepoint2 = subjectIDs_timepoint2 - subjectIDs_timepoint1

    # Log warnings for subjectIDs present only in one timepoint
    if subjectIDs_only_in_timepoint1:
        logging.warning(
            f"The following subjectIDs are present only in timepoint '{timepoint_1}' and not in timepoint '{timepoint_2}': {subjectIDs_only_in_timepoint1}"
        )

    if subjectIDs_only_in_timepoint2:
        logging.warning(
            f"The following subjectIDs are present only in timepoint '{timepoint_2}' and not in timepoint '{timepoint_1}': {subjectIDs_only_in_timepoint2}"
        )

    results = []

    # Iterate over subjectIDs present in both timepoints
    common_subjectIDs = [
        subjectID
        for subjectID in data_dict
        if timepoint_1 in data_dict[subjectID] and timepoint_2 in data_dict[subjectID]
    ]

    if not common_subjectIDs:
        logging.warning(
            f"No common subjectIDs found between the {timepoint_1} and {timepoint_2}."
        )
        return

    for subjectID in common_subjectIDs:
        # Get DataFrames for each timepoint
        df_timepoint1 = data_dict[subjectID][timepoint_1]
        df_timepoint2 = data_dict[subjectID][timepoint_2]

        # Merge on contig and position
        # In pandas when both key columns ('gene_id' here) contain rows where the key is a null value, those rows will be matched against each other
        """
        df1 = pd.DataFrame({'key1': [1, 2, None], 'value1': ['A', 'B', 'C']})
        df2 = pd.DataFrame({'key1': [1, None, 3], 'value1': ['X', 'Y', 'Z']})

        merged_df = pd.merge(df1, df2, on='key1', how='inner')
        print(merged_df)
           key1 value1_x value1_y
        0   1.0        A        X
        1   NaN        C        Y
        """
        merged_df = pd.merge(
            df_timepoint1,
            df_timepoint2,
            on=["subjectID", "contig", "gene_id", "position", "replicate", "group"],
            suffixes=(f"_{timepoint_1}", f"_{timepoint_2}"),
            how="inner",
        )

        if merged_df.empty:
            logging.warning(f"No matching positions found for subjectID {subjectID}.")
            continue

        # Compute differences
        # Since the merge is "inner", only the positions common to both timepoints are present
        for nuc in NUCLEOTIDES:
            merged_df[f"{nuc}_diff"] = (
                merged_df[f"{nuc}_{timepoint_2}"] - merged_df[f"{nuc}_{timepoint_1}"]
            )

        # Select relevant columns
        columns_to_keep = (
            ["subjectID", "gene_id", "contig", "position", "replicate", "group"]
            + [f"{nuc}_{timepoint_1}" for nuc in NUCLEOTIDES]
            + [f"{nuc}_{timepoint_2}" for nuc in NUCLEOTIDES]
            + [f"{nuc}_diff" for nuc in NUCLEOTIDES]
        )

        results.append(merged_df[columns_to_keep])

    if not results:
        logging.warning("No allele frequency changes calculated.")
        return

    allele_changes = pd.concat(results, ignore_index=True)

    logging.info(
        f"saving allele frequency changes saved to {output_dir}/{mag_id}_allele_frequency_changes.tsv"
    )
    # Save the allele frequency changes
    allele_changes.to_csv(
        os.path.join(output_dir, f"{mag_id}_allele_frequency_changes.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip",
    )

    return allele_changes


def get_mean_change(allele_changes, mag_id, output_dir):
    """
    Calculate the mean changes in allele frequencies for subjectIDs present in the same replicate and group.

    Parameters:
        allele_changes (pd.DataFrame): DataFrame containing allele change information with columns
                                       ['contig', 'position', 'replicate', 'group', 'subjectID', 'gene_id' '<nuc>_diff'].
        mag_id (str): Identifier for the metagenome-assembled genome (MAG).
        output_dir (str): Directory where the output file will be saved.

    Returns:
        pd.DataFrame: DataFrame with mean changes in allele frequencies and count of unique subject IDs
                      for each group, with columns ['contig', 'position', 'replicate', 'group', 'gene_id',
                      '<nuc>_diff_mean', 'subjectID_count'].

    Notes:
        - The function groups the input DataFrame by ['contig', 'position', 'replicate', 'group', 'gene_id'] and
          calculates the mean of nucleotide differences and the count of unique subject IDs in each group.
        - The resulting DataFrame is saved as a compressed TSV file in the specified output directory.
    """
    logging.info(
        "Calculating mean changes in allele frequencies for subjectIDs present in the same replicate and group."
    )
    # Prepare the aggregation dictionary
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html
    agg_dict = {f"{nuc}_diff": "mean" for nuc in NUCLEOTIDES}
    agg_dict["subjectID"] = "nunique"  # Count of unique subject IDs in each group

    # Perform the groupby and aggregation
    mean_changes_df = (
        allele_changes.groupby(
            ["contig", "gene_id", "position", "replicate", "group"], dropna=False
        )
        .agg(agg_dict)
        .reset_index()
    )

    # Rename the nucleotide columns to reflect mean changes
    mean_changes_df.rename(
        columns={f"{nuc}_diff": f"{nuc}_diff_mean" for nuc in NUCLEOTIDES}, inplace=True
    )

    # Optionally, rename 'subjectID' to 'subjectID_count' to make it clear it's a count
    mean_changes_df.rename(columns={"subjectID": "subjectID_count"}, inplace=True)

    logging.info(
        f"Saving mean change in allele frequency for MAG {mag_id} to {output_dir}"
    )

    mean_changes_df.to_csv(
        os.path.join(output_dir, f"{mag_id}_allele_frequency_changes_mean.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )
    return mean_changes_df


def perform_tests(mean_changes_df, mag_id, output_dir, cpus, min_sample_num=4):
    # Get unique groups
    groups = mean_changes_df["group"].unique()
    if len(groups) != 2:
        logging.error(
            f"Expected exactly 2 groups for paired tests, but found {len(groups)} groups: {groups}. Exiting...."
        )
        sys.exit(1)

    group_1, group_2 = groups

    # Check that pairing key exists
    if "replicate" not in mean_changes_df.columns:
        logging.error("Column 'replicate' not found in the data.")
        return

    # Group the data
    logging.info("Grouping data by contig, gene_id and position")
    grouped = mean_changes_df.groupby(["contig", "gene_id", "position"], dropna=False)
    # import itertools

    # grouped = list(itertools.islice(grouped, 1000000))

    num_tests = len(grouped)

    logging.info(
        f"Performing {num_tests:,} unpaired tests between {group_1} and {group_2} using {cpus} cores."
    )
    func_unpaired = partial(
        run_unpaired_tests,
        group_1=group_1,
        group_2=group_2,
        min_sample_num=min_sample_num,
    )
    perform_unpaired_tests(
        func_unpaired, grouped, group_1, group_2, cpus, num_tests, output_dir, mag_id
    )

    logging.info(
        f"Performing {num_tests:,} paired tests (paired by replicate) between {group_1} and {group_2} using {cpus} cores."
    )
    func_paired = partial(
        run_paired_tests,
        group_1=group_1,
        group_2=group_2,
        min_sample_num=min_sample_num,
    )
    perform_paired_tests(func_paired, grouped, cpus, num_tests, output_dir, mag_id)

    logging.info(f"Performing tests for {num_tests:,} positions using {cpus} cores.")
    func_one_sample = partial(
        run_one_sample_tests,
        groups=groups,
        min_sample_num=min_sample_num,
    )

    perform_one_sample_tests(
        func_one_sample, grouped, cpus, num_tests, output_dir, mag_id
    )


def perform_unpaired_tests(
    func_unpaired, grouped, group_1, group_2, cpus, num_tests, output_dir, mag_id
):
    start_time = time.time()

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(func_unpaired, grouped)
        records = []
        for result in tqdm(
            results_iter, desc="Performing significance tests", total=num_tests
        ):
            name_tuple, p_values, num_samples_group1, num_samples_group2, notes = result
            contig, gene_id, position = name_tuple
            record = {
                "contig": contig,
                "gene_id": gene_id,
                "position": position,
                **p_values,
                f"num_samples_{group_1}": num_samples_group1,
                f"num_samples_{group_2}": num_samples_group2,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logging.info(
        f"2 sample unpaired tests performed in {end_time - start_time:.2f} seconds"
    )
    test_results = pd.DataFrame(records)
    # Identify p-value columns
    p_value_columns = [col for col in test_results.columns if "_p_value" in col]

    # Remove rows where all p-value columns are NaN
    test_results.dropna(subset=p_value_columns, how="all", inplace=True)

    logging.info(
        f"Saving 2 sample unpaired significance results for MAG {mag_id} to {output_dir}"
    )
    test_results.to_csv(
        os.path.join(output_dir, f"{mag_id}_two_sample_unpaired.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )


def run_unpaired_tests(args, group_1, group_2, min_sample_num):
    name_tuple, group = args
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    p_values = {}
    notes = ""
    for nucleotide in NUCLEOTIDES:
        p_values[f"{nucleotide}_p_value_tTest"] = np.nan
        p_values[f"{nucleotide}_p_value_MannWhitney"] = np.nan

    # Counts of samples in each group
    num_samples_group1 = group1.shape[0]
    num_samples_group2 = group2.shape[0]

    # Only perform the t-test if both groups have at least min_sample_num data points
    if num_samples_group1 >= min_sample_num and num_samples_group2 >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            nuc_col = f"{nucleotide}_diff_mean"
            mean1 = np.mean(group1[nuc_col])
            mean2 = np.mean(group2[nuc_col])
            # ddof of 1 is used as we are calculating sample variance. "0" is used for population variance
            var1 = np.var(group1[nuc_col], ddof=1)
            var2 = np.var(group2[nuc_col], ddof=1)
            # In t-test if both groups have identical values, the p-value is NaN, eg. 5,5,5,5 and 5,5,5,5,5,5,5,5
            # I'm setting it 1 for consistency with Mann Whitney
            if var1 == 0 and var2 == 0 and mean1 == mean2:
                p_values[f"{nucleotide}_p_value_tTest"] = 1
                notes += f"{nucleotide}: identical values in both groups, p-value for t-test is set to 1; "
            else:
                # Perform t-test
                res_para = stats.ttest_ind(
                    group1[nuc_col],
                    group2[nuc_col],
                    equal_var=False,
                    nan_policy="raise",
                    alternative="two-sided",
                )
                p_values[f"{nucleotide}_p_value_tTest"] = res_para.pvalue

            # Perform Mann-Whitney U test
            res_non_para = stats.mannwhitneyu(
                group1[nuc_col],
                group2[nuc_col],
                alternative="two-sided",
                nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
            )
            p_values[f"{nucleotide}_p_value_MannWhitney"] = res_non_para.pvalue

    return (name_tuple, p_values, num_samples_group1, num_samples_group2, notes)


def perform_paired_tests(func_unpaired, grouped, cpus, num_tests, output_dir, mag_id):
    start_time = time.time()

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(func_unpaired, grouped)
        records = []
        for result in tqdm(
            results_iter, desc="Performing paired significance tests", total=num_tests
        ):
            name_tuple, p_values, num_pairs, notes = result
            contig, gene_id, position = name_tuple
            record = {
                "contig": contig,
                "gene_id": gene_id,
                "position": position,
                **p_values,
                "num_pairs": num_pairs,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logging.info(f"Paired tests performed in {end_time - start_time:.2f} seconds")
    test_results = pd.DataFrame(records)
    # Identify p-value columns
    p_value_columns = [col for col in test_results.columns if "_p_value" in col]

    # Remove rows where all p-value columns are NaN
    test_results.dropna(subset=p_value_columns, how="all", inplace=True)

    logging.info(
        f"Saving 2 sample paired significance results for MAG {mag_id} to {output_dir}"
    )
    test_results.to_csv(
        os.path.join(output_dir, f"{mag_id}_two_sample_paired.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )


def run_paired_tests(args, group_1, group_2, min_sample_num):
    name_tuple, group = args
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Merge the two groups on 'replicate_id', 'contig', and 'position'
    merged_data = pd.merge(
        group1,
        group2,
        on=["replicate", "contig", "position"],
        suffixes=("_group1", "_group2"),
        how="inner",
    )

    # Initialize p_values with NaN
    p_values = {}
    notes = ""
    for nucleotide in NUCLEOTIDES:
        p_values[f"{nucleotide}_p_value_paired_tTest"] = np.nan
        p_values[f"{nucleotide}_p_value_Wilcoxon"] = np.nan

    # Number of pairs
    num_pairs = merged_data.shape[0]

    # Only perform the tests if there are at least min_sample_num pairs
    if num_pairs >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            data1 = merged_data[f"{nucleotide}_diff_mean_group1"]
            data2 = merged_data[f"{nucleotide}_diff_mean_group2"]
            # Check for identical values
            d = data1 - data2
            if np.all(d == 0):
                p_values[f"{nucleotide}_p_value_paired_tTest"] = (
                    1.0  # Paired t-test outputs NaN if both groups have identical values
                )
                p_values[f"{nucleotide}_p_value_Wilcoxon"] = (
                    1.0  # Wilcoxon gives an error if both groups have identical values
                )
                notes += f"{nucleotide}: identical values in both groups, p-value for both tests is set to 1; "
            else:
                # Perform paired t-test
                res_ttest = stats.ttest_rel(
                    data1,
                    data2,
                    nan_policy="raise",
                    alternative="two-sided",
                )
                p_values[f"{nucleotide}_p_value_paired_tTest"] = res_ttest.pvalue

                res_wilcoxon = stats.wilcoxon(
                    data1,
                    data2,
                    alternative="two-sided",
                    nan_policy="raise",
                )
                p_values[f"{nucleotide}_p_value_Wilcoxon"] = res_wilcoxon.pvalue

    return (name_tuple, p_values, num_pairs, notes)


def perform_one_sample_tests(
    func_one_sample, grouped, cpus, num_tests, output_dir, mag_id
):
    start_time = time.time()

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(func_one_sample, grouped)
        records = []
        for result in tqdm(
            results_iter, desc="Performing one-sample tests", total=num_tests
        ):
            name_tuple, p_values, num_samples_dict, notes = result
            contig, gene_id, position = name_tuple
            record = {
                "contig": contig,
                "gene_id": gene_id,
                "position": position,
                **p_values,
                **num_samples_dict,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logging.info(f"Tests performed in {end_time - start_time:.2f} seconds")
    test_results = pd.DataFrame(records)
    # Identify p-value columns
    p_value_columns = [col for col in test_results.columns if "_p_value" in col]

    # Remove rows where all p-value columns are NaN
    test_results.dropna(subset=p_value_columns, how="all", inplace=True)
    logging.info(
        f"Saving single-sample significance results for MAG {mag_id} to {output_dir}"
    )
    test_results.to_csv(
        os.path.join(output_dir, f"{mag_id}_single_sample.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )


def run_one_sample_tests(args, groups, min_sample_num):
    name_tuple, group = args
    # Initialize p_values with NaN and notes
    p_values = {}
    notes = ""
    num_samples_dict = {}
    # For each group
    for group_name in groups:
        group_data = group[group["group"] == group_name]
        num_samples_group = group_data.shape[0]
        # Set the number of samples
        key_num_samples = f"num_samples_{group_name}"
        num_samples_dict[key_num_samples] = num_samples_group

        # Initialize p-values for all nucleotides for this group to NaN
        for nucleotide in NUCLEOTIDES:
            p_values[f"{nucleotide}_p_value_tTest_{group_name}"] = np.nan
            p_values[f"{nucleotide}_p_value_Wilcoxon_{group_name}"] = np.nan

        # Only perform the tests if the group has at least min_sample_num data points
        if num_samples_group >= min_sample_num:
            for nucleotide in NUCLEOTIDES:
                nuc_col = f"{nucleotide}_diff_mean"
                data = group_data[nuc_col]
                # Check for zero variance and zero mean ie. 0,0,0,0. T-test is NA, and wilcoxon gives an error is this case. P-value is set at 1.
                var = np.var(data, ddof=1)
                mean = np.mean(data)
                if var == 0 and mean == 0:
                    # Store p-value as 1
                    p_values[f"{nucleotide}_p_value_tTest_{group_name}"] = 1.0
                    p_values[f"{nucleotide}_p_value_Wilcoxon_{group_name}"] = 1.0

                    notes += f"{nucleotide} in {group_name}: identical values, p-value for both tests is set to 1; "
                else:
                    # Perform one-sample tests
                    res_tTest = stats.ttest_1samp(
                        data,
                        0.0,
                        alternative="two-sided",
                        nan_policy="raise",
                    )
                    p_values[f"{nucleotide}_p_value_tTest_{group_name}"] = (
                        res_tTest.pvalue
                    )

                    # Perform Wilcoxon signed-rank test
                    res_wilcoxon = stats.wilcoxon(
                        data, alternative="two-sided", nan_policy="raise"
                    )
                    p_values[f"{nucleotide}_p_value_Wilcoxon_{group_name}"] = (
                        res_wilcoxon.pvalue
                    )

    return (name_tuple, p_values, num_samples_dict, notes)


def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Analyze allele frequency and perform significance tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--magID",
        help="MAG ID to process",
        type=str,
        required=True,
        metavar="str",
    )
    parser.add_argument(
        "--mag_metadata_file",
        help="Path to metadata file",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--fasta",
        help="Path to FASTA file with contigs",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--breath_threshold",
        help="Breath threshold to use for MAGs.",
        type=float,
        default=0.1,
        metavar="float",
    )

    parser.add_argument(
        "--cpus",
        help=f"Number of processors to use.",
        default=cpu_count(),
        metavar="int",
        type=int,
    )

    parser.add_argument(
        "--min_sample_num",
        help="Minimum number of samples per group to perform the significance test",
        type=int,
        default=4,
        metavar="int",
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()
    start_time = time.time()
    # Calculate MAG sizes
    mag_size_dict = calculate_mag_sizes(args.fasta)
    mag_id = args.magID

    # Load per-MAG metadata file and get required data structures
    metadata_dict, sample_files_with_mag_id = load_mag_metadata_file(
        args.mag_metadata_file, mag_id, args.breath_threshold
    )

    # Process the samples
    number_of_processes = min(args.cpus, len(sample_files_with_mag_id))

    logging.info(
        f"Processing {len(sample_files_with_mag_id)} samples for MAG {mag_id} using {number_of_processes} processes."
    )

    # Load sample data in parallel
    with Pool(
        processes=number_of_processes,
        initializer=init_worker,
        initargs=(metadata_dict, mag_size_dict),
    ) as pool:
        data_list = list(
            pool.imap_unordered(process_mag_files, sample_files_with_mag_id)
        )
    # Filter out None values (samples with breadth < 50%)
    data_list = [df for df in data_list if df is not None]

    if not data_list:
        logging.info(f"No samples for MAG {mag_id} passed the breadth threshold.")
        # os.makedirs(args.output_dir, exist_ok=True)
        # nuc_fPath = os.path.join(
        #     args.output_dir, f"{mag_id}_nucleotide_frequencies.tsv"
        # )
        # Path(nuc_fPath).touch()
        return  # Exit the program

    data_dict = create_data_dict(data_list)

    # Release memory from data_list
    del data_list
    gc.collect()

    save_nucleotide_frequencies(data_dict, args.output_dir, mag_id)

    allele_changes = calculate_allele_frequency_changes(
        data_dict, args.output_dir, mag_id
    )

    mean_changes_df = get_mean_change(allele_changes, mag_id, args.output_dir)

    perform_tests(
        mean_changes_df, mag_id, args.output_dir, args.cpus, args.min_sample_num
    )

    end_time = time.time()
    logging.info(f"Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
