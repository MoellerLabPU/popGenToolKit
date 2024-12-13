import argparse
import gc
import logging
import os
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
    required_columns = {"sample_id", "file_path", "subjectID", "group"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in mag metadata file: {missing_columns}")

    # Convert columns to string and fill missing values with 'Unknown'
    df = df.astype({"sample_id": str, "file_path": str, "subjectID": str, "group": str})
    df = df.fillna({"subjectID": "Unknown", "group": "Unknown"})

    # Build metadata_dict
    metadata_dict = df.set_index("sample_id")[["group", "subjectID"]].to_dict(
        orient="index"
    )

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


def calculate_frequencies(df):
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
    logging.info("Calculating nucleotide frequencies.")
    # Calculate frequencies directly
    df["A_frequency"] = df["A"] / total_coverage
    df["C_frequency"] = df["C"] / total_coverage
    df["T_frequency"] = df["T"] / total_coverage
    df["G_frequency"] = df["G"] / total_coverage

    return df


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
    metadata_info = metadata_dict.get(
        sample_id, {"group": "Unknown", "subjectID": "Unknown"}
    )
    df["group"] = metadata_info["group"]
    df["subjectID"] = metadata_info["subjectID"]
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


def run_wilcoxon_test(args, group_1, group_2, min_sample_num):
    """
    Perform the Wilcoxon signed-rank test on paired samples from two groups.

    Parameters:
    args (tuple): A tuple containing the name and the grouped dataframe.
    group_1 (str): The name of the first group.
    group_2 (str): The name of the second group.
    min_sample_num (int): The minimum number of paired samples required to perform the test.

    Returns:
    tuple: A tuple containing:
        - name_tuple (tuple): The name tuple from the input args.
        - p_values (dict): A dictionary with nucleotides as keys and their corresponding p-values as values.
        - num_samples (int): The number of paired samples used in the test.
        - num_samples (int): The number of paired samples used in the test (duplicate).
        - notes (str): Notes regarding the test, including any special conditions encountered.
    """
    name_tuple, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Initialize notes
    notes = ""

    # subjectID is the main column to merge on here, as that is what's same in the before and after samples
    merge_cols = ["subjectID", "contig", "position", "gene"]
    merged = pd.merge(
        group1, group2, on=merge_cols, suffixes=("_group1", "_group2"), how="inner"
    )

    # Check if we have enough paired samples
    num_samples = merged.shape[0]
    if num_samples >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            x = merged[f"{nucleotide}_group1"]
            y = merged[f"{nucleotide}_group2"]
            # Compute differences
            d = x - y
            # Check if all differences are zero
            # https://stackoverflow.com/a/65227113/12671809 statitics doesn't make sense if all differences are zero
            # https://stackoverflow.com/a/18402696/12671809
            if np.all(d == 0):
                # All differences are zero, set p-value to 1.0
                p_values[f"{nucleotide}_p_value"] = 1.0
                notes += f"{nucleotide}: all differences zero, p-value is set to 1; "
            else:
                res = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
                )
                p_values[f"{nucleotide}_p_value"] = res.pvalue

    return (name_tuple, p_values, num_samples, num_samples, notes)


def run_mannwhitneyu_test(args, group_1, group_2, min_sample_num):
    """
    Perform the Mann-Whitney U test on allele frequencies between two groups.

    Parameters:
    args (tuple): A tuple containing the name and the grouped dataframe.
    group_1 (str): The name of the first group.
    group_2 (str): The name of the second group.
    min_sample_num (int): The minimum number of samples required in each group to perform the test.

    Returns:
    tuple: A tuple containing:
        - name_tuple (tuple): The name tuple from args.
        - p_values (dict): A dictionary with nucleotides as keys and their corresponding p-values.
        - num_samples_group1 (int): The number of samples in the first group.
        - num_samples_group2 (int): The number of samples in the second group.
        - notes (str): Additional notes (currently empty).
    """
    name_tuple, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Initialize notes
    notes = ""

    # Counts of samples in each group
    num_samples_group1 = group1.shape[0]
    num_samples_group2 = group2.shape[0]

    # Only perform the t-test if both groups have at least 2 data points
    if num_samples_group1 >= min_sample_num and num_samples_group2 >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            res = stats.mannwhitneyu(
                group1[nucleotide],
                group2[nucleotide],
                alternative="two-sided",
                nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
            )

            p_values[f"{nucleotide}_p_value"] = res.pvalue

    return (name_tuple, p_values, num_samples_group1, num_samples_group2, notes)


def perform_tests_parallel(df, cpus, group_1, group_2, min_sample_num=6, paired=False):
    """
    Perform statistical tests (Mann-Whitney U test or Wilcoxon signed-rank test) in parallel on grouped data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data to be tested.
    cpus (int): Number of CPU cores to use for parallel processing.
    group_1 (str): Name of the first group for comparison.
    group_2 (str): Name of the second group for comparison.
    min_sample_num (int, optional): Minimum number of samples required for the test. Default is 6.
    paired (bool, optional): If True, perform Wilcoxon signed-rank test for paired samples. If False, perform Mann-Whitney U test for independent samples. Default is False.

    Returns:
    pd.DataFrame: DataFrame containing the test results with p-values and other relevant information.
    """
    start_time = time.time()

    # Group the data
    logging.info("Grouping data by scaffold and position")
    grouped = df.groupby(["contig", "position", "gene_id"])
    num_test = len(grouped)

    # Mann-Whitney U Test (or Wilcoxon rank-sum test) is used for independent samples
    # Wilcoxon signed-rank test is used for comparing two paired samples
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#mannwhitneyu
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#wilcoxon

    if paired:
        logging.info(
            f"Performing {num_test:,} Wilcoxon signed-rank tests for dependent samples in parallel using {cpus} cores."
        )
        my_func = partial(
            run_wilcoxon_test,
            group_1=group_1,
            group_2=group_2,
            min_sample_num=min_sample_num,
        )
    else:
        logging.info(
            f"Performing {num_test:,} Mann-Whitney U tests for independent samples in parallel using {cpus} cores."
        )
        my_func = partial(
            run_mannwhitneyu_test,
            group_1=group_1,
            group_2=group_2,
            min_sample_num=min_sample_num,
        )

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(my_func, grouped)
        records = []
        for result in tqdm(
            results_iter, desc="Performing significance tests", total=num_test
        ):
            name_tuple, p_values, num_samples_group1, num_samples_group2, notes = result
            contig, position, gene_id = name_tuple
            record = {
                "contig": contig,
                "position": position,
                "gene_id": gene_id,
                **p_values,
                f"num_samples_{group_1}": num_samples_group1,
                f"num_samples_{group_2}": num_samples_group2,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logging.info(
        f"Groups added to a list and tests performed in {end_time - start_time:.2f} seconds"
    )
    test_results = pd.DataFrame(records)
    p_value_columns = [nucleotide + "_p_value" for nucleotide in NUCLEOTIDES]
    test_results["min_p_value"] = test_results[p_value_columns].min(axis=1)
    return apply_bh_correction(test_results)


def apply_bh_correction(test_results):
    """
    Apply Benjamini-Hochberg correction to p-values in the given test results DataFrame.

    This function takes a DataFrame containing p-values for allele frequencies (A, T, G, C)
    and applies the Benjamini-Hochberg correction to control the false discovery rate (FDR).
    The corrected p-values are added to the DataFrame as new columns with the suffix '_adj'.

    Parameters:
    test_results (pd.DataFrame): A DataFrame containing p-values for allele frequencies.
                                 The DataFrame must have the following columns:
                                 - "A_frequency_p_value"
                                 - "T_frequency_p_value"
                                 - "G_frequency_p_value"
                                 - "C_frequency_p_value"

    Returns:
    pd.DataFrame: A new DataFrame with the original p-values and the adjusted p-values.
                  The adjusted p-values are added as new columns with the suffix '_adj'.

    Raises:
    ValueError: If NaN values are found in the p-value columns after cleaning.
    """

    # Only keep rows where no p-value column is NaN
    p_value_columns = [nucleotide + "_p_value" for nucleotide in NUCLEOTIDES]
    test_results_cleaned = test_results.dropna(subset=p_value_columns, how="any").copy()

    # Ensure there are no NaN values in p-value columns
    if test_results_cleaned[p_value_columns].isnull().values.any():
        logging.error("NaNs found in p-value columns after cleaning.")
        raise ValueError("NaNs found in p-value columns after cleaning.")

    if test_results_cleaned.empty:
        logging.warning("No p-values to correct.")
        # raise ValueError("No p-values to correct.")
        return test_results_cleaned

    for col in p_value_columns:
        # Apply Benjamini-Hochberg correction
        logging.info(f"Applying Benjamini-Hochberg correction to {col}.")
        p_values = test_results_cleaned[col]
        _, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

        # Create a Series with adjusted p-values
        adjusted_p_values = pd.Series(p_adjusted, index=p_values.index)

        # Assign adjusted p-values back to the DataFrame
        test_results_cleaned.loc[adjusted_p_values.index, f"{col}_adj"] = (
            adjusted_p_values
        )

    p_value_columns_adj = [column + "_adj" for column in p_value_columns]
    test_results_cleaned["min_p_value_adj"] = test_results_cleaned[
        p_value_columns_adj
    ].min(axis=1)
    return test_results_cleaned


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Perform Mann-Whitney U test with BH correction for nucleotide frequencies at each site.",
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
        "--group_1",
        help="First group to compare",
        type=str,
        required=True,
        metavar="str",
    )

    parser.add_argument(
        "--group_2",
        help="Second group to compare",
        type=str,
        required=True,
        metavar="str",
    )

    parser.add_argument(
        "--min_sample_num",
        help="Minimum number of samples per group to perform the significance test",
        type=int,
        default=6,
        metavar="int",
    )

    parser.add_argument(
        "--paired",
        help="Samples are paired or dependent (e.g., before and after treatment)",
        action="store_true",
        default=False,
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

    # Extract mag_id from the mag_metadata_file name
    # mag_id = os.path.basename(args.mag_metadata_file).replace("_samples.tsv", "")
    mag_id = args.magID
    # Load per-MAG metadata file and get required data structures
    metadata_dict, sample_files_with_mag_id = load_mag_metadata_file(
        args.mag_metadata_file, mag_id, args.breath_threshold
    )

    groups_in_metadata = set(entry["group"] for entry in metadata_dict.values())

    # Check if args.group_1 and args.group_2 are in the metadata groups
    if args.group_1 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_1} not found in metadata file.")

    if args.group_2 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_2} not found in metadata file.")

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
        logging.info(
            f"No samples for MAG {mag_id} passed the breadth threshold. Writing empty files and exiting..... :("
        )
        os.makedirs(args.output_dir, exist_ok=True)
        nuc_fPath = os.path.join(
            args.output_dir, f"{mag_id}_nucleotide_frequencies.tsv"
        )
        test_fPath = os.path.join(args.output_dir, f"{mag_id}_test_results.tsv")
        Path(nuc_fPath).touch()
        Path(test_fPath).touch()
        return  # Exit the program

    # Concatenate data for this MAG
    mag_df = pd.concat(data_list, ignore_index=True)

    # Release memory from data_list
    del data_list
    gc.collect()

    # Calculate frequencies and perform statistical tests
    mag_df = calculate_frequencies(mag_df)

    test_results = perform_tests_parallel(
        mag_df,
        args.cpus,
        args.group_1,
        args.group_2,
        args.min_sample_num,
        args.paired,
    )

    # This adds MAG_ID as the first column
    test_results.insert(0, "MAG_ID", mag_id)

    # Save test results for this MAG
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Saving results for MAG {mag_id} to {args.output_dir}")

    mag_df_sorted = mag_df.sort_values(
        by=["contig", "position"], ascending=[True, True]
    )
    mag_df_sorted.to_csv(
        os.path.join(args.output_dir, f"{mag_id}_nucleotide_frequencies.tsv"),
        index=False,
        sep="\t",
    )
    test_results_sorted = test_results.sort_values(
        by=["contig", "position"], ascending=[True, True]
    )
    test_results_sorted.to_csv(
        os.path.join(args.output_dir, f"{mag_id}_test_results.tsv"),
        index=False,
        sep="\t",
    )

    # Release memory from mag_df and test_results if necessary
    del mag_df, test_results
    gc.collect()
    logging.info(f"Processing of MAG {mag_id} completed.")

    end_time = time.time()
    logging.info(f"Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
