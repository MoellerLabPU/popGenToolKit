#!/usr/bin/env python

import argparse
import glob
import logging
import os
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]


def load_metadata(metadata_file):
    """
    Load metadata from a given file and return it as a dictionary.

    Parameters:
        metadata_file (str): The path to the metadata file in tab-separated values (TSV) format.

    Returns:
        dict: A dictionary where the keys are sample identifiers and the values are dictionaries
              containing 'diet' and 'mouse' information for each sample.
    """

    logging.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t")
    # Create a dictionary from the metadata where file_prefix is the key and group is the value
    metadata_dict = metadata.set_index("sample")[["diet", "mouse"]].to_dict(
        orient="index"
    )
    return metadata_dict


def load_snv_files(input_dir, metadata_dict):
    """
    Load inStrain SNV files from a specified directory and return a dictionary of DataFrames.

    Parameters:
        input_dir (str): The directory containing the SNV files with the extension ".IS_SNVs.tsv".
        metadata_dict (dict): A dictionary containing metadata where keys are file prefixes and values are group names.

    Returns:
        dict: A dictionary where keys are file prefixes and values are pandas DataFrames. Each DataFrame contains the following columns:
            - scaffold: The scaffold identifier.
            - position: The position of the SNV.
            - position_coverage: The coverage at the position.
            - A: The count of 'A' nucleotides.
            - C: The count of 'C' nucleotides.
            - T: The count of 'T' nucleotides.
            - G: The count of 'G' nucleotides.
            - MAG_ID: The MAG (Metagenome-Assembled Genome) ID extracted from the scaffold column.
            - group: The group name from the metadata dictionary, defaulting to 'Unknown' if not found.

    Raises:
        FileNotFoundError: If the input directory does not exist.
        pd.errors.EmptyDataError: If any of the SNV files are empty.
        KeyError: If required columns are missing from the SNV files.
    """
    # Dictionary to store DataFrames with dynamic names
    dataframes = {}
    for input_file in glob.glob(os.path.join(input_dir, "*.IS_SNVs.tsv")):
        # Extract the prefix (e.g., SLG1121 or SLG441) from the file name
        file_prefix = os.path.basename(input_file).split(".")[0]

        logging.info(f"Reading SNV file {input_file} for sample {file_prefix}")

        # Read the file into a DataFrame
        df = pd.read_csv(
            input_file,
            sep="\t",
            usecols=[
                "scaffold",
                "position",
                "position_coverage",
                # "gene",
                "A",
                "C",
                "T",
                "G",
            ],
        )

        # Extract the MAG ID from the scaffold column by splitting on '.fa'
        # df["MAG_ID"] = df["scaffold"].str.partition(".fa")[0]

        # Add metadata columns: diet, replicate, and mouse
        # Default to 'Unknown' if not found in metadata
        metadata_info = metadata_dict.get(
            file_prefix, {"diet": "Unknown", "mouse": "Unknown"}
        )
        df["group"] = metadata_info["diet"]
        # df["replicate"] = metadata_info['replicate']
        df["mouse"] = metadata_info["mouse"]

        dataframes[f"{file_prefix}"] = df
    return dataframes


def calculate_frequencies(dataframes):
    """
    Calculate nucleotide frequencies for each position in the given DataFrames and concatenate the results.

    This function takes a dictionary of DataFrames, where each DataFrame represents nucleotide counts at various positions.
    It calculates the frequency of each nucleotide (A, C, T, G) at each position by dividing the count of each nucleotide
    by the position coverage. It then concatenates the results from all DataFrames into a single DataFrame.

    Parameters:
        dataframes (dict): A dictionary where keys are sample IDs and values are pandas DataFrames. Each DataFrame must
                           contain the columns 'A', 'C', 'T', 'G', 'position_coverage', 'MAG_ID', 'scaffold', 'position',
                           and 'group'.

    Returns:
        pandas.DataFrame: A concatenated DataFrame containing the nucleotide frequencies for all positions and scaffolds
                          across all samples. The resulting DataFrame includes the columns 'sample_id', 'MAG_ID',
                          'scaffold', 'position', 'A_frequency', 'T_frequency', 'G_frequency', 'C_frequency', and 'group'.
    """
    for df_name, df in dataframes.items():
        logging.info(f"Calculating nucleotide frequencies for {df_name}")
        position_coverage = df["position_coverage"]

        # Calculate frequencies directly
        df["A_frequency"] = df["A"] / position_coverage
        df["C_frequency"] = df["C"] / position_coverage
        df["T_frequency"] = df["T"] / position_coverage
        df["G_frequency"] = df["G"] / position_coverage

    # Initialize an empty list to store DataFrames for all positions and scaffolds for all MAGs
    all_results = []

    # Loop through each sample DataFrame
    for sample_id, df in dataframes.items():
        # Make a copy of df to avoid modifying the original DataFrame
        df = df.copy()
        # Add sample_id as a new column
        df["sample_id"] = sample_id
        # Select the columns you need
        df_subset = df[
            [
                "sample_id",
                # "MAG_ID",
                "scaffold",
                "position",
                # "gene",
                "A_frequency",
                "T_frequency",
                "G_frequency",
                "C_frequency",
                "group",
                # "replicate",
                "mouse",
            ]
        ]
        # Append to the list
        all_results.append(df_subset)

    # Concatenate all the DataFrames in the list.
    # This is the final DataFrame with all positions and scaffolds for all MAGs.
    logging.info(f"Concatenating all dataframes together.")
    final_table_all_positions = pd.concat(all_results, ignore_index=True)

    return final_table_all_positions


def run_wilcoxon_test(args, group_1, group_2):
    """
    Perform the Wilcoxon signed-rank test on paired samples from two groups.

    Parameters:
    args (tuple): A tuple containing the name and the grouped DataFrame.
    group_1 (str): The name of the first group.
    group_2 (str): The name of the second group.

    Returns:
    tuple: A tuple containing:
        - name (str): The name from the args.
        - p_values (dict): A dictionary with nucleotides as keys and their corresponding p-values.
        - num_samples (int): The number of paired samples used in the test.
        - num_samples (int): The number of paired samples used in the test (repeated for clarity).

    Notes:
    - The function assumes that the DataFrame is grouped by 'scaffold' and 'position'.
    - The p-values are initialized to NaN and will only be calculated if both groups have at least 2 data points.
    - If there are NaN values in the data, a ValueError will be raised due to the 'nan_policy' set to 'raise'.
    """
    name, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Mouse is the main column to merge on here, as that is what's same in the before and after samples
    merge_cols = [
        "mouse",
        "scaffold",
        "position",
    ]
    merged = pd.merge(
        group1, group2, on=merge_cols, suffixes=("_group1", "_group2"), how="inner"
    )

    # Check if we have enough paired samples
    num_samples = merged.shape[0]
    if num_samples >= 2:
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
            else:
                res = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
                )
                p_values[f"{nucleotide}_p_value"] = res.pvalue

    # Always return (name, p_values)
    return (name, p_values, num_samples, num_samples)


def run_mannwhitneyu_test(args, group_1, group_2):
    """
    Perform the Mann-Whitney U test on two groups within a grouped DataFrame.

    Parameters:
    args (tuple): A tuple containing the name and the grouped DataFrame.
    group_1 (str): The name of the first group to compare.
    group_2 (str): The name of the second group to compare.

    Returns:
    tuple: A tuple containing:
        - name (str): The name from the args tuple.
        - p_values (dict): A dictionary with nucleotides as keys and their corresponding p-values as values.
        - num_samples_group1 (int): The number of samples in the first group.
        - num_samples_group2 (int): The number of samples in the second group.

    Notes:
    - The function assumes that the DataFrame is grouped by 'scaffold' and 'position'.
    - The p-values are initialized to NaN and will only be calculated if both groups have at least 2 data points.
    - If there are NaN values in the data, a ValueError will be raised due to the 'nan_policy' set to 'raise'.
    """
    name, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Counts of samples in each group
    num_samples_group1 = group1.shape[0]
    num_samples_group2 = group2.shape[0]

    # Only perform the t-test if both groups have at least 2 data points
    if num_samples_group1 >= 2 and num_samples_group2 >= 2:
        for nucleotide in NUCLEOTIDES:
            res = stats.mannwhitneyu(
                group1[nucleotide],
                group2[nucleotide],
                alternative="two-sided",
                nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
            )

            p_values[f"{nucleotide}_p_value"] = res.pvalue

    # Always return (name, p_values)
    return (name, p_values, num_samples_group1, num_samples_group2)


def perform_tests_parallel(df, cpus, group_1, group_2, paired=False):
    """
    Perform statistical tests in parallel on grouped data.

    This function groups the input DataFrame by 'scaffold' and 'position', then performs either the Mann-Whitney U test
    for independent samples or the Wilcoxon signed-rank test for paired samples in parallel using the specified number
    of CPU cores.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be tested.
    cpus (int): The number of CPU cores to use for parallel processing.
    group_1 (str): The name of the first group for comparison.
    group_2 (str): The name of the second group for comparison.
    paired (bool, optional): If True, perform the Wilcoxon signed-rank test for paired samples. If False, perform the
                             Mann-Whitney U test for independent samples. Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the test results with p-values and sample sizes, with Benjamini-Hochberg
                  correction applied.
    """

    start_time = time.time()

    # Group the data
    logging.info("Grouping data by scaffold and position")
    grouped = df.groupby(["scaffold", "position"])

    # Prepare arguments for mapping
    logging.info(
        "Adding all groups to a list for parallisation. This may take some time..."
    )
    groups = list(grouped)

    # Mann-Whitney U Test (or Wilcoxon rank-sum test) is used for independent samples
    # Wilcoxon signed-rank test is used for comparing two paired samples
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#mannwhitneyu
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#wilcoxon

    if paired:
        logging.info(
            f"Performing Wilcoxon signed-rank tests for dependent samples in parallel using {cpus} cores."
        )
        my_func = partial(run_wilcoxon_test, group_1=group_1, group_2=group_2)
    else:
        logging.info(
            f"Performing Mann-Whitney U tests for independent samples in parallel using {cpus} cores."
        )
        my_func = partial(run_mannwhitneyu_test, group_1=group_1, group_2=group_2)

    with Pool(cpus) as pool:
        try:
            results = pool.map(my_func, groups)
        except ValueError as err:
            logging.error(f"An error occurred during multiprocessing: {err}")
            pool.terminate()
            pool.join()
            raise

    end_time = time.time()
    logging.info(
        f"Groups added to a list and tests performed in {end_time - start_time:.2f} seconds"
    )
    # Convert results to DataFrame
    records = []
    for name, p_values, num_samples_group1, num_samples_group2 in results:
        scaffold, position = name
        record = {
            "scaffold": scaffold,
            "position": position,
            **p_values,
            f"num_samples_{group_1}": num_samples_group1,
            f"num_samples_{group_2}": num_samples_group2,
        }
        records.append(record)
    test_results = pd.DataFrame(records)
    # print(test_results)
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

    p_value_columns = [
        "A_frequency_p_value",
        "T_frequency_p_value",
        "G_frequency_p_value",
        "C_frequency_p_value",
    ]
    test_results_cleaned = test_results.dropna(subset=p_value_columns, how="all").copy()

    # Ensure there are no NaN values in p-value columns
    if test_results_cleaned[p_value_columns].isnull().values.any():
        logging.error("NaNs found in p-value columns after cleaning.")
        raise ValueError("NaNs found in p-value columns after cleaning.")

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
    # Extract the MAG ID from the scaffold column by splitting on '.fa'
    test_results_cleaned["MAG_ID"] = test_results_cleaned["scaffold"].str.partition(
        ".fa"
    )[0]

    # Make MAG_ID as the first column
    cols = test_results_cleaned.columns.tolist()

    # Remove 'MAG_ID' from the list
    cols.remove("MAG_ID")

    # Place 'MAG_ID' at the beginning
    cols = ["MAG_ID"] + cols

    # Reindex the DataFrame
    test_results_cleaned = test_results_cleaned[cols]

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
        "--input_dir",
        help="Path to directory with input files",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--metadata_file",
        help="Path to metadata file",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--cpus",
        help=f"Number of processors to use.",
        default=16,
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

    metadata_dict = load_metadata(args.metadata_file)
    # Extract all unique group names from the metadata
    groups_in_metadata = set(entry["diet"] for entry in metadata_dict.values())

    # Check if args.group_1 and args.group_2 are in the metadata groups
    if args.group_1 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_1} not found in metadata file.")

    if args.group_2 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_2} not found in metadata file.")

    dataframes = load_snv_files(args.input_dir, metadata_dict)
    final_table_all_positions = calculate_frequencies(dataframes)
    test_results = perform_tests_parallel(
        final_table_all_positions, args.cpus, args.group_1, args.group_2, args.paired
    )

    # Save results to output directory
    os.makedirs(args.output_dir, exist_ok=True)

    final_table_all_positions.to_csv(
        os.path.join(args.output_dir, "nucleotide_frequencies_all.tsv"),
        index=False,
        sep="\t",
    )

    test_results.to_csv(
        os.path.join(args.output_dir, "test_results.tsv"), index=False, sep="\t"
    )
    logging.info(f"Results saved to {args.output_dir}")

    end_time = time.time()
    logging.info(f"Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
