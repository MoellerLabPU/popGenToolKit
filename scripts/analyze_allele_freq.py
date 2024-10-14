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


def load_metadata(metadata_file):
    """
    Load metadata from a given file and create a dictionary mapping sample names to their corresponding diet groups.

    Parameters:
        metadata_file (str): Path to the metadata file in tab-separated values (TSV) format with column, "sample" and "group".

    Returns:
        dict: A dictionary where keys are sample names and values are diet groups.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        pd.errors.ParserError: If there is an error parsing the metadata file.
    """
    logging.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t")
    # Create a dictionary from the metadata where file_prefix is the key and group is the value
    metadata_dict = dict(zip(metadata["sample"], metadata["diet"]))
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
                "A",
                "C",
                "T",
                "G",
            ],
        )

        # Extract the MAG ID from the scaffold column by splitting on '.fa'
        df["MAG_ID"] = df["scaffold"].str.partition(".fa")[0]

        # Add the group column from metadata
        df["group"] = metadata_dict.get(
            file_prefix, "Unknown"
        )  # Default to 'Unknown' if not found in metadata

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
                "MAG_ID",
                "scaffold",
                "position",
                "A_frequency",
                "T_frequency",
                "G_frequency",
                "C_frequency",
                "group",
            ]
        ]
        # Append to the list
        all_results.append(df_subset)

    # Concatenate all the DataFrames in the list.
    # This is the final DataFrame with all positions and scaffolds for all MAGs.
    logging.info(f"Concatenating all dataframes together {df_name}")
    final_table_all_positions = pd.concat(all_results, ignore_index=True)

    return final_table_all_positions


def run_significanceTest(args, group_1, group_2, paired=False):
    """
    Perform a significance test (Mann-Whitney U Test or Wilcoxon signed-rank test) on allele frequencies for each nucleotide between two groups.

    Parameters:
    args (tuple): A tuple containing the name and the grouped dataframe.
    group_1 (str): The label for the first group in the dataframe.
    group_2 (str): The label for the second group in the dataframe.
    paired (bool, optional): If True, perform the Wilcoxon signed-rank test for paired samples.
                             If False, perform the Mann-Whitney U Test for independent samples. Default is False.

    Returns:
    tuple: A tuple containing the name and a dictionary of p-values for each nucleotide frequency
           ('A_frequency', 'T_frequency', 'G_frequency', 'C_frequency'). If the test is not performed due to insufficient data,
           p-values will be NaN.
    """
    name, group = args
    # group is basically the dataframe, grouped by 'MAG_ID', 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {
        f"{nuc}_p_value": np.nan
        for nuc in ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]
    }

    # Only perform the t-test if both groups have at least 2 data points
    if group1.shape[0] >= 2 and group2.shape[0] >= 2:
        for nucleotide in ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]:

            # Mann-Whitney U Test (or Wilcoxon rank-sum test) is used for independent samples
            # Wilcoxon signed-rank test is used for comparing two paired samples
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#mannwhitneyu
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#wilcoxon

            if paired:
                res = stats.wilcoxon(
                    group1[nucleotide],
                    group2[nucleotide],
                    alternative="two-sided",
                    nan_policy="omit",
                )

            else:
                res = stats.mannwhitneyu(
                    group1[nucleotide],
                    group2[nucleotide],
                    alternative="two-sided",
                    nan_policy="omit",
                )

            p_values[f"{nucleotide}_p_value"] = res.pvalue

    # Always return (name, p_values)
    return (name, p_values)


def perform_tests_parallel(df, cpus, group_1, group_2, paired=False):
    """
    Perform statistical tests in parallel on grouped data.

    This function groups the input DataFrame by 'MAG_ID', 'scaffold', and 'position',
    and then performs either Wilcoxon signed-rank tests (for dependent samples) or
    Mann-Whitney U tests (for independent samples) in parallel using the specified
    number of CPU cores.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be tested.
    cpus (int): The number of CPU cores to use for parallel processing.
    group_1 (str): The name of the first group for the statistical test.
    group_2 (str): The name of the second group for the statistical test.
    paired (bool, optional): If True, perform Wilcoxon signed-rank tests for dependent samples.
                             If False, perform Mann-Whitney U tests for independent samples.
                             Default is False.

    Returns:
    pd.DataFrame: A DataFrame containing the test results with Benjamini-Hochberg correction applied.
    """
    start_time = time.time()

    # Group the data
    logging.info("Grouping data by MAG_ID, scaffold and position")
    grouped = df.groupby(["MAG_ID", "scaffold", "position"])

    # Prepare arguments for mapping
    logging.info(
        "Adding all groups to a list for parallisation. This may take some time..."
    )
    groups = list(grouped)

    if paired:
        logging.info(
            f"Performing Wilcoxon signed-rank tests for dependent samples in parallel using {cpus} cores."
        )
    else:
        logging.info(
            f"Performing Mann-Whitney U tests for independent samples in parallel using {cpus} cores."
        )

    with Pool(cpus) as pool:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap
        # https://stackoverflow.com/a/4463621/12671809 and https://stackoverflow.com/a/5443941/12671809
        my_func = partial(
            run_significanceTest, group_1=group_1, group_2=group_2, paired=paired
        )
        results = pool.map(my_func, groups)

    end_time = time.time()
    logging.info(
        f"Groups added to a list and tests performed in {end_time-start_time:.2f} seconds"
    )

    # Convert results to DataFrame
    records = []
    for name, p_values in results:
        mag_id, scaffold, position = name
        record = {
            "MAG_ID": mag_id,
            "scaffold": scaffold,
            "position": position,
            **p_values,
        }
        records.append(record)
    test_results = pd.DataFrame(records)
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
    if args.group_1 not in metadata_dict.values():
        raise ValueError(f"Group {args.group_1} not found in metadata file.")

    if args.group_2 not in metadata_dict.values():
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
