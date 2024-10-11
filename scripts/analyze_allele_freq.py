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
    logging.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t")
    # Create a dictionary from the metadata where file_prefix is the key and group is the value
    metadata_dict = dict(zip(metadata["sample"], metadata["diet"]))
    return metadata_dict


def load_snv_files(input_dir, metadata_dict):
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
    Perform a significance test (Mann-Whitney U Test or Wilcoxon signed-rank test) on allele frequencies between two groups.

    Parameters:
    args (tuple): A tuple containing the name and the grouped dataframe.
    group_1 (str): The name of the first group.
    group_2 (str): The name of the second group.
    paired (bool, optional): If True, perform the Wilcoxon signed-rank test for paired samples.
                             If False, perform the Mann-Whitney U Test for independent samples. Default is False.

    Returns:
    tuple: A tuple containing the name and a dictionary of p-values for each nucleotide frequency
           (A_frequency, T_frequency, G_frequency, C_frequency). If the test is not performed due to insufficient data,
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
