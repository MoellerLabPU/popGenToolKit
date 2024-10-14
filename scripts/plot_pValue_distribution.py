#!/usr/bin/env python

import argparse
import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def plot_histogram(
    mag_df, dirPath, filePrefix, pvalue_columns, adj_pvalue_columns, mag_id
):
    """
    Plots histograms of original and adjusted p-values for a given MAG (Metagenome-Assembled Genome).

    Parameters:
    mag_df (pd.DataFrame): DataFrame containing p-values and other relevant data.
    dirPath (str): Directory path where the plot image will be saved.
    filePrefix (str): Prefix for the output file name.
    pvalue_columns (list of str): List of column names containing original p-values.
    adj_pvalue_columns (list of str): List of column names containing adjusted p-values.
    mag_id (str): Identifier for the MAG being plotted.

    Returns:
    None
    """
    os.makedirs(dirPath, exist_ok=True)

    if not mag_df.empty:
        # logging.info(f"Plotting histograms for MAG_ID: {mag_id}")

        # Prepare the DataFrame to contain both original and adjusted p-values
        long_df_original = mag_df.melt(
            id_vars=["MAG_ID", "scaffold", "position"],
            value_vars=pvalue_columns,
            var_name="Nucleotide",
            value_name="P-value",
        )

        long_df_original["Type"] = "Original"

        long_df_adjusted = mag_df.melt(
            id_vars=["MAG_ID", "scaffold", "position"],
            value_vars=adj_pvalue_columns,
            var_name="Nucleotide",
            value_name="P-value",
        )

        long_df_adjusted["Type"] = "Adjusted"

        # Clean up the nucleotide names in both dataframes
        # This converts "A_frequency_p_value" to "A" and "A_frequency_p_value_adj" to "A"
        long_df_original["Nucleotide"] = long_df_original["Nucleotide"].str.replace(
            "_frequency_p_value", ""
        )
        long_df_adjusted["Nucleotide"] = long_df_adjusted["Nucleotide"].str.replace(
            "_frequency_p_value_adj", ""
        )

        # Concatenate the original and adjusted p-values into one DataFrame
        long_df = pd.concat([long_df_original, long_df_adjusted], ignore_index=True)

        # Adds "Nucleotide_Type" column with values like "A (Original), A (Adjusted)"
        long_df["Nucleotide_Type"] = (
            long_df["Nucleotide"] + " (" + long_df["Type"] + ")"
        )

        plt.figure(figsize=(10, 6))

        # Plot both original and adjusted p-values on the same plot
        sns.histplot(
            data=long_df,
            x="P-value",
            bins=50,
            kde=True,
            edgecolor="black",
            hue="Nucleotide_Type",
            multiple="dodge",
        )

        plt.xlabel("P-value")
        plt.ylabel("Frequency")
        plt.title(f"{filePrefix} for {mag_id}")
        plt.tight_layout()
        filename = f"{filePrefix}_{mag_id}.png"
        plt.savefig(os.path.join(dirPath, filename), dpi=300)
        # plt.show()
        plt.close()


def process_mag(
    df, mag_id, base_path, pvalue_columns, adj_pvalue_columns, filter_condition, desc
):
    """
    Processes a DataFrame to filter and plot p-value distributions for a specific MAG (Metagenome-Assembled Genome).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        mag_id (str): The MAG ID to filter the DataFrame.
        base_path (str): The base directory path where plots will be saved.
        pvalue_columns (list of str): List of column names containing p-values.
        adj_pvalue_columns (list of str): List of column names containing adjusted p-values.
        filter_condition (function): A function that takes a DataFrame and returns a filtered DataFrame.
        desc (str): A description used for naming the output directory and file prefix.

    Returns:
        None
    """
    mag_df = df[df["MAG_ID"] == mag_id]
    mag_df_subset = filter_condition(mag_df)

    dir_path = os.path.join(base_path, desc)
    file_prefix = desc

    plot_histogram(
        mag_df_subset,
        dir_path,
        file_prefix,
        pvalue_columns,
        adj_pvalue_columns,
        mag_id,
    )


def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Make histogram for p-value frequencies in MAGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_file",
        help="Path to the input TSV file containing p-values.",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory.",
        type=str,
        required=True,
        metavar="directory path",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep="\t")
    logging.info(f"Successfully read input file: {args.input_file}")

    pvalue_columns = [
        "A_frequency_p_value",
        "T_frequency_p_value",
        "G_frequency_p_value",
        "C_frequency_p_value",
    ]
    adj_pvalue_columns = [col + "_adj" for col in pvalue_columns]

    # Check if required columns exist
    required_columns = set(
        ["MAG_ID", "scaffold", "position"] + pvalue_columns + adj_pvalue_columns
    )
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logging.error(f"Missing columns in input file: {missing_columns}")
        raise ValueError(f"Missing columns in input file: {missing_columns}")

    # Get unique MAG IDs
    mag_ids = df["MAG_ID"].unique()

    filter_conditions = {
        "all_p_values": lambda x: x,
        "p_values_any_0.05": lambda x: x[(x[pvalue_columns] < 0.05).any(axis=1)],
        "p_values_all_0.05": lambda x: x[(x[pvalue_columns] < 0.05).all(axis=1)],
    }

    # Process each MAG ID
    for mag_id in tqdm(
        mag_ids, desc="Processing and plotting MAGs p-value distributions"
    ):
        for desc, filter_condition in filter_conditions.items():
            process_mag(
                df,
                mag_id,
                args.output_dir,
                pvalue_columns,
                adj_pvalue_columns,
                filter_condition,
                desc,
            )


if __name__ == "__main__":
    main()
