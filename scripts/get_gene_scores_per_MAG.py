#!/usr/bin/env python

import argparse
import logging
import os

import pandas as pd


def calculate_group_scores(df):
    group_scores = df.groupby("gene_id").agg(
        total_sites=("position", "count"),
        significant_sites=("is_significant", "sum"),
    )
    group_scores["score"] = (
        group_scores["significant_sites"] / group_scores["total_sites"]
    ) * 100
    group_scores = group_scores.reset_index()
    group_scores = group_scores.sort_values(by="score", ascending=False)
    return group_scores


def calculate_significant_scores(df, p_value_threshold=0.05):

    # Define the p-value columns
    p_value_columns = [
        "A_frequency_p_value",
        "T_frequency_p_value",
        "G_frequency_p_value",
        "C_frequency_p_value",
    ]

    # Check for missing values in p_value_columns
    if df[p_value_columns].isnull().values.any():
        logging.warning(
            "Missing value found in p-value column. Dropping rows with NaNs."
        )
        df = df.dropna(subset=p_value_columns)

    # Create 'is_significant' column
    df["is_significant"] = df[p_value_columns].lt(p_value_threshold).any(axis=1)

    ### First Output: Current Functionality ###
    # Overlapping genes are kept as combined entities
    group_scores_combined = calculate_group_scores(df)

    ### Second Output: Individual Genes ###
    # Overlapping positions contribute to each gene separately
    df_individual = df.copy()
    # Split 'gene_id' into a list if multiple genes are present
    df_individual["gene_id"] = df_individual["gene_id"].str.split(",")
    # Explode the DataFrame to have one gene_id per row
    df_individual = df_individual.explode("gene_id")
    # Trim whitespace from gene_ids
    df_individual["gene_id"] = df_individual["gene_id"].str.strip()
    # Calculate group scores
    group_scores_individual = calculate_group_scores(df_individual)

    ### Third Output: Overlapping Genes Only ###
    # Identify overlapping genes
    overlapping_rows = df[df["gene_id"].str.contains(",", na=False)].copy()
    if not overlapping_rows.empty:
        group_scores_overlapping = calculate_group_scores(overlapping_rows)
    else:
        # If no overlapping genes, create an empty DataFrame
        logging.info("No overlapping genes found, creating empty DataFrame.")
        group_scores_overlapping = pd.DataFrame(
            columns=["gene_id", "total_sites", "significant_sites", "score"]
        )

    return group_scores_combined, group_scores_individual, group_scores_overlapping


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser(
        description="Calculate significance score for each gene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--pValue_table",
        help="Path to table with p-values.",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--pValue_threshold",
        help="p-value threshold to use.",
        default=0.05,
        metavar="float",
        type=float,
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory.",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--prefix",
        help="File prefix to use.",
        metavar="str",
        type=str,
    )

    args = parser.parse_args()

    logging.info("Reading p-value table.")
    df = pd.read_csv(args.pValue_table, sep="\t")

    logging.info("Calculating significant scores...")

    group_scores_combined, group_scores_individual, group_scores_overlapping = (
        calculate_significant_scores(df, args.pValue_threshold)
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine file prefix
    prefix = args.prefix if args.prefix else ""

    # Save the three DataFrames to separate files with optional prefix
    output_combined = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_combined.tsv"
    )
    group_scores_combined.to_csv(output_combined, index=False, sep="\t")
    logging.info(f"Combined gene scores saved to {output_combined}")

    output_individual = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_individual.tsv"
    )
    group_scores_individual.to_csv(output_individual, index=False, sep="\t")
    logging.info(f"Individual gene scores saved to {output_individual}")

    output_overlapping = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_overlapping.tsv"
    )
    group_scores_overlapping.to_csv(output_overlapping, index=False, sep="\t")
    logging.info(f"Overlapping gene scores saved to {output_overlapping}")


if __name__ == "__main__":
    main()