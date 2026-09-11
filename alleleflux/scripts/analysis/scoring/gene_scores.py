#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd

from alleleflux.scripts.utilities.utilities import (
    calculate_score,
    extract_relevant_columns,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def get_scores(df, p_value_threshold=0.05):

    test_columns_dict = extract_relevant_columns(df, capture_str="p_value_")
    # First Output: Overlapping genes are kept as combined entities
    logger.info("Calculating scores for combined genes.")
    group_scores_combined = calculate_score(
        df, test_columns_dict, "gene_id", p_value_threshold
    )

    # Second Output: Overlapping positions contribute to each gene separately
    df_individual = df.copy()
    # Split 'gene_id' into a list if multiple genes are present
    df_individual["gene_id"] = df_individual["gene_id"].str.split(",")
    # Explode the DataFrame to have one gene_id per row
    df_individual = df_individual.explode("gene_id")
    # Trim whitespace from gene_ids
    df_individual["gene_id"] = df_individual["gene_id"].str.strip()
    # Calculate group scores
    logger.info("Calculating scores for individual genes.")
    group_scores_individual = calculate_score(
        df_individual, test_columns_dict, "gene_id", p_value_threshold
    )

    # Third Output: Overlapping Genes Only
    logger.info("Calculating scores for overlapping genes.")
    overlapping_rows = df[df["gene_id"].str.contains(",", na=False)].copy()
    if not overlapping_rows.empty:
        group_scores_overlapping = calculate_score(
            overlapping_rows, test_columns_dict, "gene_id", p_value_threshold
        )
    else:
        # If no overlapping genes, create an empty DataFrame
        logger.info("No overlapping genes found, creating empty DataFrame.")
        group_scores_overlapping = pd.DataFrame(
            columns=["gene_id", "total_sites", "significant_sites", "score"]
        )

    return group_scores_combined, group_scores_individual, group_scores_overlapping


def main():

    setup_logging()
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
        metavar="dirpath",
    )

    parser.add_argument(
        "--prefix",
        help="File prefix to use.",
        metavar="str",
        type=str,
        default="sample",
    )

    args = parser.parse_args()

    logger.info("Reading p-value table.")
    df = pd.read_csv(args.pValue_table, sep="\t")

    if df.empty:
        raise ValueError("Input p-value table is empty.")

    if df["gene_id"].isna().all():
        raise ValueError("No gene IDs are present in the input table.")

    logger.info("Calculating significant scores...")

    group_scores_combined, group_scores_individual, group_scores_overlapping = (
        get_scores(df, args.pValue_threshold)
    )

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine file prefix
    prefix = args.prefix if args.prefix else "sample"

    # Save the three DataFrames to separate files with optional prefix
    output_combined = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_combined.tsv"
    )
    group_scores_combined.to_csv(output_combined, index=False, sep="\t")
    logger.info(f"Combined gene scores saved to {output_combined}")

    output_individual = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_individual.tsv"
    )
    group_scores_individual.to_csv(output_individual, index=False, sep="\t")
    logger.info(f"Individual gene scores saved to {output_individual}")

    output_overlapping = os.path.join(
        args.output_dir, f"{prefix}_gene_scores_overlapping.tsv"
    )
    group_scores_overlapping.to_csv(output_overlapping, index=False, sep="\t")
    logger.info(f"Overlapping gene scores saved to {output_overlapping}")


if __name__ == "__main__":
    main()
