#!/usr/bin/env python

import argparse
import logging
import os
import time

import pandas as pd
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def compute_p_value_binomial(s_obs, n_gene, p):
    """
    Compute the p-value using the binomial survival function.
    """
    p_val = binom.sf(s_obs - 1, n_gene, p)
    return p_val


def load_mag_df(mag_fPath):
    """
    Load the MAG scores from the specified file.
    """
    mag_df = pd.read_csv(mag_fPath, sep="\t")
    logging.info(f"Loaded MAG scores from {mag_fPath}")
    required_mag_columns = {"MAG_ID", "score"}
    if not required_mag_columns.issubset(mag_df.columns):
        logging.error(
            f"MAG file {mag_fPath} does not contain required columns: {required_mag_columns}. Exiting."
        )
        exit(1)
    return mag_df


def compute_p_values(mag_fPath, gene_fPath, out_fPath, fwer=0.05):
    mag_df = load_mag_df(mag_fPath)

    # Load gene data
    try:
        genes_df = pd.read_csv(gene_fPath, sep="\t")
        logging.info(f"Loaded gene scores from {gene_fPath}")
    except Exception as e:
        logging.error(f"Error loading gene file {gene_fPath}: {e}")
        exit(1)

    # Filter genes with a single gene_id (exclude rows with multiple genes)
    genes_df = genes_df[~genes_df["gene_id"].str.contains(",")]

    # Extract 'MAG_ID' from 'gene_id' by taking everything before '.fa'
    genes_df["MAG_ID"] = genes_df["gene_id"].str.split(".fa").str[0]

    # List to store results from all MAGs
    all_results = []

    # Process each MAG
    for index, mag_row in tqdm(
        mag_df.iterrows(), total=mag_df.shape[0], desc="Processing MAGs"
    ):

        mag_id = mag_row["MAG_ID"]
        p = mag_row["score"] / 100  # Convert score to probability

        # Get subset of genes for the current MAG
        mag_genes_df = genes_df[genes_df["MAG_ID"] == mag_id]

        if mag_genes_df.empty:
            logging.warning(f"No genes found for MAG {mag_id}. Skipping.")
            continue

        required_gene_columns = {"gene_id", "significant_sites", "total_sites", "score"}
        if not required_gene_columns.issubset(mag_genes_df.columns):
            logging.error(
                f"Gene data for MAG {mag_id} does not have required columns: {required_gene_columns}. Skipping this MAG."
            )
            continue

        # For each gene in the MAG, compute p-values
        gene_p_values = []
        for _, gene_row in mag_genes_df.iterrows():
            gene_id = gene_row["gene_id"]
            s_gene = gene_row["significant_sites"]
            n_gene = gene_row["total_sites"]

            # Expected number of significant sites under the null hypothesis
            lambda_ = n_gene * p

            # Compute p-value using the binomial distribution
            p_val = compute_p_value_binomial(s_gene, n_gene, p)

            gene_p_values.append(
                {
                    "mag_id": mag_id,
                    "gene_id": gene_id,
                    "significant_sites_gene": s_gene,
                    "total_sites_gene": n_gene,
                    "p_value": p_val,
                    "gene_score": gene_row["score"],
                    "mag_score": mag_row["score"],
                }
            )

        # Convert to DataFrame and sort by p-value
        mag_genes_results_df = pd.DataFrame(gene_p_values)

        # Adjust p-values for multiple testing within the MAG
        mag_genes_results_df["p_value_adjusted"] = multipletests(
            mag_genes_results_df["p_value"], method="fdr_bh", alpha=fwer
        )[1]

        # Append the results to the list
        all_results.append(mag_genes_results_df)

    # Combine all results into a single DataFrame
    all_genes_df = pd.concat(all_results, ignore_index=True)
    all_genes_df = all_genes_df.sort_values(by="p_value")

    # Save the combined results to the output file
    all_genes_df.to_csv(out_fPath, sep="\t", index=False)
    logging.info(f"Outlier genes written to {out_fPath}")


def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Calculate outlier genes using Binomial distribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mag_file",
        required=True,
        help="File containing MAG scores.",
        type=str,
    )
    parser.add_argument(
        "--gene_file",
        required=True,
        help="File containing gene scores.",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="File to write output.",
        type=str,
    )
    parser.add_argument(
        "--fwer",
        default=0.05,
        help="Family-wise error rate. The probability of at least 1 false positive when multiple comparisons are being tested.",
        type=float,
    )

    args = parser.parse_args()
    start_time = time.time()
    compute_p_values(args.mag_file, args.gene_file, args.output_file, args.fwer)
    end_time = time.time()
    logging.info(f"Done! Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
