#!/usr/bin/env python
import argparse
import glob
import logging
import os
import time

import pandas as pd
from scipy.stats import binom, poisson
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def compute_p_value_binomial(s_obs, n_gene, p):
    p_val = binom.sf(
        s_obs - 1, n_gene, p
    )  # Subtract 1 to include observed_count, so we can do greater than equal to rather than just greater than
    return p_val


def compute_p_value_poisson(s_obs, lambda_):
    p_val = poisson.sf(s_obs - 1, lambda_)  # Subtract 1 to include observed_count
    return p_val


def load_mag_df(mag_fPath):
    mag_df = pd.read_csv(mag_fPath, sep="\t")
    logging.info(f"Loaded MAG scores from {mag_fPath}")
    required_mag_columns = {"MAG_ID", "score"}
    if not required_mag_columns.issubset(mag_df.columns):
        logging.error(
            f"MAG file {mag_fPath} does not contain required columns: {required_mag_columns}. Exiting."
        )
        exit(1)

    return mag_df


def compute_p_values(mag_fPath, gene_dir, outDir, fwer=0.05):
    mag_df = load_mag_df(mag_fPath)

    # List to store results
    results = []

    for index, mag_row in tqdm(
        mag_df.iterrows(), total=mag_df.shape[0], desc="Processing MAGs"
    ):
        mag_id = mag_row["MAG_ID"]
        # logging.info(f"Processing MAG {mag_id}.")
        p = mag_row["score"] / 100  # Overall significance rate

        # Construct gene file name
        gene_file = os.path.join(gene_dir, f"{mag_id}_gene_scores_individual.tsv")
        # Check if gene file exists
        if not os.path.exists(gene_file):
            logging.error(
                f"Gene file for MAG {mag_id} not found: {gene_file}. Skipping this MAG."
            )
            continue

        gene_df = pd.read_csv(gene_file, sep="\t")
        # logging.info(f"Loaded gene scores from {gene_file}")

        required_gene_columns = {"gene_id", "significant_sites", "total_sites", "score"}
        if not required_gene_columns.issubset(gene_df.columns):
            logging.error(
                f"Gene file {gene_file} does not have required columns: {required_gene_columns}. Skipping this MAG."
            )
            continue

        # For each gene, compute p-values
        gene_p_values = []
        for _, gene_row in gene_df.iterrows():
            gene_id = gene_row["gene_id"]
            s_gene = gene_row["significant_sites"]
            n_gene = gene_row["total_sites"]

            lambda_ = n_gene * p  # Expected number of significant sites

            # Decide whether to use binomial or Poisson
            # if n_gene * p < 5 and n_gene < 1000:
            #     # Use binomial distribution
            #     p_val = compute_p_value_binomial(s_gene, n_gene, p)
            #     method = "binomial"
            # else:
            #     # Use Poisson distribution
            #     p_val = compute_p_value_poisson(s_gene, lambda_)
            #     method = "poisson"
            p_val = compute_p_value_binomial(s_gene, n_gene, p)
            # p_val = compute_p_value_poisson(s_gene, lambda_)
            # p_val_bi = compute_p_value_binomial(s_gene, n_gene, p)
            # p_val_poisson = compute_p_value_poisson(s_gene, lambda_)

            gene_p_values.append(
                {
                    "mag_id": mag_id,
                    "gene_id": gene_id,
                    "significant_sites_gene": s_gene,
                    "total_sites_gene": n_gene,
                    "p_value": p_val,
                    # "p_value_binomial": p_val_bi,
                    # "p_value_poisson": p_val_poisson,
                    "gene_score": gene_row["score"],
                    "mag_score": mag_row["score"],
                    # "method": method,
                }
            )
        genes_df = pd.DataFrame(gene_p_values).sort_values(by="p_value")
        genes_df["p_value_adjusted"] = multipletests(
            genes_df["p_value"], method="fdr_bh", alpha=fwer
        )[1]
        out_fPath = os.path.join(outDir, f"{mag_id}_outliers.tsv")
        genes_df.to_csv(out_fPath, sep="\t", index=False)


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Calculate outlier genes using Binomial disctribution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mag_file",
        required=True,
        help="File containing MAG scores.",
        type=str,
    )
    parser.add_argument(
        "--gene_dir",
        required=True,
        help="Directory containing gene score files.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write output files.",
        type=str,
    )
    parser.add_argument(
        "--fwer",
        default=0.05,
        help="Family wise error rate. The probability of at least 1 false positive when multiple comparisons are being tested",
        type=float,
    )

    args = parser.parse_args()
    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    compute_p_values(args.mag_file, args.gene_dir, args.output_dir, args.fwer)
    end_time = time.time()
    logging.info(f"Done! \n Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
