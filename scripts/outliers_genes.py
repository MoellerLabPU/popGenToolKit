#!/usr/bin/env python
import argparse
import logging
import os
import time

import pandas as pd
from scipy.stats import binom, poisson
from tqdm import tqdm


def compute_p_value_binomial(s_obs, n_gene, p):
    """
    Compute the p-value using the binomial distribution.

    Parameters:
    s_obs (int): The observed number of successes.
    n_gene (int): The number of trials (genes).
    p (float): The probability of success on each trial.

    Returns:
    float: The p-value computed using the survival function (sf) of the binomial distribution.

    Notes:
    The survival function (sf) is used to compute the probability of observing at least `s_obs` successes.
    The observed count is adjusted by subtracting 1 to include the observed count itself in the calculation.
    """
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    p_val = binom.sf(
        s_obs - 1, n_gene, p
    )  # Subtract 1 to include observed_count, so we can do greater than equal to rather than just greater than
    return p_val


def compute_p_value_poisson(s_obs, lambda_):
    """
    Compute the p-value for a given observed count using the Poisson distribution.

    Parameters:
    s_obs (int): The observed count.
    lambda_ (float): The expected count (mean of the Poisson distribution).

    Returns:
    float: The p-value computed using the survival function (sf) of the Poisson distribution.

    Notes:
    The survival function (sf) is used to compute the probability of observing at least `s_obs` successes.
    The observed count is adjusted by subtracting 1 to include the observed count itself in the calculation.
    """

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html
    p_val = poisson.sf(s_obs - 1, lambda_)  # Subtract 1 to include observed_count
    return p_val


def load_mag_df(mag_fPath, mag_id):
    """
    Load a DataFrame from a specified file path and filter it by a given MAG ID.

    Parameters:
    mag_fPath (str): The file path to the MAG data file in CSV format.
    mag_id (str): The MAG ID to filter the DataFrame by.

    Returns:
    pandas.DataFrame: A DataFrame containing rows that match the specified MAG ID.

    Raises:
    SystemExit: If no rows are found for the specified MAG ID, the function logs an error and exits the program.
    """

    df = pd.read_csv(mag_fPath, sep="\t")
    mag_df = df[df["MAG_ID"] == mag_id]
    if mag_df.empty:
        logging.error(f"No rows found for MAG ID '{mag_id}' in {mag_fPath}. Exiting...")
        exit(1)
    return mag_df


def extract_score_columns(df):
    """
    Extracts columns from a DataFrame that start with "score_" and groups them by their suffix.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing columns to be extracted.

    Returns:
        dict: A dictionary where the keys are the suffixes of the score columns and the values are the original column names.
    """

    score_columns = [col for col in df.columns if col.startswith("score_")]
    grouped = {}
    for col in score_columns:
        suffix = col.replace("score_", "").replace("(%)", "").strip()
        grouped[suffix] = col
    return grouped


def compute_p_values(mag_fPath, mag_id, gene_fPath):
    """
    Compute p-values for gene scores based on MAG scores.

    This function loads MAG scores and gene scores from specified file paths,
    extracts relevant score columns, and computes p-values for each gene based
    on the MAG scores. The results are returned as a DataFrame.

    Parameters:
    mag_fPath (str): File path to the MAG scores file.
    mag_id (str): Identifier for the MAG.
    gene_fPath (str): File path to the gene scores file.

    Returns:
    pd.DataFrame: A DataFrame containing the computed p-values and related information
    for each gene.
    """

    logging.info(f"Loading MAG scores from {mag_fPath}")
    mag_df = load_mag_df(mag_fPath, mag_id)
    logging.info(f"Loading gene scores from {gene_fPath}")
    gene_df = pd.read_csv(gene_fPath, sep="\t")

    # Extract score columns and their suffixes
    mag_score_columns = extract_score_columns(mag_df)
    gene_score_columns = extract_score_columns(gene_df)

    # Ensure common suffixes between MAG and gene files
    common_suffixes = set(mag_score_columns.keys()).intersection(
        gene_score_columns.keys()
    )
    if not common_suffixes:
        logging.error(
            "No matching score groups found between MAG and gene files. Exiting."
        )
        exit(1)

    results = []
    for _, gene_row in tqdm(
        gene_df.iterrows(), total=gene_df.shape[0], desc="Identifying outliers in genes"
    ):
        # Common fields for this gene
        base_result = {
            "gene_id": gene_row["gene_id"],
        }

        for suffix in common_suffixes:
            mag_score_col = mag_score_columns[suffix]  # e.g. "score_tTest_fat (%)"
            gene_score_col = gene_score_columns[suffix]  # e.g. "score_tTest_fat (%)"

            # Derive corresponding total_sites_ and significant_sites_ columns
            gene_total_sites_col = f"total_sites_per_group_{suffix}"
            gene_significant_sites_col = f"significant_sites_per_group_{suffix}"

            # Check if they exist in the gene dataframe. If not, skip this suffix.
            if (
                gene_total_sites_col not in gene_df.columns
                or gene_significant_sites_col not in gene_df.columns
            ):
                logging.warning(
                    f"Missing columns for suffix '{suffix}' in gene file. Skipping."
                )
                continue

            # Convert from percent to fraction
            mag_score = mag_df.iloc[0][mag_score_col] / 100

            # Compute p-values
            s_gene = gene_row[gene_significant_sites_col]
            n_gene = gene_row[gene_total_sites_col]
            lambda_ = n_gene * mag_score

            p_val_bi = compute_p_value_binomial(s_gene, n_gene, mag_score)
            p_val_poisson = compute_p_value_poisson(s_gene, lambda_)

            # Add suffix-specific fields
            base_result.update(
                {
                    f"mag_score_{suffix} (%)": mag_df.iloc[0][mag_score_col],
                    f"gene_score_{suffix} (%)": gene_row[gene_score_col],
                    f"total_sites_gene_{suffix}": n_gene,
                    f"significant_sites_gene_{suffix}": s_gene,
                    f"p_value_binomial_{suffix}": p_val_bi,
                    f"p_value_poisson_{suffix}": p_val_poisson,
                }
            )

        results.append(base_result)

    # Combine results into a DataFrame
    combinded_df = pd.DataFrame(results)
    return combinded_df


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Calculate outlier genes using Binomial and Poisson distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mag_file",
        required=True,
        help="File containing MAG scores.",
        type=str,
    )
    parser.add_argument(
        "--mag_id",
        required=True,
        help="MAG ID to filter on within the mag_file.",
    )
    parser.add_argument(
        "--gene_file",
        required=True,
        help="File containing gene score files.",
        type=str,
    )

    parser.add_argument(
        "--out_fPath",
        required=True,
        help="Directory to write output files.",
        type=str,
    )

    args = parser.parse_args()
    start_time = time.time()
    combined_df = compute_p_values(args.mag_file, args.mag_id, args.gene_file)

    combined_df.to_csv(args.out_fPath, sep="\t", index=False)
    logging.info(f"Output written to: {args.out_fPath}")
    end_time = time.time()
    logging.info(f"Done! Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
