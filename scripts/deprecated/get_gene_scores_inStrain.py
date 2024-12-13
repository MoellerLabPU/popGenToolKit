#!/usr/bin/env python

import argparse
import logging
import os
import time
from collections import defaultdict

import pandas as pd
from Bio import SeqIO
from intervaltree import IntervalTree


def map_genes(prodigal_fasta):
    # Initialize a list to store gene information
    logging.info("Parsing Prodigal FASTA file to extract gene information.")
    genes_data = []

    # Parse the FASTA file
    for record in SeqIO.parse(prodigal_fasta, "fasta"):
        header = record.description
        # Split the header at '#' to get the gene coordinates
        # https://github.com/hyattpd/prodigal/wiki/understanding-the-prodigal-output#protein-translations
        parts = header.split("#")
        if len(parts) == 5:
            gene_id = parts[0]
            start = (
                int(parts[1]) - 1
            )  # pySam uses 0 based indexing while Prodigal uses 1 based
            end = int(parts[2]) - 1
            # Append gene information to the list
            genes_data.append(
                {
                    "gene_id": gene_id,
                    "start": start,
                    "end": end,
                }
            )
        else:
            logging.warning(f"Warning: Unexpected header format: {header}.")
            raise ValueError("Unexpected header format in the FASTA file.")

    # Convert the list to a DataFrame
    genes_df = pd.DataFrame(genes_data)
    genes_df["contig"] = genes_df["gene_id"].apply(extract_contigID)
    logging.info("Gene information extracted successfully.")
    return genes_df


def extract_contigID(gene_id):
    contig = "_".join(gene_id.split("_")[:-1])
    return contig


def create_intervalTree(genes_df):
    logging.info(f"Creating IntervalTree for the genes dataframe.")
    start_time = time.time()

    contig_trees = defaultdict(IntervalTree)
    for idx, row in genes_df.iterrows():
        contig = row["contig"]
        start = row["start"]
        end = row["end"]
        gene_id = row["gene_id"].rstrip()

        # contig_trees is a dictionary where each key is a contig and each value is an IntervalTree containing intervals for the genes and the gene_ID
        # intervaltree uses half-open intervals, including the lower bound but not the upper bound. We add 1 to include the end position
        # https://github.com/chaimleib/intervaltree/issues/128#issuecomment-1515796584
        # gene_id, is stored as the data attribute of the interval
        contig_trees[contig].addi(start, end + 1, gene_id)

    end_time = time.time()
    logging.info(f"IntervalTree created in {end_time - start_time:.2f} seconds")
    return contig_trees


def map_positions_to_genes(positions_df, contig_trees):
    logging.info("Mapping positions to genes...")
    start_time = time.time()

    # Initialize a list to collect DataFrames
    result_dfs = []

    # Group positions by `contig`
    grouped = positions_df.groupby("scaffold")

    for contig, group in grouped:
        tree = contig_trees.get(contig)
        if tree:
            # Get the positions
            positions = group["position"].values
            # Prepare a list to store gene IDs
            gene_ids = []
            # Query the IntervalTree for each position
            for pos in positions:
                overlaps = tree.at(pos)
                if overlaps:
                    gene_id = ",".join([interval.data for interval in overlaps])
                else:
                    gene_id = None
                gene_ids.append(gene_id)
            # Assign the gene IDs to the group
            # Create a copy to avoid modifying the original DataFrame
            group_copy = group.copy()
            group_copy["gene_id"] = gene_ids
        else:
            group_copy = group.copy()
            group_copy["gene_id"] = None
        # Collect the processed groups
        result_dfs.append(group_copy)

    # Concatenate all groups back into a single DataFrame
    mapped_positions_df = pd.concat(result_dfs, ignore_index=True)

    end_time = time.time()
    logging.info(f"Positions mapped to genes in {end_time - start_time:.2f} seconds")
    return mapped_positions_df


def calculate_significant_scores(mapped_positions_df, p_value_threshold=0.05):
    # Define the p-value columns and threshold
    p_value_columns = [
        "A_frequency_p_value",
        "T_frequency_p_value",
        "G_frequency_p_value",
        "C_frequency_p_value",
    ]

    # Check for missing values in p_value_columns
    any_missing = mapped_positions_df[p_value_columns].isnull().values.any()

    if any_missing:
        logging.warning(
            f"Missing value found in p-value column. Dropping rows with NaNs."
        )
        mapped_positions_df = mapped_positions_df.dropna(subset=p_value_columns)

    # Create 'is_significant' column
    mapped_positions_df["is_significant"] = (
        mapped_positions_df[p_value_columns].lt(p_value_threshold).any(axis=1)
    )

    # Group by 'gene_id' and calculate required aggregations
    # https://pandas.pydata.org/docs/user_guide/groupby.html#named-aggregation
    group_scores = mapped_positions_df.groupby("gene_id").agg(
        total_sites=("position", "count"),
        significant_sites=("is_significant", "sum"),
    )

    # Calculate percentage of significant sites per gene
    group_scores["score"] = (
        group_scores["significant_sites"] / group_scores["total_sites"]
    ) * 100

    # Reset index to turn 'gene_id' back into a column
    group_scores = group_scores.reset_index()

    # Sort the DataFrame by score in descending order
    group_scores = group_scores.sort_values(by="score", ascending=False)
    return group_scores


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
        "--prodigal_fasta",
        help="Path to Prodigal predicted genes (DNA).",
        type=str,
        required=True,
        metavar="filepath",
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
        help="Path to output file.",
        type=str,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()
    genes_df = map_genes(args.prodigal_fasta)
    contig_trees = create_intervalTree(genes_df)

    df_mag = pd.read_csv(args.pValue_table, sep="\t")
    mapped_positions_df = map_positions_to_genes(df_mag, contig_trees)
    os.makedirs(args.output_dir, exist_ok=True)
    outPath = os.path.join(args.output_dir, "genes_mapped.tsv")
    mapped_positions_df.to_csv(outPath, index=False, sep="\t")
    logging.info(f"Positions mapped to genes saved to {outPath}")
    logging.info("Calculating significant scores...")
    score_df = calculate_significant_scores(mapped_positions_df, args.pValue_threshold)
    score_df.to_csv(
        os.path.join(args.output_dir, "gene_scores.tsv"), index=False, sep="\t"
    )
    logging.info(f"Significant scores saved to {args.output_dir}/gene_scores.tsv")


if __name__ == "__main__":
    main()
