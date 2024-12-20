#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd
from utilities import calculate_score, extract_test_columns


def read_gtdb(gtdb_fpath):
    gtdb_df = pd.read_csv(
        gtdb_fpath,
        sep="\t",
        usecols=["user_genome", "classification"],
    )
    gtdb_df = gtdb_df.rename(columns={"user_genome": "MAG_ID"})
    taxon_data = gtdb_df["classification"].apply(parse_classification)
    taxon_df = pd.DataFrame(taxon_data.tolist())
    gtdb_df = pd.concat([gtdb_df, taxon_df], axis=1)
    gtdb_df = gtdb_df.drop(columns=["classification"])
    return gtdb_df


def parse_classification(classification_str):
    """
    Parses a taxonomic classification string and returns a dictionary with taxonomic ranks.

    Parameters:
        classification_str (str): A semicolon-separated string containing taxonomic classifications
                                  with prefixes (e.g., "d__Bacteria;p__Proteobacteria;c__Gammaproteobacteria").

    Returns:
        dict: A dictionary where keys are taxonomic ranks (e.g., "domain", "phylum", "class", etc.)
              and values are the corresponding names from the classification string. If a rank is
              not specified in the input string, its value will be "unclassified".
    """
    # Define taxonomic ranks and prefixes
    taxonomic_ranks = [
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    rank_prefixes = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]
    # Set all taxonomic ranks to unclassified
    taxon_dict = {rank: "unclassified" for rank in taxonomic_ranks}
    taxa = classification_str.split(";")
    for taxon in taxa:
        for prefix, rank in zip(rank_prefixes, taxonomic_ranks):
            if taxon.startswith(prefix):
                # remove the prefix and get the name
                name = taxon.replace(prefix, "").strip()
                if name == "":
                    name = "unclassified"
                taxon_dict[rank] = name
                # Exit the loop to avoid further unnecessary iterations
                break
    return taxon_dict


def get_scores(df, group_by_column="MAG_ID", p_value_threshold=0.05):

    allowed_columns = [
        "MAG_ID",
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    if group_by_column not in allowed_columns:
        raise ValueError(f"Invalid group_by_column. Must be one of {allowed_columns}")

    test_columns_dict = extract_test_columns(df)
    # Compute significance scores for all tests and merge them
    merged_results = calculate_score(
        df, test_columns_dict, group_by_column, p_value_threshold
    )

    # Determine which taxonomic columns to include
    if group_by_column == "MAG_ID":
        relevant_taxa_columns = allowed_columns
    else:
        allowed_without_mag = [c for c in allowed_columns if c != "MAG_ID"]
        index_of_group_by = allowed_without_mag.index(group_by_column)
        # Include group_by_column
        relevant_taxa_columns = allowed_without_mag[: index_of_group_by + 1]

    # Remove duplicates and keep relevant taxonomic information
    group_taxonomy = df[relevant_taxa_columns].drop_duplicates()

    # Merge taxonomy into the final results
    final_table = pd.merge(
        merged_results, group_taxonomy, on=group_by_column, how="left"
    )

    # Reorder columns: taxonomy columns first, then total_sites_per_group, and all test columns
    test_total_cols = [
        c for c in final_table.columns if c.startswith("total_sites_per_group_")
    ]
    test_sig_cols = [
        col
        for col in final_table.columns
        if col.startswith("significant_sites_per_group_")
    ]
    test_score_cols = [col for col in final_table.columns if col.startswith("score_")]

    columns_order = (
        relevant_taxa_columns + test_total_cols + test_sig_cols + test_score_cols
    )
    final_table = final_table[columns_order]

    # Sort by one of the score columns, if desired. For example, by the first test score:
    if test_score_cols:
        final_table = final_table.sort_values(
            by=test_score_cols[0], ascending=False
        ).reset_index(drop=True)

    final_table["grouped_by"] = group_by_column

    return final_table


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Group MAGs and calculate significance score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--gtdb_taxonomy",
        help="GTDB-Tk taxonomy file (gtdbtk.bac120.summary.tsv).",
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
        "--group_by_column",
        help="Column to group the MAGs by.",
        type=str,
        choices=[
            "MAG_ID",
            "domain",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ],
        default="MAG_ID",
    )

    parser.add_argument(
        "--pValue_threshold",
        help="p-value threshold to use.",
        default=0.05,
        metavar="float",
        type=float,
    )

    parser.add_argument(
        "--out_fPath",
        help="Path to output file.",
        type=str,
        metavar="filepath",
        default="significant_score_<group_by_column>.tsv",
    )

    args = parser.parse_args()
    logging.info("Parsing GTDB taxa table.")
    gtdb_df = read_gtdb(args.gtdb_taxonomy)

    logging.info("Reading p-value table.")
    pValue_table = pd.read_csv(args.pValue_table, sep="\t")
    if "MAG_ID" not in pValue_table.columns:
        pValue_table["MAG_ID"] = pValue_table["contig"].str.split(".fa").str[0]

    logging.info("Merging p-value table with GTDB taxonomy.")
    merged_df = pd.merge(pValue_table, gtdb_df, on="MAG_ID", how="left")

    logging.info("Calculating significance score.")
    final_table = get_scores(merged_df, args.group_by_column, args.pValue_threshold)
    if not args.out_fPath:
        baseDir = os.path.dirname(args.pValue_table)
        outFpath = os.path.join(baseDir, f"significant_taxa_{args.group_by_column}.tsv")
    else:
        outFpath = args.out_fPath

    logging.info(f"Writing results to file: {outFpath}")
    final_table.to_csv(outFpath, sep="\t", index=False)


if __name__ == "__main__":
    main()
