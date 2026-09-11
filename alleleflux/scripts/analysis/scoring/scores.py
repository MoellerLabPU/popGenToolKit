#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd

from alleleflux.scripts.utilities.utilities import (
    calculate_score,
    extract_mag_id,
    extract_relevant_columns,
    load_mag_mapping,
    read_gtdb,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


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

    # Extract relevant columns
    test_columns_dict = extract_relevant_columns(df, capture_str="p_value_")

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

    final_table = reorder_columns(final_table, relevant_taxa_columns, group_by_column)

    return final_table


def reorder_columns(final_table, relevant_taxa_columns, group_by_column):
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
    setup_logging()

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
        "--mag_mapping_file",
        help="Path to tab-separated file mapping contig names to MAG IDs. "
        "Must have columns 'contig_name' and 'mag_id'.",
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
        "--out_fPath",
        help="Path to output file. Default is `significant_score_<group_by_column>.tsv`.",
        type=str,
        metavar="filepath",
    )

    args = parser.parse_args()
    logger.info("Parsing GTDB taxa table.")
    gtdb_df = read_gtdb(args.gtdb_taxonomy)

    logger.info("Reading p-value table.")
    pValue_table = pd.read_csv(args.pValue_table, sep="\t")

    if pValue_table.empty:
        raise ValueError("Input p-value table is empty.")

    if "MAG_ID" not in pValue_table.columns:
        mag_mapping_dict = load_mag_mapping(args.mag_mapping_file)
        for contig in pValue_table["contig"]:
            pValue_table["MAG_ID"] = extract_mag_id(contig, mag_mapping_dict)
        # pValue_table["MAG_ID"] = pValue_table["contig"].str.split(".fa").str[0]

    logger.info("Merging p-value table with GTDB taxonomy.")
    merged_df = pd.merge(pValue_table, gtdb_df, on="MAG_ID", how="left")

    logger.info("Calculating significance score.")
    final_table = get_scores(merged_df, args.group_by_column, args.pValue_threshold)

    if not args.out_fPath:
        baseDir = os.path.dirname(args.pValue_table)
        outFpath = os.path.join(baseDir, f"significant_taxa_{args.group_by_column}.tsv")
    else:
        outFpath = args.out_fPath

    logger.info(f"Writing results to file: {outFpath}")
    final_table.to_csv(outFpath, sep="\t", index=False)


if __name__ == "__main__":
    main()
