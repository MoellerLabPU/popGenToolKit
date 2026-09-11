#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd

from alleleflux.scripts.utilities.utilities import extract_relevant_columns
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def taxa_scores(df, group_col):
    columns_dict = extract_relevant_columns(df, capture_str="sites_per_group_")
    logger.info(f"Calculating scores for {len(columns_dict)} tests.")

    taxonomy_order = [
        "domain",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    if group_col not in taxonomy_order:
        raise ValueError(
            f"Grouping column {group_col} not found in taxonomy_order list."
        )
    # Get all higher-level ranks above the chosen one
    index_of_group_by = taxonomy_order.index(group_col)

    # For ranks above the chosen one, take all columns that come before it
    relevant_taxa_columns = taxonomy_order[: index_of_group_by + 1]

    agg_dict = {}
    for test_name, cols in columns_dict.items():
        # Find the total and significant columns for this test
        total_col = [c for c in cols if c.startswith("total_sites_per_group_")]
        significant_col = [
            c for c in cols if c.startswith("significant_sites_per_group_")
        ]

        if len(total_col) != 1 or len(significant_col) != 1:
            raise ValueError(
                f"Expected exactly one total and one significant column for test '{test_name}', "
                f"found {total_col} and {significant_col}."
            )

        agg_dict[total_col[0]] = "sum"
        agg_dict[significant_col[0]] = "sum"

    # Count how many MAGs per group
    agg_dict["MAG_ID"] = "nunique"

    # Perform the aggregation
    grouped = df.groupby(group_col).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={"MAG_ID": "unique_MAGs"})

    # Compute the new scores for each test
    for test_name, cols in columns_dict.items():
        total_col = [c for c in cols if c.startswith("total_sites_per_group_")][0]
        significant_col = [
            c for c in cols if c.startswith("significant_sites_per_group_")
        ][0]

        new_score_col = f"score_{test_name} (%)"
        grouped[new_score_col] = (grouped[significant_col] / grouped[total_col]) * 100

    group_taxonomy = df[relevant_taxa_columns].drop_duplicates(subset=group_col)

    final_table = pd.merge(group_taxonomy, grouped, on=group_col, how="left")
    final_table["grouped_by"] = group_col
    return final_table


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Group a table by a specified taxonomic level and calculate new scores, retaining higher-level taxonomy columns."
    )
    parser.add_argument(
        "--input_df", required=True, help="Path to the input dataframe."
    )
    parser.add_argument(
        "--group_by_column",
        choices=["domain", "phylum", "class", "order", "family", "genus", "species"],
        help="Taxonomic column to group by.",
        default="genus",
    )
    parser.add_argument(
        "--out_fPath",
        help="Path to the output TSV file.",
        default="grouped_scores_<group_by_column>.tsv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_df, sep="\t")
    grouped_scores = taxa_scores(df, args.group_by_column)

    if not args.out_fPath:
        baseDir = os.path.dirname(args.input_df)
        outFpath = os.path.join(baseDir, f"grouped_scores_{args.group_by_column}.tsv")
    else:
        outFpath = args.out_fPath

    logger.info(f"Writing results to file: {outFpath}")
    grouped_scores.to_csv(outFpath, sep="\t", index=False)


if __name__ == "__main__":
    main()
