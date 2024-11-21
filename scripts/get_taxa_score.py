#!/usr/bin/env python
import argparse
import logging
import os

import pandas as pd


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


def calculate_significant_scores(df, group_by_column="MAG_ID", p_value_threshold=0.05):
    """
    Calculate the percentage of significant sites per taxonomic group.

    This function identifies significant sites based on p-value thresholds and
    calculates the percentage of significant sites for each group specified by
    the `group_by_column`. The results are returned in a DataFrame that includes
    taxonomic information and the calculated scores.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data with p-value columns and taxonomic information.
    group_by_column (str): The column name to group by. Must be one of the allowed taxonomic columns.
                           Default is "MAG_ID".
    p_value_threshold (float): The threshold for determining significance. Default is 0.05.

    Returns:
    pd.DataFrame: A DataFrame containing the taxonomic information and the percentage of significant
                  sites for each group, sorted by the score in descending order. The DataFrame also
                  includes a column indicating the grouping column used.

    Raises:
    ValueError: If the `group_by_column` is not one of the allowed taxonomic columns.
    """

    # Ensure the group_by_column is one of the allowed columns
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

    df_merged = df.copy()
    # Identify significant sites
    df_merged["is_significant"] = (
        df_merged[p_value_columns].lt(p_value_threshold).any(axis=1)
    )

    # Group by the specified column
    group = df_merged.groupby(group_by_column)

    # Total number of sites per group
    total_sites_per_group = group.size()

    # Number of significant sites per group
    significant_sites_per_group = group["is_significant"].sum()

    # Calculate percentage of significant sites per group
    percentage_significant = (significant_sites_per_group / total_sites_per_group) * 100

    # Create a DataFrame with the results
    # group_scores = percentage_significant.reset_index(name="score")
    group_scores = pd.DataFrame(
        {
            group_by_column: total_sites_per_group.index,
            "total_sites_per_group": total_sites_per_group.values,
            "significant_sites_per_group": significant_sites_per_group.values,
            "score": percentage_significant.values,
        }
    )

    if group_by_column == "MAG_ID":
        # If grouping by MAG_ID, include all taxonomic columns
        group_taxonomy = df_merged[allowed_columns].drop_duplicates()
        relevant_taxa_columns = allowed_columns
    else:
        # Determine taxonomic columns up to the group_by_column
        allowed_columns.remove("MAG_ID")
        index_of_group_by = allowed_columns.index(group_by_column)
        relevant_taxa_columns = allowed_columns[
            : index_of_group_by + 1
        ]  # Include group_by_column

        # Remove duplicates and keep relevant taxonomic info
        group_taxonomy = df_merged[relevant_taxa_columns].drop_duplicates()
    # Merge the scores with taxonomic information
    final_table = pd.merge(
        group_scores, group_taxonomy, on=group_by_column, how="inner"
    )

    # Reorder columns for clarity
    # columns_order = relevant_taxa_columns + ["score"]
    columns_order = relevant_taxa_columns + [
        "total_sites_per_group",
        "significant_sites_per_group",
        "score",
    ]
    final_table = final_table[columns_order]

    # Sort the table by score in descending order
    final_table = final_table.sort_values(by="score", ascending=False).reset_index(
        drop=True
    )
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
        "--output_dir",
        help="Path to output file.",
        type=str,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()
    logging.info("Parsing GTDB taxa table.")
    gtdb_df = read_gtdb(args.gtdb_taxonomy)
    logging.info("Reading p-value table.")
    pValue_table = pd.read_csv(
        args.pValue_table,
        sep="\t",
        usecols=[
            "MAG_ID",
            "A_frequency_p_value",
            "T_frequency_p_value",
            "G_frequency_p_value",
            "C_frequency_p_value",
        ],
    )

    logging.info("Merging p-value table with GTDB taxonomy.")
    merged_df = pd.merge(pValue_table, gtdb_df, on="MAG_ID", how="inner")

    # Make MAG_ID as the first column
    cols = merged_df.columns.tolist()

    # Remove 'MAG_ID' from the list
    cols.remove("MAG_ID")

    # Place 'MAG_ID' at the beginning
    cols = ["MAG_ID"] + cols

    # Reindex the DataFrame
    merged_df = merged_df[cols]
    logging.info("Calculating significance score.")
    final_table = calculate_significant_scores(
        merged_df, args.group_by_column, args.pValue_threshold
    )
    outFpath = os.path.join(
        args.output_dir, f"significant_taxa_{args.group_by_column}.tsv"
    )
    logging.info("Writing results to file.")
    final_table.to_csv(outFpath, sep="\t", index=False)


if __name__ == "__main__":
    main()
