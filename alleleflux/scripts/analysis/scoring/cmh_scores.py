#!/usr/bin/env python3
"""
calculate_cmh_scores.py

Calculate differential significance scores from CMH test outputs.

This script:
  - Loads CMH results from either a single combined file or two separate files for different timepoints.
  - Computes the intersection of contig-position pairs across both timepoints (denominator).
  - Identifies sites significant in the specified focus timepoint but not in the other (numerator).
  - Calculates and reports the score = numerator / denominator.
  - Optionally writes the list of differential sites to a file.

Usage with separate files:
  python calculate_cmh_scores.py \
    --tp1-file path/to/mag_cmh_time1.tsv.gz --tp1-name time1 \
    --tp2-file path/to/mag_cmh_time2.tsv.gz --tp2-name time2 \
    --focus time1 [--threshold 0.05] [--output-sites diff_sites.tsv]

Usage with combined file:
  python calculate_cmh_scores.py \
    --combined-file path/to/mag_cmh_combined.tsv.gz \
    --tp1-name time1 --tp2-name time2 \
    --focus time1 [--threshold 0.05] [--output-sites diff_sites.tsv]

Note: When using --combined-file, the file must contain a 'time' column with values
      matching the specified timepoint names (--tp1-name and --tp2-name).
"""
import argparse
import logging

import pandas as pd

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import read_gtdb

logger = logging.getLogger(__name__)


def load_cmh_results(file_path, timepoint=None):
    """
    Load CMH results from TSV or TSV.GZ file into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CMH results file.
        timepoint (str, optional): If provided, will filter the dataframe to this timepoint value.
            Used when loading from a combined file with multiple timepoints.

    Returns:
        pd.DataFrame: DataFrame containing CMH results, filtered by timepoint if specified.
    """
    logger.info(f"Loading CMH results from {file_path}")
    df = pd.read_csv(file_path, sep="\t")

    if timepoint is not None:
        if "time" in df.columns:
            available_timepoints = df["time"].unique().tolist()
            logger.info(f"Filtering results to timepoint: {timepoint}")
            df = df[df["time"] == timepoint].copy()
            if df.empty:
                logger.warning(
                    f"No data found for timepoint '{timepoint}' in file {file_path}. "
                    f"Available timepoints: {available_timepoints}"
                )
        else:
            logger.warning(
                f"Timepoint '{timepoint}' specified but 'time' column not found in {file_path}"
            )

    return df


def get_common_sites(df1, df2):
    """Return set of contig-position pairs common to both DataFrames."""
    logger.info("Identifying common sites between two dataframes")
    set1 = set(zip(df1["contig"], df1["position"]))
    set2 = set(zip(df2["contig"], df2["position"]))
    return set1.intersection(set2)


def get_focus_and_other(args, df1, df2):
    """Determine focus and other DataFrames based on args.focus."""
    if args.focus == args.tp1_name:
        logger.info(f"Focus timepoint set to '{args.tp1_name}'")
        return df1, df2
    else:
        logger.info(f"Focus timepoint set to '{args.tp2_name}'")
        return df2, df1


def compute_diff_significance(focus_df, other_df, intersect_set, threshold):
    """
    Compute the number of differential significant sites between two dataframes based on a p-value threshold.

    This function identifies positions (defined by 'contig' and 'position') that are significant in the focus dataframe
    but not in the other dataframe, considering only those positions present in the provided intersection set.

    Parameters:
        focus_df (pd.DataFrame): DataFrame containing 'contig', 'position', and 'p_value_CMH' columns for the focus timepoint.
        other_df (pd.DataFrame): DataFrame containing 'contig', 'position', and 'p_value_CMH' columns for the comparison timepoint.
        intersect_set (set): Set of (contig, position) tuples representing the intersection of positions to consider.
        threshold (float): P-value threshold for determining significance.

    Returns:
        int: The number of positions that are significant in the focus dataframe, not significant in the other dataframe, and present in the intersection set.
    """

    logger.info(
        f"Identifying differential significant sites with threshold {threshold}"
    )
    # Identify significant positions for focus timepoint
    mask_focus = focus_df["p_value_CMH"] < threshold
    sig_focus = set(
        zip(focus_df.loc[mask_focus, "contig"], focus_df.loc[mask_focus, "position"])
    )
    # Identify significant positions for other timepoint
    mask_other = other_df["p_value_CMH"] < threshold
    sig_other = set(
        zip(other_df.loc[mask_other, "contig"], other_df.loc[mask_other, "position"])
    )
    diff_sig = (sig_focus - sig_other) & intersect_set
    return len(diff_sig)


def merge_with_taxonomy(score_df, gtdb_path):
    """Merge scores DataFrame with taxonomy information from GTDB."""
    logger.info(f"Merging score DataFrame with taxonomy from {gtdb_path}")
    gtdb_df = read_gtdb(gtdb_path)
    return pd.merge(score_df, gtdb_df, on="MAG_ID", how="left")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Compute CMH differential significance scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create a mutually exclusive group for input file options
    input_group = parser.add_mutually_exclusive_group(required=True)

    # Option 1: Combined file with both timepoints
    input_group.add_argument(
        "--combined-file",
        help="Path to combined CMH output file containing data for both timepoints. "
        "The file must contain a 'time' column with values matching --tp1-name and --tp2-name.",
    )

    # Option 2: Separate files for each timepoint (existing behavior)
    input_group.add_argument("--tp1-file", help="Path to CMH output for timepoint 1")

    parser.add_argument(
        "--tp1-name",
        required=True,
        help="Name identifier for timepoint 1 (not required when using --combined-file)",
    )
    parser.add_argument(
        "--tp2-file",
        help="Path to CMH output for timepoint 2 (not required when using --combined-file)",
    )
    parser.add_argument(
        "--tp2-name", required=True, help="Name identifier for timepoint 2"
    )
    parser.add_argument(
        "--focus",
        required=True,
        help="Timepoint to consider for differential significance (must match --tp1-name or --tp2-name)",
    )
    parser.add_argument(
        "--gtdb_taxonomy",
        help="GTDB-Tk taxonomy file (gtdbtk.bac120.summary.tsv).",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--mag_id",
        help="MAG ID to process. ",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="P-value threshold for significance (default: 0.05)",
    )
    parser.add_argument(
        "--out_fPath",
        help="Path to output file. Default is `CMH_score.tsv`.",
        type=str,
        metavar="filepath",
        default="CMH_score.tsv",
    )
    args = parser.parse_args()

    # Validate input arguments based on mode
    if not args.combined_file and (not args.tp1_file or not args.tp2_file):
        parser.error(
            "When not using --combined-file, both --tp1-file and --tp2-file must be provided."
        )

    # Validate focus
    if args.focus not in {args.tp1_name, args.tp2_name}:
        raise ValueError(
            f"Error: focus '{args.focus}' must be one of '{{args.tp1_name}}','{{args.tp2_name}}'",
        )

    # Load CMH results into DataFrames based on input mode
    if args.combined_file:
        # Using a combined file with 'time' column
        logger.info(f"Using combined file mode: {args.combined_file}")
        df1 = load_cmh_results(args.combined_file, timepoint=args.tp1_name)
        df2 = load_cmh_results(args.combined_file, timepoint=args.tp2_name)
    else:
        # Using separate files for each timepoint
        if not args.tp1_file or not args.tp2_file:
            raise ValueError(
                "Both --tp1-file and --tp2-file must be provided when not using --combined-file"
            )

        logger.info(f"Using separate files mode: {args.tp1_file} and {args.tp2_file}")
        df1 = load_cmh_results(args.tp1_file)
        df2 = load_cmh_results(args.tp2_file)

    # Check if either timepoint returned empty results
    if df1.empty or df2.empty:
        empty_tps = []
        if df1.empty:
            empty_tps.append(args.tp1_name)
        if df2.empty:
            empty_tps.append(args.tp2_name)
        logger.warning(
            f"No CMH results for timepoint(s) {empty_tps} for MAG '{args.mag_id}'. "
            f"Writing score=0 output."
        )
        total_sites = 0
        significant_sites = 0
        score = 0.0
    else:
        # Identify positions common to both timepoints
        intersect_set = get_common_sites(df1, df2)
        logger.info(
            f"Identified {len(intersect_set):,} common positions between timepoints."
        )
        total_sites = len(intersect_set)

        if total_sites == 0:
            logger.warning(
                f"No common positions found between timepoints for MAG '{args.mag_id}'. "
                f"Writing score=0 output."
            )
            significant_sites = 0
            score = 0.0
        else:
            # Determine which DataFrame is focus and which is other
            focus_df, other_df = get_focus_and_other(args, df1, df2)

            # Compute differential significance
            significant_sites = compute_diff_significance(
                focus_df, other_df, intersect_set, args.threshold
            )

            # Calculate score as percentage
            score = (significant_sites / total_sites) * 100

    logger.info(
        f"Differential significance score for '{args.focus}': {significant_sites}/{total_sites} = {score:.4f}"
    )

    # Build result DataFrame
    score_df = pd.DataFrame(
        {
            "MAG_ID": args.mag_id,
            "focus_timepoint": args.focus,
            "significant_sites_per_group_CMH": significant_sites,
            "total_sites_per_group_CMH": total_sites,
            "score_CMH (%)": score,
            "grouped_by": "MAG_ID",
        },
        index=[0],
    )

    merged_df = merge_with_taxonomy(score_df, args.gtdb_taxonomy)
    logger.info(f"Writing results to {args.out_fPath}")
    merged_df.to_csv(args.out_fPath, sep="\t", index=False)

    logger.info("Results successfully written.")


if __name__ == "__main__":
    main()
