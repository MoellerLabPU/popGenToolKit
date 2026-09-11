import argparse
import json
import logging
import os

import pandas as pd

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import load_allele_freq_inputs

# Constants
NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]

logger = logging.getLogger(__name__)


def preprocess_data(
    df,
    max_zero_count,
    output_dir,
    mag_id,
    group,
    min_positions=1,
    min_sample_num=4,
    status_dir=None,
):
    """
    Preprocess data by removing positions where more than max_zero_count replicates
    have _diff_mean value zero for all nucleotides.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing nucleotide frequency data
    max_zero_count : int
        Maximum allowed replicates per position where all nucleotide_diff_mean values are zero
    output_dir : str
        Directory to save the processed data
    mag_id : str
        MAG identifier
    group : str
        Group identifier
    min_positions : int
        Minimum number of eligible positions required after filtering for eligibility
    min_sample_num : int
        Minimum number of samples per position required for the position to be considered eligible
    status_dir : str or None
        Directory to write status file. If None, uses output_dir.

    Returns:
    --------
    dict
        Status dictionary with row counts and eligibility flag
    """
    logger.info(
        f"Preprocessing: Removing positions where more than {max_zero_count} replicates have _diff_mean value zero for all nucleotides."
    )

    input_rows = len(df)

    # List the columns we want to check.
    diff_mean_cols = [f"{nuc}_diff_mean" for nuc in NUCLEOTIDES]

    # For each replicate, check if all nucleotide diff_mean columns are zero.
    df["all_zeros"] = (df[diff_mean_cols] == 0).all(axis=1)

    # Group by position (using contig and position) and count replicates with all zeros.
    # No need to include gene_id here as we are not using that again for the final dataframe as with the significance tests
    zeros_count = df.groupby(["contig", "position"], dropna=False)["all_zeros"].sum()

    # Identify positions where the number of replicates with all zeros exceeds the threshold.
    positions_to_remove = zeros_count[zeros_count > max_zero_count].index
    logger.info(
        f"Removing {len(positions_to_remove):,} positions based on zero count threshold."
    )

    # Remove these positions from the dataframe.
    df = df.set_index(["contig", "position"])
    df = df.drop(positions_to_remove).reset_index()

    # Cleanup temporary column
    df = df.drop(columns=["all_zeros"])

    output_rows = len(df)

    # Calculate position-level eligibility: count positions with >= min_sample_num samples
    total_positions = 0
    positions_eligible = 0
    if output_rows > 0:
        position_grouped = df.groupby(["contig", "position"], dropna=False)
        total_positions = len(position_grouped)
        for (contig, position), pos_df in position_grouped:
            num_samples = len(pos_df)
            if num_samples >= min_sample_num:
                positions_eligible += 1
        logger.info(
            f"Position-level eligibility: {total_positions:,} total positions, "
            f"{positions_eligible:,} have >= {min_sample_num} samples"
        )

    eligible = positions_eligible >= min_positions

    df.to_csv(
        os.path.join(
            output_dir,
            f"{mag_id}_{group}_allele_frequency_changes_mean_zeros_processed.tsv.gz",
        ),
        index=False,
        sep="\t",
        compression="gzip",
    )

    # Write preprocessing status file
    status = {
        "mag_id": mag_id,
        "preprocess_type": "within_groups",
        "group": group,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "min_positions": min_positions,
        "min_sample_num": min_sample_num,
        "total_positions": total_positions,
        "positions_eligible": positions_eligible,
        "eligible": eligible,
        "max_zero_count": max_zero_count,
    }

    # Determine status file location
    if status_dir:
        os.makedirs(status_dir, exist_ok=True)
        status_file = os.path.join(
            status_dir, f"{mag_id}_{group}_preprocessing_status.json"
        )
    else:
        status_file = os.path.join(
            output_dir, f"{mag_id}_{group}_preprocessing_status.json"
        )

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    logger.info(f"Preprocessing status written to {status_file}")

    return status


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Script to preprocess within group data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--df_fPath",
        required=True,
        help="Path to mean changes dataframe",
        type=str,
    )
    parser.add_argument(
        "--max_zero_count",
        type=int,
        default=4,
        help="Maximum allowed replicates per position where all nucleotide_diff_mean values are zero. Positions where the count of such replicates exceeds this threshold are removed.",
    )
    parser.add_argument(
        "--mag_id",
        help="MAG ID to process",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--group",
        help="Group to perform tests on",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--min_positions",
        type=int,
        default=1,
        help="Minimum number of eligible positions required after filtering for a MAG to be considered eligible",
    )
    parser.add_argument(
        "--min_sample_num",
        type=int,
        default=4,
        help="Minimum number of samples per position required for the position to be considered eligible",
    )
    parser.add_argument(
        "--status_dir",
        type=str,
        default=None,
        help="Directory to write preprocessing status file. If not provided, status file is written to the same directory as output.",
    )

    args = parser.parse_args()

    input_df = load_allele_freq_inputs(args.df_fPath)

    # Filter the dataframe to only include positions from the specified group.
    input_df = input_df[input_df["group"] == args.group]
    logger.info(
        f"Data filtered for group '{args.group}', resulting in {input_df.shape[0]:,} rows."
    )

    os.makedirs(args.output_dir, exist_ok=True)

    preprocess_data(
        df=input_df,
        max_zero_count=args.max_zero_count,
        output_dir=args.output_dir,
        mag_id=args.mag_id,
        group=args.group,
        min_positions=args.min_positions,
        min_sample_num=args.min_sample_num,
        status_dir=args.status_dir,
    )


if __name__ == "__main__":
    main()
