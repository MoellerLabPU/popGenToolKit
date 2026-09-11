#!/usr/bin/env python3
"""
Prepare and combine metadata tables for the allele tracking workflow.

This script reads an input metadata table, standardizes its columns
(e.g., 'order' -> 'day'), and adds a 'sample_profile_dir' column
based on a provided base directory.

It safely appends the cleaned data to an output metadata file,
allowing you to combine multiple disparate metadata tables into one
clean file ready for 'track_allele_frequencies.py'.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def load_input_metadata(metadata_in_path):
    """Load and return the input metadata DataFrame.

    Args:
        metadata_in_path (Path): Path to the input metadata file.

    Returns:
        pd.DataFrame: The loaded metadata DataFrame.
    """
    logger.info(f"Loading input metadata from: {metadata_in_path}")
    df = pd.read_csv(metadata_in_path, sep="\t")
    return df


def validate_and_rename_columns(
    df, sample_col, group_col, time_col, day_col, replicate_col, subject_col
):
    """Validate columns exist and rename to standard format.

    Args:
        df (pd.DataFrame): Input metadata DataFrame.
        sample_col (str): Name of the sample ID column.
        group_col (str): Name of the group column.
        time_col (str): Name of the timepoint column.
        day_col (str): Name of the day/order column (optional).
        replicate_col (str): Name of the replicate column (optional).
        subject_col (str): Name of the subject ID column.

    Returns:
        pd.DataFrame: DataFrame with standardized column names.
    """
    # Required columns
    required_map = {
        sample_col: "sample_id",
        group_col: "group",
        time_col: "time",
        subject_col: "subjectID",
    }

    # Optional columns
    optional_map = {
        day_col: "day",
        replicate_col: "replicate",
    }

    # Check required columns
    missing_required = set(required_map.keys()) - set(df.columns)
    if missing_required:
        logger.error(
            f"Input file is missing required columns: {missing_required}. "
            f"Check your --sample-col, --group-col, etc. arguments."
        )
        sys.exit(1)

    # Build final column map with available optional columns
    column_map = required_map.copy()
    for orig_col, new_col in optional_map.items():
        if orig_col in df.columns:
            column_map[orig_col] = new_col
            logger.info(
                f"Optional column '{orig_col}' found and will be included as '{new_col}'"
            )
        else:
            logger.info(f"Optional column '{orig_col}' not found in input file")

    # Rename columns to the standard format
    df = df.rename(columns=column_map)
    return df


def add_sample_profile_dir(df, base_profile_dir):
    """Add the sample_profile_dir column.

    Args:
        df (pd.DataFrame): Input metadata DataFrame.
        base_profile_dir (str): Base directory path for sample profiles.

    Returns:
        pd.DataFrame: DataFrame with added sample_profile_dir column.
    """
    # Ensure no trailing slash
    base_dir = base_profile_dir.rstrip("/")
    df["sample_profile_dir"] = df["sample_id"].apply(lambda s: f"{base_dir}/{s}")
    logger.info(f"Added 'sample_profile_dir' column for {len(df)} samples.")
    return df


def load_existing_output(output_path, final_cols):
    """Load existing output file if it exists.

    Args:
        output_path (Path): Path to the output metadata file.
        final_cols (list): List of final column names.

    Returns:
        tuple: (existing_df, existing_samples) where existing_samples is the set of sample IDs.
    """
    existing_df = pd.DataFrame(columns=final_cols)

    if output_path.exists():
        logger.info(f"Loading existing output file to append to: {output_path}")
        existing_df = pd.read_csv(output_path, sep="\t")

    return existing_df


def combine_and_save_metadata(existing_df, new_df, output_path, final_cols):
    """Combine DataFrames and save to file.

    Args:
        existing_df (pd.DataFrame): Existing metadata DataFrame.
        new_df (pd.DataFrame): New metadata DataFrame.
        output_path (Path): Path to save the combined metadata.
        final_cols (list): List of final column names.

    Returns:
        pd.DataFrame: The final combined DataFrame.
    """
    # Select only the columns we need
    new_df = new_df[final_cols]

    # Check for duplicate sample_ids
    new_samples = set(new_df["sample_id"])
    existing_samples = set(existing_df["sample_id"])
    duplicates = new_samples.intersection(existing_samples)

    if duplicates:
        raise ValueError(f"Found {len(duplicates)} duplicate sample_ids that are ")

    # Combine DataFrames
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Drop any exact duplicates that might have arisen from re-running
    combined_df = combined_df.drop_duplicates()

    # Save to file
    combined_df.to_csv(output_path, sep="\t", index=False)
    logger.info(
        f"Successfully saved combined metadata ({len(combined_df)} total rows) "
        f"to: {output_path}"
    )

    return combined_df


def main():
    """Main function for the metadata preparation CLI."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Clean, standardize, and combine metadata tables for allele tracking."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metadata-in",
        help="Path to the input metadata table to process.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--metadata-out",
        help=(
            "Path to the final, combined metadata file. "
            "Data will be appended if this file already exists."
        ),
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base-profile-dir",
        help=(
            "The base directory containing sample profile subdirectories "
            "(e.g., '/path/to/run/profiles')."
        ),
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sample-col",
        help="The name of the sample ID column in the input file.",
        default="sample_id",
    )
    parser.add_argument(
        "--group-col",
        help="The name of the group column in the input file.",
        default="group",
    )
    parser.add_argument(
        "--time-col",
        help="The name of the timepoint column in the input file.",
        default="time",
    )
    parser.add_argument(
        "--day-col",
        help="The name of the day/order column in the input file (optional).",
        default="day",
    )
    parser.add_argument(
        "--replicate-col",
        help="The name of the replicate column in the input file (optional).",
        default="replicate",
    )
    parser.add_argument(
        "--subject-col",
        help="The name of the subject ID column in the input file (required).",
        default="subjectID",
    )

    args = parser.parse_args()

    # Input file validation
    if not args.metadata_in.exists():
        logger.error(f"Input file not found: {args.metadata_in}")
        sys.exit(1)

    # 1. Load input metadata
    df = load_input_metadata(args.metadata_in)

    # 2. Validate and rename columns
    df = validate_and_rename_columns(
        df,
        args.sample_col,
        args.group_col,
        args.time_col,
        args.day_col,
        args.replicate_col,
        args.subject_col,
    )

    # Define final columns based on what's present in the DataFrame
    final_cols = ["sample_id", "group", "time", "subjectID"]
    if "day" in df.columns:
        final_cols.append("day")
    if "replicate" in df.columns:
        final_cols.append("replicate")
    final_cols.append("sample_profile_dir")  # Always last

    # 3. Add sample_profile_dir column
    df = add_sample_profile_dir(df, args.base_profile_dir)

    # 4. Load existing output file
    existing_df = load_existing_output(args.metadata_out, final_cols)

    # 5. Combine and save metadata
    combine_and_save_metadata(existing_df, df, args.metadata_out, final_cols)


if __name__ == "__main__":
    main()
