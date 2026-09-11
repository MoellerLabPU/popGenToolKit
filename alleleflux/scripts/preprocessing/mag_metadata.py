import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def find_mag_files(root_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Finds and organizes MAG (Metagenome-Assembled Genome) files in a given root directory.

    This function iterates over subdirectories in the specified root directory. Each subdirectory
    is treated as a sample directory. Within these subdirectories, it searches for files
    that match the pattern '{sample_id}_{MAG_ID}_profiled.tsv.gz'. It extracts the sample
    ID from the subdirectory name and the MAG ID from the filename, and stores this information
    in a dictionary.

    Parameters:
        root_dir (str): The path to the root directory containing the subdirectories with MAG files.

    Returns:
        dict: A dictionary where the keys are MAG IDs and the values are lists of dictionaries,
              each containing 'sample_id' and 'file_path' keys corresponding to the sample ID and
              the absolute path to the MAG file, respectively.

    Example:
        >>> mag_files = find_mag_files("/path/to/root_dir")
        >>> print(mag_files)
        {
            'MAG1': [{'sample_id': 'sample1', 'file_path': '/path/to/root_dir/sample1/sample1_MAG1_profiled.tsv.gz'}],
            'MAG2': [{'sample_id': 'sample2', 'file_path': '/path/to/root_dir/sample2/sample2_MAG2_profiled.tsv.gz'}]
        }
    """
    mag_dict = defaultdict(list)
    root_path = Path(root_dir)

    if not root_path.is_dir():
        logger.error(f"Root directory not found: {root_dir}")
        return {}

    logger.info(f"Scanning for MAG files in: {root_path}")
    for subdir_path in root_path.iterdir():
        if not subdir_path.is_dir():
            continue

        sample_id = subdir_path.name
        logger.info(f"Looking for MAGs in sample directory: {sample_id}")

        # Define the prefix and suffix for slicing based on the sample_id
        prefix = f"{sample_id}_"
        suffix = "_profiled.tsv.gz"

        for file_path in subdir_path.glob(f"*_profiled.tsv.gz"):
            filename = file_path.name
            # Check if the filename starts with the expected sample_id prefix
            if filename.startswith(prefix):
                # Extract MAG ID using slicing, which is robust to underscores in sample_id
                mag_id = filename[len(prefix) : -len(suffix)]
                mag_dict[mag_id].append(
                    {"sample_id": sample_id, "file_path": str(file_path.resolve())}
                )
            else:
                logger.info(
                    f"Filename does not match expected sample prefix: {filename}. Skipped."
                )

    return mag_dict


def merge_metadata(
    mag_dict: Dict[str, List[Dict[str, str]]],
    metadata_file: str,
    timepoints: List[str],
    groups: List[str],
    data_type: str,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Merges metadata from a file into a dictionary of metagenome-assembled genomes (MAGs).

    This function now performs extensive validation:
    1.  Normalizes all relevant ID columns (strips whitespace, sets type).
    2.  Checks for duplicate sample_ids in the metadata.
    3.  Validates that user-provided groups and timepoints exist in the data.
    4.  Logs dropped samples and fails if any discovered sample is missing from metadata.


    Parameters:
        mag_dict (dict): A dictionary where keys are MAG IDs and values are lists of sample information dictionaries.
        metadata_file (str): Path to the metadata file in TSV format containing columns 'sample', 'subjectID', 'group', 'time'.
                             The 'replicate' column is optional.
        timepoints (list): List of timepoints to filter the metadata. Required for longitudinal data, optional for single data.
        groups (list): List of groups to filter the metadata.
        data_type (str): Type of analysis to perform, either 'single' or 'longitudinal'.

    Returns:
        dict: The updated mag_dict with metadata fields added to each sample information dictionary.
    """
    # --- 1. Load and Validate Metadata Structure ---
    # Load metadata
    metadata_df = pd.read_csv(
        metadata_file,
        sep="\t",
    )

    # Check for required columns except replicate (we'll handle it specially)
    required_cols = {"sample_id", "subjectID", "group", "bam_path"}
    # Conditionally add the 'time' column if the analysis involves timepoints.
    if data_type == "longitudinal" or (timepoints and data_type == "single"):
        required_cols.add("time")

    # Check if all required columns are present in the loaded DataFrame.
    missing_columns = required_cols - set(metadata_df.columns)
    if missing_columns:
        # If any columns are missing, raise an error with a helpful message.
        raise ValueError(
            f"Missing required columns in metadata: {sorted(list(missing_columns))}"
        )
    # --- 2. Normalize and Clean Data ---
    # This prevents subtle bugs from whitespace or incorrect data types.
    # Loop through key identifier columns to standardize them.
    for col in ["sample_id", "subjectID", "group"]:
        if col in metadata_df.columns:
            # Convert column to string type and remove leading/trailing whitespace.
            metadata_df[col] = metadata_df[col].astype(str).str.strip()

    # Specifically clean the 'time' column if it exists.
    if "time" in metadata_df.columns:
        metadata_df["time"] = metadata_df["time"].astype(str).str.strip()

    # If 'replicate' column doesn't exist, create one using subjectID values
    if "replicate" not in metadata_df.columns:
        logger.info(
            "No 'replicate' column found in metadata. Using 'subjectID' as replicate values."
        )
        metadata_df["replicate"] = metadata_df["subjectID"]
    # Ensure the 'replicate' column is also cleaned and standardized.
    metadata_df["replicate"] = metadata_df["replicate"].astype(str).str.strip()

    # --- 3. Fail-Fast Validations ---
    # Check for duplicate sample_ids, which can cause ambiguous merges.
    duplicates = metadata_df["sample_id"][
        metadata_df["sample_id"].duplicated()
    ].unique()
    if len(duplicates) > 0:
        raise ValueError(
            f"Duplicate sample_id rows found in metadata: {duplicates.tolist()}"
        )
    # Get the set of all unique sample IDs discovered by find_mag_files.
    discovered_samples = {s["sample_id"] for v in mag_dict.values() for s in v}
    metadata_samples = set(metadata_df["sample_id"])

    # Check if any discovered samples are completely missing from the metadata file.
    missing_from_metadata = discovered_samples - metadata_samples
    if missing_from_metadata:
        raise ValueError(
            f"{len(missing_from_metadata)} discovered samples were not found "
            f"in the metadata file. Examples: {sorted(list(missing_from_metadata))[:5]}"
        )
    # Check if user-provided filters actually exist in the metadata before proceeding.
    if groups:
        # Find which of the requested groups are not in the metadata's 'group' column.
        missing = set(groups) - set(metadata_df["group"].unique())
        if missing:
            raise ValueError(
                f"Requested groups not found in metadata: {sorted(list(missing))}"
            )

    if timepoints and "time" in metadata_df.columns:
        # Find which of the requested timepoints are not in the metadata's 'time' column.
        missing = set(timepoints) - set(metadata_df["time"].unique())
        if missing:
            raise ValueError(
                f"Requested timepoints not found in metadata: {sorted(list(missing))}"
            )

    # --- 4. Filter Metadata ---
    # Filter metadata based on data type
    if data_type == "longitudinal":
        # Filter metadata to only include the specified timepoints and groups
        metadata_df = metadata_df[
            metadata_df["time"].isin(timepoints) & metadata_df["group"].isin(groups)
        ]
    elif data_type == "single":  # single data type
        # Filter metadata to only include the specified groups
        metadata_df = metadata_df[metadata_df["group"].isin(groups)]
        # Check if there are multiple unique time values in the filtered data
        if "time" in metadata_df.columns:
            unique_times = metadata_df["time"].unique()
            # If there are multiple time values but the user didn't specify one, raise an error.
            if len(unique_times) > 1 and not timepoints:
                raise ValueError(
                    f"Data type is 'single' but metadata contains multiple unique time values: {unique_times}. "
                    "Please either:\n"
                    "1. Use --timepoints to specify which timepoint to use, or\n"
                    "2. Filter your metadata to include only one timepoint, or\n"
                    "3. Use data_type='longitudinal' if you want to analyze multiple timepoints."
                )
        # If a specific timepoint was provided, filter the data to just that timepoint.
        if timepoints:
            metadata_df = metadata_df[metadata_df["time"] == timepoints[0]]

    # --- 5. Merge and Report ---
    # Convert the filtered DataFrame to a dictionary for fast lookups, using sample_id as the key.
    metadata_dict = metadata_df.set_index("sample_id").to_dict(orient="index")
    filtered_mag_dict = defaultdict(list)

    # Iterate through each MAG and its associated sample files.
    for mag_id, sample_info_list in mag_dict.items():
        for sample_info in sample_info_list:
            sample_id = sample_info["sample_id"]
            # If the sample_id exists in our filtered metadata dictionary
            if sample_id in metadata_dict:
                # update the sample's info with the corresponding metadata
                sample_info.update(metadata_dict[sample_id])
                # and add it to our final dictionary.
                filtered_mag_dict[mag_id].append(sample_info)

    # Return the final dictionary containing only the MAGs with valid, merged metadata.
    return filtered_mag_dict


def write_output_files(
    mag_dict: Dict[str, List[Dict[str, str]]], output_dir: str, data_type: str
) -> None:
    """
    Writes output files for each MAG ID in the provided dictionary.

    This function creates an output directory if it does not exist and writes
    a TSV file for each MAG ID. Each TSV file contains sample information
    including sample ID, subjectID, replicate, group, and file path.

    Parameters:
    mag_dict (dict): A dictionary where keys are MAG IDs and values are lists
                     of dictionaries containing sample information. Each sample
                     information dictionary should have the keys "sample_id",
                     "subjectID", "group", "replicate" and "file_path".
    output_dir (str): The directory where the output files will be written.
    data_type (str): Type of analysis to perform, either 'single' or 'longitudinal'.

    Returns:
    None
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define base columns and add optional ones.
    columns = ["sample_id", "subjectID", "group", "replicate", "file_path", "bam_path"]
    sort_keys = ["sample_id", "replicate"]
    if data_type == "longitudinal":
        columns.insert(2, "time")
        sort_keys.append("time")

    for mag_id, sample_info_list in mag_dict.items():
        if not sample_info_list:
            logger.info(f"No samples left for MAG {mag_id} after filtering. Skipping.")
            continue

        output_df = pd.DataFrame(sample_info_list)

        # Ensure all expected columns exist, filling missing with empty strings
        for col in columns:
            if col not in output_df.columns:
                # output_df[col] = ""
                raise ValueError(f"Missing expected column: {col}")

        # Sort for deterministic output
        output_df.sort_values(by=sort_keys, inplace=True)

        output_file_path = output_dir / f"{mag_id}_metadata.tsv"
        output_df.to_csv(
            output_file_path,
            sep="\t",
            index=False,
            columns=columns,  # Enforce column order
        )
        logger.info(f"Output written for MAG {mag_id}: {output_file_path}")


def write_summary_index(
    mag_dict: Dict[str, List[Dict[str, str]]], output_dir: str
) -> None:
    """
    Writes a single summary index file of all MAGs and their metadata.

    This is useful for quick sanity checks and downstream analysis.

    Args:
        mag_dict: The final, merged dictionary of MAGs.
        output_dir: The directory where the index file will be written.
    """
    output_dir = Path(output_dir)
    rows = []
    for mag_id, sample_info_list in mag_dict.items():
        for sample in sample_info_list:
            row = sample.copy()
            row["mag_id"] = mag_id
            rows.append(row)

    if not rows:
        logger.warning("No data to write to summary index file.")
        return

    index_df = pd.DataFrame(rows)
    # Define a sensible column order for the index file
    index_cols = [
        "mag_id",
        "sample_id",
        "subjectID",
        "group",
        "time",
        "replicate",
        "file_path",
        "bam_path",
    ]
    # Filter for columns that actually exist in the DataFrame
    final_cols = [c for c in index_cols if c in index_df.columns]

    index_file_path = output_dir / "summary_mag_index.tsv"
    index_df.to_csv(index_file_path, sep="\t", index=False, columns=final_cols)
    logger.info(f"Summary index file written to: {index_file_path}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Process MAG files and generate output per MAG with metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rootDir",
        type=Path,
        help="Path to the root directory containing subdirectories.",
        required=True,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to the metadata TSV file.\n"
        "Required columns: sample_id, subjectID, group, bam_path.\n"
        "Optional columns: time, replicate.",
        required=True,
    )

    parser.add_argument(
        "--timepoints",
        nargs="*",
        help="Timepoints to include. Required for longitudinal data (exactly 2).\n"
        "Optional for single data (at most 1).",
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        help="One or more groups to include. All provided groups will be written"
        " to the metadata files. Group-pair-specific eligibility is determined"
        " downstream by eligibility_table.py.",
        required=True,
    )

    parser.add_argument(
        "--data_type",
        help="Type of analysis: 'single' timepoint or 'longitudinal' time series.",
        type=str,
        choices=["single", "longitudinal"],
        default="longitudinal",
    )

    parser.add_argument(
        "--outDir",
        help="Path to the directory where output files will be saved.",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    # Validate arguments based on data type
    if args.data_type == "longitudinal" and (
        not args.timepoints or len(args.timepoints) != 2
    ):
        raise ValueError(
            "--timepoints is required for longitudinal data type and must be exactly two"
        )

    if args.data_type == "single" and args.timepoints:
        if len(args.timepoints) > 1:
            raise ValueError(
                f"Only provide one timepoint for single data type. Provided: {args.timepoints}"
            )

    mag_dict = find_mag_files(args.rootDir)
    logger.info(f"Found profiles for {len(mag_dict):,} MAGs in the root directory.")
    if not mag_dict:
        logger.warning("No MAG files were found. Exiting.")
        return
    filtered_mag_dict = merge_metadata(
        mag_dict, args.metadata, args.timepoints, args.groups, args.data_type
    )
    if not filtered_mag_dict:
        logger.warning(
            "No MAGs remained after merging with metadata. No output files will be written."
        )
        return
    write_output_files(filtered_mag_dict, args.outDir, args.data_type)
    write_summary_index(filtered_mag_dict, args.outDir)


if __name__ == "__main__":
    main()
