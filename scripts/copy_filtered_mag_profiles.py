#!/usr/bin/env python3

import argparse
import os
import shutil
import sys

import pandas as pd


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Copy MAG profile files for samples where "
            "time is 'end' or 'post' and group is 'control' or 'fat', "
            "replicating the directory structure in the destination."
        )
    )
    parser.add_argument(
        "source_root",
        type=str,
        help="Root directory containing .sorted subdirectories.",
    )
    parser.add_argument(
        "metadata_file",
        type=str,
        help="Path to the metadata file (e.g., metadata.tsv or metadata.csv).",
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Destination root directory where structured MAG files will be copied.",
    )
    parser.add_argument(
        "--mag_id",
        type=str,
        default="SLG991_DASTool_bins_60",
        help="MAG ID to filter files (default: 'SLG991_DASTool_bins_60').",
    )
    parser.add_argument(
        "--metadata_delimiter",
        type=str,
        default="\t",
        help="Delimiter used in the metadata file (default: tab). Use ',' for CSV.",
    )
    return parser.parse_args()


def read_and_filter_metadata(metadata_file, delimiter):

    metadata = pd.read_csv(metadata_file, delimiter=delimiter)

    required_columns = {"sample", "time", "group"}

    # Apply filters
    filtered = metadata[
        (metadata["time"].isin(["end", "post"]))
        & (metadata["group"].isin(["control", "fat"]))
    ]

    if filtered.empty:
        print("No samples match the specified criteria.", file=sys.stderr)
        sys.exit(1)

    # Extract sample IDs
    sample_ids = filtered["sample"].unique()
    return sample_ids


def copy_mag_profiles_structured(source_root, sample_ids, mag_id, destination):

    # Ensure the destination root directory exists
    os.makedirs(destination, exist_ok=True)

    files_copied = 0
    files_not_found = 0
    for sample_id in sample_ids:
        sorted_dir = f"{sample_id}.sorted"
        sorted_dir_path = os.path.join(source_root, sorted_dir)

        if not os.path.isdir(sorted_dir_path):
            print(
                f"Warning: Directory '{sorted_dir_path}' does not exist. Skipping sample '{sample_id}'.",
                file=sys.stderr,
            )
            files_not_found += 1
            continue

        # Construct the expected file name
        file_name = f"{sorted_dir}_{mag_id}_profiled.tsv.gz"
        file_path = os.path.join(sorted_dir_path, file_name)

        if os.path.isfile(file_path):
            # Create corresponding directory in destination
            dest_sorted_dir = os.path.join(destination, sorted_dir)
            os.makedirs(dest_sorted_dir, exist_ok=True)

            shutil.copy2(file_path, dest_sorted_dir)
            print(f"Copied: {file_path} -> {dest_sorted_dir}")
            files_copied += 1
        else:
            print(
                f"Warning: File '{file_name}' not found in '{sorted_dir_path}'.",
                file=sys.stderr,
            )
            files_not_found += 1

    print("\nCopying Summary:")
    print(f"Total samples processed: {len(sample_ids)}")
    print(f"Files successfully copied: {files_copied}")
    print(f"Files not found or failed to copy: {files_not_found}")


def main():
    args = parse_arguments()

    sample_ids = read_and_filter_metadata(args.metadata_file, args.metadata_delimiter)
    print(f"Filtered {len(sample_ids)} samples based on metadata criteria.\n")

    copy_mag_profiles_structured(
        source_root=args.source_root,
        sample_ids=sample_ids,
        mag_id=args.mag_id,
        destination=args.destination,
    )


if __name__ == "__main__":
    main()
