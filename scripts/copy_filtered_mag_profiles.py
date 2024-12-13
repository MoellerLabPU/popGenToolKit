#!/usr/bin/env python3

import argparse
import os
import shutil
import sys

import pandas as pd


def read_and_filter_metadata(metadata_file, delimiter):

    metadata = pd.read_csv(metadata_file, delimiter=delimiter)

    required_columns = {"sample", "time", "group"}
    if not required_columns.issubset(metadata.columns):
        missing = required_columns - set(metadata.columns)
        print(
            f"Error: Missing required columns in metadata: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply filters
    filtered = metadata[
        (metadata["time"].isin(["pre", "end", "post"]))
        & (metadata["group"].isin(["control", "fat"]))
    ]

    if filtered.empty:
        print("No samples match the specified criteria.", file=sys.stderr)
        sys.exit(1)

    # Extract sample IDs
    sample_ids = filtered["sample"].unique()
    return sample_ids


# def copy_mag_profiles_structured(rootDir, sample_ids, mag_id, destination):

#     # Ensure the destination root directory exists
#     os.makedirs(destination, exist_ok=True)

#     files_copied = 0
#     files_not_found = 0
#     for sample_id in sample_ids:
#         sorted_dir = f"{sample_id}.sorted"
#         sorted_dir_path = os.path.join(rootDir, sorted_dir)

#         if not os.path.isdir(sorted_dir_path):
#             print(
#                 f"Warning: Directory '{sorted_dir_path}' does not exist. Skipping sample '{sample_id}'.",
#                 file=sys.stderr,
#             )
#             files_not_found += 1
#             continue

#         # Construct the expected file name
#         file_name = f"{sorted_dir}_{mag_id}_profiled.tsv.gz"
#         file_path = os.path.join(sorted_dir_path, file_name)

#         if os.path.isfile(file_path):
#             # Create corresponding directory in destination
#             dest_sorted_dir = os.path.join(destination, sorted_dir)
#             os.makedirs(dest_sorted_dir, exist_ok=True)

#             shutil.copy2(file_path, dest_sorted_dir)
#             print(f"Copied: {file_path} -> {dest_sorted_dir}")
#             files_copied += 1
#         else:
#             print(
#                 f"Warning: File '{file_name}' not found in '{sorted_dir_path}'.",
#                 file=sys.stderr,
#             )
#             files_not_found += 1

#     print("\nCopying Summary:")
#     print(f"Total samples processed: {len(sample_ids)}")
#     print(f"Files successfully copied: {files_copied}")
#     print(f"Files not found or failed to copy: {files_not_found}")


def copy_mag_profiles_structured(rootDir, sample_ids, mag_ids, destination):
    # Ensure the destination root directory exists
    os.makedirs(destination, exist_ok=True)

    total_files_copied = 0
    total_files_not_found = 0

    for mag_id in mag_ids:
        print(f"Processing MAG_ID: {mag_id}")
        files_copied = 0
        files_not_found = 0

        for sample_id in sample_ids:
            sorted_dir = f"{sample_id}.sorted"
            sorted_dir_path = os.path.join(rootDir, sorted_dir)

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
                # print(f"Copied: {file_path} -> {dest_sorted_dir}")
                files_copied += 1
            else:
                print(
                    f"Warning: File '{file_name}' not found in '{sorted_dir_path}'.",
                    file=sys.stderr,
                )
                files_not_found += 1

        print(f"\nSummary for MAG_ID '{mag_id}':")
        print(f"  Total samples processed: {len(sample_ids)}")
        print(f"  Files successfully copied: {files_copied}")
        print(f"  Files not found or failed to copy: {files_not_found}\n")

        total_files_copied += files_copied
        total_files_not_found += files_not_found

    print("\nOverall Copying Summary:")
    print(f"Total MAG_IDs processed: {len(mag_ids)}")
    print(f"Total samples processed: {len(sample_ids) * len(mag_ids)}")
    print(f"Total files successfully copied: {total_files_copied}")
    print(f"Total files not found or failed to copy: {total_files_not_found}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Copy MAG profile files for samples where "
            "time is 'end' or 'post' and group is 'control' or 'fat', "
            "replicating the directory structure in the destination."
        )
    )
    parser.add_argument(
        "--rootDir",
        type=str,
        required=True,
        help="Root directory containing .sorted subdirectories.",
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to the metadata file.",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="Destination root directory where structured MAG files will be copied.",
    )
    parser.add_argument(
        "--mag_ids",
        type=str,
        nargs="+",
        required=True,
        help="MAG ID to filter files.",
    )
    parser.add_argument(
        "--metadata_delimiter",
        type=str,
        default="\t",
        help="Delimiter used in the metadata file (default: tab).",
    )
    args = parser.parse_args()

    sample_ids = read_and_filter_metadata(args.metadata_file, args.metadata_delimiter)
    print(f"Filtered {len(sample_ids)} samples based on metadata criteria.\n")

    copy_mag_profiles_structured(
        rootDir=args.rootDir,
        sample_ids=sample_ids,
        mag_ids=args.mag_ids,
        destination=args.destination,
    )


if __name__ == "__main__":
    main()
