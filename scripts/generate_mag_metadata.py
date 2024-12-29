import argparse
import logging
import os
from collections import defaultdict

import pandas as pd


def find_mag_files(root_dir):
    """
    Finds and organizes MAG (Metagenome-Assembled Genome) files in a given root directory.

    This function iterates over subdirectories in the specified root directory, looking for
    subdirectories that end with '.sorted'. Within these subdirectories, it searches for files
    that match the pattern '{subDirectoryName}_{MAG_ID}_profiled.tsv.gz'. It extracts the sample
    ID from the subdirectory name and the MAG ID from the filename, and stores this information
    in a dictionary.

    Parameters:
        root_dir (str): The path to the root directory containing the subdirectories with MAG files.

    Returns:
        dict: A dictionary where the keys are MAG IDs and the values are lists of dictionaries,
              each containing 'sample_id' and 'file_path' keys corresponding to the sample ID and
              the absolute path to the MAG file, respectively.

    Raises:
        OSError: If there is an issue accessing the directories or files.

    Example:
        >>> mag_files = find_mag_files("/path/to/root_dir")
        >>> print(mag_files)
        {
            'MAG1': [{'sample_id': 'sample1', 'file_path': '/path/to/root_dir/sample1.sorted/sample1_MAG1_profiled.tsv.gz'}],
            'MAG2': [{'sample_id': 'sample2', 'file_path': '/path/to/root_dir/sample2.sorted/sample2_MAG2_profiled.tsv.gz'}]
        }
    """
    mag_dict = defaultdict(list)

    # Iterate over subdirectories in the root directory
    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)
        if os.path.isdir(subdir_path):
            # Check if subdirectory name ends with '.sorted'
            if subdir_name.endswith(".sorted"):
                logging.info(f"Looking for MAGs in {subdir_name}")
                # Extract sample ID from subdirectory name
                sample_id = subdir_name.replace(".sorted", "")
                # Iterate over files in the subdirectory
                for filename in os.listdir(subdir_path):
                    if filename.endswith("_profiled.tsv.gz"):

                        # Extract MAG ID from filename
                        # Filename format: {subDirectoryName}_{MAG_ID}_profiled.tsv.gz
                        # We need to remove {subDirectoryName}_ and _profiled.tsv.gz
                        prefix = subdir_name + "_"
                        suffix = "_profiled.tsv.gz"
                        if filename.startswith(prefix) and filename.endswith(suffix):
                            mag_id = filename[len(prefix) : -len(suffix)]
                            # Store sample ID and file path in mag_dict under mag_id
                            file_path = os.path.abspath(
                                os.path.join(subdir_path, filename)
                            )
                            mag_dict[mag_id].append(
                                {"sample_id": sample_id, "file_path": file_path}
                            )
                        else:
                            logging.info(
                                f"Filename does not match expected format: {filename}. Skipped."
                            )
            else:
                logging.info(
                    f"Subdirectory does not match expected format (sampleID.sorted): {subdir_name}. Skipped."
                )
    return mag_dict


def merge_metadata(mag_dict, metadata_file, timepoints, groups):
    """
    Merges metadata from a file into a dictionary of metagenome-assembled genomes (MAGs).

    Parameters:
        mag_dict (dict): A dictionary where keys are MAG IDs and values are lists of sample information dictionaries.
        metadata_file (str): Path to the metadata file in TSV format containing columns 'sample', 'subjectID', 'group', 'time' and 'replicate'.
        timepoints (list): List of timepoints to filter the metadata.
        groups (list): List of groups to filter the metadata.

    Returns:
        dict: The updated mag_dict with metadata fields added to each sample information dictionary.
    """
    # Load metadata
    metadata_df = pd.read_csv(
        metadata_file,
        sep="\t",
        usecols=["sample_id", "subjectID", "group", "time", "replicate"],
    )

    # Ensure 'sample_id' column is a string for consistent matching
    metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)

    # Filter metadata to only include the specified timepoints and groups
    # This ensures that samples not matching these criteria are excluded
    metadata_df = metadata_df[
        metadata_df["time"].isin(timepoints) & metadata_df["group"].isin(groups)
    ]

    # Create a metadata dictionary for quick lookup
    metadata_dict = metadata_df.set_index("sample_id").to_dict(orient="index")

    # Merge metadata into mag_dict, excluding samples with no metadata match
    filtered_mag_dict = {}
    for mag_id, sample_info_list in mag_dict.items():
        filtered_samples = []
        for sample_info in sample_info_list:
            sample_id = sample_info["sample_id"]
            metadata = metadata_dict.get(sample_id)
            if metadata:
                # Add metadata fields to sample_info
                sample_info.update(metadata)
                filtered_samples.append(sample_info)
        if filtered_samples:
            filtered_mag_dict[mag_id] = filtered_samples

    return filtered_mag_dict


def write_output_files(mag_dict, output_dir):
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

    Returns:
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write output files for each MAG ID
    for mag_id, sample_info_list in mag_dict.items():
        output_file_path = os.path.join(output_dir, f"{mag_id}_samples.tsv")
        with open(output_file_path, "w") as f:
            # Write header
            f.write("sample_id\tsubjectID\ttime\tgroup\treplicate\tfile_path\n")
            for sample_info in sample_info_list:
                sample_id = sample_info["sample_id"]
                subjectID = sample_info["subjectID"]
                group = sample_info["group"]
                time = sample_info["time"]
                replicate = sample_info["replicate"]
                file_path = sample_info["file_path"]
                f.write(
                    f"{sample_id}\t{subjectID}\t{time}\t{group}\t{replicate}\t{file_path}\n"
                )
        logging.info(f"Output written for MAG {mag_id}: {output_file_path}")


def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser(
        description="Process MAG files and generate output per MAG with metadata."
    )
    parser.add_argument(
        "--rootDir",
        type=str,
        help="Path to the root directory containing subdirectories.",
        required=True,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Path to the metadata TSV file to merge with the main data. Should contain columns 'sample_id', 'subjectID', 'replicate', 'group' and 'time'.",
        required=True,
    )

    parser.add_argument(
        "--timepoints",
        nargs=2,
        help="Two timepoints to include.",
        required=True,
    )

    parser.add_argument(
        "--groups",
        nargs=2,
        help="Two groups to include.",
        required=True,
    )

    parser.add_argument(
        "--outDir",
        help="Path to the directory where output files will be saved.",
        required=True,
        type=str,
    )
    args = parser.parse_args()

    mag_dict = find_mag_files(args.rootDir)
    mag_dict = merge_metadata(mag_dict, args.metadata, args.timepoints, args.groups)
    write_output_files(mag_dict, args.outDir)


if __name__ == "__main__":
    main()
