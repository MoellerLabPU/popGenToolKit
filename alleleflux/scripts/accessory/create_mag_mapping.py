#!/usr/bin/env python3
"""
create_mag_mapping.py

A standalone preprocessing tool that should be run BEFORE the AlleleFlux workflow.

This script processes MAG FASTA files by:
1. Modifying contig headers to prepend the MAG filename (or using existing headers if already unique)
2. Optionally concatenating all modified FASTAs into one output file
3. Optionally generating individual genome files with modified headers
4. Generating a mapping file of scaffold IDs to MAG IDs

The output mapping file can be used directly in the AlleleFlux workflow
by specifying it in the config.yml file under input.mag_mapping_path.

Example usage:
    # Standard usage with header modification
    alleleflux-create-mag-mapping --dir /path/to/mags/ --output-fasta combined.fasta --output-mapping mag_mapping.tsv --output-genomes-dir modified_genomes/

    # When headers are already unique (MAGID_contigID format)
    alleleflux-create-mag-mapping --dir /path/to/mags/ --headers-unique unique --output-mapping mag_mapping.tsv

    # Only generate mapping file (minimal usage)
    alleleflux-create-mag-mapping --dir /path/to/mags/ --output-mapping mag_mapping.tsv
"""

import argparse
import gzip
import logging
import os
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def find_fasta_files(directory, extension):
    """
    Find all FASTA files with the specified extension in a directory.

    Parameters:
        directory (str): Path to directory to search
        extension (str): File extension to match (without leading dot)

    Returns:
        list: List of Path objects for matching files
    """
    # Normalize extension
    ext = extension.lstrip(".")
    return sorted(Path(directory).glob(f"*.{ext}"))


def open_fasta(path: Path, mode: str = "rt"):
    """
    Open a FASTA, transparently handling .gz or plain files.
    mode should be 'rt' (read text) for input.
    """
    if path.suffix == ".gz":
        return gzip.open(path, mode)  # returns a file-like in text mode
    return path.open(mode)  # standard Path.open


def process_and_concatenate_fastas(
    input_files,
    output_fasta,
    mapping_file,
    output_genomes_dir,
    extension,
    headers_unique=False,
    sanitize_headers=False,
):
    """
    Process each FASTA file, modify headers, and concatenate into one file.
    Also generates a mapping file and individual genome files with modified headers.

    Parameters:
        input_files (list): List of input FASTA file paths
        output_fasta (str): Path to output concatenated FASTA file
        mapping_file (str): Path to output mapping file
        output_genomes_dir (str): Directory to store individual genome files with modified headers
        extension (str): File extension to match (without leading dot)
        headers_unique (bool): If True, headers are already in MAGID_contigID format
        sanitize_headers (bool): If True, replace '|' with '_' in contig IDs

    Returns:
        tuple: (total_seqs, total_mags) - Count of sequences and MAGs processed
    """

    # Create output directories if they don't exist
    mapping_file = Path(mapping_file)
    mapping_file.parent.mkdir(parents=True, exist_ok=True)

    # Handle optional output_fasta
    if output_fasta:
        output_fasta = Path(output_fasta)
        output_fasta.parent.mkdir(parents=True, exist_ok=True)
        out_fasta_handle = output_fasta.open("w")
    else:
        out_fasta_handle = None

    # Handle optional output_genomes_dir
    if output_genomes_dir:
        genomes_dir = Path(output_genomes_dir)
        genomes_dir.mkdir(parents=True, exist_ok=True)
    else:
        genomes_dir = None

    total_seqs = 0
    total_mags = 0
    mag_ids = set()

    # We'll collect mapping rows in a small list and flush via pandas at the end
    mapping_rows = []

    # Process each file and write to the output
    try:
        for mag_file in input_files:
            mag_basename = mag_file.name.replace(f".{extension}", "")
            # Check for duplicate MAG IDs
            if mag_basename in mag_ids:
                raise ValueError(
                    f"Duplicate MAG ID found: {mag_basename}. Skipping file."
                )
            mag_ids.add(mag_basename)

            logger.info(f"Processing {mag_file}")

            # If output_genomes_dir is specified, create a separate FASTA file for this MAG
            if genomes_dir:
                modified_mag_path = genomes_dir / f"{mag_basename}.{extension}"
                mag_out_handle = open_fasta(modified_mag_path, "wt")
            else:
                mag_out_handle = None

            with open_fasta(mag_file, "rt") as handle:
                seqs_in_file = 0
                for record in SeqIO.parse(handle, "fasta"):
                    # If headers are already unique, use them as-is
                    if headers_unique:
                        new_id = record.id
                    else:
                        new_id = f"{mag_basename}_{record.id}"

                    if sanitize_headers:
                        new_id = new_id.replace("|", "_")

                    new_record = SeqRecord(record.seq, id=new_id, description="")

                    # Write to concatenated FASTA if specified
                    if out_fasta_handle:
                        SeqIO.write(new_record, out_fasta_handle, "fasta")

                    # Append mapping row
                    mapping_rows.append((mag_basename, new_id))

                    # Write to individual genome file if requested
                    if mag_out_handle:
                        SeqIO.write(new_record, mag_out_handle, "fasta")

                    seqs_in_file += 1
                    total_seqs += 1

            # Close per-MAG output if opened
            if mag_out_handle:
                mag_out_handle.close()
                logger.info(f"Created modified genome file: {modified_mag_path}")

            total_mags += 1
            logger.info(f"Added {seqs_in_file} sequences from {mag_basename}")

        # Write mapping file using pandas
        df = (
            pd.DataFrame(mapping_rows, columns=["mag_id", "contig_id"])
            if mapping_rows
            else pd.DataFrame(columns=["mag_id", "contig_id"])
        )
        df.to_csv(mapping_file, sep="\t", index=False)

        logger.info(
            f"Processing complete: {total_seqs:,} sequences from {total_mags} MAGs"
        )
        if genomes_dir:
            logger.info(
                f"Created {total_mags} individual genome files in {genomes_dir}"
            )
        return total_seqs, total_mags

    finally:
        # Close the output FASTA file if it was opened
        if out_fasta_handle:
            out_fasta_handle.close()


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="""
        PREPROCESSING UTILITY: Create a combined FASTA file and MAG mapping file from individual MAG FASTA files.
        This tool should be run BEFORE starting the AlleleFlux workflow.
        Additionally, it can create individual genome files with modified headers.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dir",
        help="Path to directory containing MAG FASTA files",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--extension",
        help="Exact file extension to identify FASTA files (e.g., 'fasta', 'fa', 'fa.gz')",
        type=str,
        default="fa",
    )

    parser.add_argument(
        "--headers-unique",
        help="Specify how contig headers are formatted: 'standard' (default) or 'unique' (already in MAGID_contigID format)",
        type=str,
        choices=["standard", "unique"],
        default="standard",
    )

    parser.add_argument(
        "--output-fasta",
        help="Path for the concatenated output FASTA file (optional)",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-mapping",
        help="Path for the scaffold-to-bin mapping TSV file",
        type=Path,
        default="megamag_mapping.tsv",
    )

    parser.add_argument(
        "--output-genomes-dir",
        help="Directory to store individual genome files with modified headers (optional)",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sanitize-headers",
        help="Replace '|' with '_' in contig IDs (recommended for downstream compatibility)",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.dir):
        raise ValueError(f"Directory not found: {args.dir}")

    # Find FASTA files
    fasta_files = find_fasta_files(args.dir, args.extension)

    if not fasta_files:
        raise ValueError(
            f"No FASTA files with extension '{args.extension}' found in {args.dir}"
        )

    logger.info(f"Found {len(fasta_files)} FASTA files to process")

    # Process files
    headers_unique = args.headers_unique == "unique"
    total_seqs, total_mags = process_and_concatenate_fastas(
        fasta_files,
        args.output_fasta,
        args.output_mapping,
        args.output_genomes_dir,
        args.extension,
        headers_unique=headers_unique,
        sanitize_headers=args.sanitize_headers,
    )

    if total_seqs == 0:
        raise ValueError("No sequences were processed")


if __name__ == "__main__":
    main()
