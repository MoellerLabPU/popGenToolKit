import argparse
import glob
import logging
import os

import pandas as pd
from alleleflux.scripts.utilities.logging_config import setup_logging


logger = logging.getLogger(__name__)


def find_bam_path(sample_id, bam_dir, extension):
    """
    Searches for a BAM file within the specified directory matching the given sample identifier and file extension.

    Parameters:
        sample_id (str): Identifier used to match the desired file.
        bam_dir (str): Directory in which to search for the file.
        extension (str): File extension (e.g., ".bam") expected for the file.

    Returns:
        str or None: The absolute path to the BAM file if exactly one match is found; None if no match is found.

    Raises:
        ValueError: If more than one file matches the search criteria.
    """
    pattern = os.path.join(bam_dir, f"{sample_id}*{extension}")
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        return None
    elif len(matches) == 1:
        logger.info(f"Found match: {matches}")
        return os.path.abspath(matches[0])
    elif len(matches) >= 1:
        raise ValueError(
            f"More than one match identified. {matches} found for {sample_id} in {bam_dir}. Please specify a more precise sample_id or check the directory contents."
        )


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Add BAM file paths to metadata table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata", help="Input metadata TSV file, with 'sample_id' column"
    )
    parser.add_argument("--output", help="Output metadata TSV file with BAM paths")
    parser.add_argument(
        "--bam-dir", default=".", help="Directory to search for BAM files"
    )
    parser.add_argument("--bam-extension", default=".bam", help="BAM filename suffix")
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop rows without matching BAM files",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.metadata, sep="\t")

    if "sample_id" not in df.columns:
        raise ValueError(f"Column sample_id not found in metadata")

    df["bam_path"] = df["sample_id"].apply(
        lambda sample_id: find_bam_path(sample_id, args.bam_dir, args.bam_extension)
    )

    if args.drop_missing:
        logger.info("Dropping rows without matching BAM files")
        df = df[df["bam_path"].notna()]

    df.to_csv(args.output, sep="\t", index=False)
    print(f"Saved updated metadata to: {args.output}")


if __name__ == "__main__":
    main()
