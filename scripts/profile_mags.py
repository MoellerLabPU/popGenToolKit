#!/usr/bin/env python
import argparse
import gc
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pysam
from Bio import SeqIO
from intervaltree import IntervalTree
from tqdm import tqdm

# Global variables for BAM and FASTA files in worker processes
bamfile = None
reference_fasta = None


def init_worker(bam_path, fasta_path):
    """
    Initializes worker process by loading BAM and FASTA files.

    This function sets up global variables `bamfile` and `reference_fasta`
    which are used for alignment and reference sequence operations respectively.

    Parameters:
    bam_path (str): The file path to the BAM file.
    fasta_path (str): The file path to the reference FASTA file.
    """
    global bamfile
    global reference_fasta
    bamfile = pysam.AlignmentFile(bam_path, "rb")
    reference_fasta = pysam.FastaFile(fasta_path)


def process_contig(contig_name):
    """
    Processes a given contig by iterating over its pileup columns and collecting base counts at each position.

    Args:
        contig_name (str): The name of the contig to process.

    Returns:
        list: A list of dictionaries, each containing the following keys:
            - ref_name (str): The name of the reference contig.
            - position (int): The 0-based position in the contig.
            - ref_base (str): The reference base at this position.
            - total_coverage (int): The total coverage at this position.
            - A (int): The count of 'A' bases at this position.
            - C (int): The count of 'C' bases at this position.
            - G (int): The count of 'G' bases at this position.
            - T (int): The count of 'T' bases at this position.
            - N (int): The count of 'N' bases at this position.
    """

    global bamfile
    global reference_fasta

    # Initialize a list to store data for this contig
    contig_data = []

    # Iterate over the pileup columns (i.e., positions) in this contig
    for pileupcolumn in bamfile.pileup(
        contig=contig_name,
        stepper="samtools",
        ignore_orphans=True,
        min_base_quality=30,
        ignore_overlaps=True,
        max_depth=100000000,
        compute_baq=False,
    ):

        pos = pileupcolumn.reference_pos  # 0-based position
        ref_name = pileupcolumn.reference_name  # Should be the same as contig_name

        # Fetch the reference base at this position
        if ref_name not in reference_fasta.references:
            logging.warning(
                f"Contig '{ref_name}' not found in the reference FASTA. Assigning 'X' as a reference base."
            )
            ref_base = "X"
        else:
            ref_base = reference_fasta.fetch(ref_name, pos, pos + 1).upper()

        # Initialize base counts for this position
        base_counts = defaultdict(int)

        # Loop through each read covering this specific base
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip:
                # Get the base from the read at this position and convert to uppercase
                base = pileupread.alignment.query_sequence[
                    pileupread.query_position
                ].upper()
                # Convert any base other than A, T, G, C to N
                if base not in {"A", "T", "G", "C"}:
                    base = "N"
                # Increment the base count at this position
                base_counts[base] += 1

        # Calculate total coverage at this position
        total_coverage = sum(base_counts.values())

        # Skip positions where total_coverage is zero ie. no reads covering the position
        if total_coverage == 0:
            continue

        # Retrieve counts for each nucleotide
        A_count = base_counts.get("A", 0)
        C_count = base_counts.get("C", 0)
        G_count = base_counts.get("G", 0)
        T_count = base_counts.get("T", 0)
        N_count = base_counts.get("N", 0)

        # Append the data to the list
        contig_data.append(
            {
                "contig": ref_name,
                "position": pos,
                "ref_base": ref_base,
                "total_coverage": total_coverage,
                "A": A_count,
                "C": C_count,
                "G": G_count,
                "T": T_count,
                "N": N_count,
            }
        )

    return contig_data


def map_genes(prodigal_fasta):
    """
    Parses a Prodigal FASTA file to extract gene information and returns it as a DataFrame.

    Parameters:
        prodigal_fasta (str): Path to the Prodigal FASTA file.

    Returns:
        pandas.DataFrame: A DataFrame containing gene information with columns:
            - gene_id (str): The gene identifier.
            - start (int): The start position of the gene (0-based index).
            - end (int): The end position of the gene (0-based index).
            - contig (str): The contig identifier extracted from the gene_id.

    Raises:
        ValueError: If the header format in the FASTA file is unexpected.
    """
    # Initialize a list to store gene information
    logging.info("Parsing Prodigal FASTA file to extract gene information.")
    genes_data = []

    # Parse the FASTA file
    for record in SeqIO.parse(prodigal_fasta, "fasta"):
        header = record.description
        # Split the header at '#' to get the gene coordinates
        # https://github.com/hyattpd/prodigal/wiki/understanding-the-prodigal-output#protein-translations
        parts = header.split("#")
        if len(parts) == 5:
            gene_id = parts[0]
            start = (
                int(parts[1]) - 1
            )  # pySam uses 0 based indexing while Prodigal uses 1 based
            end = int(parts[2]) - 1
            # Append gene information to the list
            genes_data.append(
                {
                    "gene_id": gene_id,
                    "start": start,
                    "end": end,
                }
            )
        else:
            logging.warning(f"Warning: Unexpected header format: {header}.")
            raise ValueError("Unexpected header format in the FASTA file.")

    # Convert the list to a DataFrame
    genes_df = pd.DataFrame(genes_data)
    genes_df["contig"] = genes_df["gene_id"].apply(extract_contigID)
    logging.info("Gene information extracted successfully.")
    return genes_df


def extract_contigID(gene_id):
    """
    Extract the contig ID from a given gene ID by removing the last underscore and the following characters.

    Parameters:
        gene_id (str): The gene ID string from which to extract the contig ID.
                       Example: 'SLG1007_DASTool_bins_35.fa_k141_82760_1'

    Returns:
        str: The extracted contig ID.
             Example: 'SLG1007_DASTool_bins_35.fa_k141_82760'
    """
    contig = "_".join(gene_id.split("_")[:-1])
    return contig


def extract_mag_name(contig_name):
    """
    Extracts the MAG (Metagenome-Assembled Genome) name from a given contig name.

    Parameters:
        contig_name (str): The name of the contig file, expected to end with ".fa".
            Example: 'SLG1007_DASTool_bins_35.fa_k141_82760'

    Returns:
        str: The extracted MAG name, which is the part of the contig name before ".fa".
            Example: 'SLG1007_DASTool_bins_35'
    """
    mag_name = contig_name.split(".fa")[0]
    return mag_name


def create_intervalTree(genes_df):
    """
    Creates an IntervalTree for the genes dataframe.

    This function takes a dataframe containing gene information and constructs an IntervalTree for each contig.
    The IntervalTree allows efficient querying of intervals (gene positions) and stores the gene ID as the data attribute of each interval.

    Parameters:
        genes_df (pd.DataFrame): A pandas DataFrame containing gene information with columns:
            - 'contig': The contig identifier.
            - 'start': The start position of the gene.
            - 'end': The end position of the gene.
            - 'gene_id': The identifier of the gene.

    Returns:
        dict: A dictionary where each key is a contig and each value is an IntervalTree containing intervals for the genes and their IDs.

    """
    # logging.info(f"Creating IntervalTree for the genes dataframe..")
    start_time = time.time()

    contig_trees = defaultdict(IntervalTree)
    for idx, row in genes_df.iterrows():
        contig = row["contig"]
        start = row["start"]
        end = row["end"]
        gene_id = row["gene_id"]

        # contig_trees is a dictionary where each key is a contig and each value is an IntervalTree containing intervals for the genes and the gene_ID
        # intervaltree uses half-open intervals, including the lower bound but not the upper bound. We add 1 to include the end position
        # https://github.com/chaimleib/intervaltree/issues/128#issuecomment-1515796584
        # gene_id, is stored as the data attribute of the interval
        contig_trees[contig].addi(start, end + 1, gene_id)

    end_time = time.time()
    # logging.info(f"IntervalTree created in {end_time - start_time:.2f} seconds")
    return contig_trees


def map_positions_to_genes(positions_df, contig_trees):
    """
    Maps genomic positions to gene IDs using interval trees for each contig.

    Parameters:
    positions_df (pd.DataFrame): A DataFrame containing genomic positions with at least two columns:
                                 'contig' (contig name) and 'position' (genomic position).
    contig_trees (dict): A dictionary where keys are contig names and values are IntervalTree objects
                         representing gene intervals for the corresponding contig.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as positions_df, but with an additional column 'gene_id'
                  that contains the gene IDs corresponding to each position. If a position does not overlap
                  with any gene, the 'gene_id' will be None.
    """
    # logging.info("Mapping positions to genes...")
    # start_time = time.time()

    # Initialize a list to collect DataFrames
    result_dfs = []

    # Group positions by `contig`
    grouped = positions_df.groupby("contig")

    for contig, group in grouped:
        tree = contig_trees.get(contig)
        if tree:
            # Get the positions
            positions = group["position"].values
            # Prepare a list to store gene IDs
            gene_ids = []
            # Query the IntervalTree for each position
            for pos in positions:
                overlaps = tree.at(pos)
                if overlaps:
                    gene_id = ",".join([interval.data for interval in overlaps])
                else:
                    gene_id = None
                gene_ids.append(gene_id)
            # Assign the gene IDs to the group
            # Create a copy to avoid modifying the original DataFrame
            group_copy = group.copy()
            group_copy["gene_id"] = gene_ids
        else:
            group_copy = group.copy()
            group_copy["gene_id"] = None
        # Collect the processed groups
        result_dfs.append(group_copy)

    # Concatenate all groups back into a single DataFrame
    mapped_positions_df = pd.concat(result_dfs, ignore_index=True)

    # end_time = time.time()
    # logging.info(f"Positions mapped to genes in {end_time - start_time:.2f} seconds")
    return mapped_positions_df


def main():
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )
    parser = argparse.ArgumentParser(
        description="Profile MAGs using alignment files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bam_path",
        help="Path to sorted BAM file.",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--fasta_path",
        help="Path to reference FASTA file used for alignment.",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--prodigal_fasta",
        help="Path to Prodigal predicted genes (DNA).",
        type=str,
        required=True,
        metavar="filepath",
    )

    parser.add_argument(
        "--cpus",
        help=f"Number of processors to use.",
        default=mp.cpu_count(),
        metavar="int",
        type=int,
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Open BAM file to get the list of contigs
    if not os.path.exists(args.bam_path + ".bai"):
        logging.info("BAM file index not found. Indexing it now..")
        pysam.index(
            args.bam_path, "-o", args.bam_path + ".bai", "--threads", "args.cpus"
        )

    if not os.path.exists(args.fasta_path + ".fai"):
        logging.info("FASTA file index not found. Indexing it now..")
        pysam.faidx(args.fasta_path, "-o", args.fasta_path + ".fai")

    bamfile_main = pysam.AlignmentFile(args.bam_path, "rb")
    contig_list = bamfile_main.references
    # contig_list = [
    #     "SLG1122_DASTool_bins_50.fa_k141_183033",
    #     "SLG1122_DASTool_bins_50.fa_k141_44596",
    #     "SLG1122_DASTool_bins_67.fa_k141_18120",
    # ]
    bamfile_main.close()

    num_processes = args.cpus

    # Create a mapping from MAG names to their contigs
    mag_contigs = defaultdict(list)
    for contig in contig_list:
        mag_name = extract_mag_name(contig)
        mag_contigs[mag_name].append(contig)

    # Load genes data once
    genes_df = map_genes(args.prodigal_fasta)

    # Initialize multiprocessing pool once
    with mp.Pool(
        processes=num_processes,
        initializer=init_worker,
        initargs=(args.bam_path, args.fasta_path),
    ) as pool:

        # Process each MAG with a progress bar
        for mag_name in tqdm(
            mag_contigs, total=len(mag_contigs), desc="Processing MAGs"
        ):
            contigs = mag_contigs[mag_name]
            # logging.info(f"Processing MAG '{mag_name}' with {len(contigs)} contigs")

            # Initialize data list for this MAG
            data_list = []

            # Process contigs for this MAG
            contig_results = pool.imap_unordered(process_contig, contigs)
            for contig_data in contig_results:
                data_list.extend(contig_data)

            # Create DataFrame for this MAG
            df_mag = pd.DataFrame(data_list)

            # Filter genes_df for this MAG
            genes_df_mag = genes_df[
                genes_df["contig"].apply(extract_mag_name) == mag_name
            ]

            # Create IntervalTree for this MAG
            contig_trees = create_intervalTree(genes_df_mag)

            # Map positions to genes for this MAG
            if not df_mag.empty:
                mapped_positions_df = map_positions_to_genes(df_mag, contig_trees)

                # Write data to file
                sampleID = Path(args.bam_path).stem
                outDir = os.path.join(args.output_dir, sampleID)
                os.makedirs(outDir, exist_ok=True)
                outPath = os.path.join(outDir, f"{sampleID}_{mag_name}_profiled.tsv.gz")
                mapped_positions_df.to_csv(
                    outPath, sep="\t", index=False, compression="gzip"
                )
                # logging.info(f"Mapped positions for MAG '{mag_name}' saved to {outPath}")

                # Release memory
                del data_list, df_mag, mapped_positions_df, genes_df_mag, contig_trees
            gc.collect()

    end_time = time.time()
    logging.info(f"Total time taken {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
