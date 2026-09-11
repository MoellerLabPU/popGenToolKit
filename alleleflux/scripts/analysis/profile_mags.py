#!/usr/bin/env python
"""
Profile MAGs using samtools mpileup.

Optimized implementation that delegates pileup computation to samtools mpileup
(compiled C) via subprocess, replacing the previous Python-level pysam pileup loop.

Background on why this was rewritten:
  The original code used pysam's Python-level pileup iterator, which loops over
  every read at every position inside the Python interpreter. For a typical sample
  with 306 million covered positions at 30x coverage, this amounts to ~9 billion
  Python attribute dereferences — the dominant runtime cost at ~9.6 h/sample.

Key optimizations applied (see inline comments for per-change rationale):
  1. Skip contigs with 0 mapped reads via pysam.get_index_statistics()
     WHY: avoids dispatching worker processes to contigs that will return nothing.
  2. Pre-compute gene IntervalTree index per MAG once at startup
     WHY: the old code rebuilt the tree O(N_MAGs) times from the full gene table.
  3. Stream output directly via csv.writer + gzip (no pandas DataFrame)
     WHY: eliminates peak RAM spike from constructing a per-MAG DataFrame.
  4. Use samtools mpileup subprocess instead of pysam pileup loop
     WHY: samtools mpileup counts bases in compiled C; Python only parses text output.
"""

import argparse
import csv
import gc
import gzip
import logging
import multiprocessing as mp
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pysam
from Bio import SeqIO
from intervaltree import IntervalTree
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import extract_mag_id, load_mag_mapping

logger = logging.getLogger(__name__)

# Output header for profile TSVs (mapq_scores removed in optimization)
PROFILE_HEADER = [
    "contig", "position", "ref_base", "total_coverage",
    "A", "C", "G", "T", "N", "gene_id",
]


# ---------------------------------------------------------------------------
# samtools mpileup parsing
# ---------------------------------------------------------------------------


def _parse_mpileup_bases(ref_base, bases_str):
    """Parse samtools mpileup bases string into A, C, G, T, N counts.

    Background
    ----------
    samtools mpileup outputs one line per reference position. The bases column
    encodes every read covering that position using a compact notation that must
    be decoded token by token. This function walks the string with a single
    cursor (i), consuming one token at a time, and counts the base observed in
    each read at this position.

    Key insight: bases_str and the base quality / mapq strings are NOT the same
    length. bases_str contains extra tokens (^, $, indels) that take up
    characters but do not represent reads. The base/mapq strings have exactly
    one character per read. This function only needs the bases string so the
    cursor mismatch is not a concern here, but it is important to understand
    if you ever re-introduce mapq tracking (see previous version).

    Encoding reference
    ------------------
    The full mpileup format spec is documented in the samtools manual:
      https://www.htslib.org/doc/samtools-mpileup.html

    Token types:

      . or ,        Reference match on forward (.) or reverse (,) strand.
                    Both are counted as ref_base. Example: ".,." means three
                    reads all matching the reference.

      A/C/G/T       Explicit mismatch on the forward strand (uppercase).
      a/c/g/t       Explicit mismatch on the reverse strand (lowercase).
                    Both are uppercased and counted. Example: "Aa" = two reads
                    both calling A, one on each strand.

      N / n         Ambiguous base call — the sequencer could not resolve the
                    base. This IS a real base call (unlike * / < / >) and is
                    counted in the n counter. Example: "N" means one read
                    where the base could not be determined.

      ^<q>          Start of a read segment. The character immediately after ^
                    encodes the read's mapping quality as a flag character, but
                    this is NOT the same as the mapq_str column — it is just a
                    marker. Both ^ and the following char are skipped (i += 2).
                    Example: "^!" — new read starts, mapping quality flag is !,
                    no base is counted.

      $             End of a read segment. Single character marker, no base
                    associated. Skipped with i += 1.
                    Example: ".$" — one reference match, then a read ends.

      +<n><bases>   Insertion of n bases AFTER this position. The token is an
      -<n><bases>   annotation attached to the preceding base call — it does
                    not represent a new base call at the current position.
                    The base call before the token is counted normally; the
                    token itself is skipped. The length n can be more than one
                    digit (e.g. +12AGTCAGTCAGTC is valid).
                    Example: ".,+2GT." — three reads at this position. The
                    second read (,) has a 2-base insertion (GT) after this
                    position. All three reads count as ref_base matches.
                    See: https://www.htslib.org/doc/samtools-mpileup.html

      *             Deletion placeholder. A read is present at this position
                    but has a deletion in its alignment here. The read covers
                    the position but has NO base to report — not even N. It is
                    skipped entirely and does not increment any counter.
                    Example: "*" — one read with a deletion here.

      < or >        Reference skip. A read spans this position via a splice
                    junction or similar alignment gap. Like *, the read covers
                    the position but has NO base call here. Skipped entirely.
                    Common in RNA-seq data; can appear in metagenomics too.
                    Example: "<" — one read skipping over this position.

    The distinction between N (counted) and * / < / > (not counted):
      - N means a read WAS here and tried to call a base but failed (ambiguous)
      - * / < / > mean a read WAS here but has NO base at all to report
    Counting * / < / > as N would incorrectly imply an attempted base call.

    Parameters
    ----------
    ref_base : str
        The reference base at this position, uppercase (e.g. "A").
        Used to resolve . and , tokens into an actual base.
    bases_str : str
        The mpileup bases column string (column 5 of mpileup output).

    Returns
    -------
    tuple of (int, int, int, int, int)
        Counts of (A, C, G, T, N) observed across all reads at this position.
        N counts ambiguous base calls only — not deletions or reference skips.

    Example
    -------
    >>> _parse_mpileup_bases("A", ".,^!+2GT$*Tg")
    (2, 0, 1, 1, 0)
    #
    # Walking through ".,^!+2GT$*Tg" with ref_base="A":
    #
    #  i=0  ch='.'  → reference match forward strand → count A  (a=1)
    #  i=1  ch=','  → reference match reverse strand → count A  (a=2)
    #  i=2  ch='^'  → read start marker, skip ^ and next char '!' → i=4
    #  i=4  ch='+'  → insertion token +2GT:
    #                   skip '+' → i=5
    #                   read digits '2' → i=6
    #                   skip 2 bases 'GT' → i=8
    #  i=8  ch='$'  → read end marker, skip → i=9
    #  i=9  ch='*'  → deletion placeholder, no base → skip → i=10
    #  i=10 ch='T'  → explicit mismatch forward strand → count T  (t=1)
    #  i=11 ch='g'  → explicit mismatch reverse strand → count G  (g=1)
    #  i=12 → end of string
    #
    # Result: a=2, c=0, g=1, t=1, n=0
    """
    a = c = g = t = n = 0
    i = 0
    length = len(bases_str)

    while i < length:
        ch = bases_str[i]

        if ch == '^':
            # Read start marker. The next character is a mapping quality flag
            # for this read — NOT a base call and NOT part of the mapq_str
            # column. Skip both ^ and the following char.
            i += 2
            continue

        if ch == '$':
            # Read end marker. Single character, no base associated.
            i += 1
            continue

        if ch in ('+', '-'):
            # Indel annotation. Format: +<n><bases> or -<n><bases>.
            # This token is attached to the preceding base call and describes
            # what happens AFTER this position. The preceding base was already
            # counted; now we just skip the indel token.
            #
            # Step 1: move past the + or -
            i += 1
            # Step 2: read the length digits (may be more than one digit)
            num_str = []
            while i < length and bases_str[i].isdigit():
                num_str.append(bases_str[i])
                i += 1
            # Step 3: skip over that many inserted/deleted base characters
            i += int(''.join(num_str)) if num_str else 0
            continue

        if ch in ('*', '<', '>'):
            # * = deletion placeholder: read is present but has a gap here.
            # < / > = reference skip: read spans this position via a splice
            #         or alignment gap and has no base here.
            # Neither represents a base call — not even N. Skip without counting.
            i += 1
            continue

        # Everything else is an actual base call at this position.
        # . and , are reference matches (forward and reverse strand).
        # Any letter is an explicit base call; lowercase = reverse strand,
        # uppercase = forward strand. Uppercasing normalizes both to the same.
        # N/n is an ambiguous call — the sequencer could not resolve the base.
        if ch in ('.', ','):
            b = ref_base
        else:
            b = ch.upper()

        if b == 'A':
            a += 1
        elif b == 'C':
            c += 1
        elif b == 'G':
            g += 1
        elif b == 'T':
            t += 1
        else:
            # Catches N/n (ambiguous base calls) and any unexpected characters.
            # Note: * / < / > are already handled above and never reach here.
            n += 1

        i += 1

    return a, c, g, t, n


def process_contig_mpileup(args_tuple):
    """Process a single contig via samtools mpileup subprocess.

    WHY this function exists (Change 5):
      The original code used pysam's Python pileup iterator:

          for pileupcolumn in bamfile.pileup(contig):
              for pileupread in pileupcolumn.pileups:   # ~9 billion iters/sample
                  base = pileupread.alignment.query_sequence[pileupread.query_position]

      Each `pileupread.alignment.query_sequence[...]` is a Python attribute chain
      on a C-backed object — fast for Python, but 100–1000x slower than pure C.
      For 3,000 samples this was the dominant HPC cost.

      This function instead runs `samtools mpileup` as a subprocess, which does
      the same base-counting loop entirely in compiled C. Python only has to parse
      one text line per position (not one line per read), reducing Python work by
      a factor equal to the mean coverage depth (~30x).

    WHY subprocess.Popen instead of pysam.samtools.mpileup():
      pysam.samtools.mpileup() buffers the *entire* output in memory before
      returning. For 306 M positions that is ~20+ GB — immediate OOM.
      subprocess.Popen with stdout=PIPE streams line-by-line, keeping RAM flat.

    Parameters:
        args_tuple (tuple): (contig_name, bam_path, fasta_path,
                            ignore_orphans, min_base_quality, min_mapping_quality,
                            ignore_overlaps)
          - contig_name (str): Contig to process (e.g., 'MRGM_0050_contig_2')
          - bam_path (str): Path to sorted, indexed BAM file
          - fasta_path (str): Path to indexed reference FASTA
          - ignore_orphans (bool): If True (default), discard reads without properly
            paired mates. If False, include orphans (samtools -A flag).
          - min_base_quality (int): Minimum base quality to include in pileup (samtools -Q)
          - min_mapping_quality (int): Minimum mapping quality to include (samtools -q)
          - ignore_overlaps (bool): If True (default), remove overlapping bases from
            paired reads to avoid double-counting (samtools default). If False, allow
            double-counting by passing samtools --ignore-overlaps flag.

    Returns:
        list: List of tuples (contig, position, ref_base, total, A, C, G, T, N)
              Example: [('MRGM_0050_contig_2', 88966, 'T', 2, 0, 0, 2, 0, 0)]
    """
    # Unpack the single args_tuple argument (required by multiprocessing.Pool.imap
    # which can only pass one argument per worker call).
    (contig_name, bam_path, fasta_path,
     ignore_orphans, min_base_quality, min_mapping_quality,
     ignore_overlaps) = args_tuple

    # Build the samtools mpileup command.
    # Flags are chosen to exactly match the original pysam.pileup() call:
    #   -r contig       : restrict to one contig at a time so workers run in parallel
    #   -q MQ           : minimum mapping quality  (was pysam stepper min_mapping_quality)
    #   -Q BQ           : minimum base quality     (was pysam stepper min_base_quality)
    #   -d 1e8          : lift samtools default depth cap (8000) to match pysam's unlimited
    #   -B              : disable BAQ recalculation (matches pysam compute_baq=False)
    #   --ignore-overlaps : disable samtools' default overlap detection (only when
    #                       ignore_overlaps=False, i.e., when we WANT double-counting)
    #   -A              : include orphan reads (only when ignore_orphans=False)
    cmd = [
        "samtools", "mpileup",
        "--region", contig_name,
        "--min-MQ", str(min_mapping_quality),
        "--min-BQ", str(min_base_quality),
        "--max-depth", "100000000",
        "--no-BAQ",
    ]
    if not ignore_overlaps:
        # --ignore-overlaps disables samtools' overlap detection, allowing double-
        # counting. Only pass it when ignore_overlaps=False (user wants double-counting).
        # When ignore_overlaps=True (default), samtools default already removes overlaps.
        cmd.append("--ignore-overlaps-removal")
    if not ignore_orphans:
        # --count-orphans = do NOT discard anomalous read pairs; equivalent to ignore_orphans=False
        cmd.append("--count-orphans")
    cmd += ["-f", fasta_path, bam_path]

    contig_data = []  # Accumulates one tuple per covered position on this contig
    try:
        # Launch samtools as a child process, capturing stdout line-by-line.
        # bufsize=1 enables line-buffered mode so we get lines as soon as samtools
        # writes them rather than waiting for a full OS buffer to fill.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        # Each line from samtools mpileup has 6 tab-separated columns:
        #   [0] contig name
        #   [1] 1-based position
        #   [2] reference base
        #   [3] read depth at this position
        #   [4] base calls string  (e.g. '..A,t' — see _parse_mpileup_bases)
        #   [5] base quality scores (Phred+33 ASCII, one char per read)
        for line in proc.stdout:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 6:
                continue
            contig = parts[0]
            pos = int(parts[1]) - 1  # Convert 1-based pos to 0-based to match pysam convention
            ref_base = parts[2].upper()
            depth = int(parts[3])    # Raw read depth before any filtering

            # Skip positions with no coverage (samtools emits these when --no-output-del is absent)
            if depth == 0:
                continue

            bases_str = parts[4]   # Encoded base calls (see _parse_mpileup_bases docstring)

            # Decode the mpileup base encoding into per-base counts.
            # This is the Python replacement for the inner pysam per-read loop.
            a, c, g, t, n = _parse_mpileup_bases(ref_base, bases_str)
            total = a + c + g + t + n

            # Skip positions where every read was a deletion or reference skip
            # (total would be 0 even though depth > 0 in those edge cases).
            if total == 0:
                continue
            contig_data.append((contig, pos, ref_base, total, a, c, g, t, n))

        # Wait for the samtools process to finish and check its exit code.
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read()
            logger.warning(
                f"samtools mpileup returned {proc.returncode} for {contig_name}: {stderr}"
            )
    except Exception as e:
        logger.error(f"Error running samtools mpileup for {contig_name}: {e}")
        raise
    return contig_data


# ---------------------------------------------------------------------------
# Gene mapping helpers
# ---------------------------------------------------------------------------


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
    logger.info("Parsing Prodigal FASTA file to extract gene information.")
    genes_data = []

    # Prodigal writes gene coordinates into the FASTA header, separated by '#'.
    # Example header: >k141_82760_1 # 100 # 300 # 1 # ID=1_1;partial=00
    #   parts[0] = gene ID  (e.g. 'k141_82760_1')
    #   parts[1] = start    (1-based)
    #   parts[2] = end      (1-based, inclusive)
    #   parts[3] = strand   (1 = forward, -1 = reverse)
    #   parts[4] = extra attributes
    for record in SeqIO.parse(prodigal_fasta, "fasta"):
        header = record.description
        parts = header.split("#")
        if len(parts) == 5:
            gene_id = parts[0]
            # Convert to 0-based half-open coordinates to match Python/pysam conventions:
            # Prodigal uses 1-based inclusive [start, end], so subtract 1 from both.
            start = int(parts[1]) - 1
            end = int(parts[2]) - 1
            genes_data.append(
                {
                    "gene_id": gene_id,
                    "start": start,
                    "end": end,
                }
            )
        else:
            logger.warning(f"Warning: Unexpected header format: {header}.")
            raise ValueError("Unexpected header format in the FASTA file.")

    genes_df = pd.DataFrame(genes_data)
    # Derive the contig name for each gene so we can group genes by contig.
    # Gene IDs from Prodigal follow the pattern <contig>_<gene_number>, e.g.
    # 'MRGM_0050_contig_2_15' -> contig is 'MRGM_0050_contig_2'.
    genes_df["contig"] = genes_df["gene_id"].apply(extract_contigID)
    logger.info("Gene information extracted successfully.")
    return genes_df


def extract_contigID(gene_id):
    """
    Extract the contig ID from a given gene ID by removing the last
    underscore-delimited token (the gene number appended by Prodigal).

    Prodigal names genes as <contig>_<N> where N is the gene number on that
    contig. Splitting on '_' and dropping the last element recovers the contig.

    Parameters:
        gene_id (str): The gene ID string from which to extract the contig ID.
                       Example: 'SLG1007_DASTool_bins_35.fa_k141_82760_1'

    Returns:
        str: The extracted contig ID.
             Example: 'SLG1007_DASTool_bins_35.fa_k141_82760'
    """
    # Split on '_', discard the last element (gene number), rejoin the rest.
    # 'MRGM_0050_contig_2_15'.split('_')[:-1] -> ['MRGM', '0050', 'contig', '2']
    # '_'.join([...])                          -> 'MRGM_0050_contig_2'
    contig = "_".join(gene_id.split("_")[:-1])
    return contig





def build_mag_gene_trees(genes_df, mag_mapping_dict, mags_to_include=None):
    """Pre-compute a {mag_id: {contig: IntervalTree}} dict at startup.

    WHY this function exists (Change 3):
      The original code rebuilt the gene IntervalTree inside the per-MAG loop:

          for mag in mags:                         # 1,524 iterations
              mag_genes = genes_df[genes_df['contig'].apply(
                  lambda g: mag_mapping_dict.get(g) == mag)]   # O(total_genes) scan
              tree = create_intervalTree(mag_genes) # O(mag_genes) build

      This ran O(total_genes * N_MAGs) work — roughly 3.2 billion comparisons for
      a dataset with 2.1 M genes across 1,524 MAGs.

      This function does one O(total_genes) pass and partitions genes into per-MAG
      IntervalTrees up front. The main loop then does a simple dict lookup.

    WHY IntervalTree:
      Gene coordinates are arbitrary-length intervals. We need to answer
      "which gene (if any) contains position X on contig C?" efficiently.
      An IntervalTree answers this in O(log n + k) vs O(n) for a linear scan.
      For a contig with 500 genes, that is ~9 comparisons instead of 500.

    WHY mags_to_include:
      For a typical sample only a subset of MAGs have any mapped reads — the
      rest produce no pileup output and need no gene tree. Building trees for
      uncovered MAGs wastes both time and RAM. Passing the set of MAGs that
      have at least one covered contig (derived from get_index_statistics)
      lets us skip all genes belonging to uncovered MAGs in a single check
      per gene row, at negligible cost.

    Parameters:
        genes_df (pd.DataFrame): All genes with columns gene_id, start, end, contig.
        mag_mapping_dict (dict): contig_id -> mag_id mapping.
                                  Example: {'MRGM_0050_contig_2': 'MRGM_0050', ...}
        mags_to_include (set, optional): If provided, only build trees for MAG IDs
                                  in this set. MAGs absent from the set are skipped
                                  entirely. Pass None (default) to build for all MAGs.

    Returns:
        dict: {mag_id: defaultdict(IntervalTree)} ready for per-position lookups.
              Example:
              {
                  'MRGM_0050': {
                      'MRGM_0050_contig_2': IntervalTree([
                          Interval(88960, 89521, 'MRGM_0050_contig_2_1 '),
                      ]),
                  }
              }
    """
    # ``n_requested`` may be an int count or the literal "all" when no filter
    # is supplied — only the int branch supports the thousands separator.
    if mags_to_include is not None:
        n_requested_str = f"{len(mags_to_include):,}"
    else:
        n_requested_str = "all"
    logger.info(f"Pre-computing gene IntervalTree index for {n_requested_str} MAGs...")
    start_time = time.time()

    mag_trees = {}
    for _, row in genes_df.iterrows():
        contig = row["contig"]
        # Look up which MAG this contig belongs to; skip genes from unmapped contigs
        mag_id = mag_mapping_dict.get(contig)
        if mag_id is None:
            continue
        # Skip MAGs that have no mapped reads in this sample — their gene trees
        # would never be queried, so building them wastes time and RAM.
        if mags_to_include is not None and mag_id not in mags_to_include:
            continue
        if mag_id not in mag_trees:
            mag_trees[mag_id] = defaultdict(IntervalTree)
        # addi(start, end+1, data): IntervalTree uses half-open [start, end) intervals,
        # so we add 1 to end to make the interval inclusive at the gene's last base.
        mag_trees[mag_id][contig].addi(row["start"], row["end"] + 1, row["gene_id"])

    elapsed = time.time() - start_time
    logger.info(
        f"Built gene trees for {len(mag_trees)} MAGs in {elapsed:.2f} seconds"
    )
    return mag_trees


def lookup_gene(contig_trees, contig, position):
    """Look up gene_id for a single (contig, position) pair.

    Queries the IntervalTree for the given contig to find any intervals
    that overlap the given position. If multiple intervals overlap,
    their data (gene IDs) are comma-joined.

    Parameters:
        contig_trees (dict): {contig: IntervalTree} dictionary for a specific MAG.
        contig (str): The contig name (e.g., 'MRGM_0050_contig_2')
        position (int): The 0-based coordinate position to query (e.g., 88966)

    Returns:
        str or None: Comma-joined gene IDs if overlapping, else None.
                     Example hit: 'MRGM_0050_contig_2_gene_45'
                     Example miss: None
    """
    tree = contig_trees.get(contig)
    if tree:
        overlaps = tree.at(position)
        if overlaps:
            return ",".join(interval.data for interval in overlaps)
    return None





# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
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
        "--mag_mapping_file",
        help="Path to tab-separated file mapping contig names to MAG IDs. "
        "Must have columns 'contig_name' and 'mag_id'.",
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

    parser.add_argument(
        "--sampleID",
        help="Sample identifier to use (overrides automatic extraction from BAM filename)",
        type=str,
        required=False,
        metavar="string",
    )

    parser.add_argument(
        "--no-ignore-orphans",
        dest="ignore_orphans",
        action="store_false",
        default=True,
        help="Do not ignore orphan reads (include reads without a properly paired mate). Default behavior is to ignore orphans.",
    )

    parser.add_argument(
        "--min-base-quality",
        dest="min_base_quality",
        type=int,
        default=30,
        metavar="int",
        help="Minimum base quality score to include a base in pileup. Default: 30.",
    )

    parser.add_argument(
        "--min-mapping-quality",
        dest="min_mapping_quality",
        type=int,
        default=2,
        metavar="int",
        help="Minimum mapping quality score to include a read in pileup. Default: 2.",
    )

    parser.add_argument(
        "--no-ignore-overlaps",
        dest="ignore_overlaps",
        action="store_false",
        default=True,
        help="Do not ignore overlapping read segments (may result in double-counting). Default behavior is to ignore overlaps.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level.",
    )
    args = parser.parse_args()

    # Setup logging with the specified level
    setup_logging(level=getattr(logging, args.log_level))

    start_time = time.time()

    # Log pileup parameters
    logger.info(
        f"Pileup parameters: ignore_orphans={args.ignore_orphans}, "
        f"min_base_quality={args.min_base_quality}, "
        f"min_mapping_quality={args.min_mapping_quality}, "
        f"ignore_overlaps={args.ignore_overlaps}"
    )

    # Ensure BAM index exists
    if not os.path.exists(args.bam_path + ".bai"):
        logger.info("BAM file index not found. Indexing it now..")
        pysam.index(
            args.bam_path, "-o", args.bam_path + ".bai", "--threads", "args.cpus"
        )

    # Ensure FASTA index exists
    if not os.path.exists(args.fasta_path + ".fai"):
        logger.info("FASTA file index not found. Indexing it now..")
        pysam.faidx(args.fasta_path, "-o", args.fasta_path + ".fai")

    # Load MAG mappings
    mag_mapping_dict = load_mag_mapping(args.mag_mapping_file)

    bamfile_main = pysam.AlignmentFile(args.bam_path, "rb")

    # --- Change 1: Skip contigs with 0 mapped reads ---
    # WHY: The original code dispatched a worker process for every contig in the
    # reference (often 50,000+), even those with zero mapped reads. Workers for
    # empty contigs open the BAM, seek, find nothing, and return an empty list —
    # pure overhead. BAM index statistics are stored in the .bai file and read in
    # microseconds, making this a near-free pre-filter.
    #
    # Example: a mouse gut sample may have 52,000 reference contigs but only
    # 18,000 with any coverage. Skipping the other 34,000 saves ~34,000 subprocess
    # launches and BAM seeks.
    idx_stats = bamfile_main.get_index_statistics()
    contigs_with_reads = {s.contig for s in idx_stats if s.mapped > 0}
    total_contigs = len(bamfile_main.references)
    skipped_contigs = total_contigs - len(contigs_with_reads)
    logger.info(
        f"Index statistics: {total_contigs:,} total contigs, "
        f"{len(contigs_with_reads):,} with mapped reads, "
        f"{skipped_contigs:,} skipped (0 mapped reads)"
    )

    bamfile_main.close()

    num_processes = args.cpus

    # Build a {mag_name: [contig, ...]} dict restricted to covered contigs only.
    # This is the work queue for the multiprocessing pool — one entry per contig.
    mag_contigs = defaultdict(list)
    for contig in contigs_with_reads:
        if contig in mag_mapping_dict:
            mag_name = extract_mag_id(contig, mag_mapping_dict)
            mag_contigs[mag_name].append(contig)

    logger.info(
        f"MAGs with at least one covered contig: {len(mag_contigs):,}"
    )

    # Load all gene annotations once and build trees upfront (Change 3).
    # The genes_df is a large DataFrame (~2.1 M rows for the mouse gut database).
    # We build the trees and then immediately free the DataFrame to reclaim RAM
    # before the multiprocessing pool starts allocating per-worker memory.
    genes_df = map_genes(args.prodigal_fasta)

    # --- Change 3: Pre-compute gene IntervalTree per MAG ---
    # Only build trees for MAGs that have at least one covered contig.
    # mag_contigs was derived from get_index_statistics(), so it already
    # excludes MAGs with zero mapped reads — no point building their trees.
    mag_gene_trees = build_mag_gene_trees(
        genes_df, mag_mapping_dict, mags_to_include=set(mag_contigs.keys())
    )
    del genes_df   # Free ~500 MB before the pool starts
    gc.collect()   # Force immediate deallocation rather than waiting for GC

    sampleID = args.sampleID if args.sampleID else Path(args.bam_path).stem

    # --- Process all MAGs: Change 5 (mpileup subprocess) + Change 4 (streamed output) ---
    #
    # Pool is created once outside the MAG loop so workers are reused across MAGs.
    # Creating a new Pool per MAG would add ~0.5 s of fork/exec overhead × 1,524 MAGs.
    with mp.Pool(processes=num_processes) as pool:
        for mag_name in tqdm(
            mag_contigs, total=len(mag_contigs), desc="Processing MAGs"
        ):
            contigs = mag_contigs[mag_name]

            # Retrieve the pre-built IntervalTree for this MAG (Change 3).
            # Falls back to empty dict if the MAG had no annotated genes.
            contig_trees = mag_gene_trees.get(mag_name, {})

            outDir = os.path.join(args.output_dir, sampleID)
            os.makedirs(outDir, exist_ok=True)
            outPath = os.path.join(outDir, f"{sampleID}_{mag_name}_profiled.tsv.gz")

            # Pack all arguments into tuples because multiprocessing.Pool.imap
            # can only pass a single argument to each worker function.
            contig_args = [
                (contig, args.bam_path, args.fasta_path,
                 args.ignore_orphans, args.min_base_quality,
                 args.min_mapping_quality, args.ignore_overlaps)
                for contig in contigs
            ]

            # --- Change 4: Stream output directly to gzip via csv.writer ---
            # WHY: The original code collected all rows into a Python list, built
            # a pandas DataFrame (expensive for 1.7 M rows), then called df.to_csv().
            # For a large MAG this temporarily allocated ~3–4 GB of RAM.
            #
            # Now we open the gzip file before dispatching workers and write each
            # row as it arrives from the pool. Peak RAM is bounded by the size of
            # a single contig's results (~a few MB), not the entire MAG.
            #
            # imap_unordered is used instead of imap because we don't need contigs
            # in order — it lets workers return results as soon as they finish,
            # reducing idle time when contigs have unequal depth.
            has_data = False
            with gzip.open(outPath, 'wt', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(PROFILE_HEADER)

                for contig_data in pool.imap_unordered(
                    process_contig_mpileup, contig_args
                ):
                    for row in contig_data:
                        contig, pos = row[0], row[1]
                        # Inline gene lookup using the pre-built tree (Change 3).
                        # lookup_gene returns None for intergenic positions.
                        gene_id = lookup_gene(contig_trees, contig, pos)
                        writer.writerow(list(row) + [gene_id if gene_id else ""])
                        has_data = True

            # If samtools produced no output for this MAG (e.g. all reads filtered
            # by quality), remove the file rather than leaving a header-only stub.
            if not has_data:
                os.remove(outPath)

            # Explicitly trigger GC after each MAG to release any lingering
            # reference cycles from the pool result objects before the next MAG.
            gc.collect()

    end_time = time.time()
    logger.info(f"Total time taken {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
