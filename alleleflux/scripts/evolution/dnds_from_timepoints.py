#!/usr/bin/env python3
"""
dN/dS Analysis from Ancestral to Derived States (NG86 Path Averaging)
======================================================================

This script performs dN/dS (non-synonymous to synonymous substitution ratio) analysis
by comparing ancestral and derived states at significant sites identified by AlleleFlux.

The script implements the Nei-Gojobori (1986) method with proper path averaging for
codons where multiple positions change. This provides more accurate dN/dS estimates
than treating each changed position independently.

The script accepts the output from p_value_summary.py as input, allowing users to:
- Choose between raw p-values (min_p_value) or FDR-corrected values (q_value)
- Apply significance thresholds for site filtering
- Filter by specific test types if desired

Key Features:
- Reconstruction of ancestral sequences from profile data
- Calculation of potential synonymous and non-synonymous sites using Nei-Gojobori method
- NG86 path averaging for codons with multiple changes (k=2 or k=3)
- Codon-level analysis with fractional S/N counts
- Global dN/dS calculation avoiding double-counting of codons
- Gene-level and MAG-level summary statistics

NG86 Path Averaging:
When multiple positions change within a codon (k>1), the script considers all k!
possible mutational pathways and averages the S/NS classifications across paths.
This accounts for the path-dependence of amino acid changes and provides more
accurate evolutionary rate estimates.

Expected Input Format (from p_value_summary.py):
- TSV file with columns: mag_id, contig, position, gene_id, test_type, min_p_value, q_value
- Profile data from AlleleFlux profile_mags.py
- Prodigal-predicted gene sequences and coordinates

Usage:
    python dnds_from_timepoints.py --significant_sites p_value_summary_results.tsv \
                                  --mag_id MAG_001 \
                                  --p_value_column q_value \
                                  --p_value_threshold 0.05 \
                                  --ancestral_sample_id pre_sample \
                                  --derived_sample_id post_sample \
                                  --profile_dir /path/to/profiles \
                                  --prodigal_fasta genes.fasta \
                                  --outdir results/
"""

import argparse
import itertools
import logging
import multiprocessing as mp
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

from alleleflux.scripts.analysis.allele_frequency._allele_freq_common import (
    PROFILE_READ_DTYPES,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

# Set up logger for this script
logger = logging.getLogger(__name__)

# --- Global Constants ---
# Defines the complement for each DNA base. Used for reverse strand genes.
COMPLEMENT_MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}

# NCBI genetic code table 11 (bacterial/archaeal) used throughout the script
GENETIC_CODE_TABLE = CodonTable.unambiguous_dna_by_id[11]

# --- dN/dS Pre-computation Functions (Nei-Gojobori Method) ---


def _calculate_codon_sites(
    codon: str, table: CodonTable.CodonTable
) -> tuple[float, float]:
    """
    Helper function to calculate S and N sites for a single codon.

    For a given codon, this function iterates through all 9 possible single-base
    changes and classifies each as synonymous or non-synonymous. It returns
    the fractional count of potential S and N sites for that codon.

    The Nei-Gojobori method calculates potential sites by considering all possible
    single nucleotide changes at each position and determining what fraction would
    be synonymous vs non-synonymous. Since there are 3 possible changes at each
    of the 3 positions (9 total), each position contributes 1/3 to the final count.

    Args:
        codon: A 3-base DNA string (e.g., "ATG").
        table: A Biopython CodonTable object for translation.

    Returns:
        A tuple containing the fractional number of synonymous sites and
        non-synonymous sites for the codon.
    """
    s_codon, ns_codon = 0.0, 0.0

    # Explicitly handle stop codons and invalid codons for clarity.
    if codon in table.stop_codons:
        aa_orig = "*"  # Use '*' to represent a stop codon.
    else:
        # Use .get() to handle codons that might not be in the forward table.
        aa_orig = table.forward_table.get(codon)

    # If the codon is invalid (e.g., contains 'N' or '-'), we can't analyze it.
    if aa_orig is None:
        raise ValueError(f"Invalid codon: {codon}. Cannot analyze non-standard codons.")

    # Iterate through each of the 3 positions in the codon (0, 1, 2).
    for i in range(3):
        original_base = codon[i]
        # Iterate through the 3 possible alternative bases at this position.
        for new_base in "ATGC":
            if new_base == original_base:
                continue  # Skip the original base, as it's not a change.

            # Create the new codon by substituting the base.
            new_codon_list = list(codon)
            new_codon_list[i] = new_base
            new_codon_str = "".join(new_codon_list)

            # Check if the new codon is a stop codon.
            if new_codon_str in table.stop_codons:
                aa_new = "*"
            else:
                aa_new = table.forward_table.get(new_codon_str)

            # If the new codon is invalid, count it as non-synonymous.
            if aa_new is None:
                raise ValueError(
                    f"New codon is invalid: {new_codon_str}. Cannot analyze non-standard codons."
                )

            # Compare amino acids to classify the hypothetical mutation.
            if aa_new == aa_orig:
                s_codon += 1  # The change was synonymous.
            else:
                ns_codon += 1  # The change was non-synonymous.

    # Each site contributes a fraction (s/3 or ns/3) to the total.
    # This averages the potential outcomes over the 3 positions.
    # For example, if position 1 had 2 synonymous and 1 non-synonymous change,
    # position 2 had 1 synonymous and 2 non-synonymous, etc., we average them.
    return s_codon / 3.0, ns_codon / 3.0


def _precompute_codon_sites_cache() -> dict:
    """
    Pre-computes and caches the S and N site counts for all 64 possible codons.
    This avoids redundant calculations and significantly improves performance.

    Since there are only 64 possible codons (4^3), we can calculate the S/N sites
    for each one upfront and store them in a dictionary. This cache is used by
    calculate_potential_sites() to avoid recalculating the same codon multiple times.

    Returns:
        A dictionary mapping each codon string to its (potential_S_sites, potential_NS_sites) tuple.
        Example: {"ATG": (0.33, 2.67), "TTT": (0.67, 2.33), ...}
    """
    logger.info(
        "Pre-computing potential S and N sites for all 64 codons using NCBI table 11."
    )
    bases = "ATGC"
    cache = {}
    # Generate all 64 possible codons programmatically.
    for a in bases:
        for b in bases:
            for c in bases:
                codon = a + b + c
                # Calculate S/N sites for each codon and store in the cache.
                # The key is the codon string, the value is the tuple of (S, N) sites.
                cache[codon] = _calculate_codon_sites(codon, GENETIC_CODE_TABLE)
    return cache


# --- Pre-computed cache for all 64 codons, created once at script startup ---
_CODON_SITE_CACHE = _precompute_codon_sites_cache()

# --- NG86 Codon Pair Cache (for path averaging) ---
# There are 64×64 = 4,096 possible ancestral→derived codon pairs. This cache stores
# averaged (frac_S, frac_N, num_valid_paths) for each pair, excluding paths that pass
# through an intermediate stop codon (final stop is allowed). Built lazily on first use.
_NG86_CODON_PAIR_CACHE = None


def calculate_potential_sites_for_gene(gene_seq_str: str) -> dict:
    """
    Calculates potential synonymous (S) and non-synonymous (N) sites for a gene
    using a pre-computed cache for maximum efficiency.

    Args:
        gene_seq_str: The string of the protein-coding DNA sequence.

    Returns:
        A dictionary containing the total counts for 'NS' and 'S' sites.
        Example: {"S": 45.67, "NS": 123.33}
    """
    potential_S_sites, potential_NS_sites = 0.0, 0.0
    # Iterate over the sequence in 3-base steps (codons).
    for i in range(0, len(gene_seq_str), 3):
        # Slice out the current 3-base codon.
        codon = gene_seq_str[i : i + 3].upper()
        # Skip partial codons at the end of a sequence.
        if len(codon) != 3:
            logger.warning(
                f"Gene sequence '{gene_seq_str}' of length {len(gene_seq_str):,} is not a multiple of 3. "
                f"Skipping partial codon at position {i}."
            )
            continue

        # Look up the pre-calculated S and N values from the cache. This is much
        # faster than re-calculating for every codon in every gene.
        # 1. First, get the result from the cache into a single variable.
        potential_codon_sites = _CODON_SITE_CACHE.get(codon)

        # 2. Now, check if the result is None (meaning the codon was not found).
        if potential_codon_sites is None:
            # If the codon is invalid, log a warning and skip it.
            # This handles non-standard bases (like 'Y', 'N') gracefully.
            logger.warning(
                f"Invalid codon '{codon}' found in gene '{gene_seq_str}' at position {i}. Skipping."
            )
            continue

        # 3. If the result is valid, unpack it into the two variables.
        s_codon, ns_codon = potential_codon_sites

        # Add the fractional counts for this codon to the gene's total.
        potential_S_sites += s_codon
        potential_NS_sites += ns_codon

    # Return the final summed totals for the gene.
    return {"S": potential_S_sites, "NS": potential_NS_sites}


# --- NG86 Path Averaging Functions ---


def _enumerate_ng86_paths(
    ancestral_codon: str, derived_codon: str, table: CodonTable.CodonTable
) -> tuple[float, float, int]:
    """
    Enumerate all single-nucleotide pathways and average S/NS classifications.

    Implements the Nei-Gojobori (1986) method of considering all possible
    orders of single-nucleotide mutations when multiple positions differ.

    The NG86 method accounts for the fact that when multiple positions change
    in a codon, the classification of each individual change (S vs NS) may
    depend on the order in which the changes occur. By averaging over all
    possible orders (permutations), we get a more accurate estimate that
    doesn't depend on arbitrary assumptions about the mutational pathway.

    Args:
        ancestral_codon: 3-base ancestral codon (e.g., "ATG")
        derived_codon: 3-base derived codon (e.g., "TTG")
        table: Biopython CodonTable for translation

    Returns:
        Tuple of (frac_S, frac_NS, num_valid_paths):
        - frac_S: Average synonymous steps across all valid paths
        - frac_NS: Average non-synonymous steps across all valid paths
        - num_valid_paths: Number of pathways not passing through intermediate stops

    Algorithm:
        1. Identify changed positions (k = 1, 2, or 3)
        2. If k=1: Single-step mutation, no averaging needed (optimization)
        3. If k>1: Generate all k! permutations of change orders
        4. For each permutation:
           a. Apply changes one at a time
           b. Translate intermediate codons
           c. Exclude path if it passes through an intermediate stop codon
           d. Count S/NS steps along the path
        5. Average counts across valid paths

    Example (k=1):
        >>> table = CodonTable.unambiguous_dna_by_id[11]
        >>> frac_S, frac_NS, npaths = _enumerate_ng86_paths("ATG", "TTG", table)
        >>> # ATG (Met) → TTG (Leu) is non-synonymous
        >>> # Result: (0.0, 1.0, 1)

    Example (k=2):
        >>> frac_S, frac_NS, npaths = _enumerate_ng86_paths("ATG", "TTT", table)
        >>> # Position 0: A→T, Position 2: G→T
        >>> # Path 1: ATG → TTG → TTT (Met→Leu→Phe: 2 NS steps)
        >>> # Path 2: ATG → ATT → TTT (Met→Ile→Phe: 2 NS steps)
        >>> # Average: (0+0)/2 = 0.0 S, (2+2)/2 = 2.0 NS
        >>> # Result: (0.0, 2.0, 2)


    Example (intermediate stop excluded):
        >>> # Consider ancestral CAA (Gln) → derived TAA (Stop), k=1 (pos 0 differs)
        >>> # Only one step and it is final stop, allowed. For a k=2 illustration:
        >>> # ancestral CAA (Gln) → derived TGA (Stop) with changes at positions 0 and 1
        >>> # Path order 1: CAA→TAA (Gln→Stop: intermediate stop) → TGA (Stop)
        >>> #   This path is excluded because it hits a stop before the final codon.
        >>> # Path order 2: CAA→CGA (Gln→Arg: NS) → TGA (Arg→Stop: NS)
        >>> #   This path is valid. Averaging includes only valid paths.

    Notes:
        - Maximum k! = 6 permutations (when k=3), so performance is not a concern
        - Returns (NaN, NaN, 0) if all paths pass through intermediate stops
        - Final codon being a stop is allowed; only intermediate stops are excluded
        - Stop codons are represented as "*" in translation
    """
    # Ensure codons are uppercase for consistency
    ancestral_codon = ancestral_codon.upper()
    derived_codon = derived_codon.upper()

    # Identify which positions differ between ancestral and derived codons
    changed_positions = [i for i in range(3) if ancestral_codon[i] != derived_codon[i]]
    k = len(changed_positions)

    # Special case: no changes (shouldn't happen in practice, but handle gracefully)
    if k == 0:
        # logger.warning(
        #     f"No changes detected between codons: {ancestral_codon} → {derived_codon}, skipping."
        # )
        return 0.0, 0.0, 0

    # Optimization for k=1 (most common case): single-step mutation, no averaging needed
    if k == 1:
        # Translate both codons to determine if synonymous or non-synonymous
        if ancestral_codon in table.stop_codons:
            aa_anc = "*"
        else:
            aa_anc = table.forward_table.get(ancestral_codon, "?")

        if derived_codon in table.stop_codons:
            aa_der = "*"
        else:
            aa_der = table.forward_table.get(derived_codon, "?")

        # Handle invalid codons
        if aa_anc == "?" or aa_der == "?":
            logger.warning(
                f"Invalid codon in single-step mutation: {ancestral_codon} → {derived_codon}"
            )
            return np.nan, np.nan, 0

        # Classify as S or NS
        if aa_anc == aa_der:
            return 1.0, 0.0, 1  # Synonymous
        else:
            return 0.0, 1.0, 1  # Non-synonymous

    # For k > 1: enumerate all permutations and average
    pathways = list(itertools.permutations(changed_positions))
    valid_paths = []

    for pathway_order in pathways:
        # Start with the ancestral codon
        current_codon = list(ancestral_codon)
        s_steps = 0
        ns_steps = 0
        path_valid = True

        # Apply changes one at a time according to this permutation order
        for pos_idx in pathway_order:
            # Create the next codon by applying the change at this position
            next_codon = current_codon.copy()
            next_codon[pos_idx] = derived_codon[pos_idx]
            next_codon_str = "".join(next_codon)

            # Translate current and next codons
            if "".join(current_codon) in table.stop_codons:
                aa_before = "*"
            else:
                aa_before = table.forward_table.get("".join(current_codon), "?")

            if next_codon_str in table.stop_codons:
                aa_after = "*"
            else:
                aa_after = table.forward_table.get(next_codon_str, "?")

            # Handle invalid codons
            if aa_before == "?" or aa_after == "?":
                logger.warning(
                    f"Invalid codon in pathway: {''.join(current_codon)} → {next_codon_str}"
                )
                path_valid = False
                break

            # Check for intermediate stop codons (NG86 standard: exclude these paths)
            # If translation results in a stop (*) before reaching the final derived_codon,
            # this path is invalid. Example: CAA→TGA (k=2). Order1: CAA→TAA (Gln→Stop)→TGA
            # hits a stop at the intermediate step and is excluded; Order2 that avoids
            # an intermediate stop is included in averaging.
            if aa_after == "*" and next_codon_str != derived_codon:
                path_valid = False
                break

            # Classify this step as S or NS
            if aa_before == aa_after:
                s_steps += 1
            else:
                ns_steps += 1

            # Move to the next codon in the pathway
            current_codon = next_codon

        # If this pathway is valid (didn't hit intermediate stops), record it
        if path_valid:
            valid_paths.append({"s_steps": s_steps, "ns_steps": ns_steps})

    # Calculate fractional counts by averaging over all valid paths
    if valid_paths:
        frac_S = np.mean([p["s_steps"] for p in valid_paths])
        frac_NS = np.mean([p["ns_steps"] for p in valid_paths])
        num_valid_paths = len(valid_paths)
    else:
        # All paths passed through intermediate stops - cannot analyze this codon
        # This is expected for certain codon transitions (e.g., stop ↔ non-stop)
        logger.warning(
            f"Codon transition {ancestral_codon} → {derived_codon}: "
            f"All {len(pathways)} pathways pass through intermediate stop codons. "
            f"This is expected for some transitions (e.g., TAA → TGG or vice-versa) and the codon will be excluded from analysis."
        )
        frac_S = np.nan
        frac_NS = np.nan
        num_valid_paths = 0

    return frac_S, frac_NS, num_valid_paths


def _build_ng86_codon_pair_cache() -> dict:
    """
    Pre-compute NG86 path averages for all 4096 possible codon pairs.

    This cache enables O(1) lookup during analysis, avoiding repeated
    permutation enumeration for the same codon transitions. Since there
    are only 64 possible codons, there are 64×64 = 4,096 possible codon
    pairs to pre-compute.

    The cache is built once at script startup (or on first use) and stores
    the fractional S/N counts for each possible ancestral→derived transition.

    Returns:
        Dict mapping (ancestral_codon, derived_codon) to (frac_S, frac_NS, num_valid_paths)

    Cache Structure:
        Key: (ancestral_codon, derived_codon) - tuple of two 3-base strings
        Value: (frac_S, frac_NS, num_valid_paths) - tuple of two floats and an int

    Cache Size:
        - 64 × 64 = 4,096 entries
        - Memory: ~100-200 KB (negligible)
        - Build time: ~0.1-0.5 seconds (one-time startup cost)

    Example Entries:
        >>> cache = _build_ng86_codon_pair_cache()
        >>> cache[("ATG", "ATG")]  # No change
        (0.0, 0.0, 0)
        >>> cache[("ATG", "TTG")]  # Single change (A→T): Met→Leu
        (0.0, 1.0, 1)
        >>> cache[("ATG", "TTT")]  # Two changes (A→T, G→T)
        (0.0, 2.0, 2)

    Notes:
        - Called once and stored in global variable _NG86_CODON_PAIR_CACHE
        - Significantly faster than computing paths on-the-fly
        - Cache is deterministic and can be validated against hand calculations
        - Uses the _enumerate_ng86_paths function for actual computation
    """
    logger.info(
        "Building NG86 codon pair cache for all 4,096 codon pairs (NCBI table 11)..."
    )
    bases = "ATGC"
    cache = {}

    # Generate all 64 possible codons
    all_codons = [a + b + c for a in bases for b in bases for c in bases]

    # Pre-compute all 4,096 codon pairs
    for anc_codon in all_codons:
        for der_codon in all_codons:
            # Calculate the NG86 path averaging for this codon pair
            frac_S, frac_NS, num_paths = _enumerate_ng86_paths(
                anc_codon, der_codon, GENETIC_CODE_TABLE
            )
            cache[(anc_codon, der_codon)] = (frac_S, frac_NS, num_paths)

    logger.info(f"NG86 cache built successfully with {len(cache):,} entries.")
    return cache


def _get_ng86_cache() -> dict:
    """
    Get the NG86 codon pair cache, building it lazily if not yet initialized.

    This function provides lazy initialization of the global cache to avoid
    building it if the script is imported but not executed.

    Returns:
        The global NG86 codon pair cache dictionary
    """
    global _NG86_CODON_PAIR_CACHE
    if _NG86_CODON_PAIR_CACHE is None:
        _NG86_CODON_PAIR_CACHE = _build_ng86_codon_pair_cache()
    return _NG86_CODON_PAIR_CACHE


# --- Core Data Loading and Sequence Reconstruction ---


def setup_and_load_data(
    significant_sites_path,
    mag_id,
    p_value_column,
    p_value_threshold,
    test_type,
    group_analyzed,
):
    """
    Loads and filters the initial significant sites data.
    This function reads the significant sites file (output from p_value_summary.py)
    and applies filtering to ensure we only analyze sites that meet our criteria.
    Args:
        significant_sites_path: Path to the significant sites file from p_value_summary.py
        mag_id: The MAG ID to filter for
        p_value_column: Either 'min_p_value' or 'q_value' for significance filtering
        p_value_threshold: Threshold value for significance filtering
        test_type: Optional test type to filter for.
        group_analyzed: Optional group name to filter for.
    Returns:
        A pandas DataFrame of sites to be processed, or None if no sites are found.
    """
    logger.info(f"Loading significant sites from {significant_sites_path}")
    sig_sites_df = pd.read_csv(significant_sites_path, sep="\t")

    # Validate required columns
    required_columns = ["mag_id", "contig", "position", "gene_id", p_value_column]
    missing_columns = [
        col for col in required_columns if col not in sig_sites_df.columns
    ]
    if missing_columns:
        logger.error(
            f"Required columns {missing_columns} not found in significant_sites file. Cannot continue."
        )
        return None

    logger.info(f"Filtering significant sites for mag_id '{mag_id}'")
    sig_sites_df = sig_sites_df[sig_sites_df["mag_id"] == mag_id].copy()
    logger.info(f"Found {len(sig_sites_df):,} sites for mag_id '{mag_id}'")

    if sig_sites_df.empty:
        logger.warning(f"No sites found for mag_id '{mag_id}'.")
        return None

    # Filter by test type if specified
    if test_type:
        if "test_type" in sig_sites_df.columns:
            logger.info(f"Filtering for test type: {test_type} ")
            sig_sites_df = sig_sites_df[sig_sites_df["test_type"] == test_type].copy()
            logger.info(
                f"After test type filtering, {len(sig_sites_df):,} sites remain for mag_id '{mag_id}'."
            )
        else:
            logger.warning(
                "`--test-type` provided but 'test_type' column not found. Skipping test type filtering."
            )

    # Filter by group_analyzed if specified
    if group_analyzed:
        if "group_analyzed" in sig_sites_df.columns:
            logger.info(
                f"Filtering for group_analyzed: {group_analyzed} for mag_id '{mag_id}'"
            )
            # Ensure the column doesn't have mixed types that would break comparison
            sig_sites_df = sig_sites_df[
                sig_sites_df["group_analyzed"] == group_analyzed
            ].copy()
            logger.info(
                f"After group filtering, {len(sig_sites_df)} sites remain for mag_id '{mag_id}'."
            )
        else:
            logger.warning(
                "`--group-analyzed` provided but 'group_analyzed' column not found. Skipping group filtering."
            )

    # Apply significance threshold filtering
    logger.info(
        f"Applying significance threshold: {p_value_column} <= {p_value_threshold}"
    )
    sig_sites_df = sig_sites_df[
        sig_sites_df[p_value_column] <= p_value_threshold
    ].copy()
    logger.info(
        f"After significance filtering ({p_value_column} <= {p_value_threshold}), "
        f"{len(sig_sites_df)} significant sites remain"
    )

    if sig_sites_df.empty:
        logger.warning(
            f"No sites meet the significance threshold ({p_value_column} <= {p_value_threshold}) "
            "and other filter criteria."
        )
        return None

    logger.info(
        "Pre-filtering sites to include only those with a valid, single gene_id."
    )
    sites_to_process = sig_sites_df.dropna(subset=["gene_id"]).copy()
    sites_to_process["gene_id"] = sites_to_process["gene_id"].astype(str).str.strip()
    sites_to_process = sites_to_process[~sites_to_process["gene_id"].str.contains(",")]

    logger.info(
        f"After pre-filtering, {len(sites_to_process)} sites remain with valid single gene_id for mag_id '{mag_id}'."
    )
    if sites_to_process.empty:
        logger.warning("No processable sites found after all filtering steps.")
        return None
    return sites_to_process


def get_major_allele(row, ref_base):
    """
    Determines the major allele from allele count data in a DataFrame row.

    If there's a tie in allele counts, it prioritizes the reference base.
    If the reference base is not part of the tie, it chooses one randomly.

    Args:
        row (pd.Series): A row from the profile DataFrame containing A, T, G, C counts.
        ref_base (str): The reference allele at this position.

    Returns:
        The single-character major allele (e.g., 'A').
    """
    # Create a dictionary of allele counts from the row.
    allele_counts = {
        "A": row.get("A", 0),
        "T": row.get("T", 0),
        "G": row.get("G", 0),
        "C": row.get("C", 0),
    }
    # Find the highest allele count.
    max_count = max(allele_counts.values())
    # If there's no coverage, default to the reference base.
    if max_count == 0:
        return ref_base
    # Find all alleles that have the maximum count.
    major_alleles = [
        allele for allele, count in allele_counts.items() if count == max_count
    ]
    # If there's only one major allele, return it.
    if len(major_alleles) == 1:
        return major_alleles[0]
    else:
        # In case of a tie, prefer the reference base if it's among the tied alleles.
        # Otherwise, break the tie randomly to avoid systematic bias.
        return ref_base if ref_base in major_alleles else random.choice(major_alleles)


def reconstruct_ancestral_sequences(unique_genes, prodigal_records, profile_by_gene):
    """
    Reconstructs ancestral sequences for a list of genes using profile data.

    For each gene, it starts with the reference sequence from Prodigal and then
    substitutes the major allele from the ancestral profile at variable sites.

    Args:
        unique_genes (list): A list of unique gene IDs to reconstruct.
        prodigal_records (dict): A dictionary containing parsed info for each gene.
        profile_by_gene (pd.DataFrameGroupBy): The ancestral profile data, grouped by gene_id.

    Returns:
        A tuple containing:
        - ancestral_orfs (dict): Maps gene_id to its reconstructed sequence string.
        - ancestral_major_alleles (dict): Maps (contig, position) to the major allele on the forward strand.
    """
    ancestral_orfs = {}
    ancestral_major_alleles = {}
    # Iterate through each unique gene that needs reconstruction.
    for gene_id in unique_genes:
        # Skip if the gene ID from the sites file isn't in the annotation file.
        if gene_id not in prodigal_records:
            logger.warning(
                f"Gene ID '{gene_id}' not found in prodigal FASTA, skipping reconstruction."
            )
            continue

        gene_info = prodigal_records[gene_id]
        # Start with the original reference sequence from the Prodigal FASTA file.
        ancestral_seq_list = list(gene_info["record"].seq)

        # Check if this gene has any sites in the ancestral profile data.
        if gene_id in profile_by_gene.groups:
            gene_profile = profile_by_gene.get_group(gene_id)
            # Iterate through each variable site for this gene.
            for _, row in gene_profile.iterrows():
                contig, contig_position = row["contig"], int(row["position"])
                # Get the position of this site within the gene's own coordinate system.
                _, pos_in_gene = get_codon_from_site(
                    contig_position, gene_info, ancestral_seq_list
                )

                if pos_in_gene is not None:
                    # This is the reference base from the original Prodigal FASTA file
                    ref_base = ancestral_seq_list[pos_in_gene]
                    # SANITY CHECK Make sure that the Reference base in profile is same as the one in prodigal
                    ref_base_from_profile = row["ref_base"].upper()
                    # The ref_base from prodigal needs to be oriented to the forward strand for comparison
                    ref_base_from_prodigal_fwd = (
                        COMPLEMENT_MAP.get(ref_base.upper())
                        if gene_info["strand"] == -1
                        else ref_base.upper()
                    )

                    if ref_base_from_prodigal_fwd != ref_base_from_profile:
                        logger.warning(
                            f"Reference base mismatch at {contig_position} in {gene_id}. "
                            f"Prodigal (fwd strand): '{ref_base_from_prodigal_fwd}', "
                            f"Profile: '{ref_base_from_profile}'. Using profile 'ref_base' for tie-breaking."
                        )
                    # Determine major allele, using the profile base as the tie-breaker
                    major_allele = get_major_allele(row, ref_base_from_profile)

                    # Store the forward-strand major allele for later comparison.
                    # This is simply the major allele from the profile, as it's always on the forward strand.
                    ancestral_major_allele_fwd = major_allele.upper()
                    ancestral_major_alleles[(contig, contig_position)] = (
                        ancestral_major_allele_fwd
                    )
                    # Determine the "effective allele" to insert into the gene sequence.
                    # For reverse strand genes, this must be the complement of the major allele.
                    effective_allele = (
                        COMPLEMENT_MAP.get(major_allele.upper())
                        if gene_info["strand"] == -1
                        else major_allele
                    )

                    if effective_allele:
                        ancestral_seq_list[pos_in_gene] = effective_allele
                    else:
                        logger.warning(
                            f"Could not determine effective allele for major allele '{major_allele}' in gene {gene_id}."
                        )

                else:
                    logger.warning(
                        f"Contig position {contig_position} for gene {gene_id} is out of bounds. Skipping."
                    )

        ancestral_orfs[gene_id] = "".join(ancestral_seq_list)
    return ancestral_orfs, ancestral_major_alleles


# --- Helper and Core dN/dS Analysis Functions ---
def get_codon_from_site(position, gene_info, sequence):
    """
    Helper to calculate a site's position within a gene and extract its codon.

    This function handles coordinate conversion based on whether the gene is on
    the forward (+) or reverse (-) strand.

    COORDINATE SYSTEM EXPLANATION:
    - Input position: 0-based position on the contig (from frequency file)
    - Prodigal coordinates: 1-based start/end positions on the contig
    - Gene sequence: Always 5'->3' coding sequence, regardless of strand

    For FORWARD genes: position in gene = SNV_pos - (gene_start - 1)
    For REVERSE genes: position in gene = (gene_end - 1) - SNV_pos

    STRAND HANDLING:
    - Forward strand: Use alleles directly from frequency file
    - Reverse strand: Complement alleles to match reverse-complement sequence

    EXAMPLE:
      - Contig is 10000 bp long.
      - SNV Position (0-based from freq file): 8025

      - FORWARD GENE EXAMPLE:
        - Prodigal Header: # 8000 # 8500 # 1
        - The gene sequence starts at contig position 8000.
        - Calculation: 8025 (SNV) - (8000 - 1) (gene start) = index 26
        - This is intuitive: the SNV is 26 bases into the gene.

      - REVERSE GENE EXAMPLE:
        - Prodigal Header: # 8000 # 8500 # -1
        - The provided gene sequence is the REVERSE COMPLEMENT. Its first base
          corresponds to contig position 8500.
        - Calculation: (8500 - 1) (gene end) - 8025 (SNV) = index 474
        - This is counter-intuitive but correct: the SNV is near the 'start'
          coordinate on the contig, which means it's at the *end* of the
          reverse-complemented sequence string.

    Args:
        position (int): The genomic position on the contig.
        gene_info (dict): Metadata for the gene (start, end, strand).
        sequence (list or str): The gene's sequence.

    Returns:
        A tuple of (codon_string, pos_in_gene) or (None, None) if the position is invalid.
    """
    strand = gene_info["strand"]
    # Calculate the 0-indexed position within the gene's sequence.
    if strand == 1:
        pos_in_gene = position - (gene_info["start"] - 1)
    elif strand == -1:
        pos_in_gene = (gene_info["end"] - 1) - position
    else:
        logger.warning(
            f"Unknown strand '{strand}' for gene {gene_info['record'].id}. Cannot get codon."
        )
        return None, None
    # Validate that the calculated position is within the bounds of the sequence.
    if not (0 <= pos_in_gene < len(sequence)):
        logger.warning(
            f"Calculated position {pos_in_gene} for gene {gene_info['record'].id} is out of sequence bounds (gene len: {len(sequence)})."
        )
        return None, None

    # Find the start of the 3-base codon containing this position.
    codon_start_index = (pos_in_gene // 3) * 3
    # Handle both string and list sequence types
    codon_slice = sequence[codon_start_index : codon_start_index + 3]
    codon = "".join(codon_slice) if isinstance(codon_slice, list) else codon_slice

    # Ensure a full codon was extracted.
    if len(codon) != 3:
        logger.warning(
            f"Site at contig pos {position} in gene {gene_info['record'].id} results in a partial codon ('{codon}')."
        )
        return None, None

    return codon, pos_in_gene


def _group_sites_into_codon_events(
    sites_df: pd.DataFrame,
    ancestral_major_alleles: dict,
    derived_profile_df: pd.DataFrame,
    prodigal_records: dict,
    ancestral_seqs: dict,
) -> dict:
    """
    Group per-site data into per-codon data structures for NG86 analysis.

    Converts a DataFrame where each row is a significant site into a dictionary
    where each key is a unique codon and the value contains all changes within
    that codon. This is the critical transformation that enables path averaging.

    The function handles:
    - Coordinate conversions (contig → gene → codon positions)
    - Strand orientation (forward vs reverse)
    - Multiple sites within the same codon
    - Validation of alleles (no ambiguous bases)

    Args:
        sites_df: Per-site significant sites DataFrame with columns:
                  mag_id, contig, position, gene_id
        ancestral_major_alleles: Dict mapping (contig, position) to ancestral allele
        derived_profile_df: Profile data for derived sample
        prodigal_records: Gene metadata (coordinates, strand)
        ancestral_seqs: Reconstructed ancestral sequences

    Returns:
        Dict mapping codon_key to codon_data:
        - codon_key: (mag_id, gene_id, contig, codon_start_index, strand)
        - codon_data: {
            'gene_info': gene metadata dict,
            'ancestral_seq': str (full gene sequence),
            'contig': str,
            'changes': [
                {'pos_in_codon': 0, 'contig_pos': 1020,
                 'ancestral_allele': 'A', 'derived_allele': 'T'},
                ...
            ]
          }

    Example:
        >>> codon_events = _group_sites_into_codon_events(sites_df, ...)
        >>> # Sites at positions 1020 and 1022 in same codon (positions 0 and 2)
        >>> key = ('MAG_001', 'gene_123', 'contig_1', 1020, 1)  # codon starts at 1020
        >>> codon_events[key]['changes']
        [
            {'pos_in_codon': 0, 'contig_pos': 1020,
             'ancestral_allele': 'A', 'derived_allele': 'T'},
            {'pos_in_codon': 2, 'contig_pos': 1022,
             'ancestral_allele': 'G', 'derived_allele': 'T'}
        ]

    Notes:
        - Handles forward and reverse strand orientation via COMPLEMENT_MAP
        - Validates that all alleles are A/T/G/C (raises ValueError on ambiguous bases)
        - Uses get_codon_from_site() for coordinate conversions
        - Skips sites where gene is not in prodigal_records
    """
    # Index the derived profile for fast lookups
    derived_profile_indexed = derived_profile_df.set_index(
        ["contig", "position", "gene_id"]
    )

    # Dictionary to hold codon events: key = codon identifier, value = codon data
    codon_events = {}
    logger.debug(f"Processing sites DataFrame:\n{sites_df}")
    for _, site_row in sites_df.iterrows():
        contig = site_row["contig"]
        position = site_row["position"]
        gene_id = str(site_row["gene_id"]).strip()
        mag_id = site_row["mag_id"]

        # Skip if gene not in records
        if gene_id not in prodigal_records:
            logger.warning(
                f"Gene ID '{gene_id}' at {contig}:{position} not found in prodigal records. Skipping."
            )
            continue

        gene_info = prodigal_records[gene_id]
        ancestral_seq = ancestral_seqs.get(gene_id)

        if ancestral_seq is None:
            logger.warning(
                f"Ancestral sequence for gene '{gene_id}' not found. Skipping site {contig}:{position}."
            )
            continue

        # Get the position within the gene and the codon
        _, pos_in_gene = get_codon_from_site(position, gene_info, ancestral_seq)

        if pos_in_gene is None:
            logger.warning(
                f"Could not determine position in gene for {contig}:{position} in gene {gene_id}. Skipping."
            )
            continue

        # Calculate which codon this site belongs to
        codon_start_index = (pos_in_gene // 3) * 3
        pos_in_codon = pos_in_gene % 3

        # Get the ANCESTRAL major allele from the pre-calculated dictionary.
        ancestral_allele_fwd = ancestral_major_alleles.get((contig, position))
        if ancestral_allele_fwd is None:
            logger.warning(
                f"Ancestral allele not found for {contig}:{position}. Skipping."
            )
            continue

        # Get derived allele from derived profile
        try:
            derived_site_row = derived_profile_indexed.loc[(contig, position, gene_id)]
        except KeyError:
            logger.warning(
                f"Site {contig}:{position} for gene {gene_id} not found in derived profile. Skipping."
            )
            continue
        # Determine the DERIVED major allele using the derived_site_row as reference.
        profile_ref_base_fwd = derived_site_row["ref_base"].upper()

        # SANITY CHECK: Make sure that the Reference base in profile is same as the one in prodigal
        # This is checking the reference against the reference base from prodigal, not derived.
        original_ref_seq = str(gene_info["record"].seq)
        # Get position within the gene's coordinate system (0-indexed)
        _, pos_in_gene_ref = get_codon_from_site(position, gene_info, original_ref_seq)

        if pos_in_gene_ref is not None:
            prodigal_base_on_strand = original_ref_seq[pos_in_gene_ref]
            # The ref_base from prodigal needs to be oriented to the forward strand for comparison
            prodigal_ref_base_fwd = (
                COMPLEMENT_MAP.get(prodigal_base_on_strand.upper())
                if gene_info["strand"] == -1
                else prodigal_base_on_strand.upper()
            )

            if profile_ref_base_fwd != prodigal_ref_base_fwd:
                logger.warning(
                    f"Reference base mismatch at {contig}:{position} in {gene_id}. "
                    f"Prodigal says: '{prodigal_ref_base_fwd}', "
                    f"Profile file says: '{profile_ref_base_fwd}'. "
                    f"Using profile 'ref_base' for tie-breaking."
                )

        derived_allele_fwd = get_major_allele(derived_site_row, profile_ref_base_fwd)

        # Validate alleles - no ambiguous bases allowed
        valid_bases = set("ATGC")
        if (
            ancestral_allele_fwd not in valid_bases
            or derived_allele_fwd not in valid_bases
        ):
            raise ValueError(
                f"Ambiguous base found at {contig}:{position} in gene {gene_id}: "
                f"ancestral='{ancestral_allele_fwd}', derived='{derived_allele_fwd}'. "
                f"Cannot analyze codons with ambiguous bases. Check profile data quality."
            )

        # Only include if there's an actual substitution
        if ancestral_allele_fwd == derived_allele_fwd:
            logger.debug(
                f"Site {contig}:{position} in gene {gene_id}: "
                f"Ancestral and derived alleles are identical ({ancestral_allele_fwd}). "
                f"No substitution occurred. Skipping."
            )
            continue  # No change, skip this site

        # Create codon key (unique identifier for each codon)
        codon_key = (mag_id, gene_id, contig, codon_start_index, gene_info["strand"])
        logger.debug(f"Adding site {contig}:{position} to codon event: {codon_key}")

        # Initialize codon event if not seen before
        if codon_key not in codon_events:
            codon_events[codon_key] = {
                "gene_info": gene_info,
                "ancestral_seq": ancestral_seq,
                "contig": contig,
                "changes": [],
            }

        # Add this change to the codon's list of changes
        codon_events[codon_key]["changes"].append(
            {
                "pos_in_codon": pos_in_codon,
                "pos_in_gene": pos_in_gene,
                "contig_pos": position,
                "ancestral_allele": ancestral_allele_fwd,
                "derived_allele": derived_allele_fwd,
            }
        )

    return codon_events


def analyze_codon_substitutions_with_ng86_paths(
    sites_df: pd.DataFrame,
    ancestral_seqs: dict,
    prodigal_records: dict,
    derived_profile_df: pd.DataFrame,
    ancestral_major_alleles: dict,
    ng86_cache: dict | None = None,
) -> pd.DataFrame:
    """
    Analyze codon-level substitutions using Nei-Gojobori path averaging.

    This function replaces the old find_substitutions() function and implements
    proper NG86 methodology. Instead of treating each changed position independently,
    it groups sites by codon and applies path averaging when multiple positions
    change within the same codon.

    Key Differences from Old Approach:
    - Groups sites by codon before analysis (not per-site)
    - Uses NG86 path averaging for codons with k>1 changes
    - Returns fractional S/N counts (not integer per-site counts)
    - One row per codon event (not one row per site)

    Args:
        sites_df: DataFrame of significant sites with columns:
                  mag_id, contig, position, gene_id
        ancestral_seqs: Dict mapping gene_id to reconstructed ancestral sequence
        prodigal_records: Dict mapping gene_id to gene metadata (start, end, strand)
        derived_profile_df: Profile data for derived timepoint
        ancestral_major_alleles: Dict mapping (contig, position) to ancestral allele
        ng86_cache: Optional pre-built NG86 codon-pair cache to avoid global lookup;
            if None, the cache will be retrieved lazily via _get_ng86_cache().

    Returns:
        DataFrame with one row per codon event, containing:
        - Codon identifiers (mag_id, gene_id, contig, codon_start_index, strand)
        - Ancestral and derived codons (3-base strings)
        - Ancestral and derived alleles (comma-separated major alleles at changed positions)
        - Amino acid translations
        - Mutation type: "S" (synonymous) or "NS" (non-synonymous)
        - Changed positions (comma-separated: "0,2")
        - Contig positions involved (comma-separated)
        - Gene positions involved (comma-separated: 0-indexed positions within the gene)
        - k = number of changed positions
        - frac_S, frac_NS = fractional counts from path averaging
        - num_valid_paths = number of pathways considered
        - potential_S_codon, potential_NS_codon = per-codon potentials (for global dN/dS)
        - potential_S_sites_gene, potential_NS_sites_gene = entire gene potentials (for summaries)
        - Validity flag (False if all paths hit stops)

    Example Output Row (k=1, NS):
        mag_id='MAG_001', gene_id='gene_123', codon_start_index=21,
        ancestral_codon='ATG', derived_codon='TTG',
        ancestral_allele='A', derived_allele='T',
        ancestral_aa='M', derived_aa='L', mutation_type='NS',
        changed_positions='0', k=1,
        frac_S=0.0, frac_NS=1.0, num_valid_paths=1,
        is_valid=True

    Example Output Row (k=2, NS):
        mag_id='MAG_001', gene_id='gene_123', codon_start_index=24,
        ancestral_codon='ATG', derived_codon='TTT',
        ancestral_allele='A,G', derived_allele='T,T',
        ancestral_aa='M', derived_aa='F', mutation_type='NS',
        changed_positions='0,2', k=2,
        frac_S=0.0, frac_NS=2.0, num_valid_paths=2,
        is_valid=True

    Raises:
        ValueError: If ambiguous bases (N, Y, etc.) found in major alleles

    Notes:
        - Uses cached codon pair lookups for O(1) performance per codon
        - Paths passing through intermediate stop codons are excluded (NG86 standard)
        - If all paths are invalid, returns NaN for fractional counts and is_valid=False
        - Handles forward and reverse strand orientation automatically
    """
    logger.info("Grouping sites into codon events for NG86 analysis...")

    # Step 1: Group sites by codon
    codon_events = _group_sites_into_codon_events(
        sites_df,
        ancestral_major_alleles,
        derived_profile_df,
        prodigal_records,
        ancestral_seqs,
    )

    # Calculate total number of sites in codon events
    total_sites_in_events = sum(
        len(event["changes"]) for event in codon_events.values()
    )

    logger.info(
        f"Identified total sites {total_sites_in_events} and {len(codon_events):,} unique codon events from {len(sites_df):,} significant sites."
    )
    logger.info(
        f"Sites successfully grouped into codon events: {total_sites_in_events:,} "
        f"({len(sites_df) - total_sites_in_events:,} sites were filtered out)"
    )

    if not codon_events:
        logger.warning("No codon events identified. Returning empty DataFrame.")
        return pd.DataFrame()

    # Get the NG86 cache (use provided cache if available; otherwise build lazily)
    if ng86_cache is None:
        ng86_cache = _get_ng86_cache()

    # Step 2: Process each codon event
    results = []

    for codon_key, codon_data in codon_events.items():
        mag_id, gene_id, contig, codon_start_index, strand = codon_key
        # gene_info = codon_data["gene_info"]
        ancestral_seq = codon_data["ancestral_seq"]
        changes = codon_data["changes"]

        # Extract the ancestral codon from the ancestral sequence
        ancestral_codon_list = list(
            ancestral_seq[codon_start_index : codon_start_index + 3]
        )

        if len(ancestral_codon_list) != 3:
            logger.warning(
                f"Partial codon at {gene_id}:{codon_start_index} (length={len(ancestral_codon_list)}). Skipping."
            )
            continue

        ancestral_codon_str = "".join(ancestral_codon_list).upper()

        # Build the derived codon by applying all changes
        derived_codon_list = ancestral_codon_list.copy()

        # Build a unified list and sort by contig_pos to keep fields aligned
        entries = []
        for change in changes:
            pos_in_codon = change["pos_in_codon"]
            ancestral_allele_fwd = change["ancestral_allele"]
            derived_allele_fwd = change["derived_allele"]
            contig_pos = change["contig_pos"]
            pos_in_gene = change[
                "pos_in_gene"
            ]  # Already calculated in _group_sites_into_codon_events

            # For reverse strand genes, we need to complement the allele
            # because the sequence is stored as reverse complement
            # Alleles from profiles are from the forward strand. They must be complemented for reverse-strand genes.
            if strand == -1:
                effective_allele = COMPLEMENT_MAP.get(derived_allele_fwd.upper())
                if not effective_allele:
                    logger.warning(
                        f"Cannot complement derived allele '{derived_allele_fwd}' for reverse strand gene {gene_id}. Skipping codon."
                    )
                    continue
            else:
                effective_allele = derived_allele_fwd.upper()

            entries.append(
                {
                    "pos_in_codon": pos_in_codon,
                    "pos_in_gene": pos_in_gene,
                    "contig_pos": contig_pos,
                    "ancestral_allele_fwd": ancestral_allele_fwd,
                    "derived_allele_fwd": derived_allele_fwd,
                    "effective_allele": effective_allele,
                }
            )

        # Sort ONCE by gene position (5'→3' biological order), then derive everything from the same order
        # This keeps all fields aligned and maintains consistent ordering across forward/reverse strands
        entries.sort(key=lambda e: e["pos_in_gene"])

        # Apply edits to the codon in the sorted order (order doesn't change outcome, but keeps reporting aligned)
        for e in entries:
            derived_codon_list[e["pos_in_codon"]] = e["effective_allele"]

        derived_codon_str = "".join(derived_codon_list).upper()

        # Now output aligned, comma-joined fields (all in 5'→3' gene order)
        contig_positions_list = [e["contig_pos"] for e in entries]
        gene_positions_list = [e["pos_in_gene"] for e in entries]
        changed_positions_list = [e["pos_in_codon"] for e in entries]
        ancestral_alleles_list = [e["ancestral_allele_fwd"] for e in entries]
        derived_alleles_list = [e["derived_allele_fwd"] for e in entries]

        # Validate codons
        if len(derived_codon_str) != 3:
            logger.warning(
                f"Derived codon for {gene_id}:{codon_start_index} is not 3 bases. Skipping."
            )
            continue

        # Look up the NG86 path averaging results from cache
        cache_key = (ancestral_codon_str, derived_codon_str)
        if cache_key in ng86_cache:
            frac_S, frac_NS, num_valid_paths = ng86_cache[cache_key]
        else:
            # Fallback: compute on-the-fly if not in cache (shouldn't happen)
            logger.warning(
                f"Codon pair {cache_key} not in cache. Computing on-the-fly."
            )
            frac_S, frac_NS, num_valid_paths = _enumerate_ng86_paths(
                ancestral_codon_str, derived_codon_str, GENETIC_CODE_TABLE
            )

        # Translate codons to amino acids
        if ancestral_codon_str in GENETIC_CODE_TABLE.stop_codons:
            ancestral_aa = "*"
        else:
            ancestral_aa = GENETIC_CODE_TABLE.forward_table.get(
                ancestral_codon_str, "?"
            )

        if derived_codon_str in GENETIC_CODE_TABLE.stop_codons:
            derived_aa = "*"
        else:
            derived_aa = GENETIC_CODE_TABLE.forward_table.get(derived_codon_str, "?")

        # Get per-codon potential S/NS sites from the cache (for global dN/dS calculation)
        # These are fractional values representing the mutational opportunity at this specific codon
        potential_sites = _CODON_SITE_CACHE.get(ancestral_codon_str)
        if potential_sites:
            potential_S_codon, potential_NS_codon = potential_sites
        else:
            logger.warning(
                f"Codon {ancestral_codon_str} not in potential sites cache. Using NaN."
            )
            potential_S_codon, potential_NS_codon = np.nan, np.nan

        # Get ENTIRE gene potential S/NS sites from prodigal_records (for gene/MAG summaries)
        # These represent the total mutational opportunity across the full length of the gene
        # Note: All codon events within the same gene will have identical gene-level values
        gene_potential_S = prodigal_records.get(gene_id, {}).get(
            "potential_S_sites", np.nan
        )
        gene_potential_NS = prodigal_records.get(gene_id, {}).get(
            "potential_NS_sites", np.nan
        )

        if np.isnan(gene_potential_S) or np.isnan(gene_potential_NS):
            logger.warning(
                f"Gene {gene_id} does not have potential sites calculated. Using NaN."
            )

        # Determine if this codon event is valid
        is_valid = (
            num_valid_paths > 0 and not np.isnan(frac_S) and not np.isnan(frac_NS)
        )

        # Determine mutation type based on amino acid change
        if ancestral_aa == derived_aa:
            mutation_type = "S"  # Synonymous
        else:
            mutation_type = "NS"  # Non-synonymous

        # Store the result
        results.append(
            {
                "mag_id": mag_id,
                "contig": contig,
                "contig_position": ",".join(map(str, contig_positions_list)),
                "gene_id": gene_id,
                "gene_position": ",".join(map(str, gene_positions_list)),
                "codon_start_index": codon_start_index,
                "codon_position": ",".join(map(str, changed_positions_list)),
                "strand": strand,
                "ancestral_allele": ",".join(ancestral_alleles_list),
                "derived_allele": ",".join(derived_alleles_list),
                "ancestral_codon": ancestral_codon_str,
                "derived_codon": derived_codon_str,
                "ancestral_aa": ancestral_aa,
                "derived_aa": derived_aa,
                "mutation_type": mutation_type,
                "k": len(changed_positions_list),
                "frac_S": frac_S,
                "frac_NS": frac_NS,
                "num_valid_paths": num_valid_paths,
                "potential_S_codon": potential_S_codon,
                "potential_NS_codon": potential_NS_codon,
                "potential_S_sites_gene": gene_potential_S,
                "potential_NS_sites_gene": gene_potential_NS,
                "is_valid": is_valid,
            }
        )

    codon_events_df = pd.DataFrame(results)

    if not codon_events_df.empty:
        # Log summary statistics
        total_codons = len(codon_events_df)
        k1_codons = (codon_events_df["k"] == 1).sum()
        k2_codons = (codon_events_df["k"] == 2).sum()
        k3_codons = (codon_events_df["k"] == 3).sum()
        invalid_codons = (~codon_events_df["is_valid"]).sum()

        logger.info(
            f"\nCodon Event Summary:\n"
            f"  Total codons analyzed: {total_codons:,}\n"
            f"  Codons with k=1 (single change): {k1_codons:,}\n"
            f"  Codons with k=2 (two changes): {k2_codons:,}\n"
            f"  Codons with k=3 (three changes): {k3_codons:,}\n"
            f"  Invalid codons (all paths hit stops): {invalid_codons:,}"
        )

    return codon_events_df


# --- Global dN/dS Calculation ---
def calculate_global_dnds_for_sites(substitution_results_df):
    """
    Calculates a single, global dN/dS ratio using NG86 fractional counts.

    This function now works with codon-level events (not per-site) and uses
    fractional S/N counts from path averaging. It still avoids double-counting
    potential sites by tracking unique codons.

    Key Changes from Previous Version:
    - Works with codon events (one row per codon) instead of per-site substitutions
    - Sums fractional frac_S and frac_N counts instead of integer S/NS tallies
    - Includes additional metrics (k distribution, invalid codons)

    Args:
        substitution_results_df (pd.DataFrame): DataFrame of codon events from
            analyze_codon_substitutions_with_ng86_paths(). Expected columns:
            - mag_id, gene_id, codon_start_index
            - frac_S, frac_NS (fractional counts)
            - k (number of changes)
            - is_valid (bool)
            - potential_S_codon, potential_NS_codon (per-codon potentials for NG86)
            - potential_S_sites_gene, potential_NS_sites_gene (entire gene potentials, used in summaries)

    Returns:
        pd.DataFrame: Global dN/dS calculation summary with fractional counts, or None if no data.

    Example Output:
        summary_df with rows like:
        - Total Unique Codons Analyzed: 245
        - Codons with k=1: 198
        - Codons with k=2: 42
        - Codons with k=3: 5
        - Invalid Codons: 3
        - Total Potential Non-Synonymous Sites (ns): 612.33
        - Total Potential Synonymous Sites (s): 122.67
        - Observed Non-Synonymous Substitutions (NS): 56.75  # Sum of frac_NS
        - Observed Synonymous Substitutions (S): 12.25  # Sum of frac_S
        - pNS (NS / ns): 0.0927
        - pS (S / s): 0.0999
        - Global dN/dS (pN/pS): 0.9279

    Notes:
        - Input is now codon-centric (one row per codon, not per site)
        - Observed S/N are fractional sums, not integer counts
        - Invalid codons (is_valid=False) are excluded from calculations
        - Denominators are sums of potential sites from unique codons
    """
    logger.info(
        f"Calculating global dN/dS for {len(substitution_results_df):,} codon events..."
    )
    if substitution_results_df.empty:
        logger.warning("No codon events found, cannot calculate global dN/dS.")
        return None

    # Filter out invalid codons (where all paths hit stops)
    valid_codons_df = substitution_results_df[
        substitution_results_df["is_valid"] == True
    ].copy()
    invalid_count = len(substitution_results_df) - len(valid_codons_df)

    if valid_codons_df.empty:
        logger.warning(
            f"All {len(substitution_results_df):,} codon events are invalid (all paths hit stops). "
            f"Cannot calculate global dN/dS."
        )
        return None

    logger.info(
        f"Using {len(valid_codons_df):,} valid codon events "
        f"({invalid_count:,} invalid codons excluded)."
    )

    total_potential_S = 0.0
    total_potential_NS = 0.0

    # Use a set to track unique codons that have been processed to avoid double-counting.
    # The key will be a tuple: (gene_id, codon_start_index)
    processed_codons = set()

    # 1. Calculate the total potential S and N sites for the UNIQUE codons
    # Each codon should only contribute its potential sites ONCE
    for _, codon_row in valid_codons_df.iterrows():
        gene_id = str(codon_row["gene_id"]).strip()
        codon_start_index = codon_row["codon_start_index"]

        # Create unique identifier for this codon
        codon_identifier = (gene_id, codon_start_index)

        # Only process this codon if we haven't seen it before (avoid double-counting)
        if codon_identifier not in processed_codons:
            # Get per-codon potential S/NS sites from the DataFrame
            potential_S_codon = codon_row.get("potential_S_codon", np.nan)
            potential_NS_codon = codon_row.get("potential_NS_codon", np.nan)

            if not np.isnan(potential_S_codon) and not np.isnan(potential_NS_codon):
                total_potential_S += potential_S_codon
                total_potential_NS += potential_NS_codon
                processed_codons.add(codon_identifier)
            else:
                logger.warning(
                    f"Codon at {gene_id}:{codon_start_index} has NaN potential sites. Skipping."
                )

    # 2. Sum the observed fractional S and NS counts from the codon events
    # This is the key difference: we now sum fractional counts, not integer tallies
    observed_S = valid_codons_df["frac_S"].sum()
    observed_NS = valid_codons_df["frac_NS"].sum()

    # 3. Calculate pNS, pS, and the dN/dS ratio
    # Use np.nan for invalid calculations to distinguish from true zero rates
    pNS = observed_NS / total_potential_NS if total_potential_NS > 0 else np.nan
    pS = observed_S / total_potential_S if total_potential_S > 0 else np.nan
    dNdS_ratio = pNS / pS if pS > 0 and not np.isnan(pS) else np.nan

    # 4. Get distribution of k values for reporting
    k1_count = (valid_codons_df["k"] == 1).sum()
    k2_count = (valid_codons_df["k"] == 2).sum()
    k3_count = (valid_codons_df["k"] == 3).sum()

    # 5. Create a summary DataFrame with additional NG86 metrics
    summary_data = {
        "Metric": [
            "Total Unique Codons Analyzed",
            "Codons with k=1 (single change)",
            "Codons with k=2 (two changes)",
            "Codons with k=3 (three changes)",
            "Invalid Codons (all paths hit stops)",
            "Total Potential Non-Synonymous Sites (ns)",
            "Total Potential Synonymous Sites (s)",
            "Observed Non-Synonymous Substitutions (NS)",
            "Observed Synonymous Substitutions (S)",
            "pNS (NS / ns)",
            "pS (S / s)",
            "Global dN/dS (pNS/pS)",
        ],
        "Value": [
            len(valid_codons_df),
            k1_count,
            k2_count,
            k3_count,
            invalid_count,
            total_potential_NS,
            total_potential_S,
            observed_NS,  # Now fractional
            observed_S,  # Now fractional
            pNS,
            pS,
            dNdS_ratio,
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    logger.info(
        "\n--- Global dN/dS Summary (NG86 Path Averaging) ---\n"
        + summary_df.to_string(index=False)
    )

    return summary_df


# --- Summarization and Output Functions ---
def summarize_results(results_df):
    """
    Generates gene-level and MAG-level summary tables with dN/dS ratios.

    Updated to work with codon-level events and fractional S/N counts from NG86.

    Args:
        results_df: DataFrame of codon events with columns:
            - mag_id, gene_id, codon_start_index
            - frac_S, frac_NS (fractional counts)
            - potential_S_codon, potential_NS_codon (per-codon potentials, not used here)
            - potential_S_sites_gene, potential_NS_sites_gene (entire gene potentials)
            - is_valid (bool)

    Returns:
        Dict with keys 'gene' and 'mag' containing summary DataFrames.

    Notes:
        - Sums fractional frac_S and frac_NS counts across codon events
        - Denominators use ENTIRE gene potentials (potential_S_sites_gene, potential_NS_sites_gene)
        - Only includes valid codons (is_valid=True)
    """
    if results_df.empty:
        logger.warning("Results dataframe is empty. Cannot generate summaries.")
        return {}

    # Filter to only valid codons
    valid_results = results_df[results_df["is_valid"] == True].copy()
    invalid_count = len(results_df) - len(valid_results)
    logger.info(
        f"Using {len(valid_results):,} valid codon events "
        f"({invalid_count:,} invalid codons excluded)."
    )

    if valid_results.empty:
        logger.warning("No valid codon events. Cannot generate summaries.")
        return {}

    # --- Gene-level Statistics ---
    logger.info("Summarizing results at the gene level...")

    # Group by gene and aggregate fractional counts and gene potentials
    # Note: potential_S_sites_gene and potential_NS_sites_gene are the same for all codons in a gene,
    # so we can use 'first' to get the gene-level value
    gene_stats = (
        valid_results.groupby(["mag_id", "gene_id"])
        .agg(
            total_frac_s=("frac_S", "sum"),
            total_frac_ns=("frac_NS", "sum"),
            num_codon_events=("codon_start_index", "count"),
            potential_S_sites_gene=("potential_S_sites_gene", "first"),
            potential_NS_sites_gene=("potential_NS_sites_gene", "first"),
        )
        .reset_index()
    )

    # dN = (total fractional NS) / (total potential NS sites for entire gene)
    # Use np.nan for invalid calculations (denominator = 0 or NaN)
    gene_stats["dN"] = gene_stats.apply(
        lambda row: (
            row["total_frac_ns"] / row["potential_NS_sites_gene"]
            if row["potential_NS_sites_gene"] > 0
            and not np.isnan(row["potential_NS_sites_gene"])
            else np.nan
        ),
        axis=1,
    )
    # dS = (total fractional S) / (total potential S sites for entire gene)
    # Use np.nan for invalid calculations (denominator = 0 or NaN)
    gene_stats["dS"] = gene_stats.apply(
        lambda row: (
            row["total_frac_s"] / row["potential_S_sites_gene"]
            if row["potential_S_sites_gene"] > 0
            and not np.isnan(row["potential_S_sites_gene"])
            else np.nan
        ),
        axis=1,
    )
    # dN/dS ratio. Return NaN if dS is 0 to avoid division by zero errors.
    gene_stats["dN_dS_ratio"] = gene_stats.apply(
        lambda r: r["dN"] / r["dS"] if r["dS"] > 0 else np.nan, axis=1
    )

    # --- MAG-level Statistics ---
    logger.info("Summarizing results at the MAG level...")
    # Sum gene potentials across all genes in the MAG (skipping NaN values)
    # Each gene contributes its full potential exactly once
    mag_stats = (
        gene_stats.groupby("mag_id")
        .agg(
            total_frac_ns=("total_frac_ns", "sum"),
            total_frac_s=("total_frac_s", "sum"),
            num_genes=("gene_id", "count"),
            potential_NS_sites=(
                "potential_NS_sites_gene",
                lambda x: x.sum(skipna=True),
            ),
            potential_S_sites=("potential_S_sites_gene", lambda x: x.sum(skipna=True)),
        )
        .reset_index()
    )

    # Use np.nan for invalid calculations (denominator = 0 or NaN)
    mag_stats["dN"] = mag_stats.apply(
        lambda row: (
            row["total_frac_ns"] / row["potential_NS_sites"]
            if row["potential_NS_sites"] > 0 and not np.isnan(row["potential_NS_sites"])
            else np.nan
        ),
        axis=1,
    )
    mag_stats["dS"] = mag_stats.apply(
        lambda row: (
            row["total_frac_s"] / row["potential_S_sites"]
            if row["potential_S_sites"] > 0 and not np.isnan(row["potential_S_sites"])
            else np.nan
        ),
        axis=1,
    )
    mag_stats["dN_dS_ratio"] = mag_stats.apply(
        lambda r: r["dN"] / r["dS"] if r["dS"] > 0 else np.nan, axis=1
    )

    return {"gene": gene_stats, "mag": mag_stats}


def write_output_files(
    summaries,
    results_df,
    args,
    ancestral_sequences,
    prodigal_records,
    global_dnds_summary_df,
):
    """
    Writes the main events table, summary tables, and ancestral sequences to disk.
    Skips writing files if the corresponding dataframe is empty or None.

    Updated for NG86: results_df now contains codon-level events with fractional counts.
    """
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- Codon events (replaces all_substitutions) ---
    path_codon_events = args.outdir / f"{args.prefix}_codon_events_ng86.tsv"
    if results_df is not None and not results_df.empty:
        results_df.to_csv(path_codon_events, sep="\t", index=False)
        logger.info(
            f"Successfully wrote {len(results_df)} codon events to {path_codon_events}"
        )
    else:
        logger.warning(f"No codon events found. Skipping file: {path_codon_events}")

    # --- Gene and MAG summaries ---
    if summaries:
        for summary_type in ["gene", "mag"]:
            path = args.outdir / f"{args.prefix}_{summary_type}_summary_ng86.tsv"
            df = summaries.get(summary_type)
            if df is not None and not df.empty:
                df.to_csv(path, sep="\t", index=False)
                logger.info(f"Wrote {summary_type} summary (NG86) to {path}")
            else:
                logger.warning(
                    f"No {summary_type} summary was generated. Skipping file: {path}"
                )
    else:
        logger.warning(
            "No summaries were generated. Skipping gene and MAG summary files."
        )

    # --- Global dN/dS summary ---
    path_global_summary = args.outdir / f"{args.prefix}_global_dnds_ng86_summary.tsv"
    if global_dnds_summary_df is not None and not global_dnds_summary_df.empty:
        global_dnds_summary_df.to_csv(path_global_summary, sep="\t", index=False)
        logger.info(f"Wrote global dN/dS (NG86) summary to {path_global_summary}")
    else:
        logger.warning(
            f"No global dN/dS summary was generated. Skipping file: {path_global_summary}"
        )

    # --- Reconstructed ancestral sequences ---
    path_ancestral_fasta = args.outdir / f"{args.prefix}_ancestral_orfs.ffn"
    logger.info(f"Writing ancestral sequences to {path_ancestral_fasta}...")
    if ancestral_sequences:
        ancestral_records = []
        for gene_id, seq_str in ancestral_sequences.items():
            if gene_id in prodigal_records:
                original_description = prodigal_records[gene_id]["record"].description
                rest_of_description = original_description.replace(
                    gene_id, "", 1
                ).strip()
                record = SeqRecord(
                    Seq(seq_str),
                    id=f"ancestral_{gene_id}",
                    description=rest_of_description,
                )
                ancestral_records.append(record)
        with open(path_ancestral_fasta, "w") as output_handle:
            SeqIO.write(ancestral_records, output_handle, "fasta")
        logger.info(f"Successfully wrote {len(ancestral_records)} ancestral sequences.")
    else:
        logger.warning(
            f"No ancestral sequences to write. Skipping file: {path_ancestral_fasta}"
        )


def process_mag(mag_id, args, prodigal_records):
    """Process a single MAG and write output files."""
    # logger.info(f"--- Processing MAG: {mag_id} ---")

    # Create a copy of the args object to avoid modifying the original
    mag_args = argparse.Namespace(**vars(args))
    # Update the prefix and outdir for this specific MAG
    mag_args.prefix = f"{args.prefix}_{mag_id}_{args.test_type}_{args.p_value_column}"
    mag_args.outdir = args.outdir / mag_id

    # 1. Load and filter significant sites for this MAG
    sites_to_process = setup_and_load_data(
        args.significant_sites,
        mag_id,
        mag_args.p_value_column,
        mag_args.p_value_threshold,
        mag_args.test_type,
        mag_args.group_analyzed,
    )
    if sites_to_process is None or sites_to_process.empty:
        #     logger.warning(
        #         f"No processable sites found for MAG {mag_id} after filtering. Writing empty files and skipping."
        #     )
        #     write_output_files(None, None, mag_args, None, None, None, None)
        return

    # 2. Load profile data for this MAG
    ancestral_profile_path = (
        mag_args.profile_dir
        / f"{mag_args.ancestral_sample_id}/{mag_args.ancestral_sample_id}_{mag_id}_profiled.tsv.gz"
    )
    derived_profile_path = (
        mag_args.profile_dir
        / f"{mag_args.derived_sample_id}/{mag_args.derived_sample_id}_{mag_id}_profiled.tsv.gz"
    )

    if not ancestral_profile_path.exists() or not derived_profile_path.exists():
        logger.warning(
            f"Profile file(s) for MAG {mag_id} not found at {ancestral_profile_path} or {derived_profile_path}. Skipping."
        )
        return

    ancestral_profile_df = pd.read_csv(
        ancestral_profile_path, sep="\t", dtype=PROFILE_READ_DTYPES
    )
    derived_profile_df = pd.read_csv(
        derived_profile_path, sep="\t", dtype=PROFILE_READ_DTYPES
    )

    # 3. Reconstruct Ancestral Sequences
    logger.info(
        f"Reconstructing Ancestral Sequences for {mag_id} (genes: {sites_to_process['gene_id'].nunique()})"
    )
    ancestral_profile_df.dropna(subset=["gene_id"], inplace=True)
    ancestral_profile_df["gene_id"] = (
        ancestral_profile_df["gene_id"].astype(str).str.strip()
    )
    profile_by_gene = ancestral_profile_df.groupby("gene_id")
    unique_genes = sites_to_process["gene_id"].unique()
    ancestral_sequences, ancestral_major_alleles = reconstruct_ancestral_sequences(
        unique_genes, prodigal_records, profile_by_gene
    )

    if not ancestral_sequences:
        logger.warning(f"No ancestral sequences reconstructed for MAG {mag_id}.")

    # 4. Calculate potential S/N sites for each entire ancestral gene
    for gene_id, seq_str in ancestral_sequences.items():
        if gene_id in prodigal_records:
            potential_sites = calculate_potential_sites_for_gene(seq_str)
            prodigal_records[gene_id]["potential_S_sites"] = potential_sites["S"]
            prodigal_records[gene_id]["potential_NS_sites"] = potential_sites["NS"]

    # 5. Analyze codon substitutions using NG86 path averaging
    derived_profile_df.dropna(subset=["gene_id"], inplace=True)
    derived_profile_df["gene_id"] = (
        derived_profile_df["gene_id"].astype(str).str.strip()
    )

    # Check for duplicate (contig, position, gene_id) combinations
    duplicates = derived_profile_df[
        derived_profile_df.duplicated(
            subset=["contig", "position", "gene_id"], keep=False
        )
    ]

    if not duplicates.empty:
        duplicate_count = len(duplicates)
        unique_duplicates = len(
            duplicates.drop_duplicates(subset=["contig", "position", "gene_id"])
        )
        raise ValueError(
            f"MAG {mag_id}: Found {duplicate_count} duplicate entries across {unique_duplicates} "
            f"unique (contig, position, gene_id) combinations in derived profile. "
            f"First few examples:\n{duplicates[['contig', 'position', 'gene_id']].head(10)}"
        )

    logger.info(f"Analyzing substitutions for {mag_id}...")

    # Retrieve NG86 cache once per worker. On Linux (fork), this references the pre-warmed
    # cache from parent via copy-on-write. On Windows/macOS (spawn), builds lazily here.
    ng86_cache = _get_ng86_cache()

    results_df = analyze_codon_substitutions_with_ng86_paths(
        sites_to_process,
        ancestral_sequences,
        prodigal_records,
        derived_profile_df,
        ancestral_major_alleles,
        ng86_cache=ng86_cache,
    )

    # 6. Summarize results
    summaries = summarize_results(results_df)

    # 7. Calculate global dN/dS
    global_dnds_summary_df = calculate_global_dnds_for_sites(results_df)

    # 8. Write outputs
    write_output_files(
        summaries,
        results_df,
        mag_args,
        ancestral_sequences,
        prodigal_records,
        global_dnds_summary_df,
    )


def main():
    """Main execution function to orchestrate the entire analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze dN/dS from a reconstructed ancestral state to a derived state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--significant_sites",
        required=True,
        type=Path,
        help="Path to the significant sites dataframe (TSV) from p_value_summary.py. "
        "Expected columns: mag_id, contig, position, gene_id, test_type, min_p_value, q_value.",
    )
    parser.add_argument(
        "--mag_ids",
        nargs="+",
        required=True,
        help="One or more MAG IDs to process.",
    )
    parser.add_argument(
        "--p_value_column",
        choices=["min_p_value", "q_value"],
        default="q_value",
        help="Column to use for significance filtering: 'min_p_value' (raw p-values) or 'q_value' (FDR-corrected).",
    )
    parser.add_argument(
        "--p_value_threshold",
        type=float,
        default=0.05,
        help="Threshold for significance filtering using the selected p-value column.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=[
            "two_sample_unpaired_tTest",
            "two_sample_unpaired_MannWhitney",
            "two_sample_unpaired_tTest_abs",
            "two_sample_unpaired_MannWhitney_abs",
            "two_sample_paired_tTest",
            "two_sample_paired_Wilcoxon",
            "two_sample_paired_tTest_abs",
            "two_sample_paired_Wilcoxon_abs",
            "single_sample_tTest",
            "single_sample_Wilcoxon",
            "CMH",
            "LMM",
            "LMM_abs",
            # "lmm_across_time",
            # "cmh_across_time",
        ],
        default="two_sample_paired_tTest",
        help="A test type to filter the `significant_sites` dataframe.",
    )
    parser.add_argument(
        "--group-analyzed",
        type=str,
        help="Filter the `significant_sites` dataframe for a specific group, if the 'group_analyzed' column is present.",
    )
    parser.add_argument(
        "--ancestral_sample_id",
        required=True,
        help="The sample ID for the ANCESTRAL timepoint.",
    )
    parser.add_argument(
        "--derived_sample_id",
        required=True,
        help="The sample ID for the DERIVED timepoint for comparison.",
    )
    parser.add_argument(
        "--profile_dir",
        required=True,
        type=Path,
        help="Directory containing the profile files from profile_mags.py.",
    )
    parser.add_argument(
        "--prodigal_fasta",
        required=True,
        type=Path,
        help="Path to the Prodigal-predicted FASTA file for gene sequences and coordinates.",
    )
    parser.add_argument(
        "--outdir", required=True, type=Path, help="Directory to save the output files."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="ancestral_dnds",
        help="Prefix for the output file names.",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=mp.cpu_count(),
        help="Number of CPUs to use for multiprocessing. Default: use all available CPUs.",
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

    # Load Prodigal records once for all MAGs
    logger.info("Loading gene annotation file...")
    prodigal_raw = SeqIO.to_dict(SeqIO.parse(args.prodigal_fasta, "fasta"))
    prodigal_records = {}
    for rec in prodigal_raw.values():
        header_parts = [p.strip() for p in rec.description.split("#")]
        prodigal_records[rec.id] = {
            "record": rec,
            "start": int(header_parts[1]),
            "end": int(header_parts[2]),
            "strand": int(header_parts[3]),
        }

    # Determine number of CPUs to use, ensuring it doesn't exceed the number of MAGs
    num_cpus = min(args.cpus, len(args.mag_ids))
    logger.info(f"Using {num_cpus} CPUs to process {len(args.mag_ids)} MAGs")
    
    # Log which samples are being used as ancestral and derived
    logger.info(
        f"Sample configuration for dN/dS analysis: "
        f"Ancestral (Time 1) = '{args.ancestral_sample_id}', "
        f"Derived (Time 2) = '{args.derived_sample_id}'"
    )

    # Pre-warm the NG86 cache (4,096 codon pairs) once in parent before forking.
    # On Linux: workers inherit via copy-on-write (shared memory, ~0.5s saved per worker).
    # On Windows/macOS: no effect (spawn creates fresh processes), but doesn't hurt.
    _ = _get_ng86_cache()

    # Process each MAG using multiprocessing
    sorted_mag_ids = sorted(args.mag_ids)

    # Create a partial function with fixed args and prodigal_records
    process_mag_partial = partial(
        process_mag, args=args, prodigal_records=prodigal_records
    )

    # Use multiprocessing with imap_unordered and tqdm
    with mp.Pool(processes=num_cpus) as pool:
        # Use imap_unordered to get results as they complete
        list(
            tqdm(
                pool.imap_unordered(process_mag_partial, sorted_mag_ids),
                total=len(sorted_mag_ids),
                desc="Processing MAGs",
            )
        )

    logger.info("Analysis completed successfully for all processed MAGs!")


if __name__ == "__main__":
    main()
