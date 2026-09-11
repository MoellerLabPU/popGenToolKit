#!/usr/bin/env python3
"""
Consolidate Per-MAG Regional Contrast Results and Apply Cross-MAG FDR Correction
==================================================================================

This script discovers all per-MAG regional contrast summary files produced by
``regional_contrast.py``, merges them into one table per region type
(gene / window), and applies a second round of Benjamini–Hochberg FDR
correction across all MAGs combined.

Background
----------
``regional_contrast.py`` runs independently per MAG and applies FDR correction
within each MAG.  These per-MAG q-values are preserved in the output as-is
(column names unchanged, e.g. ``q_value_wilcoxon_greater``).

This script adds a complementary *global* FDR correction that treats every
region across **all** MAGs as a single hypothesis pool, yielding
``q_value_*_global`` columns.  The global correction is applied separately
for each test direction (greater / less) and test type (Wilcoxon / t-test /
Fisher), so each directional hypothesis pool retains its own FDR budget —
mirroring the logic used in ``p_value_summary.py``.

Expected input file naming convention (produced by the Snakemake workflow)::

    {input_dir}/{mag_id}_regional_contrast_{gene|window}_region_summary.tsv

Output files::

    {outdir}/{prefix}_gene_region_summary.tsv
    {outdir}/{prefix}_window_region_summary.tsv

Usage
-----
::

    alleleflux-regional-contrast-summary \\
        --input-dir /path/to/regional_contrast/regional_contrast_pre_post-treatment_control \\
        --outdir /path/to/output \\
        --prefix regional_contrast_combined

"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────
#
# P-value columns written by regional_contrast.py that get global FDR correction.
# These represent different statistical tests and directions:
#   - Wilcoxon: nonparametric signed-rank test (greater/less direction)
#   - ttest: parametric one-sample t-test (greater/less direction)
#   - fisher: empirical p-values combined via Fisher's method (treatment/control)
#
# Fisher columns are optional and only present when --use_fisher was set
# in regional_contrast.py. The skipping logic in apply_all_global_fdr() handles
# missing columns gracefully.
_PVAL_COLS = [
    "p_value_wilcoxon_greater",
    "p_value_wilcoxon_less",
    "p_value_ttest_greater",
    "p_value_ttest_less",
    "p_value_fisher_treatment",
    "p_value_fisher_control",
]

# Filename fragment that separates {mag_id} from the rest of the filename.
# Used to extract MAG identifiers from region summary files.
# Expected format: {mag_id}_regional_contrast_{region_type}_region_summary.tsv
_FILE_TAG = "_regional_contrast_"


# ── File discovery ──────────────────────────────────────────────────────


def find_summary_files(input_dir: Path, region_type: str) -> list[Path]:
    """Discover all regional contrast summary files for a given region type.

    Searches for files matching the Snakemake workflow output naming convention:
    ``*_regional_contrast_{region_type}_region_summary.tsv``. These files are
    produced independently for each MAG by the regional_contrast.py script.

    Parameters
    ----------
    input_dir : Path
        Directory to search (non-recursive). Should point to a single timepoint-group
        combination output directory from the workflow (e.g.,
        ``regional_contrast_pre_post-treatment_control``).
    region_type : str
        Filter by region type: ``"gene"`` or ``"window"``.

    Returns
    -------
    list[Path]
        Sorted list of matching file paths (empty list when none are found).
        Files are sorted lexicographically for reproducibility.
    """
    pattern = f"*{_FILE_TAG}{region_type}_region_summary.tsv"
    files = sorted(input_dir.glob(pattern))
    logger.info(
        f"Found {len(files)} {region_type} region summary file(s) in {input_dir}"
    )
    return files


# ── MAG ID extraction ───────────────────────────────────────────────────


def extract_mag_id(filepath: Path, region_type: str) -> str:
    """Extract the MAG identifier from a regional contrast summary filename.

    Parses the Snakemake workflow output naming convention:

    .. code-block:: text

        {mag_id}_regional_contrast_{region_type}_region_summary.tsv
                                     ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^ ^
                                     region_type   fixed suffix        extension

    The MAG ID is everything before the ``_regional_contrast_{region_type}``
    fragment, allowing MAG IDs to themselves contain underscores.

    Parameters
    ----------
    filepath : Path
        Path to the summary TSV file.
    region_type : str
        ``"gene"`` or ``"window"`` — must match the region type in the filename.

    Returns
    -------
    str
        Extracted MAG ID string (portion before ``_regional_contrast_{region_type}``).

    Raises
    ------
    ValueError
        If the expected filename suffix is not found. Indicates either:
        - The file is not a regional contrast output file, or
        - The region_type parameter does not match the actual file name.
    """
    suffix = f"{_FILE_TAG}{region_type}_region_summary"
    stem = filepath.stem  # drop .tsv
    if suffix not in stem:
        raise ValueError(
            f"Cannot extract MAG ID from '{filepath.name}'. "
            f"Expected filename containing '{suffix}'."
        )
    return stem.split(suffix)[0]


# ── Loading & consolidation ─────────────────────────────────────────────


def load_and_consolidate(input_dir: Path, region_type: str) -> pd.DataFrame:
    """Load all per-MAG summary files for *region_type* and concatenate them.

    A ``mag_id`` column is prepended to the returned DataFrame.

    Parameters
    ----------
    input_dir : Path
        Directory containing per-MAG summary TSV files.
    region_type : str
        ``"gene"`` or ``"window"``.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with ``mag_id`` as the first column, or an empty
        DataFrame if no files are found.
    """
    # Discover all matching files for this region type
    files = find_summary_files(input_dir, region_type)
    if not files:
        logger.warning(f"No {region_type} summary files found — skipping.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for filepath in tqdm(
        files, desc=f"Loading {region_type} files", unit="file", total=len(files)
    ):
        # Extract MAG ID from filename for traceability in the consolidated output
        mag_id = extract_mag_id(filepath, region_type)

        # Load the per-MAG summary TSV; per-MAG q-values are preserved unchanged
        df = pd.read_csv(filepath, sep="\t")
        if df.empty:
            logger.warning(f"Empty file, skipping: {filepath.name}")
            continue

        # Insert MAG ID as the first column for easy identification and grouping
        df.insert(0, "mag_id", mag_id)
        frames.append(df)

    if not frames:
        logger.warning(f"All {region_type} files were empty or unreadable.")
        return pd.DataFrame()

    # Concatenate all per-MAG DataFrames; region counts may vary across MAGs
    # because regions are filtered by eligibility (min_replicates, etc.) in
    # regional_contrast.py. The global FDR correction below treats all regions
    # as a single hypothesis pool across all MAGs.
    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Consolidated {len(frames)} MAG(s) → {len(combined):,} {region_type} regions"
    )
    return combined


# ── Global FDR correction ───────────────────────────────────────────────


def apply_global_fdr(
    df: pd.DataFrame,
    p_col: str,
    q_col: str,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply FDR correction to a p-value column across all MAGs (global scope).

    This function implements a second-stage FDR correction that treats every
    region across all MAGs as a single hypothesis pool. Only non-NaN p-values
    participate in the correction; NaN rows receive NaN q-values and do not
    count as tested hypotheses. This preserves the same logic as
    ``adjust_pvalues`` in ``regional_contrast.py`` (per-MAG) but now at the
    global level.

    Note: The per-MAG q-values (unchanged column names, e.g.
    ``q_value_wilcoxon_greater``) are preserved in the output for comparison
    and transparency. The global q-values are added in new columns with a
    ``_global`` suffix.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table across all MAGs for a single region type (modified
        in-place and returned).
    p_col : str
        Source p-value column name (e.g., ``"p_value_wilcoxon_greater"``)
        that contains the p-values to correct.
    q_col : str
        Destination q-value column name (written as a new column). Typically
        derived by replacing ``p_value_`` with ``q_value_`` and appending
        ``_global``.
    method : str
        Correction method passed to ``statsmodels.stats.multitest.multipletests``.
        Default: ``"fdr_bh"`` (Benjamini–Hochberg).

    Returns
    -------
    pd.DataFrame
        Same frame with *q_col* appended as a new column.
    """
    # Identify rows with valid (non-NaN) p-values for this test direction
    mask = df[p_col].notna()
    n_tests = mask.sum()

    # Edge case: no testable p-values in this column (e.g., subset of tests failed)
    if n_tests == 0:
        df[q_col] = np.nan
        return df

    # Apply BH-FDR correction to non-NaN p-values only.
    # multipletests returns (reject_bool_array, p_corrected, alpha_sidak, alpha_bonf);
    # we extract p_corrected (index 1), which are the adjusted p-values (q-values).
    _, q_vals, _, _ = multipletests(df.loc[mask, p_col].values, method=method)

    # Initialize q-value column with NaN, then fill in corrected values at masked positions
    df[q_col] = np.nan
    df.loc[mask, q_col] = q_vals

    # Summary: report how many regions pass significance threshold (q < 0.05)
    n_sig = (df[q_col] < 0.05).sum()
    logger.info(
        f"Global FDR ({method}) on '{p_col}': "
        f"{n_sig:,} regions with q < 0.05 out of {n_tests:,} tested"
    )
    return df


def apply_all_global_fdr(df: pd.DataFrame, method: str = "fdr_bh") -> pd.DataFrame:
    """Apply global FDR correction for all testable p-value columns.

    Iterates through all known p-value column names in ``_PVAL_COLS``, applies
    global FDR correction to each that is present in the DataFrame, and appends
    a corresponding ``*_global`` q-value column.

    This two-stage approach allows users to see both:
    1. Per-MAG q-values (preserved from ``regional_contrast.py``):  e.g.
       ``q_value_wilcoxon_greater`` (how significant within that MAG)
    2. Global q-values (computed here): e.g.
       ``q_value_wilcoxon_greater_global`` (how significant across all MAGs)

    Missing p-value columns (e.g., Fisher columns when ``--use_fisher`` was False
    in ``regional_contrast.py``) are silently skipped, ensuring the script works
    regardless of which optional tests were run.

    FDR correction is applied **independently per test direction** so that each
    directional hypothesis pool (greater / less) retains its own α budget,
    improving statistical power for each direction individually.

    Parameters
    ----------
    df : pd.DataFrame
        Combined table across all MAGs for a single region type (genes or windows).
    method : str
        BH-FDR method string passed to ``multipletests``. Default: ``"fdr_bh"``.

    Returns
    -------
    pd.DataFrame
        Same frame with global q-value columns (one per p-value column present).
        Columns are named ``q_value_{test}_{direction}_global``.
    """
    # Process each known p-value column, skipping those not present in the DataFrame
    for p_col in _PVAL_COLS:
        if p_col not in df.columns:
            # Column not present (e.g., Fisher tests not run); skip gracefully
            continue

        # Derive the corresponding global q-value column name by:
        #   1. Replacing "p_value_" prefix with "q_value_"
        #   2. Appending "_global" suffix
        # Examples:
        #   p_value_wilcoxon_greater     → q_value_wilcoxon_greater_global
        #   p_value_fisher_treatment     → q_value_fisher_treatment_global
        q_col = p_col.replace("p_value_", "q_value_", 1) + "_global"

        # Apply FDR correction for this test direction's p-value pool
        df = apply_global_fdr(df, p_col=p_col, q_col=q_col, method=method)

    return df


# ── Output ──────────────────────────────────────────────────────────────


def write_output(df: pd.DataFrame, out_path: Path) -> None:
    """Write DataFrame to disk as a tab-separated file.

    Creates parent directories if they do not exist. Output files are uncompressed
    TSVs for easy viewing and downstream analysis in spreadsheet applications.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write.
    out_path : Path
        Target file path (typically named ``{prefix}_{region_type}_region_summary.tsv``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {len(df):,} rows → {out_path}")


# ── Entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the regional contrast summary script."""
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate per-MAG regional contrast results and apply "
            "cross-MAG (global) BH-FDR correction.  Produces one output "
            "table for gene regions and one for window regions."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing per-MAG regional contrast summary TSV files "
            "named ``{mag_id}_regional_contrast_{gene|window}_region_summary.tsv``."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory (created if absent).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="region_contrast_summary",
        help="Prefix for output file names.",
    )
    parser.add_argument(
        "--region-types",
        nargs="+",
        choices=["gene", "window"],
        default=["gene", "window"],
        help="Region type(s) to process.",
    )
    args = parser.parse_args()

    setup_logging()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Process each region type (gene and/or window) independently
    # This ensures separate q-value distributions and FDR budgets for each type
    any_output = False
    for region_type in args.region_types:
        logger.info(f"── Processing region type: {region_type} ──")

        # Step 1: Load and consolidate all per-MAG results for this region type
        combined = load_and_consolidate(args.input_dir, region_type)
        if combined.empty:
            logger.warning(f"No data for region type '{region_type}'. Skipping.")
            continue

        # Step 2: Apply global (across all MAGs) FDR correction
        # This adds *_global q-value columns while preserving per-MAG q-values
        logger.info(f"Applying global FDR correction for {region_type} regions …")
        combined = apply_all_global_fdr(combined, method="fdr_bh")

        # Step 3: Write consolidated results with both per-MAG and global q-values
        out_path = args.outdir / f"{args.prefix}_{region_type}_region_summary.tsv"
        write_output(combined, out_path)
        any_output = True

    # Validation: ensure at least one output file was produced
    if not any_output:
        logger.error(
            "No output was produced. Please verify:\n"
            "  - Input directory contains *_regional_contrast_*_region_summary.tsv files\n"
            "  - Requested region types match available files (gene/window)"
        )
        sys.exit(1)

    logger.info("Consolidation and global FDR correction complete.")


if __name__ == "__main__":
    main()
