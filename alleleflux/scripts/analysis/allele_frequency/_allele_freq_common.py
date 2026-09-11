"""Shared helpers for allele frequency cache and analysis steps.

Pipeline architecture
---------------------
Allele frequency processing is split into two stages to eliminate redundant
work when the same timepoint appears in multiple (timepoint_combination,
group_combination) pairs:

  Stage 1 — allele_freq_cache.py  (``alleleflux-cache-allele-freq``)
      Runs ONCE per (MAG, timepoint).
      Reads raw profile TSV files → computes per-base frequencies → writes
      one Parquet file: ``{OUTDIR}/allele_freq_cache/{mag}_{tp}.parquet``.

  Stage 2 — allele_freq.py  (``alleleflux-allele-freq``)
      Runs ONCE per (MAG, timepoint_combination, group_combination).
      Reads one or two Stage-1 Parquet files → computes per-subject diffs
      → writes ``_changes_mean.tsv.gz`` and optional ``_no_zero-diff.tsv.gz``.

This module (``_allele_freq_common.py``) holds the shared helpers that both
stages import.  Nothing else in the codebase should import from this module
directly; use the public API of allele_freq_cache.py or allele_freq.py.
"""

import logging

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names for per-base nucleotide *frequencies* (0–1 values computed
# from raw counts).  Used as the canonical list in both the cache writer
# (calculate_frequencies) and the analyzer (diff / filter steps).
NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]

# Columns retained in the Parquet cache.  Everything else written by
# process_mag_files (MAG_ID, ref_base, N, breadth, genome_size,
# average_coverage) is dropped before writing to avoid bloating the file
# with data no downstream consumer reads.
#
# Downstream consumers and the columns each needs:
#   allele_freq.py (Stage 2): contig, position, gene_id, total_coverage,
#       sample_id, group, subjectID, replicate, time, *_frequency
#   CMH.py (raw-count mode): contig, gene_id, position, group, replicate,
#       A, T, G, C, time
#   LMM.py (across_time):    subjectID, contig, gene_id, position, group,
#       replicate, time, *_frequency
CACHE_COLUMNS_BASE = [
    "contig", "position", "gene_id",
    "total_coverage",
    "A", "C", "G", "T",
    "sample_id", "group", "subjectID", "replicate",
    "A_frequency", "T_frequency", "G_frequency", "C_frequency",
]
# Longitudinal adds the "time" column so Stage 2 can pair timepoints.
CACHE_COLUMNS_LONGITUDINAL = CACHE_COLUMNS_BASE + ["time"]

# Explicit dtypes for reading raw profile TSVs.  Default pandas dtypes use
# int64/float64 which double the in-memory footprint of large profiles
# unnecessarily — coverage/positions/counts fit comfortably in int32
# (range ±2.1B), and 0–1 frequencies need only float32 precision (~7
# decimal digits).  Without these casts, a single MAG×timepoint cache
# can balloon to hundreds of GB in pandas.
PROFILE_READ_DTYPES = {
    "position": "int32",
    "total_coverage": "int32",
    "A": "int32",
    "C": "int32",
    "G": "int32",
    "T": "int32",
    # gene_id may be non-numeric (e.g. "k141_0_1") — keep as string.
    "gene_id": "string",
    "contig": "string",
    # N and ref_base are dropped from the cache; leaving them at default
    # dtype is fine since they're discarded immediately after column select.
}

# Low/medium-cardinality string columns that should be encoded as Arrow
# dictionary (pandas category) in the Parquet cache.  This is the largest
# single memory saving in the cache: pandas object strings are ~50 B/row,
# dictionary-encoded categories collapse to ~4 B/row (an int32 code).
# Excluded: gene_id and contig (high cardinality per MAG — dictionary
# helps but only modestly, and the code overhead is real).  We still
# dictionary-encode them in Parquet via use_dictionary=True at write time,
# but keep them as plain string in pandas to avoid category-merging cost
# during downstream concat.
CACHE_CATEGORICAL_COLUMNS = [
    "sample_id",
    "group",
    "subjectID",
    "replicate",
    "time",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QC loading
# ---------------------------------------------------------------------------

def load_qc_results(qc_file_path, mag_id):
    """Load a per-MAG QC TSV and return only the samples that passed.

    Called by ``allele_freq_cache.py``'s ``union_qc_samples()`` to identify
    which samples to include in the cache.  May be called with multiple QC
    files (one per combination) so that the cache union covers every sample
    relevant to this timepoint across all group combinations.

    Parameters
    ----------
    qc_file_path : str
        Path to the QC TSV for this MAG (written by quality_control.py).
    mag_id : str
        MAG identifier used only for logging.

    Returns
    -------
    pd.DataFrame
        Rows where ``coverage_threshold_passed == True``, with at minimum:
        sample_id, file_path, group, subjectID, replicate, breadth,
        average_coverage, genome_size, and ``time`` (for longitudinal data).

    Raises
    ------
    ValueError
        If the QC file has no ``coverage_threshold_passed`` column, or if
        no samples passed the threshold.
    """
    logger.info(f"Loading QC results from {qc_file_path}")

    qc_df = pd.read_csv(qc_file_path, sep="\t", dtype={"sample_id": str})

    if "coverage_threshold_passed" not in qc_df.columns:
        raise ValueError(
            f"QC file {qc_file_path} missing 'coverage_threshold_passed' column. "
            "Ensure quality_control.py was run properly."
        )

    # Keep only samples that cleared both breadth and average-coverage thresholds.
    # This mirrors the original allele_freq.py filter — the QC step has already
    # computed the boolean; we just respect it here.
    passed_df = qc_df[qc_df["coverage_threshold_passed"] == True].copy()

    if passed_df.empty:
        raise ValueError(
            f"No samples passed coverage threshold for MAG {mag_id}. "
            f"Total samples in QC file: {len(qc_df)}"
        )

    logger.info(
        f"Loaded {len(passed_df)} samples (out of {len(qc_df)} total) that passed coverage and breadth thresholds"
    )

    return passed_df


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------

def init_worker(metadata, data_type):
    """Initialise each worker process with shared, read-only state.

    Called as the ``initializer`` argument of ``multiprocessing.Pool``.
    Both ``allele_freq_cache.py`` and (previously) ``allele_freq.py`` use
    this to avoid pickling the potentially-large ``metadata`` dict on every
    ``imap_unordered`` call.

    After initialisation, worker processes access ``metadata_dict`` and
    ``DATA_TYPE`` as module-level globals inside ``process_mag_files``.

    Parameters
    ----------
    metadata : dict
        Maps ``sample_id -> {group, subjectID, replicate, breadth, ...}``.
        Built by ``build_metadata_from_qc()``.
    data_type : str
        ``"single"`` or ``"longitudinal"``.  Controls whether a ``time``
        column is attached to each row in ``process_mag_files``.
    """
    global metadata_dict
    global DATA_TYPE
    metadata_dict = metadata
    DATA_TYPE = data_type


def calculate_frequencies(df):
    """Compute per-base nucleotide frequencies in-place and return the frame.

    Divides each of the four raw count columns (A, T, G, C) by
    ``total_coverage`` to produce 0–1 frequency values stored as float32.
    The resulting columns match the ``NUCLEOTIDES`` constant defined above.

    Float32 precision (~7 decimal digits) is more than sufficient for
    frequencies in [0, 1]; using float64 doubles the memory footprint of
    these four columns without any meaningful downstream benefit.  The
    division is performed against a float32 ``total_coverage`` view so the
    output dtype is float32 without an extra promotion-then-cast pass.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``A``, ``T``, ``G``, ``C``, ``total_coverage``.

    Returns
    -------
    pd.DataFrame
        The same frame with four additional float32 columns:
        ``A_frequency``, ``T_frequency``, ``G_frequency``, ``C_frequency``.
    """
    # Compute against float32 total_coverage so each division stays in float32
    # rather than promoting through float64 and back.
    total_coverage_f32 = df["total_coverage"].astype("float32", copy=False)
    df["A_frequency"] = (df["A"] / total_coverage_f32).astype("float32", copy=False)
    df["T_frequency"] = (df["T"] / total_coverage_f32).astype("float32", copy=False)
    df["G_frequency"] = (df["G"] / total_coverage_f32).astype("float32", copy=False)
    df["C_frequency"] = (df["C"] / total_coverage_f32).astype("float32", copy=False)
    return df


def process_mag_files(args):
    """Worker function: load one sample's profile TSV and annotate with metadata.

    This is the core I/O function executed in parallel by ``allele_freq_cache.py``.
    Each call handles exactly one sample at one timepoint.

    Previously this logic lived inside ``allele_freq.py``.  It was extracted
    here so the cache writer (``allele_freq_cache.py``) can reuse it without
    duplicating the expensive ``pd.read_csv`` + metadata-join + frequency
    calculation for samples that appear in multiple group combinations.

    Flow
    ----
    1. Read the gzipped profile TSV from ``filepath``.
    2. Attach ``sample_id`` (so rows can be traced back after concatenation).
    3. Look up per-sample metadata (group, subjectID, replicate, time) from
       the ``metadata_dict`` global populated by ``init_worker``.
    4. Call ``calculate_frequencies`` to add the four frequency columns.
    5. Select only the columns in ``CACHE_COLUMNS_*`` — dropping MAG_ID,
       ref_base, N, breadth, genome_size, average_coverage which no
       downstream cache consumer reads.

    Parameters
    ----------
    args : tuple
        ``(sample_id, filepath, mag_id)`` — the only data passed per-call
        to keep pickle overhead minimal.

    Returns
    -------
    pd.DataFrame or None
        Annotated frame with only the cache columns on success; ``None`` if
        the sample_id is missing from ``metadata_dict`` (logged as an error).
    """
    sample_id, filepath, mag_id = args

    # Read the per-base coverage profile written by profile_mags.py.
    # Explicit dtypes (int32 counts/coverage/position; string contig/gene_id)
    # are critical for memory: default pandas inference uses int64/float64
    # which doubles the in-memory size of the cache for no precision benefit.
    df = pd.read_csv(filepath, sep="\t", dtype=PROFILE_READ_DTYPES)

    df["sample_id"] = sample_id

    # Retrieve the pre-built metadata row for this sample.  The dict was
    # populated in the parent process by build_metadata_from_qc() and
    # broadcast to workers via init_worker().
    metadata_info = metadata_dict.get(sample_id)
    if metadata_info is None:
        logger.error(f"Sample ID not found for sample {sample_id}")
        return None

    df["group"] = metadata_info["group"]
    df["subjectID"] = metadata_info["subjectID"]
    df["replicate"] = metadata_info["replicate"]

    # ``time`` is only meaningful for longitudinal data; for single data the
    # QC file may not have a time column at all.
    if DATA_TYPE == "longitudinal" and "time" in metadata_info:
        df["time"] = metadata_info["time"]

    df = calculate_frequencies(df)

    # Drop columns not consumed by any downstream cache reader.
    # Excluded: MAG_ID (redundant — derivable from filename), ref_base and N
    # (never read by Stage 2/CMH/LMM), breadth/genome_size/average_coverage
    # (QC metrics that downstream rules re-read from QC TSVs if needed).
    keep = (
        CACHE_COLUMNS_LONGITUDINAL
        if DATA_TYPE == "longitudinal" and "time" in df.columns
        else CACHE_COLUMNS_BASE
    )
    df = df[[c for c in keep if c in df.columns]]

    # Convert low-cardinality string columns to categorical so the resulting
    # pyarrow Table uses dictionary encoding.  This is by far the largest
    # memory saving in Stage 2: a 50 B/row object string becomes a 4 B/row
    # int code, and pandas reads them back as ``category`` automatically.
    # gene_id / contig stay as plain strings (high cardinality per MAG; the
    # parquet writer still dictionary-encodes them on disk via use_dictionary
    # but we don't pay the pandas category overhead in downstream readers).
    for col in CACHE_CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def build_metadata_from_qc(qc_df):
    """Convert a QC DataFrame into the structures needed by the worker pool.

    Returns
    -------
    metadata : dict
        Maps ``sample_id (str) -> {group, subjectID, replicate, breadth,
        genome_size, average_coverage, [time]}``.  Passed to ``init_worker``
        so every worker process can look up its sample's metadata without
        re-reading the QC file.
    tuples : list of (sample_id, file_path)
        One entry per sample.  The caller appends ``mag_id`` to make the
        three-element ``args`` tuple that ``process_mag_files`` expects.

    Notes
    -----
    The ``mag_id`` slot is intentionally left out here so this function
    stays generic — the caller fills it in before handing off to the pool.
    """
    metadata = {}
    tuples = []
    for _, row in qc_df.iterrows():
        sample_id = str(row["sample_id"])
        meta = {
            "group": str(row["group"]),
            "subjectID": row["subjectID"],
            "replicate": row["replicate"],
            "breadth": row["breadth"],
            "genome_size": row["genome_size"],
            "average_coverage": row["average_coverage"],
        }
        # Only include ``time`` when the column is present and non-null.
        # Absence is normal for single-data QC files.
        if "time" in row and pd.notna(row["time"]):
            meta["time"] = row["time"]
        metadata[sample_id] = meta
        tuples.append((sample_id, row["file_path"]))
    return metadata, tuples
