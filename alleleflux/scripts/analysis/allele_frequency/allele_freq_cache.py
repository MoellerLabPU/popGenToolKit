#!/usr/bin/env python3
"""Per-(MAG, timepoint) allele-frequency cache writer.  [Stage 1 of 2]

Role in the two-stage pipeline
-------------------------------
This script is **Stage 1**.  It runs ONCE per (MAG, timepoint) regardless of
how many (timepoint_combination, group_combination) pairs include that
timepoint.

    alleleflux-cache-allele-freq
        --magID       MRGM_1079
        --qc_files    {OUTDIR}/QC/QC_5mo_10mo/MRGM_1079_QC.tsv
        --timepoint   5mo
        --data_type   longitudinal
        --output_path {OUTDIR}/allele_freq_cache/5mo/MRGM_1079_5mo_allele_frequency.parquet
        --cpus        16

Output consumed by
------------------
- ``allele_freq.py`` (Stage 2): reads exactly two cache files per combination
  to compute per-subject allele-frequency diffs.
- ``CMH.py``, ``LMM.py`` (across-time / between-groups tests): read one or
  two cache files directly when the longitudinal per-sample data is needed
  (replaces the old ``_allele_frequency_longitudinal.tsv.gz``).

Why this exists
---------------
Before this refactor, ``allele_freq.py`` was invoked once per
(timepoint_combination, group_combination).  For a config with timepoints
``[pre, post, end]`` and two combinations ``pre_post`` and ``pre_end``, the
``pre`` profile files were read, frequencies computed, and ``pre`` rows written
to disk **twice** — once inside each combination's longitudinal TSV.

This script eliminates that duplication: profile reads and frequency
computations for ``pre`` happen once; the resulting Parquet file is read by
both downstream combinations.

Shared helpers (from ``_allele_freq_common.py``)
-------------------------------------------------
- ``load_qc_results``        — load QC TSV and filter to passing samples
- ``init_worker``            — broadcast metadata dict to worker processes
- ``process_mag_files``      — read one profile TSV + compute frequencies (worker fn)
- ``build_metadata_from_qc`` — convert QC rows into pool-friendly structures

Note on QC input
-----------------
The Snakemake rule ``compute_allele_freq_per_timepoint`` passes exactly ONE
canonical QC file per cache job (the QC from the first timepoint_combination
in config order that contains the requested timepoint).  ``filter_qc_to_timepoint``
handles this single-file case; there is no need to union across combinations.
"""

import argparse
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from alleleflux.scripts.analysis.allele_frequency._allele_freq_common import (
    build_metadata_from_qc,
    init_worker,
    load_qc_results,
    process_mag_files,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


def filter_qc_to_timepoint(qc_path, mag_id, timepoint):
    """Load the canonical QC file and filter to a single timepoint.

    The Snakemake rule always supplies exactly one QC path per cache job —
    the canonical QC file for this (gr_combo, timepoint) pair (the first
    timepoint_combination in config order that contains this timepoint).
    This function loads it, filters rows to ``timepoint``, and returns the
    result ready for ``build_metadata_from_qc``.

    Parameters
    ----------
    qc_path : str
        Path to the canonical per-MAG QC TSV for this (gr_combo, timepoint).
    mag_id : str
        Used for logging and error messages.
    timepoint : str
        Timepoint label to retain (matched against the ``time`` column after
        casting both sides to ``str`` to avoid int/str mismatches).

    Returns
    -------
    pd.DataFrame
        QC rows for samples at this timepoint that passed coverage/breadth QC.

    Raises
    ------
    ValueError
        If the QC file has no ``time`` column, or if no samples remain after
        filtering to the requested timepoint.
    """
    qc_df = load_qc_results(qc_path, mag_id)

    if "time" not in qc_df.columns:
        raise ValueError(
            f"QC file for MAG {mag_id} has no 'time' column; "
            "cannot filter to a single timepoint. This rule expects longitudinal QC data."
        )

    # Cast to str on both sides to handle numeric timepoints (e.g. 5 vs "5").
    qc_df["time"] = qc_df["time"].astype(str)
    qc_df = qc_df[qc_df["time"] == str(timepoint)]

    if qc_df.empty:
        raise ValueError(
            f"No QC-passing samples for MAG {mag_id} at timepoint '{timepoint}'."
        )

    logger.info(
        f"Found {len(qc_df)} samples for MAG {mag_id} at timepoint '{timepoint}'."
    )
    return qc_df


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: Compute per-sample allele frequencies for a MAG at a single "
            "timepoint and write to a Parquet cache file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--magID",
        required=True,
        type=str,
        help="MAG ID to process.",
    )
    parser.add_argument(
        "--qc_files",
        required=True,
        nargs="+",
        type=str,
        help=(
            "Path(s) to per-MAG QC TSV file(s).  Exactly one file is expected "
            "for longitudinal data (the canonical QC file for this gr_combo and "
            "timepoint); one file for single data.  The nargs='+' form is kept "
            "so the Snakemake rule can expand the list naturally."
        ),
    )
    parser.add_argument(
        "--timepoint",
        required=False,
        default=None,
        type=str,
        help=(
            "Timepoint label to extract from the QC file.  Required for "
            "'longitudinal' data_type (filters the QC file's 'time' column).  "
            "Not used for 'single' data_type — the QC file covers one timepoint "
            "and no row-level filtering is needed."
        ),
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["single", "longitudinal"],
        default="longitudinal",
        help=(
            "Data type.  For 'longitudinal', rows in the QC file are filtered "
            "to --timepoint.  For 'single', all QC-passing samples are loaded."
        ),
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help=(
            "Destination Parquet file.  Consumed by allele_freq.py (Stage 2) "
            "and by across-time statistical tests (CMH, LMM)."
        ),
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Worker processes for parallel profile loading.",
    )

    args = parser.parse_args()
    start_time = time.time()
    mag_id = args.magID

    # ------------------------------------------------------------------
    # 1. Determine which samples to load
    # ------------------------------------------------------------------
    if args.data_type == "longitudinal":
        if args.timepoint is None:
            parser.error("--timepoint is required when --data_type is 'longitudinal'.")
        # The Snakemake rule passes one canonical QC file per (gr_combo, timepoint).
        # Filter its rows to the single requested timepoint.
        qc_df = filter_qc_to_timepoint(args.qc_files[0], mag_id, args.timepoint)
    else:
        # Single data: the QC file covers exactly one timepoint — load all passing
        # samples without any timepoint-based row filtering.
        qc_df = load_qc_results(args.qc_files[0], mag_id)

    # ------------------------------------------------------------------
    # 2. Build worker inputs
    # ------------------------------------------------------------------
    # build_metadata_from_qc returns:
    #   metadata_dict: {sample_id -> metadata row} — broadcast to workers
    #   base_tuples:   [(sample_id, file_path), ...] — one entry per sample
    metadata_dict, base_tuples = build_metadata_from_qc(qc_df)

    # Append mag_id to each tuple to match the (sample_id, filepath, mag_id)
    # signature expected by process_mag_files.
    sample_file_tuples = [(sid, fpath, mag_id) for sid, fpath in base_tuples]

    # ------------------------------------------------------------------
    # 3. Load profiles in parallel and stream to Parquet
    # ------------------------------------------------------------------
    # Each worker:
    #   - reads one profile TSV (the expensive step) with int32 / string dtypes
    #   - attaches metadata (group, subjectID, replicate, time) as categoricals
    #   - computes per-base nucleotide frequencies as float32
    #
    # Rather than collect every per-sample frame in the parent and then
    # concat + write (which doubles parent-process memory at the moment of
    # concat), we stream each frame straight into a single Parquet file via
    # pyarrow.parquet.ParquetWriter.  Each per-sample table becomes one row
    # group; pyarrow handles per-row-group dictionaries for the categorical
    # columns automatically.
    n_proc = min(args.cpus, len(sample_file_tuples))
    tp_label = args.timepoint if args.data_type == "longitudinal" else "N/A (single)"
    logger.info(
        f"Computing allele frequencies for {len(sample_file_tuples)} samples "
        f"of MAG {mag_id} (timepoint={tp_label}) using {n_proc} processes."
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then rename atomically.  If the job is
    # killed mid-write (SLURM SIGTERM/SIGKILL) or hits an OOM, the partial
    # file stays at the .tmp path and Snakemake correctly re-runs the job
    # because the final output is still missing.
    tmp_path = output_path.with_suffix(".parquet.tmp")

    writer = None
    total_rows = 0
    try:
        with Pool(
            processes=n_proc,
            initializer=init_worker,     # broadcasts metadata_dict + DATA_TYPE
            initargs=(metadata_dict, args.data_type),
        ) as pool:
            for df in pool.imap_unordered(process_mag_files, sample_file_tuples):
                # None means the worker logged an error for that sample;
                # skip it and continue with the rest.
                if df is None:
                    continue

                # Convert to Arrow.  Categorical pandas columns map to Arrow
                # dictionary type; pyarrow then writes them as dictionary
                # pages in the row group.  preserve_index=False avoids an
                # extra __index_level_0__ column.
                table = pa.Table.from_pandas(df, preserve_index=False)

                if writer is None:
                    # Lock the schema on the first table.  All subsequent
                    # tables must match this schema (dtype + column order);
                    # process_mag_files produces them deterministically so
                    # this holds in practice.
                    writer = pq.ParquetWriter(
                        tmp_path,
                        table.schema,
                        compression="snappy",
                        use_dictionary=True,
                    )

                writer.write_table(table)
                total_rows += table.num_rows
                # Drop the per-sample frame ASAP — the next iteration will
                # build the next one and we don't want N frames live at once.
                del df, table
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        logger.error(
            f"No sample profiles could be loaded for MAG {mag_id} "
            f"at timepoint {args.timepoint}."
        )
        # Remove the empty/missing-schema .tmp so the next attempt starts
        # from a clean slate.
        if tmp_path.exists():
            tmp_path.unlink()
        sys.exit(1)

    logger.info(f"Wrote {total_rows:,} rows to {output_path}")
    tmp_path.rename(output_path)

    logger.info(f"Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
