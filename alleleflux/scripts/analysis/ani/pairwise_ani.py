#!/usr/bin/env python3
"""Pairwise conANI / popANI between the sample pairs of one MAG.

THE runnable entry point of the ani package (console script
``alleleflux-pairwise-ani`` -> ``main()``); the other four modules are libraries
this file orchestrates: profile_io loads, null_model calibrates, engine compares,
classify judges.

Two samples are compared at every genome position where BOTH have at least
``--min_cov`` reads.  At each such position their consensus SETS (bases tied at
the max count) and credible-allele sets are compared:

    consensus SNP   <=> consensus sets disjoint (the majorities differ)
    population SNP  <=> credible sets disjoint  (no allele shared: fixed difference)

    conANI = (compared - consensus SNPs) / compared
    popANI = (compared - population SNPs) / compared

popANI answers "same strain?"; read either number alongside
``percent_genome_compared`` -- a high ANI over 3% of the genome means little.
Sample selection reuses AlleleFlux's QC verdict; ``--min_cov`` (default 5, part of
the metric definition; 1 disables the depth gate) is stamped into every output row.
Only samples that take part in at least one pair are loaded and summarised (with
``--pairs within_subject`` on DRiDO that skips ~38% of the roster); ``--cpus``
parallelises the profile loading, which dominates runtime.

Outputs, one set per MAG in ``--output_dir``:

    {mag}_pairwise_ani.tsv                  one row per pair + metadata both sides
    {mag}_pairwise_ani_samples.tsv          per-sample breadth / depth summary
    {mag}_pairwise_ani_snp_locations.tsv.gz per-position SNP rows for the pairs
                                            chosen by --store_snp_locations

Worked example (real data, verified 2026-08-27)::

    alleleflux-pairwise-ani \\
        --mag SLG1007_DASTool_bins_35 \\
        --profiles_dir .../AlleleFlux_mapq20/longitudinal/profiles \\
        --qc_files .../QC_pre_end-fat_control/SLG1007_DASTool_bins_35_QC.tsv \\
                   .../QC_pre_post-fat_control/SLG1007_DASTool_bins_35_QC.tsv \\
        --fasta .../redo_representative_megamag.fa \\
        --mag_mapping .../megamag_mapping.tsv \\
        --output_dir /tmp/ani_check

    -> 43 QC-passing samples, 903 pairs; the SLG1007 vs SLG1102 row reads
       compared 1,649,007 (54.8%), consensus 212, population 29,
       conANI 0.999871, popANI 0.9999824.

Method credit: Olm et al. 2021, Nat Biotechnol, doi:10.1038/s41587-020-00797-0.
"""

import argparse
import itertools
import logging
import math
import multiprocessing
import os

import pandas as pd

from alleleflux.scripts.analysis.ani.engine import compare_all_pairs
from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.ani.profile_io import (
    contig_lengths_for_mag,
    dense_contig_counts,
    load_profile,
    profile_path,
    qc_passing_samples,
)
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Per-sample summary table schema (fixed so empty outputs still carry a header).
# breadth is kept for convenience but VERIFIED against QC's recorded value (guard
# below); mean_coverage is deliberately absent -- QC's average_coverage is the one
# authoritative copy, and a near-duplicate here (ours would exclude N) is an
# inconsistency factory.
SAMPLE_COLUMNS = [
    "MAG_ID",
    "sample_id",
    "subjectID",
    "group",
    "time",
    "replicate",
    "n_positions",
    "breadth",
    "breadth_minCov",
]

# QC metadata copied onto each pair row for both sides (as subjectID_1/_2 etc.),
# so downstream analyses can filter by mouse/group/timepoint without any join.
_PAIR_META_FIELDS = ("subjectID", "group", "time", "replicate")


def parse_transitions(specs: list[str]) -> list[tuple[str, str]]:
    """Turn ``["5mo:10mo", "5mo:22mo"]`` into ``[("5mo", "10mo"), ("5mo", "22mo")]``.

    One place for the format so this CLI and the strain-turnover tool can never
    disagree on it.  Each spec is ``EARLIER:LATER``; a degenerate ``T:T`` is
    rejected because a sample cannot transition to its own timepoint.
    """
    transitions = []
    for spec in specs:
        if spec.count(":") != 1:
            raise ValueError(f"Transition {spec!r} must look like EARLIER:LATER (e.g. 5mo:22mo)")
        earlier, later = (part.strip() for part in spec.split(":"))
        if not earlier or not later:
            raise ValueError(f"Transition {spec!r} has an empty timepoint")
        if earlier == later:
            raise ValueError(f"Degenerate transition {spec!r}: both sides are the same timepoint")
        transitions.append((earlier, later))
    return transitions


def enumerate_pairs(
    samples_df: pd.DataFrame, mode: str, transitions: list[tuple[str, str]] | None = None
) -> list[tuple[str, str]]:
    """Ordered, de-duplicated sample pairs from the QC roster.

    ``all`` compares every unordered pair; ``within_subject`` keeps only pairs of
    samples from the same mouse; ``transitions`` keeps only same-mouse pairs whose
    two timepoints are one of the configured ``(earlier, later)`` transitions --
    the comparisons the strain-turnover analysis actually consumes (on DRiDO:
    970 pairs instead of within_subject's 2,570, because the extra 1,600 compare
    two LATER timepoints nothing downstream reads).  Every mode returns pairs as
    ``(sample1, sample2)`` with ``sample1 < sample2`` by id, sorted, so outputs
    are deterministic and join across runs.

    Example: roster S1(m1, pre), S2(m2, pre), S3(m1, end) ->
    all: (S1,S2),(S1,S3),(S2,S3); within_subject: (S1,S3);
    transitions [(pre, end)]: (S1,S3); transitions [(pre, post)]: nothing.
    """
    # Sort so combinations() emits each pair exactly once with sample1 < sample2.
    sample_ids = sorted(samples_df["sample_id"].astype(str))
    if mode == "all":
        return list(itertools.combinations(sample_ids, 2))
    if mode not in ("within_subject", "transitions"):
        raise ValueError(f"Unknown --pairs mode: {mode}")
    if "subjectID" not in samples_df.columns:
        raise ValueError(f"--pairs {mode} requires a subjectID column in the QC file")
    # sample_id -> subjectID lookup; a pair is kept only when both sides match.
    subject_of = dict(
        zip(samples_df["sample_id"].astype(str), samples_df["subjectID"].astype(str))
    )
    if mode == "within_subject":
        return [p for p in itertools.combinations(sample_ids, 2)
                if subject_of[p[0]] == subject_of[p[1]]]

    # mode == "transitions"
    if not transitions:
        raise ValueError("--pairs transitions requires --transitions EARLIER:LATER [...]")
    if "time" not in samples_df.columns:
        raise ValueError("--pairs transitions requires a time column in the QC file")
    time_of = dict(zip(samples_df["sample_id"].astype(str), samples_df["time"].astype(str)))
    # One sample per (mouse, timepoint).  A mouse sampled twice at the same
    # timepoint has no single "before" or "after" and is a roster error, so raise
    # rather than silently pairing both (0 such cases in DRiDO's 2,250 samples).
    sample_at: dict[tuple[str, str], str] = {}
    for sample in sample_ids:
        key = (subject_of[sample], time_of[sample])
        if key in sample_at:
            raise ValueError(
                f"Mouse {key[0]} has two samples at timepoint {key[1]}: "
                f"{sample_at[key]} and {sample}"
            )
        sample_at[key] = sample

    # For every transition and every mouse: if the mouse has a sample at BOTH the
    # earlier and the later timepoint, that is one pair; otherwise nothing.
    pairs = set()
    for earlier, later in transitions:
        for subject in set(subject_of.values()):
            before = sample_at.get((subject, earlier))
            after = sample_at.get((subject, later))
            if before is not None and after is not None:
                pairs.add(tuple(sorted((before, after))))  # sample1 < sample2
    return sorted(pairs)


def _load_one_sample(job: tuple) -> tuple[str, dict, dict]:
    """Load one profile into dense arrays plus its summary stats.

    Module-level (not a closure) so it pickles into ``multiprocessing.Pool``
    workers.  ``job`` = (sample_id, profile_path, contig_lengths, min_cov,
    genome_length).  Returns ``(sample_id, dense_by_contig, stats)`` where
    ``stats`` holds ``n_positions``, ``breadth`` (>=1x) and ``breadth_minCov``.

    Example: a 6 bp contig covered at 20x everywhere except position 5 at 2x,
    min_cov 5 -> n_positions 6, breadth 1.0, breadth_minCov 5/6.
    """
    sample_id, path, contig_lengths, min_cov, genome_length = job
    profile = load_profile(path)
    dense = dense_contig_counts(profile, contig_lengths)
    # Stats while the frame is still in hand, so the parent never re-reads.
    coverage = profile["coverage"].to_numpy()
    stats = {
        "n_positions": len(profile),                                    # covered >=1x
        "breadth": len(profile) / genome_length,                        # fraction >=1x
        # Fraction deep enough to ever be compared (QC has no min_cov-aware breadth).
        "breadth_minCov": int((coverage >= min_cov).sum()) / genome_length,
    }
    return sample_id, dense, stats


def round_up_the_sample_pairs(args: argparse.Namespace) -> int:
    """Load everything, compare every requested pair, write the three tables.

    The orchestrator: each step below is one library call, in the order the data
    flows.  Returns a process exit code (0 = success); contract violations raise.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. How big is the board?  contig -> length, via the .fai fast path.
    contig_lengths = contig_lengths_for_mag(args.fasta, args.mag_mapping, args.mag)
    genome_length = int(sum(contig_lengths.values()))
    logger.info(f"MAG {args.mag}: {len(contig_lengths)} contigs, {genome_length:,} bp")

    # ---- 2. Who plays?  QC-passing samples, unioned across the QC files.
    # MAY be empty (low-abundance MAG): everything below still runs and produces
    # header-only outputs, so a pipeline over all MAGs never crashes here.
    samples = qc_passing_samples(args.qc_files, args.mag)

    # ---- consistency guard: QC recorded the genome size IT used (same FASTA +
    # mapping, at QC time).  If it disagrees with the .fai-derived sum we are
    # sizing arrays with, QC ran against a DIFFERENT reference and every breadth
    # anywhere is suspect -- die loudly rather than mix the two silently.
    if "genome_size" in samples.columns and len(samples):
        qc_sizes = set(pd.to_numeric(samples["genome_size"]).astype(int))
        if qc_sizes != {genome_length}:
            raise ValueError(
                f"MAG {args.mag}: QC genome_size {sorted(qc_sizes)} != reference "
                f"length {genome_length:,}. QC and this run used different "
                "references or MAG mappings."
            )

    # ---- 3. The error model: coverage -> minimum reads for a credible allele.
    model = build_error_model(min_base_quality=args.min_base_quality, fdr=args.fdr)

    # ---- 4. Decide the pairs FIRST, so only samples that take part in at least
    # one pair are ever loaded.  On DRiDO with --pairs within_subject that skips
    # ~38% of the QC-passing roster (samples whose mouse lacks the partner
    # timepoint) -- each skipped load is ~3.6 s and ~34 MB.
    pairs = enumerate_pairs(samples, args.pairs, transitions=args.transitions)
    if args.store_snp_locations == "all":
        location_pairs = set(pairs)  # the 7.7 GB option: explicit only
    elif args.store_snp_locations == "within_subject":
        # Intersect with `pairs` so --pairs within_subject + this default agree.
        location_pairs = set(enumerate_pairs(samples, "within_subject")) & set(pairs)
    else:  # "none"
        location_pairs = set()

    # Roster = paired samples in the QC file's own order; keeping that order is
    # what makes the parallel loader's ordered map() reproduce the serial layout.
    paired_ids = {sample for pair in pairs for sample in pair}
    roster = samples[samples["sample_id"].astype(str).isin(paired_ids)]
    n_unpaired = len(samples) - len(roster)
    if not pairs:
        logger.warning(
            f"MAG {args.mag}: {len(samples)} usable sample(s) -> no pairs under "
            f"--pairs {args.pairs}; writing header-only outputs so downstream rules resolve"
        )
    elif n_unpaired:
        logger.info(
            f"MAG {args.mag}: {n_unpaired} QC-passing sample(s) take part in no pair "
            f"under --pairs {args.pairs}; not loaded"
        )

    # ---- 5. Load the paired samples' dense arrays, in parallel if asked.
    # Memory: genome_length x 4 x 2 bytes per sample (34 MB for a 4.3 Mb MAG),
    # ALL held at once -- mem_per_job is sized from that (see config template).
    jobs = []
    for row in roster.itertuples(index=False):
        sample_id = str(row.sample_id)
        path = profile_path(args.profiles_dir, sample_id, args.mag)
        if not os.path.exists(path):
            # A QC-passing sample MUST have a profile: QC read it to pass the
            # sample.  Checked for EVERY sample before any loading starts, so an
            # inconsistent tree fails in milliseconds, not after 10 minutes.
            raise FileNotFoundError(
                f"MAG {args.mag}: profile missing for QC-passing sample {sample_id}: {path}"
            )
        jobs.append((sample_id, path, contig_lengths, args.min_cov, genome_length))

    if args.cpus > 1 and len(jobs) > 1:
        # Loading (gzip + parse + scatter) dominates runtime and is embarrassingly
        # parallel.  PROCESSES, not threads: pandas' CSV parser holds the GIL for
        # most of the work, so threads would serialise anyway (the same lesson
        # allele_freq.py records for its parquet reads).  multiprocessing.Pool is
        # this codebase's convention (profile_mags, coverage_and_allele_stats);
        # .map() returns results in submission order, so the samples table comes
        # out in exactly the layout a serial run produces.
        with multiprocessing.Pool(processes=args.cpus) as pool:
            loaded = pool.map(_load_one_sample, jobs)
    else:
        loaded = [_load_one_sample(job) for job in jobs]
    logger.info(f"MAG {args.mag}: loaded {len(loaded)} profiles with {args.cpus} cpu(s)")

    dense_by_sample: dict[str, dict] = {}
    sample_records = []
    for row, (sample_id, dense, stats) in zip(roster.itertuples(index=False), loaded):
        # ---- consistency guard #2: QC computed this SAME breadth from this SAME
        # profile when the sample passed.  A mismatch means the profile file
        # changed after QC ran (re-profiled tree, stale QC) -- refuse to mix eras.
        qc_breadth = getattr(row, "breadth", None)
        if qc_breadth is not None and not math.isclose(
            stats["breadth"], float(qc_breadth), rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"MAG {args.mag}: sample {sample_id} breadth from its profile "
                f"({stats['breadth']:.6g}) != QC's recorded breadth ({float(qc_breadth):.6g}). "
                "The profile appears to have changed since QC ran."
            )
        dense_by_sample[sample_id] = dense
        sample_records.append(
            {
                "MAG_ID": args.mag,
                "sample_id": sample_id,
                # getattr default: 'time' is absent for single-timepoint data.
                "subjectID": str(getattr(row, "subjectID", "")),
                "group": str(getattr(row, "group", "")),
                "time": str(getattr(row, "time", "")),
                "replicate": str(getattr(row, "replicate", "")),
                "n_positions": stats["n_positions"],
                "breadth": stats["breadth"],  # verified against QC just above
                "breadth_minCov": stats["breadth_minCov"],
            }
        )

    sample_table = pd.DataFrame(sample_records, columns=SAMPLE_COLUMNS)
    sample_out = os.path.join(args.output_dir, f"{args.mag}_pairwise_ani_samples.tsv")
    sample_table.to_csv(sample_out, sep="\t", index=False)
    logger.info(f"Wrote {sample_out} ({len(sample_table)} samples)")

    # ---- 6. The engine, single-threaded: one call over every pair.  Threads were
    # tried and dropped -- NumPy releases the GIL only in parts of this loop, the
    # gain was minor next to loading, and chunked results had to be re-sorted to
    # stay reproducible.  --cpus therefore means exactly one thing: loader
    # workers.  Empty `pairs` flows through here too and yields empty tables with
    # full schemas -- one code path for every case.
    pair_table, snp_locations = compare_all_pairs(
        dense_by_sample,
        contig_lengths,
        pairs,
        model,
        min_freq=args.min_freq,
        min_cov=args.min_cov,
        genome_length=genome_length,
        mag_id=args.mag,
        snp_location_pairs=location_pairs,
    )

    # ---- 7. Attach mouse/group/time for both sides of every pair, so no
    # downstream analysis ever needs to re-join against QC files.
    metadata = samples.set_index(samples["sample_id"].astype(str))
    for side in ("1", "2"):
        for field in _PAIR_META_FIELDS:
            if field in metadata.columns:
                # .map: sample id -> that sample's metadata value (row-aligned).
                pair_table[f"{field}_{side}"] = (
                    pair_table[f"sample{side}"].map(metadata[field]).astype(str)
                )

    # ---- 8. Write both tables (gzip for locations: it can be large for 'all').
    pair_out = os.path.join(args.output_dir, f"{args.mag}_pairwise_ani.tsv")
    location_out = os.path.join(
        args.output_dir, f"{args.mag}_pairwise_ani_snp_locations.tsv.gz"
    )
    pair_table.to_csv(pair_out, sep="\t", index=False)
    snp_locations.to_csv(location_out, sep="\t", index=False, compression="gzip")
    logger.info(
        f"Wrote {pair_out} ({len(pair_table)} pairs) and "
        f"{location_out} ({len(snp_locations)} flagged positions)"
    )
    return 0


def main():
    setup_logging()  # once, here, per house rule -- never in the libraries
    parser = argparse.ArgumentParser(
        description="Pairwise conANI/popANI between the QC-passing sample pairs of "
                    "one MAG, over all genome positions covered >= min_cov in both "
                    "samples. See the module docstring for method and worked example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mag", required=True, help="MAG identifier")
    parser.add_argument(
        "--profiles_dir",
        required=True,
        help="Directory holding per-sample profile subdirectories",
    )
    parser.add_argument(
        "--qc_files",
        required=True,
        nargs="+",
        help="Per-MAG QC TSV(s), one per timepoint combination; samples "
        "with coverage_threshold_passed anywhere are compared",
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Reference FASTA (config input.fasta_path)",
    )
    parser.add_argument(
        "--mag_mapping",
        required=True,
        help="MAG-to-contig mapping TSV with columns mag_id, contig_id "
        "(config input.mag_mapping_path)",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--min_cov",
        type=int,
        default=5,
        help="Minimum reads in BOTH samples for a position to be compared. "
        "Part of the metric definition (stamped into the output); "
        "1 disables the depth gate -- exploration only",
    )
    parser.add_argument(
        "--min_freq",
        type=float,
        default=0.05,
        help="Minimum fraction of a position's reads for an allele to "
        "count as present (popANI only; conANI never uses it)",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=1e-6,
        help="Tolerated per-position probability that sequencing error "
        "alone explains an allele (popANI only)",
    )
    parser.add_argument(
        "--min_base_quality",
        type=int,
        default=30,
        help="Phred floor used when the profiles were built; sets the "
        "error model's assumed per-base error rate",
    )
    parser.add_argument(
        "--pairs",
        choices=["all", "within_subject", "transitions"],
        default="within_subject",
        help="all = every pair; within_subject = every same-mouse pair; transitions = "
        "only same-mouse pairs matching --transitions (what strain-turnover consumes)",
    )
    parser.add_argument(
        "--transitions",
        nargs="+",
        default=None,
        metavar="EARLIER:LATER",
        help="Timepoint transitions for --pairs transitions, e.g. 5mo:10mo 5mo:22mo "
        "(the pipeline builds these from analysis.timepoints_combinations)",
    )
    parser.add_argument(
        "--store_snp_locations",
        choices=["none", "within_subject", "all"],
        default="within_subject",
        help="Which pairs also get per-position SNP rows ('all' was 7.7 GB "
        "for 90 samples in inStrain -- use deliberately)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="Worker processes for loading profiles -- the runtime bottleneck "
        "(~3.6 s per DRiDO profile). Comparison is single-threaded; outputs are "
        "identical for any value",
    )
    args = parser.parse_args()
    # Validate the transition specs here, at the door, so a typo fails before
    # any profile is read; None stays None for the other modes.
    if args.pairs == "transitions" and not args.transitions:
        parser.error("--pairs transitions requires --transitions EARLIER:LATER [...]")
    args.transitions = parse_transitions(args.transitions) if args.transitions else None

    logger.info(
        f"pairwise-ani: mag={args.mag} min_cov={args.min_cov} min_freq={args.min_freq} "
        f"fdr={args.fdr:.0e} min_base_quality={args.min_base_quality} pairs={args.pairs} "
        f"store_snp_locations={args.store_snp_locations} cpus={args.cpus} "
        f"transitions={args.transitions}"
    )
    raise SystemExit(round_up_the_sample_pairs(args))


if __name__ == "__main__":
    main()
