#!/usr/bin/env bash
# generate_bams.sh
# ----------------
# Regenerates the synthetic BAM files in bams/ from the bundled reference FASTA.
#
# What this script does
# ---------------------
# For each of the 8 samples it runs a three-step pipeline:
#
#   1. wgsim  — simulates paired-end Illumina reads from the reference FASTA
#   2. bwa mem — aligns those reads back to the reference
#   3. samtools sort + index — produces a sorted, indexed BAM
#
# Why separate seeds per sample
# ------------------------------
# All 8 samples are generated from the identical reference genome with
# identical parameters, but each uses a different random seed. This means
# the reads differ slightly between samples (different positions get covered,
# slightly different noise) but there is NO systematic treatment effect —
# treatment and control samples are statistically indistinguishable.
#
# To add a real treatment effect you would need to modify the reference FASTA
# itself before simulating (e.g. introduce SNPs at 8% of positions) and use
# the modified reference only for treatment post-timepoint samples.
# generate_synthetic_data.py takes the alternative approach: it injects the
# treatment effect directly into the allele count tables, bypassing reads
# entirely.
#
# Requirements
# ------------
# wgsim and bwa are NOT included in the alleleflux conda environment.
# Install them first:
#   conda install -c bioconda bwa wgsim
# samtools IS included in the alleleflux environment.
#
# Usage
# -----
#   cd docs/source/examples
#   bash generate_bams.sh                          # default: example_data_longitudinal/
#   bash generate_bams.sh --data-dir my_test_data  # use a freshly-generated dataset
#
# This script lives alongside generate_synthetic_data.py in docs/source/examples/.
# By default it reads the bundled reference at docs/source/examples/example_data_longitudinal/
# and writes BAMs into example_data_longitudinal/bams/. Pass --data-dir <path>
# to target a different dataset (e.g. example_data_single/ or one produced ad-hoc
# by generate_synthetic_data.py).
#
# The --data-dir directory must contain reference/combined_mags.fasta; the BAMs
# will be written into <data-dir>/bams/. Path can be absolute or relative to the
# current working directory.
#
# Approximate runtime: < 5 seconds for the bundled 7 kb reference

set -euo pipefail   # exit on any error, treat unset vars as errors, propagate pipe failures

# Resolve the directory this script lives in regardless of where it is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/example_data_longitudinal"   # default; overridable via --data-dir

# ---- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --data-dir requires a path argument" >&2
                exit 1
            fi
            # Resolve to absolute so subsequent paths stay anchored even if the
            # user passes a relative arg from a cwd we don't know about.
            DATA_DIR="$(cd "$2" 2>/dev/null && pwd || true)"
            if [[ -z "$DATA_DIR" ]]; then
                echo "ERROR: --data-dir path does not exist: $2" >&2
                exit 1
            fi
            shift 2
            ;;
        -h|--help)
            sed -n '2,52p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: bash generate_bams.sh [--data-dir <dir>]" >&2
            exit 1
            ;;
    esac
done

REF="$DATA_DIR/reference/combined_mags.fasta"
OUTDIR="$DATA_DIR/bams"

if [[ ! -f "$REF" ]]; then
    echo "ERROR: reference FASTA not found at $REF" >&2
    echo "Make sure the data directory contains reference/combined_mags.fasta" >&2
    exit 1
fi

echo "[generate_bams] Script dir: $SCRIPT_DIR"
echo "[generate_bams] Data dir:   $DATA_DIR"

# ---- Dependency check --------------------------------------------------------
for cmd in wgsim bwa samtools; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found."
        echo "Install with: conda install -c bioconda bwa wgsim"
        exit 1
    fi
done

mkdir -p "$OUTDIR"

# ---- Index the reference (skip if already done) ------------------------------
# bwa needs 5 index files (.amb .ann .bwt .pac .sa) before it can align.
# We check for .bwt as a proxy for all five.
if [[ ! -f "$REF.bwt" ]]; then
    echo "[generate_bams] Indexing reference with bwa..."
    bwa index "$REF" 2>/dev/null
fi

# samtools faidx is needed to read contig lengths for the coverage calculation.
if [[ ! -f "$REF.fai" ]]; then
    samtools faidx "$REF"
fi

# ---- Compute number of read pairs for ~50× coverage -------------------------
# Formula: N_pairs = (target_coverage × genome_size) / (2 × read_length)
#          = (50 × TOTAL_BP) / (2 × 150)  = TOTAL_BP / 6
#
# Read length is 150 bp to match the most common Illumina NovaSeq output
# (2×150 paired-end). NovaSeq X Plus can produce 2×100 or 2×250, but 2×150
# is the de facto standard for whole-genome and metagenomic sequencing.
#
# Example: reference is ~7,173 bp → N_pairs = 7173 / 6 ≈ 1,195 pairs per sample
# At 150 bp per read: 1195 × 2 × 150 = 358,500 bp → 358,500 / 7,173 ≈ 50×
TOTAL_BP=$(awk '{s+=$2} END {print s}' "$REF.fai")
N_PAIRS=$(( (50 * TOTAL_BP) / 300 ))
echo "[generate_bams] Reference: ${TOTAL_BP} bp | Read pairs per sample: ${N_PAIRS} (~50×)"

# ---- Sample list (derived from metadata, not hardcoded) ----------------------
# Sample IDs and ordering come from <data-dir>/metadata/sample_metadata.tsv so
# this script works on ANY dataset produced by generate_synthetic_data.py, not
# just the bundled 8-sample longitudinal default. Seeds are assigned by row
# order (1, 2, 3, ...) which is deterministic — the generator always writes
# samples in the same order for a given config, so BAMs stay reproducible
# across reruns.
META="$DATA_DIR/metadata/sample_metadata.tsv"
if [[ ! -f "$META" ]]; then
    echo "ERROR: metadata file not found at $META" >&2
    echo "Run generate_synthetic_data.py first to produce it." >&2
    exit 1
fi
SAMPLES=()
while IFS=$'\t' read -r sample_id _; do
    SAMPLES+=("$sample_id")
done < <(tail -n +2 "$META")
echo "[generate_bams] Samples: ${#SAMPLES[@]} (from metadata)"

# Use a temp directory for intermediate FASTQ files; clean up automatically on exit.
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# ---- Main loop: simulate → align → sort → index -----------------------------
SEED=0
for SAMPLE in "${SAMPLES[@]}"; do
    SEED=$((SEED + 1))   # 1-indexed, monotonic, reproducible
    BAM="$OUTDIR/${SAMPLE}.bam"
    echo "[generate_bams] $SAMPLE (seed=$SEED)"

    # Step 1 — wgsim: simulate paired-end reads from the reference
    #   -e 0.005  base error rate 0.5% (controls substitution frequency)
    #   -N        number of read pairs
    #   -1 150    read 1 length = 150 bp  (Illumina NovaSeq 2×150 standard)
    #   -2 150    read 2 length = 150 bp
    #   -r 0.001  SNP mutation rate (additional variants beyond the reference)
    #   -R 0.0    indel fraction = 0 (no indels, keeps alignment simple)
    #   -X 0.0    indel extension probability = 0
    #   -S        random seed (reproducibility)
    #
    # Output: r1.fq and r2.fq in the temp directory.
    # Note: wgsim assigns Phred quality ~23 to all bases regardless of error rate.
    # This is below the default alleleflux-profile threshold of 30, which is why
    # the BAM-mode configs (config_with_bams_*.yml) set min_base_quality: 0
    # for this synthetic dataset.
    wgsim \
        -e 0.005 \
        -N "$N_PAIRS" \
        -1 150 -2 150 \
        -r 0.001 -R 0.0 -X 0.0 \
        -S "$SEED" \
        "$REF" \
        "$TMPDIR_LOCAL/r1.fq" "$TMPDIR_LOCAL/r2.fq" \
        > /dev/null 2>&1

    # Step 2 — bwa mem: align reads to the reference
    #   -t 4          use 4 threads
    #   -R "@RG\t..."  read-group tag; samtools and many downstream tools expect
    #                  SM (sample name) to be set for per-sample identification
    #
    # Step 3 — samtools sort: convert SAM → coordinate-sorted BAM
    #   -@4  use 4 threads for sorting
    #   -o   output file path
    #   -    read SAM from stdin (piped directly from bwa mem)
    bwa mem -t 4 \
        -R "@RG\tID:${SAMPLE}\tSM:${SAMPLE}\tLB:${SAMPLE}\tPL:ILLUMINA" \
        "$REF" "$TMPDIR_LOCAL/r1.fq" "$TMPDIR_LOCAL/r2.fq" 2>/dev/null \
    | samtools sort -@4 -o "$BAM" -

    # Step 4 — samtools index: create .bai index so alleleflux-profile can seek
    # to arbitrary genomic positions without reading the whole BAM.
    samtools index "$BAM"
done

echo "[generate_bams] Done. BAMs written to: $OUTDIR"
# Closing tip — list the BAM-flavor config(s) that exist in the data dir.
# Both bundled datasets (example_data_longitudinal/, example_data_single/) and
# any ad-hoc generate_synthetic_data.py output write a ``config_with_bams_*.yml``
# we can point the user at directly.
echo "[generate_bams] Run the full pipeline with:"
for cfg in "$DATA_DIR"/config_with_bams_*.yml; do
    if [[ -f "$cfg" ]]; then
        echo "  alleleflux run --config $cfg --threads 4"
    fi
done
