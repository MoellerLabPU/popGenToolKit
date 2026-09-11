# Tutorial

This tutorial uses the bundled synthetic dataset in
`docs/source/examples/example_data_longitudinal` to run AlleleFlux end-to-end on a normal
desktop computer. No HPC access or GPU is required.

---

## Dataset Overview

| Property | Value |
|---|---|
| MAGs | 2 (`MAG_001`, `MAG_002`) |
| Samples | 8 (4 control, 4 treatment; 2 subjects × 2 timepoints each) |
| Design | Longitudinal (pre → post) |
| Genome size | ~3,600–3,650 positions per MAG (2 contigs each) |

A simulated treatment effect is present in the pre-generated profiles: at ~8%
of positions in treatment post-timepoint samples, the dominant allele is shifted
to an alternative base. This signal is what AlleleFlux is designed to detect.

---

## Prerequisites

AlleleFlux installed (see [Installation](../getting_started/installation.md)):

```bash
conda activate alleleflux
```

Clone the repository and navigate to the example directory:

```bash
git clone https://github.com/MoellerLabPU/AlleleFlux.git
cd AlleleFlux/docs/source/examples/example_data_longitudinal
```

---

## Directory Structure

```
examples/
├── generate_synthetic_data.py      # Regenerate profiles + reference files
├── generate_bams.sh                # Regenerate BAMs from reference FASTA
└── example_data_longitudinal/
    ├── config_generated_longitudinal.yml   # Run from pre-generated profiles (Option A)
    ├── config_with_bams_longitudinal.yml   # Run from BAM files, includes profiling (Option B)
    ├── reference/
    │   ├── combined_mags.fasta
    │   ├── prodigal_genes.fna
    │   ├── mag_mapping.tsv
    │   └── gtdbtk_taxonomy.tsv
    ├── metadata/
    │   ├── sample_metadata.tsv             # For Option A
    │   └── sample_metadata_bams.tsv        # For Option B
    ├── profiles/                       # Pre-generated profiles with treatment effect
    │   └── {sample}/{sample}_{MAG}_profiled.tsv.gz
    ├── bams/                           # Synthetic BAM files (~940 KB)
    │   └── {sample}.bam[.bai]
    └── significant_sites/
        └── significant_sites.tsv
```

---

## Option A — Pre-generated profiles (recommended)

The profiling step is skipped; pre-generated profiles are loaded directly. These
profiles were created with a treatment effect baked in (see
[Data generation details](#data-generation-details)), making the statistical
results meaningful even with a small dataset.

> **Caveat:** The profiles are synthetically generated, not derived from real
> BAM files. They are designed to demonstrate statistical detection, not to
> reflect real sequencing noise.

### Dry-run

```bash
alleleflux run --config config_generated_longitudinal.yml --dry-run --threads 4
```

Expected: Snakemake prints 4 initial jobs (metadata, QC, eligibility, all) and
notes that checkpoint-based jobs will expand the DAG after running. No files
are created.

### Run end-to-end

```bash
alleleflux run --config config_generated_longitudinal.yml --threads 4
```

Outputs land under `example_output/longitudinal/`.

### Inspect outputs

```bash
OUT=example_output/longitudinal

# QC eligibility table — both MAGs pass with 2 replicates/group
column -t -s $'\t' $OUT/eligibility_table_pre_post-treatment_control.tsv

# Allele frequency changes for MAG_001
zcat $OUT/allele_analysis/allele_analysis_pre_post-treatment_control/MAG_001_allele_frequency_changes_mean.tsv.gz | head -5

# Paired t-test results for MAG_001
zcat $OUT/significance_tests/two_sample_paired_pre_post-treatment_control/MAG_001_two_sample_paired.tsv.gz | head -5

# MAG-level parallel-evolution scores by phylum
column -t -s $'\t' \
  $OUT/scores/processed/combined/phylum/scores_two_sample_paired-pre_post-treatment_control-phylum.tsv

# dN/dS output directories (one per subject pair)
ls $OUT/dnds_analysis/pre_post-treatment_control/
```

---

## Option B — Full pipeline from BAM files

Runs the complete pipeline including the profiling step, starting from the
bundled synthetic BAM files.

> **Caveat:** The BAMs were generated with `wgsim` using the same parameters for
> all samples — there is **no treatment effect** at the read level. Any
> statistically significant sites in this run are artefacts of the small n rather
> than real biological signal. This option is provided to demonstrate that the
> profiling step runs correctly, not to demonstrate statistical detection.
>
> Additionally, with only 2 subjects per group, the paired longitudinal analysis
> yields n=2 paired differences per position. Non-parametric tests (Wilcoxon
> signed-rank, Mann-Whitney) require n≥5 to achieve p<0.05, so they will report
> zero significant sites. The paired t-test can produce sub-0.05 p-values at n=2
> but those results are statistically unreliable at this sample size.


### Run end-to-end

```bash
alleleflux run --config config_with_bams_longitudinal.yml --threads 4
```

Outputs land under `example_output_bams/longitudinal/`, including a `profiles/`
subdirectory with profiles generated from the BAMs.

---

## Expected Output Structure

```
longitudinal/
├── eligibility_table_pre_post-treatment_control.tsv   # Both MAGs: all tests eligible
├── QC/                                                 # Per-sample coverage stats
├── allele_analysis/                                    # Allele frequency changes per position
│   └── allele_analysis_pre_post-treatment_control/
├── significance_tests/                                 # Per-position p-values
│   ├── two_sample_paired_pre_post-treatment_control/
│   ├── two_sample_unpaired_pre_post-treatment_control/
│   └── single_sample_pre_post-treatment_control/
├── p_value_summary/                                    # FDR-corrected q-values
├── scores/                                             # Parallel-evolution scores by taxonomy
│   └── processed/combined/{phylum,genus,MAG}/
├── outlier_genes/                                      # Per-gene outlier scores
├── dnds_analysis/                                      # dN/dS per subject pair
│   └── pre_post-treatment_control/{subj1,subj2,subj3,subj4}/
└── preprocessing_eligibility/                          # Positions passing preprocessing filters
```

---

## Expected Run Time

Measured on a node pinned to 4 CPU cores:

| Option | Config | Wall time | Peak memory |
|---|---|---|---|
| A — Pre-generated profiles | `config_generated_longitudinal.yml` | ~94 s | < 150 MB |
| B — Full pipeline from BAMs | `config_with_bams_longitudinal.yml` | ~74 s | < 150 MB |

Option B is faster in this demo because the wgsim-derived BAMs produce sparser
profiles (not every position achieves uniform coverage), so there is less data
in downstream steps. In a real experiment with whole-genome sequencing data,
profiling is the most computationally intensive step and will dominate runtime.


---

## Single-Timepoint Walkthrough

The repository also bundles a **single-timepoint** dataset in
`docs/source/examples/example_data_single/`. Use this when your real data
has only one timepoint per subject (no pre/post pairing). Same Option A and
Option B structure as the longitudinal flow; just different commands and a
different statistical model under the hood.

### What's different from the longitudinal flow

A compact reference for readers who already know the longitudinal walkthrough:

| Aspect | Longitudinal | Single-timepoint |
|--------|--------------|------------------|
| Directory | `example_data_longitudinal/` | `example_data_single/` |
| Sample naming | `{group}_subj{N}_{pre,post}` | `{group}_sample{N}` |
| Samples per subject | 2 (one per timepoint) | 1 |
| Statistical design | Paired (within-subject Δ across timepoints) | Unpaired (between-group at one timepoint) |
| Allele-frequency input | Δ post − pre per subject | Frequencies at the single timepoint |
| Primary tests | Paired t-test, Wilcoxon signed-rank, paired CMH | Unpaired t-test, Mann-Whitney U, unpaired CMH |
| Config (Option A) | `config_generated_longitudinal.yml` | `config_generated_single.yml` |
| Config (Option B) | `config_with_bams_longitudinal.yml` | `config_with_bams_single.yml` |
| Pipeline output subdir | `output_*/longitudinal/` | `output_*/single/` |
| dN/dS analysis | Runs per subject pair across timepoints | Skipped (no ancestor pairing available) |
| Regional contrast | Configurable | Configurable |

Everything else — QC, allele-frequency caching, scoring by taxonomy,
p-value summary, preprocessing eligibility — works identically. The
generators (`generate_synthetic_data.py` and `generate_bams.sh`) produce
both datasets from the same code path with the `--data_type` flag.

### Option A — Pre-generated profiles (single-timepoint)

This option is the fastest way to verify the pipeline runs and produces
output for the single-timepoint design. The bundled `profiles/` already
contain a simulated between-group difference at ~8% of positions in the
treatment-group samples.

```bash
conda activate alleleflux
cd docs/source/examples/example_data_single
```

#### Dry-run

```bash
alleleflux run --config config_generated_single.yml --dry-run --threads 4
```

Expected: Snakemake prints the planned jobs (metadata, QC, eligibility,
unpaired tests, scoring) and notes which downstream targets the checkpoint
mechanism will discover at runtime. No files are created.

#### Run end-to-end

```bash
alleleflux run --config config_generated_single.yml --threads 4
```

#### Inspect outputs

```bash
# QC eligibility table — both MAGs pass with 4 replicates/group
cat output/single/eligibility_table_*-treatment_control.tsv

# Allele frequencies for MAG_001 at the single timepoint
zcat output/single/allele_analysis/allele_analysis_*-treatment_control/MAG_001_allele_frequency_single.parquet | head

# Unpaired t-test results for MAG_001
zcat output/single/significance_tests/two_sample_unpaired_*-treatment_control/MAG_001_two_sample_unpaired.tsv.gz | head -3

# MAG-level scores aggregated by taxonomy
cat output/single/scores/processed/combined/MAG/scores_two_sample_unpaired-*-treatment_control-MAGs.tsv
```

There is no `dnds_analysis/` directory — dN/dS requires ancestor-derived
pairs across timepoints, which the single-timepoint design lacks.

---

### Option B — Full pipeline from BAM files (single-timepoint)

Mirrors Option B in the longitudinal flow: profiles are derived from BAMs
at runtime rather than read from the committed `profiles/`. This is the
mode the regression test exercises for the `single` scenario.

```bash
conda activate alleleflux
cd docs/source/examples/example_data_single
```

#### Run end-to-end

```bash
alleleflux run --config config_with_bams_single.yml --threads 4
```

Outputs land under `output_bams/single/`, including a `profiles/`
subdirectory with profiles generated from the BAMs.

---

## Expected Output Structure (Single-Timepoint)

```
single/
├── eligibility_table_<timepoint>-treatment_control.tsv  # Both MAGs eligible
├── QC/                                                   # Per-sample coverage stats
├── allele_analysis/                                      # Allele frequencies (not Δ)
│   └── allele_analysis_<timepoint>-treatment_control/
├── significance_tests/                                   # Per-position p-values
│   ├── two_sample_unpaired_<timepoint>-treatment_control/
│   └── single_sample_<timepoint>-treatment_control/
├── p_value_summary/                                      # FDR-corrected q-values
├── scores/                                               # Parallel-evolution scores
│   └── processed/combined/{phylum,genus,MAG}/
├── outlier_genes/                                        # Per-gene outlier scores
└── preprocessing_eligibility/                            # Positions passing preprocessing
```

Note the absence of `dnds_analysis/` and any `two_sample_paired_*`
directory — both require pre/post pairing.

---

## Regenerating Test Data

The bundled example dataset can be rebuilt from scratch using two scripts:

- [`generate_synthetic_data.py`](generate_synthetic_data.py) — produces the
  pre-generated **profiles** (with the simulated treatment effect baked in)
  plus the reference FASTA, Prodigal genes, MAG mapping, GTDB taxonomy,
  metadata, significant-sites table, and a ready-to-run config.
- [`generate_bams.sh`](generate_bams.sh) — produces the
  **BAM files** by simulating reads from the existing reference FASTA with
  `wgsim` and aligning them back with `bwa mem`.

The two scripts are complementary: the Python generator bypasses reads entirely
and writes the count tables directly (so it can inject a controlled treatment
effect), while the bash script produces real BAMs that exercise the profiling
step (but with no treatment effect — see the caveat in Option B above).

### Regenerating the profiles

```bash
# Reproduce the exact bundled dataset (matches the committed profiles/)
python generate_synthetic_data.py \
    --output_dir my_test_data \
    --num_mags 2 \
    --num_samples 8 \
    --seed 42
```

Expected output (truncated):

```text
[2026-05-28 13:27:33 INFO] Generating synthetic data in: my_test_data
[2026-05-28 13:27:33 INFO] 2 MAGs, 8 samples, longitudinal design
[2026-05-28 13:27:33 INFO] Wrote reference FASTA: my_test_data/reference/combined_mags.fasta
[2026-05-28 13:27:33 INFO] Wrote Prodigal genes: my_test_data/reference/prodigal_genes.fna
[2026-05-28 13:27:33 INFO] Wrote MAG mapping: my_test_data/reference/mag_mapping.tsv
[2026-05-28 13:27:33 INFO] Wrote GTDB taxonomy: my_test_data/reference/gtdbtk_taxonomy.tsv
[2026-05-28 13:27:33 INFO] Wrote metadata: my_test_data/metadata/sample_metadata.tsv
[2026-05-28 13:27:33 INFO] Generating profiles for 8 samples...
[2026-05-28 13:27:33 INFO]   8/8: treatment_subj4_post
[2026-05-28 13:27:33 INFO] Wrote significant sites: my_test_data/significant_sites/significant_sites.tsv
[2026-05-28 13:27:33 INFO] Wrote config: my_test_data/config_generated_longitudinal.yml
[2026-05-28 13:27:33 INFO] Done. Total size: 174.5 KB
[2026-05-28 13:27:33 INFO] Run with: alleleflux run --config my_test_data/config_generated_longitudinal.yml
```

Then run the pipeline directly on the generated config:

```bash
alleleflux run --config my_test_data/config_generated_longitudinal.yml --threads 4
```

**Common variations:**

```bash
# Larger dataset (more MAGs, more samples — useful for testing LMM/CMH)
python generate_synthetic_data.py \
    --output_dir large_test \
    --num_mags 10 \
    --num_samples 20 \
    --genes_per_contig 5

# Single-timepoint design (compares groups without pre/post pairing)
python generate_synthetic_data.py \
    --output_dir single_tp_test \
    --data_type single \
    --num_samples 8 \
    --timepoints t1

# Custom group names and longer contigs
python generate_synthetic_data.py \
    --output_dir custom_test \
    --groups healthy disease \
    --contig_length_min 2000 \
    --contig_length_max 5000
```

Notable flags (see `--help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--num_mags` | 2 | Number of MAGs to generate |
| `--num_samples` | 8 | Must be divisible by `groups × timepoints` |
| `--num_contigs_per_mag` | 2 | Contigs per MAG |
| `--genes_per_contig` | 3 | Genes placed per contig (non-overlapping) |
| `--data_type` | `longitudinal` | `single` for one-timepoint designs |
| `--groups` | `control treatment` | Space-separated group names |
| `--timepoints` | `pre post` | Space-separated timepoint labels |
| `--coverage_min` / `--coverage_max` | 20 / 80 | Per-position coverage band |
| `--seed` | 42 | Same seed → byte-identical dataset |

### Regenerating the bundled BAM files

To rebuild the BAMs that live in `example_data_longitudinal/bams/` from the committed reference:

```bash
# wgsim and bwa are NOT in the alleleflux env — install once if missing:
conda install -c bioconda bwa wgsim

# From inside docs/source/examples/:
bash generate_bams.sh
```

Expected output:

```text
[generate_bams] Script dir: /path/to/AlleleFlux/docs/source/examples
[generate_bams] Data dir:   /path/to/AlleleFlux/docs/source/examples/example_data_longitudinal
[generate_bams] Indexing reference with bwa...
[generate_bams] Reference: 7173 bp | Read pairs per sample: 1195 (~50×)
[generate_bams] Samples: 8 (from metadata)
[generate_bams] control_subj1_pre (seed=1)
[generate_bams] control_subj1_post (seed=2)
... (one line per sample) ...
[generate_bams] Done. BAMs written to: /path/to/.../example_data_longitudinal/bams
[generate_bams] Run the full pipeline with:
  (cd /path/to/.../example_data_longitudinal && alleleflux run --config config_with_bams_longitudinal.yml --threads 4)
```

The script uses the `reference/combined_mags.fasta` already committed in the
repository. Sample names and order come from `metadata/sample_metadata.tsv` —
seeds are assigned by row index, so reruns produce identical BAMs.
Runtime is under 5 seconds for the bundled 7 kb reference.

> **Note:** `wgsim` produces reads with uniform Phred quality ~23, which is
> below the default `alleleflux-profile` threshold of 30. That's why
> `config_with_bams_longitudinal.yml` (and `config_with_bams_single.yml`) sets `min_base_quality: 0` for this synthetic dataset.

### From-scratch BAM workflow (fresh dataset → BAMs → pipeline)

If you want a brand-new dataset *and* BAMs derived from it (different from the
bundled example), `generate_bams.sh` accepts `--data-dir <path>` so it can
operate on any directory produced by `generate_synthetic_data.py`. Three
commands, no manual config edits:

```bash
cd docs/source/examples

# 1. Generate reference + profiles + both configs (profile-mode and BAM-mode)
python generate_synthetic_data.py --output_dir my_bam_test --num_mags 2 --seed 42

# 2. Generate BAMs from the freshly-written reference; samples come from
#    my_bam_test/metadata/sample_metadata.tsv automatically.
bash generate_bams.sh --data-dir my_bam_test

# 3. Run the full pipeline (includes profiling step)
alleleflux run --config my_bam_test/config_with_bams_longitudinal.yml --threads 4
```

Each `generate_synthetic_data.py` run now emits **two** ready-to-go configs in
the output dir:

| File | Mode | What runs |
|---|---|---|
| `config_generated_{single,longitudinal}.yml` | Profile mode | Skips profiling, reads pre-generated profiles |
| `config_with_bams_{single,longitudinal}.yml` | BAM mode | Full pipeline including profiling from BAMs |

For single-timepoint designs, swap `--data_type single --num_samples 4` into
step 1 and use `config_with_bams_single.yml` in step 3 — the rest of the flow
is identical.

The `generate_bams.sh` script derives the sample list from
`<data-dir>/metadata/sample_metadata.tsv`, so it works on any dataset shape
(2 MAGs or 20, single or longitudinal, custom group/timepoint names) without
edits.

## Data Generation Details

The pre-generated profiles were created with
[`generate_synthetic_data.py`](generate_synthetic_data.py) (seed = 42):

- Each MAG has 2 contigs (~800–2,000 bp, 3 genes per contig)
- Coverage drawn uniformly from 20–80× at every position
- The dominant allele receives 85–98% of reads; the remainder are split randomly
  among the other three bases (capped at 3 reads each to stay below typical
  variant-caller minor-allele thresholds)

**Simulated treatment effect:**

```python
if is_treatment and is_post and random.random() < treatment_effect_rate:
    ref_idx = BASES.index(ref_base)
    dominant_base = BASES[(ref_idx + 1) % 4]
```

In treatment post-timepoint samples only, each position has an 8% probability of
shifting its dominant allele one step forward in the `BASES = [A, T, G, C]`
cycle relative to the reference base (so `A → T`, `T → G`, `G → C`, `C → A`).
The `% 4` wraps the index so `C` cycles back to `A` cleanly. This creates real
but modest allele frequency differences between groups that AlleleFlux is
designed to detect.

The BAMs were generated with `wgsim` (150 bp paired-end, matching Illumina
NovaSeq 2×150 output, 0.5% error rate, unique seed per sample) and aligned with
`bwa mem`. All samples were generated from the same reference with the same
parameters, so no treatment effect is present at the read level.


---

## File Format Reference

### Profile files (`*_profiled.tsv.gz`)

| Column | Type | Description |
|---|---|---|
| `contig` | string | Contig identifier |
| `position` | int | 0-based genomic position |
| `ref_base` | string | Reference base (A/C/G/T) |
| `total_coverage` | int | Total read depth |
| `A`, `C`, `G`, `T` | int | Per-base read counts |
| `N` | int | Ambiguous base count |
| `gene_id` | string | Overlapping gene ID (empty if intergenic) |

### Sample metadata

| Column | Description |
|---|---|
| `sample_id` | Unique sample identifier |
| `bam_path` | Path to BAM file or pre-generated profile directory |
| `subjectID` | Biological replicate / subject ID |
| `group` | Experimental group (`control` / `treatment`) |
| `replicate` | Replicate letter within group |
| `time` | Timepoint (`pre` / `post`) |
