# Configuration Reference

Complete reference for AlleleFlux configuration options.

## Quick Start

Copy the template configuration:

```bash
cp alleleflux/smk_workflow/config.template.yml my_config.yml
```

Edit `my_config.yml` with your file paths and analysis parameters.

## Core Parameters

**run_name** (optional)

Unique identifier for this analysis.

```yaml
run_name: my_study_2024
```

---

### input

Paths to required input files.

| Parameter | Description |
|-----------|-------------|
| `fasta_path` | Path to the combined reference FASTA file containing all MAG contigs. Header format should be `<MAG_ID>.fa_<contig_ID>`. |
| `prodigal_path` | Path to Prodigal gene predictions (nucleotide FASTA). Gene IDs must match contig IDs in the reference FASTA. |
| `metadata_path` | Path to sample metadata TSV file. Must contain columns: `sample_id`, `bam_path`, `subjectID`, `group`, `replicate`. For longitudinal data, also include `time`. |
| `gtdb_path` | Path to GTDB-Tk taxonomy file (`gtdbtk.bac120.summary.tsv`). Used for taxonomic aggregation of scores. |
| `mag_mapping_path` | Path to contig-to-MAG mapping file (TSV with `contig_name` and `mag_id` columns). |
| `profiles_path` | *(Optional)* Path to an existing `profiles/` directory from a prior run. Reuses **only** the Step 1 pileup output; QC and the allele-frequency cache are still rebuilt. See [Artifact Reuse and Null Runs](../usage/artifact_reuse_and_null_runs.md). |
| `reuse_from` | *(Optional)* Path to a completed run's **data-type output dir** (e.g. `.../output/longitudinal`). Reuses profiles **+ QC + allele-frequency cache** — only the group-dependent tail re-runs. Works standalone (no permutation required). See [Artifact Reuse and Null Runs](../usage/artifact_reuse_and_null_runs.md). |

**Example:**

```yaml
input:
  fasta_path: /path/to/combined_mags.fasta
  prodigal_path: /path/to/prodigal_genes.fna
  metadata_path: /path/to/sample_metadata.tsv
  gtdb_path: /path/to/gtdbtk.bac120.summary.tsv
  mag_mapping_path: /path/to/mag_mapping.tsv
```

---

### output

Output directory configuration.

| Parameter | Description |
|-----------|-------------|
| `root_dir` | Root directory for all output files. Subdirectories will be created for each analysis step. |

**Example:**

```yaml
output:
  root_dir: ./alleleflux_output
```

---

### analysis

Core analysis settings.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_type` | `longitudinal` | Type of analysis: `single` (one timepoint) or `longitudinal` (multiple timepoints). |
| `allele_analysis_only` | `false` | If true, only run allele frequency analysis without statistical tests. |
| `use_lmm` | `true` | Enable Linear Mixed Models (LMM) for repeated measures/longitudinal data. Best for accounting for subject-level variation. |
| `use_significance_tests` | `true` | Enable two-sample (t-test, Mann-Whitney) and single-sample statistical tests. Best for simple comparisons. |
| `use_cmh` | `true` | Enable Cochran-Mantel-Haenszel tests for stratified categorical analysis. Best for detecting consistent directional changes. |
| `use_dnds` | `true` | Enable dN/dS (Nei-Gojobori) evolutionary-rate analysis. **Longitudinal data only.** Set to `false` to skip the dN/dS step entirely (its parameters live under the top-level [`dnds`](#dnds) section). |
| `timepoints_combinations` | Required | List of timepoint combinations to analyze (see below). |
| `groups_combinations` | Required | List of group pairs to compare (see below). |

:::{seealso}
For detailed information about statistical tests and score calculations, see [Statistical Tests Reference](statistical_tests.md).
:::

**Timepoints Configuration:**

For longitudinal analysis, specify pairs of timepoints and a **focus timepoint**:

```yaml
analysis:
  data_type: longitudinal
  timepoints_combinations:
    - timepoint: [pre, post]
      focus: post      # The later/derived timepoint
    - timepoint: [pre, mid]
      focus: mid
```

**Understanding the Focus Timepoint:**

The focus timepoint represents the **derived** or **later** state in evolutionary comparisons:

- **For dN/dS analysis**: The focus timepoint is treated as the "derived" (Time 2) state, while the other timepoint is "ancestral" (Time 1). AlleleFlux calculates evolutionary changes in the direction: ancestral → derived.
- **For CMH scores**: The score measures differential significance relative to the focus timepoint (sites significant at focus but not at the other timepoint).
- **Selection guideline**: Always choose the **later** or **endpoint** timepoint as focus.
- **Default behavior**: If not specified, defaults to the second timepoint in the list.

**Examples:**

```yaml
# Typical longitudinal study: Day 0 → Day 30
timepoints_combinations:
  - timepoint: [day0, day30]
    focus: day30        # day30 is derived, day0 is ancestral

# Treatment study: Baseline → Post-treatment
timepoints_combinations:
  - timepoint: [baseline, post_treatment]
    focus: post_treatment  # Post is derived state
```

For single timepoint analysis:

```yaml
analysis:
  data_type: single
  timepoints_combinations:
    - timepoint: [baseline]  # No focus needed for single timepoint
```

**Groups Configuration:**

Specify pairs of groups to compare:

```yaml
analysis:
  groups_combinations:
    - [treatment, control]
    - [high_fat, standard]
```

---

### quality_control

Parameters for filtering samples and positions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_sample_num` | `4` | Minimum number of samples per group required for statistical tests. MAGs with fewer valid samples are marked as ineligible. |
| `breadth_threshold` | `0.1` | Minimum breadth of coverage (fraction of genome with ≥1x coverage). Range: 0.0-1.0. |
| `coverage_threshold` | `1.0` | Minimum average coverage depth required. Samples below this are excluded. |
| `disable_zero_diff_filtering` | `false` | If true, keep positions where allele frequencies do not change. By default, constant positions are filtered out. |

**Example:**

```yaml
quality_control:
  min_sample_num: 4
  breadth_threshold: 0.1
  coverage_threshold: 1.0
  disable_zero_diff_filtering: false
```

---

### profiling

Parameters for BAM file processing during profiling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ignore_orphans` | `true` | Ignore orphan reads (unpaired reads). Set to `false` to include unpaired reads. |
| `min_base_quality` | `30` | Minimum Phred base quality score to include a base in the pileup. |
| `min_mapping_quality` | `2` | Minimum mapping quality (MAPQ) score to include a read. |
| `ignore_overlaps` | `true` | Ignore overlapping segments of read pairs to avoid double-counting. |

**Example:**

```yaml
profiling:
  ignore_orphans: true
  min_base_quality: 30
  min_mapping_quality: 2
  ignore_overlaps: true
```

:::{note}
Higher `min_base_quality` values (e.g., 30) reduce sequencing errors but may also reduce coverage. For high-quality data, the default of 30 is recommended.
:::

---

### statistics

Parameters for statistical testing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filter_type` | `t-test` | Type of initial filter for preprocessing positions. |
| `preprocess_between_groups` | `true` | Enable preprocessing for between-group comparisons. |
| `preprocess_within_groups` | `true` | Enable preprocessing for within-group comparisons. |
| `max_zero_count` | `4` | Maximum number of zero-frequency samples allowed per position in preprocessing. |
| `p_value_threshold` | `0.05` | Significance threshold (alpha) for statistical tests. |
| `fdr_group_by_mag_id` | `false` | If true, apply FDR correction within each MAG. If false, apply across all positions. |
| `min_positions_after_preprocess` | `1` | Minimum number of positions required after preprocessing to proceed with analysis. |

**Example:**

```yaml
statistics:
  filter_type: t-test
  preprocess_between_groups: true
  preprocess_within_groups: true
  max_zero_count: 4
  p_value_threshold: 0.05
  fdr_group_by_mag_id: false
  min_positions_after_preprocess: 1
```

---

### dnds

Parameters for dN/dS (synonymous/non-synonymous) ratio calculations.

:::{note}
dN/dS analysis is **only applicable to longitudinal data** (`data_type: longitudinal`). To turn it on or off, use the [`analysis.use_dnds`](#analysis) toggle (defaults to `true`); the parameters below are read only when that toggle is enabled.
:::

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_value_column` | `q_value` | Column name to use for significance in dN/dS calculations. |
| `dn_ds_test_type` | `two_sample_unpaired_tTest` | Type of statistical test to use for dN/dS analysis. |

**Example:**

```yaml
dnds:
  p_value_column: q_value
  dn_ds_test_type: two_sample_unpaired_tTest
```

---

### regional_contrast

Parameters for regional contrast analysis (longitudinal data only). Detects genes or sliding windows where treatment and control groups show consistently different allele-frequency evolution across paired hosts.

:::{note}
Regional contrast analysis is **only applicable to longitudinal data** (`data_type: longitudinal`). It operates on raw allele-frequency changes without statistical preprocessing to avoid selection bias.
:::

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `both` | Region type(s) to analyze: `gene` (gene annotations), `window` (sliding windows), or `both`. |
| `window_size` | `1000` | Non-overlapping tile width in base pairs for sliding window analysis. Used when mode is `window` or `both`. |
| `agg_method` | `median` | How to summarize site scores within a region: `median` (robust), `mean` (simple average), or `trimmed_mean` (robust mean with custom tail trimming). |
| `trim_fraction` | `0.1` | Fraction of values to trim from each tail when using `agg_method: trimmed_mean`. Ignored for other aggregation methods. |
| `min_informative_sites` | `5` | Minimum number of variable sites required per region. Regions with fewer sites are excluded. Set to `0` to disable. **Note:** Sites with `site_score == 0` (perfect evolutionary stasis) are counted if they exist in the input. |
| `min_informative_fraction` | `0.0` | Minimum fraction of region length that must be covered by informative sites (0.0–1.0). Set to `0.0` to disable. |
| `use_fisher` | `true` | Also compute Fisher combined p-values from percentile-derived empirical p-values (secondary/exploratory analysis). Set to `false` to skip this computationally intensive step. |
| `use_regional_contrast` | `true` | Enable or disable regional contrast analysis entirely. Set to `false` to skip this analysis. |

**Example with default settings:**

```yaml
regional_contrast:
  mode: both
  window_size: 1000
  agg_method: median
  min_informative_sites: 5
  min_informative_fraction: 0.0
  use_fisher: true
  use_regional_contrast: true
```

**Example with stringent filtering:**

```yaml
regional_contrast:
  mode: both
  window_size: 1000
  agg_method: trimmed_mean
  trim_fraction: 0.15
  min_informative_sites: 10
  min_informative_fraction: 0.5
  use_fisher: false
```

**Understanding the parameters:**

- **mode**: `gene` focuses on annotated genes; `window` focuses on fixed-size genomic tiles; `both` runs both analyses in parallel.
- **window_size**: Typical values range from 500 bp (fine-grained) to 5000 bp (coarse-grained). Smaller windows increase power for localized signals; larger windows increase robustness to sparse data.
- **agg_method**: `median` is recommended for skewed data; `trimmed_mean` is robust to outliers; `mean` is simple but sensitive to extreme values.
- **min_informative_sites**: Higher values improve statistical power but reduce the number of analyzable regions. A region with all sites showing `site_score == 0` still counts as having full informative sites (no signal ≠ sparse data).
- **min_informative_fraction**: Ensures that only well-sampled regions are tested. A value of 0.5 requires at least 50% of the region to have observed variable sites.
- **use_fisher**: Fisher combined p-values provide an orthogonal statistical perspective but require additional computation. Set to `false` for large datasets if runtime is a concern.

---

### Multiple Group Combinations

AlleleFlux supports running regional contrast (and all other analyses) on **multiple group pairs simultaneously**. Each pair is analyzed independently:

```yaml
analysis:
  groups_combinations:
    - treatment: "high_fat"
      control: "control"
    - treatment: "high_fat"
      control: "standard"
```

For each group combination:
- A separate eligibility table is generated based on sample quality metrics for that specific pair
- Regional contrast analysis runs independently with separate output directories:
  - `regional_contrast/regional_contrast_{timepoints}-high_fat_control/`
  - `regional_contrast/regional_contrast_{timepoints}-high_fat_standard/`
- Results files are segregated by group combination, preventing cross-contamination

**Key note:** The `treatment` and `control` wildcards in the Snakemake rule are constrained to valid values from your configuration, ensuring that only defined group pairs are processed.

---

### resources

Computational resource allocation for cluster execution.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threads_per_job` | `16` | Number of CPU threads allocated to each job. |
| `mem_per_job` | `8G` | Memory allocation per job. Formats: `8G`, `16GB`, `8192M`. |
| `time` | `24:00:00` | Maximum wall time per job in HH:MM:SS format. |

**Example:**

```yaml
resources:
  threads_per_job: 16
  mem_per_job: 8G
  time: '24:00:00'
```

---

## Complete Configuration Example

```yaml
run_name: diet_microbiome_study

input:
  fasta_path: /data/mags/combined_mags.fasta
  prodigal_path: /data/mags/prodigal_genes.fna
  metadata_path: /data/metadata/samples_with_bam.tsv
  gtdb_path: /data/taxonomy/gtdbtk.bac120.summary.tsv
  mag_mapping_path: /data/mags/mag_mapping.tsv

output:
  root_dir: ./results

log_level: INFO

analysis:
  data_type: longitudinal
  allele_analysis_only: false
  use_lmm: true
  use_significance_tests: true
  use_cmh: true
  use_dnds: true
  timepoints_combinations:
    - timepoint: [pre, post]
      focus: post
  groups_combinations:
    - [high_fat, control]

quality_control:
  min_sample_num: 4
  breadth_threshold: 0.1
  coverage_threshold: 1.0
  disable_zero_diff_filtering: false

profiling:
  ignore_orphans: true
  min_base_quality: 30
  min_mapping_quality: 2
  ignore_overlaps: true

statistics:
  filter_type: t-test
  preprocess_between_groups: true
  preprocess_within_groups: true
  max_zero_count: 4
  p_value_threshold: 0.05
  fdr_group_by_mag_id: false
  min_positions_after_preprocess: 1

dnds:
  p_value_column: q_value
  dn_ds_test_type: two_sample_unpaired_tTest

regional_contrast:
  mode: both
  window_size: 1000
  agg_method: median
  min_informative_sites: 5
  min_informative_fraction: 0.0
  use_fisher: true
  use_regional_contrast: true

resources:
  threads_per_job: 16
  mem_per_job: 8G
  time: '24:00:00'
```

## Quick Tips

- **breadth_threshold**: Start with `0.1` (10% coverage); increase for high-coverage data
- **min_sample_num**: Minimum `4` samples per group for robust inference
- **min_base_quality**: Keep at `30` for Illumina; lower to `20` for older data
- **Resource allocation**: Adjust `threads_per_job` and `mem_per_job` based on MAG sizes

For worked examples, see [Use Cases](../examples/use_cases.md).
