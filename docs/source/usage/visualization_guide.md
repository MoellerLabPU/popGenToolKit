# Visualization Guide

AlleleFlux provides a complete visualization workflow for exploring allele frequency dynamics across samples and timepoints. This guide walks through the four-step visualization pipeline, from identifying significant sites to generating publication-ready plots.

:::{note}
The visualization workflow requires completing the main AlleleFlux pipeline first to generate significant sites. Alternatively, you can use the provided example data to learn the workflow.
:::

---

## Visualization Workflow Overview

The visualization pipeline consists of four sequential steps:

```text
1. Prepare Metadata      →  Standardize sample metadata with profile paths
2. Terminal Nucleotide   →  Identify terminal (endpoint) alleles at significant sites
3. Track Alleles         →  Build frequency tables for anchor alleles across samples
4. Plot Trajectories     →  Generate line plots, box plots, and violin plots
```

Each step produces output files that serve as input to the next step.

---

## Step 1: Prepare Metadata

**Command:** `alleleflux-prepare-metadata`

This step standardizes your metadata and adds profile file paths, creating a unified metadata file for the visualization workflow.

### Input Requirements

**Original Metadata File** (TSV):

| Column | Required | Description |
|--------|----------|-------------|
| `sample_id` | Yes | Unique sample identifier |
| `group` | Yes | Experimental group (e.g., "treatment", "control") |
| `time` | Yes | Timepoint identifier (e.g., "pre", "post", "day1") |
| `subjectID` | Yes | Biological replicate/subject ID for pairing samples |
| `day` | No | Numeric day for continuous time axis |
| `replicate` | No | Replicate identifier within group |

**Profile Directory Structure:**

```text
profiles/
├── sample1/
│   ├── sample1_MAG_001_profiled.tsv.gz
│   └── sample1_MAG_002_profiled.tsv.gz
├── sample2/
│   └── ...
```

### Usage

```bash
alleleflux-prepare-metadata \
    --metadata-in original_metadata.tsv \
    --metadata-out visualization_metadata.tsv \
    --base-profile-dir /path/to/profiles \
    --sample-col sample_id \
    --group-col group \
    --time-col timepoint
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--metadata-in` | Required | Input metadata file |
| `--metadata-out` | Required | Output standardized metadata (appends if exists) |
| `--base-profile-dir` | Required | Base directory containing sample profile subdirectories |
| `--sample-col` | `sample_id` | Column name for sample IDs in input |
| `--group-col` | `group` | Column name for experimental groups |
| `--time-col` | `time` | Column name for timepoints |

### Output

**Standardized Metadata** (`visualization_metadata.tsv`):

```text
sample_id    group       time    subjectID    file_path
sample1      treatment   pre     subj1        /path/to/profiles/sample1
sample2      treatment   post    subj1        /path/to/profiles/sample2
sample3      control     pre     subj2        /path/to/profiles/sample3
sample4      control     post    subj2        /path/to/profiles/sample4
```

---

## Step 2: Terminal Nucleotide Analysis

**Command:** `alleleflux-terminal-nucleotide`

This step identifies the "terminal" (endpoint) nucleotide at each significant genomic site. Two methods are used:

1. **Mean Frequency Method**: Calculates mean allele frequencies and selects the most frequent allele
2. **Majority Vote Method**: Each sample votes for its dominant allele; the most-voted allele wins

### Input Requirements

**Significant Sites File** (TSV):

| Column | Required | Description |
|--------|----------|-------------|
| `mag_id` | Yes | MAG identifier |
| `contig` | Yes | Contig identifier |
| `position` | Yes | 0-based genomic position |
| `gene_id` | Yes | Gene identifier |
| `min_p_value` | Yes* | Minimum p-value (used for filtering) |
| `q_value` | Yes* | FDR-adjusted p-value (used for filtering) |

At least one of `min_p_value` or `q_value` required, depending on `--p_value_column` setting.

**Example significant_sites.tsv:**

```text
mag_id      contig                  position    gene_id                     test_type               min_p_value    q_value
MAG_001     MAG_001.fa_contig1      120         MAG_001.fa_contig1_gene1    two_sample_paired_tTest 1.5e-06        2.3e-05
MAG_001     MAG_001.fa_contig1      145         MAG_001.fa_contig1_gene1    two_sample_paired_tTest 3.2e-05        1.8e-04
```

### Usage

```bash
alleleflux-terminal-nucleotide \
    --significant_sites p_value_summary.tsv \
    --profile_dir profiles/ \
    --metadata visualization_metadata.tsv \
    --group treatment \
    --timepoint post \
    --output results/terminal/ \
    --p_value_column q_value \
    --p_value_threshold 0.05 \
    --cpus 16
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--significant_sites` | Required | Path to significant sites table |
| `--profile_dir` | Required | Directory containing sample profile subdirectories |
| `--metadata` | Required | Standardized metadata file from Step 1 |
| `--group` | Required | Target group for terminal nucleotide calculation |
| `--timepoint` | Required | Target timepoint (typically endpoint) |
| `--output` | Required | Output directory |
| `--p_value_column` | `q_value` | Column for filtering: `min_p_value` or `q_value` |
| `--p_value_threshold` | 0.05 | Maximum p-value to include site |

### Output Files

**Per-MAG Terminal Nucleotides** (`{output}/{mag_id}/{mag_id}_terminal_nucleotides.tsv`):

| Column | Description |
|--------|-------------|
| `contig` | Contig identifier |
| `position` | Genomic position |
| `gene_id` | Gene identifier |
| `terminal_nucleotide_mean_freq` | Terminal allele (mean frequency method) |
| `terminal_nucleotide_majority_vote` | Terminal allele (majority vote method) |
| `min_p_value`, `q_value` | Original significance values |

**Full Frequency Data** (`{mag_id}_frequencies.tsv`): Complete allele frequencies for all samples at each site.

**Summary File** (`terminal_nucleotide_analysis_summary.tsv`): Aggregated results across all MAGs.

---

## Step 3: Track Alleles

**Command:** `alleleflux-track-alleles`

This step creates frequency tables that track the "anchor" allele (terminal nucleotide) across all samples and timepoints. The output is formatted for direct use in plotting.

### Input Requirements

- **Anchor File**: Output from Step 2 (terminal nucleotides)
- **Enhanced Metadata**: Metadata with `file_path` column from Step 1

### Usage

```bash
alleleflux-track-alleles \
    --mag-id MAG_001 \
    --anchor-file results/terminal/MAG_001/MAG_001_terminal_nucleotides.tsv \
    --metadata visualization_metadata.tsv \
    --output-dir results/tracking/ \
    --anchor-column terminal_nucleotide_mean_freq \
    --min-cov-per-site 5 \
    --cpus 16
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--mag-id` | Required | MAG identifier to process |
| `--anchor-file` | Required | Terminal nucleotides file from Step 2 |
| `--metadata` | Required | Enhanced metadata with `file_path` column |
| `--output-dir` | Required | Output directory |
| `--anchor-column` | `terminal_nucleotide_mean_freq` | Which anchor method to use |
| `--min-cov-per-site` | 0 | Minimum coverage required (sites below excluded) |

### Output Files

**Wide-Format Frequency Table** (`{mag_id}_frequency_table.wide.tsv`):

Sites as rows, samples as columns:

```text
contig    position    gene_id    anchor_allele    sample1_pre    sample1_post    sample2_pre    sample2_post
contig1   120         gene1      A                0.92           0.45            0.88           0.42
contig1   145         gene1      G                0.85           0.72            0.91           0.68
```

**Long-Format Frequency Table** (`{mag_id}_frequency_table.long.tsv`):

Tidy format for R/ggplot2:

```text
contig    position    gene_id    anchor_allele    sample_id    frequency    group       time    subjectID
contig1   120         gene1      A                sample1      0.92         treatment   pre     subj1
contig1   120         gene1      A                sample1      0.45         treatment   post    subj1
contig1   120         gene1      A                sample2      0.88         control     pre     subj2
```

---

## Step 4: Plot Allele Trajectories

**Command:** `alleleflux-plot-trajectories`

This step generates visualization plots showing allele frequency trajectories. Multiple plot types are available:

- **Line Plots**: Track individual site trajectories over time
- **Box Plots**: Show distribution of frequencies at each timepoint
- **Violin Plots**: Show frequency distributions with density

### Input Requirements

**Long-Format Frequency Table** (from Step 3):

Required columns:

| Column | Description |
|--------|-------------|
| `contig`, `position`, `anchor_allele` | Site identification |
| `frequency` | Allele frequency value (0-1) |
| `group` | Experimental group |
| `subjectID` | Subject/replicate identifier |
| `time` or `day` | Temporal information for x-axis |

### Usage

```bash
alleleflux-plot-trajectories \
    --input_file results/tracking/MAG_001_frequency_table.long.tsv \
    --value_col q_value \
    --n_sites_line 20 \
    --n_sites_dist all \
    --x_col time \
    --x_order pre post \
    --plot_types line box violin \
    --per_site \
    --n_sites_per_site 10 \
    --output_dir plots/MAG_001 \
    --output_format pdf \
    --group_by_replicate
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_file` | Required | Long-format frequency table from Step 3 |
| `--value_col` | `min_p_value` | Column for ranking sites: `min_p_value`, `q_value` |
| `--n_sites_line` | 10 | Number of top sites for line plots (or "all") |
| `--n_sites_dist` | all | Number of sites for box/violin plots |
| `--x_col` | `time` | X-axis column: `time` or `day` |
| `--x_order` | None | Custom x-axis order (space-separated) |
| `--plot_types` | `line` | Plot types: `line`, `box`, `violin` |
| `--per_site` | False | Generate individual plots per site |
| `--n_sites_per_site` | None | Number of sites for per-site plots |
| `--output_dir` | `./plots` | Output directory |
| `--output_format` | `png` | Format: `png`, `pdf`, `svg` |
| `--group_by_replicate` | False | Aggregate trajectories by replicate |

### Advanced Options

**Day Binning** (for continuous time data):

```bash
--x_col day --bin_width 7 --min_samples_per_bin 3
```

**Custom Styling:**

```bash
--line_alpha 0.6 --output_format svg
```

### Output Files

**Combined Plots:**

- `{mag_id}_line_plot.{format}` - Line trajectories for top N sites
- `{mag_id}_box_plot.{format}` - Box plots across timepoints
- `{mag_id}_violin_plot.{format}` - Violin plots with density

**Per-Site Plots** (when `--per_site` enabled):

- `per_site/{contig}_{position}_{gene}_line.{format}`
- `per_site/{contig}_{position}_{gene}_by_replicate.{format}`

---

## Example: Complete Visualization Workflow

Using the provided example data:

```bash
# Navigate to example data directory
cd docs/source/examples/example_data_longitudinal

# Step 1: Prepare metadata (already done - metadata provided)
# alleleflux-prepare-metadata ...

# Step 2: Terminal nucleotide analysis
alleleflux-terminal-nucleotide \
    --significant_sites significant_sites/significant_sites.tsv \
    --profile_dir profiles/ \
    --metadata metadata/sample_metadata.tsv \
    --group treatment \
    --timepoint post \
    --output results/terminal/ \
    --p_value_column q_value \
    --p_value_threshold 0.05

# Step 3: Track alleles
alleleflux-track-alleles \
    --mag-id TEST_MAG_001 \
    --anchor-file results/terminal/TEST_MAG_001/TEST_MAG_001_terminal_nucleotides.tsv \
    --metadata metadata/sample_metadata.tsv \
    --output-dir results/tracking/

# Step 4: Generate plots
alleleflux-plot-trajectories \
    --input_file results/tracking/TEST_MAG_001_frequency_table.long.tsv \
    --n_sites_line 10 \
    --plot_types line box \
    --x_order pre post \
    --output_dir plots/TEST_MAG_001 \
    --output_format png
```

---

## Expected Plot Outputs

After running the visualization pipeline, you should see plots like:

**Line Plot** (`MAG_001_line_plot.png`):

Shows allele frequency trajectories for top significant sites. Each line represents a single genomic site, with x-axis showing timepoints and y-axis showing anchor allele frequency (0-1). Different colors may represent different genes or groups.

:::{note}
**To generate example plots:**

Run the complete workflow above with the example data, then find plots in the `plots/` directory. For publication-quality figures, use `--output_format pdf` or `--output_format svg`.
:::

**Box Plot** (`MAG_001_box_plot.png`):

Distribution of allele frequencies at each timepoint across all significant sites. Useful for seeing overall trends in allele frequency shifts.

**Violin Plot** (`MAG_001_violin_plot.png`):

Similar to box plots but shows the full distribution density, helpful for identifying bimodal patterns or skewed distributions.

---

## Generating Screenshots

To generate the example plots shown in this documentation:

1. Run the example workflow above
2. Screenshots will be in `plots/TEST_MAG_001/`

For generating documentation images:

```bash
# Generate high-resolution PNG for documentation
alleleflux-plot-trajectories \
    --input_file results/tracking/TEST_MAG_001_frequency_table.long.tsv \
    --plot_types line box violin \
    --x_order pre post \
    --output_dir docs/source/_static/images/visualization_examples \
    --output_format png
```

---

## Troubleshooting

**No sites passing p-value threshold:**

- Lower `--p_value_threshold` (e.g., 0.1)
- Check if `--p_value_column` matches your data (`q_value` vs `min_p_value`)

**Empty frequency tables:**

- Ensure `--profile_dir` structure matches expected format
- Verify `--metadata` has correct `file_path` column
- Check `--min-cov-per-site` isn't too stringent

**Missing timepoints in plots:**

- Verify `--x_order` matches values in your `time` column exactly
- Check metadata has samples at all specified timepoints

**Memory errors with large datasets:**

- Process one MAG at a time
- Reduce `--cpus` to limit parallel processing

---

## See Also

- [CLI Reference](../reference/cli_reference.md) - Complete CLI documentation
- [Output Files Reference](../reference/outputs.md) - Output file specifications
- [Tutorial](../examples/tutorial.md) - Full workflow tutorial
