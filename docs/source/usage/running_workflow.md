# Running the Workflow

AlleleFlux uses Snakemake to manage the workflow. This guide explains how to run the workflow and understand its components.

## Workflow Overview

AlleleFlux is a unified Snakemake pipeline that performs:

1. **Profiling**: Process BAM files to extract allele frequencies for each MAG
2. **Quality Control**: Filter samples based on coverage breadth and depth
3. **Eligibility**: Determine which MAGs qualify for each statistical test
4. **Analysis**: Analyze allele frequencies across samples
5. **Statistical Testing**: Run appropriate tests based on your experimental design
6. **Scoring**: Calculate parallelism and divergence scores
7. **Outlier Detection**: Identify genes with exceptionally high scores
8. **dN/dS Analysis**: Calculate evolutionary rates for genes under selection

The pipeline automatically manages dependencies between these steps using Snakemake checkpoints.

## Running with the AlleleFlux CLI

The recommended way to run AlleleFlux is using the `alleleflux run` command:

```bash
alleleflux run --config config.yml
```

This command handles Snakemake invocation and resource management automatically.

**Common Options:**

```bash
# Specify threads and memory for local execution
alleleflux run --config config.yml --threads 16 --memory 64G

# Use a cluster profile for HPC execution
alleleflux run --config config.yml --profile slurm_profile/

# Dry run to see what would be executed
alleleflux run --config config.yml --dry-run

# Unlock a stuck working directory after a crash
alleleflux run --config config.yml --unlock

# Pass additional Snakemake arguments
alleleflux run --config config.yml -- --forceall --reason
```

## Running with Snakemake Directly

You can also run Snakemake directly for more control:

```bash
# Find the Snakefile location
alleleflux info

# Run Snakemake directly
snakemake \
    --snakefile $(python -c "from alleleflux.workflow import get_snakefile; print(get_snakefile())") \
    --configfile /path/to/config.yml \
    --cores 16
```

## Running on HPC Clusters (SLURM)

AlleleFlux ships with a SLURM profile for cluster execution:

```bash
# Copy the bundled SLURM profile
cp -r $(python -c "import alleleflux; print(alleleflux.__path__[0])")/smk_workflow/slurm_profile ./

# Edit slurm_profile/config.yaml to match your cluster
# Then run with the profile
alleleflux run --config config.yml --profile ./slurm_profile
```

The SLURM profile submits each rule as a separate job via `sbatch`, using the resource settings from your config file (`resources.threads_per_job`, `resources.mem_per_job`, `resources.time`).

For other schedulers (PBS, SGE, LSF), create a custom Snakemake profile following the [Snakemake documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles).

## Available Statistical Tests

AlleleFlux supports several statistical testing approaches, controlled by config flags:

### Between-Group Tests

Compare allele frequencies between experimental groups (e.g., treatment vs. control):

- **Two-sample unpaired** (t-test, Mann-Whitney U): For independent samples from different individuals
- **Two-sample paired** (paired t-test, Wilcoxon signed-rank): For matched samples from the same individuals
- **LMM** (Linear Mixed Models): For repeated measures with subject-level random effects
- **CMH** (Cochran-Mantel-Haenszel): For stratified categorical analysis controlling for replicates

### Within-Group Tests

Analyze changes within a single experimental group:

- **Single-sample** (one-sample t-test): Tests if allele frequencies deviate from reference
- **LMM across time**: Within-group temporal changes with random effects
- **CMH across time**: Within-group categorical changes over time

Enable or disable tests in your config:

```yaml
analysis:
  use_significance_tests: true  # Two-sample and single-sample tests
  use_lmm: true                 # Linear Mixed Models
  use_cmh: true                 # Cochran-Mantel-Haenszel tests
```

See [Statistical Tests Reference](../reference/statistical_tests.md) for detailed methodology and score calculation formulas.

## Customizing the Workflow

The AlleleFlux config file controls all aspects of the pipeline. Below is an example configuration matching the structure of `config.template.yml`:

```yaml
run_name: "my_analysis"

analysis:
  data_type: "longitudinal"
  use_significance_tests: true
  use_lmm: true
  use_cmh: true
  timepoints_combinations:
    - timepoint: ["pre", "post"]
      focus: "post"
  groups_combinations:
    - ["treatment", "control"]

input:
  fasta_path: "/path/to/reference.fa"
  prodigal_path: "/path/to/genes.fna"
  metadata_path: "/path/to/metadata.tsv"
  gtdb_path: "/path/to/gtdbtk.tsv"
  mag_mapping_path: "/path/to/mag_mapping.tsv"

output:
  root_dir: "/path/to/output"

quality_control:
  min_sample_num: 4
  breadth_threshold: 0.1
  coverage_threshold: 1.0
  disable_zero_diff_filtering: false

statistics:
  filter_type: "t-test"
  p_value_threshold: 0.05
```

Generate a full config template with all options using:

```bash
alleleflux init --output config.yml
```

:::{tip}
**Re-running on the same data?** Set `input.reuse_from` to a completed run's data-type
output dir to skip re-profiling, QC, and the allele-frequency cache — only the
group-dependent tail re-runs. This works standalone (add a comparison, change statistics,
recover a crash) and also powers null/permutation runs. See
[Artifact Reuse and Null Runs](artifact_reuse_and_null_runs.md).
:::

### Data Type

The `data_type` setting is the most important configuration choice:

```yaml
# For studies with multiple timepoints per subject
analysis:
  data_type: "longitudinal"
  timepoints_combinations:
    - timepoint: ["pre", "post"]
      focus: "post"
  groups_combinations:
    - ["treatment", "control"]

# For cross-sectional studies (one timepoint)
analysis:
  data_type: "single"
  timepoints_combinations:
    - timepoint: ["baseline"]
  groups_combinations:
    - ["disease", "healthy"]
```

### Quality Control Thresholds

Adjust QC stringency based on your data quality:

```yaml
quality_control:
  breadth_threshold: 0.1        # Minimum fraction of genome covered (0-1)
  coverage_threshold: 1.0       # Minimum average read depth
  min_sample_num: 4             # Minimum samples per group for statistical tests
  disable_zero_diff_filtering: false  # Keep constant positions if true
```

:::{tip}
- Start with `breadth_threshold: 0.1` (10% coverage); increase for high-coverage data
- Use `min_sample_num: 4` as a minimum; more replicates improve statistical power
- Set `disable_zero_diff_filtering: false` to focus on variable positions (recommended)
:::

### Profiling Parameters

Control read filtering during allele extraction:

```yaml
profiling:
  min_base_quality: 30          # Phred score cutoff (30 = 99.9% accuracy)
  min_mapping_quality: 2        # MAPQ score cutoff
  ignore_orphans: false         # Include unpaired reads
  ignore_overlaps: true         # Avoid double-counting overlapping read pairs
```

### Statistical Parameters

Fine-tune the preprocessing and testing:

```yaml
statistics:
  filter_type: "t-test"          # "t-test", "wilcoxon", or "both"
  preprocess_between_groups: true # Filter positions before between-group tests
  preprocess_within_groups: true  # Filter positions before within-group tests
  max_zero_count: 4              # Max zero-frequency samples per position
  p_value_threshold: 0.05        # Significance threshold
  fdr_group_by_mag_id: false     # Apply FDR per-MAG (true) or genome-wide (false)
  min_positions_after_preprocess: 1  # Min positions to proceed with analysis
```

For a complete configuration reference, see [Configuration Reference](../reference/configuration.md).

## Resource Management

AlleleFlux provides flexible resource control for both local workstations and HPC clusters.

### Resource Configuration

Set computational resources per job in your config file under the `resources` section:

```yaml
resources:
  threads_per_job: 16           # Threads allocated per job
  mem_per_job: "8G"             # Memory per job (formats: "8G", "100G", "102400M")
  time: "24:00:00"              # Wall time limit (HH:MM:SS)
```

These values are used differently depending on your execution mode:

- **Local execution**: Snakemake uses `threads_per_job` to determine how many jobs can run in parallel given the `--threads` limit. For example, with `--threads 16` and `threads_per_job: 4`, up to 4 jobs run concurrently. Keep `mem_per_job` conservative to avoid out-of-memory errors.
- **Cluster execution (SLURM)**: These values are passed directly to `sbatch` as `--cpus-per-task`, `--mem`, and `--time`. SLURM allocates resources per node, so you can set higher values.

### Per-Rule Resource Overrides

Power users can override resources for specific rules using the `resources_override` section:

```yaml
resources_override:
  profile:
    threads_per_job: 8
    mem_per_job: "16G"
    time: "08:00:00"
  allele_analysis:
    mem_per_job: "32G"
  statistical_tests:
    threads_per_job: 4
    mem_per_job: "64G"
    time: "12:00:00"
  dnds_from_timepoints:
    threads_per_job: 1
    mem_per_job: "200G"
    time: "48:00:00"
```

### SLURM Profile Setup

The bundled SLURM profile uses the `cluster-generic` executor plugin. To set it up:

1. Copy the profile to your working directory:

   ```bash
   cp -r $(python -c "import alleleflux; print(alleleflux.__path__[0])")/smk_workflow/slurm_profile ./
   ```

2. Edit `slurm_profile/config.yaml` if needed. Key settings include:

   ```yaml
   executor: cluster-generic
   jobs: 100                    # Max concurrent SLURM jobs
   latency-wait: 60            # Seconds to wait for NFS file propagation
   restart-times: 0            # Number of automatic retries on failure
   keep-going: True            # Continue with independent jobs if one fails
   rerun-incomplete: True      # Rerun jobs that were incomplete
   ```

3. Run with the profile:

   ```bash
   alleleflux run --config config.yml --profile ./slurm_profile
   ```

SLURM job logs are written to `logs/{run_name}/{date}/{rule}/` in your working directory.

## Monitoring Progress

### Checking Snakemake Output

While the pipeline runs, Snakemake prints progress to the terminal showing which rules are executing, completed, or pending. Key information includes:

- **Rule names and wildcards**: Shows which MAG and timepoint/group combination is being processed
- **Job counts**: Displays total jobs remaining and completed
- **Timestamps**: Each log line is timestamped for tracking duration

### Verifying the DAG Before Running

Use `--dry-run` to inspect the execution plan without running any jobs:

```bash
# See what rules would be executed
alleleflux run --config config.yml --dry-run

# With Snakemake directly, add --reason to see why each rule will run
snakemake --snakefile <snakefile_path> --configfile config.yml -n --reason
```

This is useful for verifying that your config changes produce the expected set of jobs before committing to a full run.

### Checking Completed Rules

After a run (or during a partial run), you can check which output files exist:

```bash
# Check Snakemake's execution summary
snakemake --snakefile <snakefile_path> --configfile config.yml --summary

# List Snakemake log files for the run
ls output/longitudinal/.snakemake/log/

# View the most recent log
less output/longitudinal/.snakemake/log/$(ls -t output/longitudinal/.snakemake/log/ | head -1)
```

## Restarting Failed Runs

### Automatic Resume

Snakemake tracks completed steps via output file timestamps. If a run is interrupted (e.g., by a crash, `Ctrl+C`, or a cluster timeout), simply re-run the same command:

```bash
alleleflux run --config config.yml --threads 16
```

Snakemake will skip all rules whose output files already exist and are newer than their inputs, resuming from where it left off.

### Unlocking Stuck Directories

If the pipeline was killed abruptly (e.g., `kill -9`, node failure), Snakemake may leave a lock file that prevents re-execution. You will see an error like:

```
Error: Directory cannot be locked. ...
```

Unlock the directory with:

```bash
alleleflux run --config config.yml --unlock
```

Then re-run the pipeline as normal.

### Force-Rerunning Specific Rules

To rerun a specific rule and everything downstream of it:

```bash
# Rerun a specific rule
alleleflux run --config config.yml -- --forcerun allele_analysis

# Rerun everything from scratch
alleleflux run --config config.yml -- --forceall

# Rerun and see the reason each job is triggered
alleleflux run --config config.yml -- --forceall --reason
```

:::{note}
`--forcerun` marks the specified rule's outputs as outdated, causing Snakemake to regenerate them and all dependent downstream outputs. Use this when you have changed parameters that affect a specific step but the input files have not changed.
:::

## Output Structure

The workflow generates organized output directories:

```text
{root_dir}/
└── {data_type}/                    # "single" or "longitudinal"
    ├── profiles/                   # Per-sample allele profiles
    ├── inputMetadata/              # Per-MAG sample metadata
    ├── QC/                         # Quality control metrics
    ├── eligibility_table_*.tsv     # MAG eligibility per test
    ├── allele_analysis/            # Allele frequency analysis
    ├── significance_tests/         # Statistical test results
    │   ├── two_sample_unpaired_*/
    │   ├── two_sample_paired_*/
    │   ├── single_sample_*/
    │   ├── lmm_*/
    │   └── cmh_*/
    ├── scores/                     # Parallelism & divergence scores
    │   ├── intermediate/           #   Per-MAG raw scores
    │   └── processed/
    │       ├── combined/           #   MAG-level & taxonomic summaries
    │       └── gene_scores_*/      #   Gene-level scores
    ├── outlier_genes/              # Outlier gene detection
    └── dnds_analysis/              # dN/dS results
```

See [Output Files Reference](../reference/outputs.md) for file format details.

## Troubleshooting

### Locked Working Directory

If the pipeline was interrupted, the working directory may be locked. You will see:

```
Error: Directory cannot be locked. Please make sure that no other Snakemake process
is trying to create the same files in the following directory: ...
```

**Solution:**

```bash
alleleflux run --config config.yml --unlock
```

### rpy2 or R Errors

R is required for CMH tests. Common errors include:

**`ModuleNotFoundError: No module named 'rpy2'`** -- Install rpy2 in your environment:

```bash
pip install rpy2
```

**`RRuntimeError: Error in library(tidyr) : there is no package called 'tidyr'`** -- Install the R `tidyr` package:

```bash
R -e "install.packages('tidyr', repos='https://cloud.r-project.org')"
```

**`R_HOME` not set** -- Ensure R is on your `PATH` and `R_HOME` is set:

```bash
R --version
# If using conda, the environment.yml includes R dependencies automatically
```

### Memory Errors (OOM)

**Symptoms**: Jobs killed with `Killed`, `MemoryError`, or SLURM `OUT_OF_MEMORY` status.

**Solution**: Increase per-job memory in the config:

```yaml
resources:
  mem_per_job: "32G"
```

Or use per-rule overrides for memory-intensive steps:

```yaml
resources_override:
  allele_analysis:
    mem_per_job: "64G"
  dnds_from_timepoints:
    mem_per_job: "200G"
```

For local execution, also limit parallelism to reduce total memory usage:

```bash
alleleflux run --config config.yml --threads 4 --memory 32G
```

### No MAGs Eligible for Tests

**Symptoms**: Pipeline completes quickly with no statistical test output files, or log messages like `No eligible MAGs found`.

Check the eligibility table to see why MAGs were filtered:

```bash
column -t -s $'\t' output/longitudinal/eligibility_table_pre_post-treatment_control.tsv
```

**Common causes and fixes:**

| Cause | Fix |
|-------|-----|
| `breadth_threshold` is too high for your coverage | Lower to `0.05` or `0.01` |
| `min_sample_num` exceeds your number of replicates | Reduce to match your smallest group size |
| BAM paths in metadata are incorrect or files are missing | Verify paths with `head metadata.tsv` and check files exist |
| Contig names in BAM don't match reference FASTA | Ensure BAM was aligned to the same FASTA specified in config |

### Missing or Empty Output Files

**Symptoms**: Expected output files are not created, or files exist but are empty (0 bytes).

```bash
# Check Snakemake logs for errors
ls output/longitudinal/.snakemake/log/
less output/longitudinal/.snakemake/log/$(ls -t output/longitudinal/.snakemake/log/ | head -1)

# Check SLURM job logs (if using cluster profile)
ls logs/<run_name>/
```

**Common causes:**

- A required input file was not generated by a previous step. Check for errors in earlier rules.
- The MAG had no positions passing QC filters. Try lowering `coverage_threshold` or `breadth_threshold`.
- A worker process crashed silently. Check for `ERROR` lines in the log files.

### Prodigal Gene ID Mismatches

**Symptoms**: dN/dS analysis produces no results, or gene scores are empty.

Ensure Prodigal was run on the same combined FASTA used as the reference:

```bash
# Check that gene headers match contig names in the reference
grep ">" prodigal_genes.fna | head -5
grep ">" combined_mags.fasta | head -5
# Gene headers should start with contig IDs from the FASTA
```

### Config Validation Errors

**Symptoms**: Pipeline fails immediately with a configuration error.

```bash
# Validate your config against the template
alleleflux init --output config_template.yml
diff config.yml config_template.yml
```

Ensure all required keys are present (`input.fasta_path`, `input.metadata_path`, `input.mag_mapping_path`) and that `data_type` matches your `timepoints_combinations` structure.

### SLURM Job Failures

**Symptoms**: Jobs submitted but fail with SLURM errors.

```bash
# Check SLURM job status
sacct -j <job_id> --format=JobID,State,ExitCode,MaxRSS,Elapsed

# Common SLURM states:
# TIMEOUT     -> Increase resources.time
# OUT_OF_MEMORY -> Increase resources.mem_per_job
# FAILED      -> Check the job .out log file in logs/
```

If jobs fail due to `TIMEOUT`, increase the wall time:

```yaml
resources:
  time: "48:00:00"
```

### Getting Help

- Check the [FAQ](../reference/faq.md) for common questions
- Open a [GitHub issue](https://github.com/MoellerLabPU/AlleleFlux/issues) with your error log
- Include `alleleflux info` output and the full error traceback
