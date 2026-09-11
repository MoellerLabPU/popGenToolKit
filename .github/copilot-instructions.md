# AI Instructions for AlleleFlux

## Project Overview
AlleleFlux is a Snakemake workflow for analyzing allele frequencies in metagenomic data, organized as a unified pipeline with checkpoint-based DAG resolution:
- **Profiling & QC**: Profile samples, generate metadata, QC filtering, eligibility tables
- **Analysis & Scoring**: Statistical tests, scoring, outlier detection, dN/dS

The workflow analyzes MAG (Metagenome-Assembled Genome) populations across samples, detecting parallel evolution through allele frequency changes.

See [README.md](../README.md) for installation and quick start. See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and PR process.

## Architecture & Data Flow

### Core Components
```
Input BAM files → Step 1 (profile_mags.py) → Per-MAG metadata files → QC filtering → Eligibility tables
                                                                                      ↓
Output: Scores, outliers, p-values ← Step 2 (Statistical tests + scoring) ← Analyze allele frequencies
```

### Script Organization
- **`alleleflux/scripts/`**: All CLI tools (installed as `alleleflux-*` commands)
  - `analysis/`: Core allele frequency analysis (`allele_freq.py`), scoring (`scores.py`, `gene_scores.py`, `taxa_scores.py`, `cmh_scores.py`), outlier detection (`outliers_genes.py`)
  - `preprocessing/`: Metadata generation (`mag_metadata.py`), QC (`quality_control.py`), eligibility tables (`eligibility_table.py`), preprocessing for statistical tests (`preprocess_*.py`)
  - `statistics/`: Statistical tests - LMM (`LMM.py`), CMH (`CMH.py`), two-sample (paired/unpaired), single-sample
  - `accessory/`: MAG mapping (`create_mag_mapping.py`), coverage stats (`coverage_and_allele_stats.py`)
  - `utilities/`: Shared functions (`utilities.py`), logging setup (`logging_config.py`)
  - `evolution/`: dN/dS analysis (`dnds_from_timepoints.py`)
  
- **`alleleflux/smk_workflow/`**: Snakemake workflow (packaged in pip installation)
  - `alleleflux_pipeline/Snakefile`: Unified entry point with checkpoint-based DAG
  - `alleleflux_pipeline/shared/common.smk`: Config parsing, resource helpers, eligibility functions
  - `alleleflux_pipeline/shared/dynamic_targets.smk`: Checkpoint-aware target generation
  - `alleleflux_pipeline/rules/`: 13 rule modules (profiling, QC, analysis, scoring, etc.)

### Configuration System
- **Central config**: `alleleflux/smk_workflow/config.yml` with nested sections
  - `input`: Paths to BAM files, FASTA, metadata, Prodigal genes, GTDB taxonomy, MAG mapping
  - `output.root_dir`: Base output directory (subdirectories auto-created by data type)
  - `analysis`: Controls workflow behavior
    - `data_type`: `"single"` or `"longitudinal"` - **critical**: determines timepoint handling, statistical tests, and output structure
    - `timepoints_combinations`: List of dicts with `timepoint` (list) and `focus` keys (for longitudinal)
    - `groups_combinations`: List of group pairs to compare (e.g., `[["fat", "control"]]`)
    - `use_lmm`, `use_significance_tests`, `use_cmh`: Boolean flags to enable/disable analysis modules
  - `quality_control`: `min_sample_num`, `breadth_threshold`
  - `statistics`: p-value thresholds, FDR settings, preprocessing filters
  - `resources`: CPUs, memory, time limits per Snakemake rule
  
- **Runtime configs**: Auto-generated with timestamps in `{output_dir}/config_runtime_{timestamp}.yml`
- **CLI overrides**: Command-line args update config via `update_config()` functions in runner scripts
- **Data type implications**:
  - `"single"`: Compares groups at single timepoints, uses unpaired tests
  - `"longitudinal"`: Tracks changes across timepoints, uses paired tests, CMH stratified by replicate

### Logging Conventions
```python
from alleleflux.scripts.utilities.logging_config import setup_logging
setup_logging()  # Call ONCE in main()
logger = logging.getLogger(__name__)

# Standard format (already configured by setup_logging):
# "[%(asctime)s %(levelname)s] %(name)s: %(message)s"
```

**Important**: Never reconfigure logging after `setup_logging()` - it handles all setup centrally.

## Development Patterns

### CLI Script Template
All CLI scripts follow this structure:
```python
#!/usr/bin/env python3
import argparse
import logging
import multiprocessing
from functools import partial
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)

def process_single_mag(mag_metadata_path: Path, other_args) -> str:
    """Worker function for a single MAG."""
    mag_id = mag_metadata_path.stem.split("_metadata")[0]
    # Process logic here
    return output_path

def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="...",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    parser.add_argument("--rootDir", required=True, help="Directory with *_metadata.tsv files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cpus", type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args()
    
    # Find all MAG metadata files
    metadata_files = glob(os.path.join(args.rootDir, "*_metadata.tsv"))
    logger.info(f"Found {len(metadata_files)} MAG(s) to process")
    
    # Parallel processing with progress bar
    worker = functools.partial(process_single_mag, other_args=...)
    with multiprocessing.Pool(processes=args.cpus) as pool:
        for result in tqdm(pool.imap_unordered(worker, metadata_files), total=len(metadata_files)):
            pass  # Results collected in worker
```

### MAG-Based Processing Pattern
- **File discovery**: Scripts find `*_metadata.tsv` files (one per MAG) in a directory
- **MAG ID extraction**: `mag_id = filename.split("_metadata")[0]` or similar
- **Metadata structure**: TSV with columns `sample_id, file_path, group, time` (group/time optional)
- **Output naming**: `{mag_id}_{script_output_type}.tsv` or `.tsv.gz`

### Statistical Analysis Integration

#### R Integration (CMH Tests)
```python
# In CMH.py - rpy2 pattern
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
from rpy2.rinterface_lib.embedded import RRuntimeError

pandas2ri.activate()  # Enable pandas ↔ R conversion

def run_cmh_test_in_r(pivoted_filtered: pd.DataFrame) -> Optional[float]:
    ro.globalenv["r_df"] = pivoted_filtered  # Transfer to R
    r("""
        library(tidyr)
        # Reshape and run mantelhaen.test
        melted <- pivot_longer(r_df, ...)
        tab3d <- xtabs(count ~ group + nucleotide + replicate, data = melted)
        cmh_result <- mantelhaen.test(tab3d, alternative = "two.sided", correct=FALSE)$p.value
    """)
    return ro.globalenv["cmh_result"][0]
```

**Key insight**: CMH test uses 3D contingency tables stratified by replicate/subject to detect parallel allele changes while controlling for individual variation.

#### Parallelization Pattern
```python
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

grouped = df.groupby(["contig", "gene_id", "position"])
worker = partial(run_test_function, param1=val1, param2=val2)
with Pool(processes=cpus) as pool:
    results = [res for res in tqdm(
        pool.imap_unordered(worker, grouped),
        total=len(grouped),
        desc="Running tests"
    ) if res is not None]
df_results = pd.DataFrame(results)
```

### File Handling Patterns
- **Path operations**: Use `pathlib.Path` for cross-platform compatibility
- **Existence checks**: Always validate before processing:
  ```python
  if not Path(file_path).exists():
      logger.warning(f"File not found: {file_path}")
      return None  # Or skip gracefully
  ```
- **Profile file format**: TSV with columns: `contig, position, total_coverage, A, C, G, T` (ATGC are allele counts)
- **Compressed output**: Results saved as `.tsv.gz` to save space (use `compression='gzip'` in `to_csv()`)
- **MAG-contig mapping**: Required TSV with `mag_id, contig_id` columns (created by `create_mag_mapping.py`)

### Workflow Integration Points

#### QC Filtering
```python
# In preprocessing scripts - QC filtering
qc_file = Path(qc_dir) / f"{mag_id}_QC.tsv"
qc_df = pd.read_csv(qc_file, sep="\t")
passed_samples = qc_df[qc_df["breadth_threshold_passed"] == True]["sample_id"]
```

**Critical column**: `breadth_threshold_passed` determines sample inclusion in downstream analysis.

#### Eligibility Tables
Eligibility files (`eligibility_table_{timepoints}-{groups}.tsv`) control which MAGs proceed to each statistical test:
- `unpaired_test_eligible`: For two-sample unpaired and LMM
- `paired_test_eligible`: For two-sample paired and CMH
- `single_sample_eligible_{group}`: Per-group single-sample test eligibility

Snakemake helper: `get_mags_by_eligibility(timepoints, groups, test_type)` → list of eligible MAG IDs

### Memory & Performance

#### Memory Optimization
```python
# Use categorical dtypes for grouping columns
df["group"] = df["group"].astype("category")
df["time"] = df["time"].astype("category")

# Explicit dtype specifications
dtype = {
    "contig": str,
    "position": int,
    "total_coverage": float,
    **{nuc: "int32" for nuc in ["A", "C", "G", "T"]}
}
df = pd.read_csv(file, sep="\t", dtype=dtype)
```

#### Parallel Processing Guidelines
- Most scripts support `--cpus` parameter
- Use `functools.partial` to fix parameters before pool mapping
- Track progress with `tqdm` for long-running operations
- Ensure workers are stateless (no shared state across processes)

## Testing & Development

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run pipeline
alleleflux run --config config.yml --threads 16
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full development setup, testing, and PR guidelines.

### Common Utilities
```python
from alleleflux.scripts.utilities.utilities import (
    load_mag_mapping,         # Load contig→MAG mapping
    extract_mag_id,           # Extract MAG ID from contig name
    calculate_score,          # Score calculation for outlier detection
    load_and_filter_data,     # Load + apply preprocessing filters
    build_contig_length_index # For coverage-weighted calculations
)
```

## Key Implementation Details

### Error Recovery
- Scripts continue processing remaining MAGs if one fails
- Log errors clearly with MAG ID context
- Return `None` from worker functions on error (filtered out in results collection)
- Use `try-except` blocks around I/O and computation

### Output Validation
```python
if df.empty:
    logger.warning(f"No results for MAG {mag_id}")
    return None

# Check for NaN in critical columns
nan_count = df["p_value"].isna().sum()
if nan_count > 0:
    logger.warning(f"{nan_count} positions with NaN p-values")
```

### Cross-Platform Compatibility
- Use `os.path.join()` or `pathlib.Path` for paths (never hardcoded `/` or `\`)
- Bash scripts (`runner.sh`) are Linux/Mac only; Python scripts are cross-platform
- Snakemake handles subprocess calls portably

## General Principles

1. **Inspect before adding**: Check existing code for similar functionality - high likelihood a utility function exists
2. **Preserve patterns**: Maintain consistency with existing architecture (e.g., MAG-based iteration, multiprocessing setup)
3. **Document non-obvious logic**: Add docstrings for complex statistical procedures or data transformations
4. **Test edge cases**: Empty DataFrames, single-sample groups, missing files, zero coverage positions
5. **Memory-conscious**: Use categorical types, explicit dtypes, and avoid loading all MAGs into memory simultaneously
6. **Fail informatively**: Clear error messages with context (MAG ID, file path, parameter values)

## Common Pitfalls to Avoid

- **Don't** assume file existence - always check and handle gracefully
- **Don't** modify global config in worker functions (use `partial` to pass parameters)
- **Don't** mix data types (single vs. longitudinal) in same analysis run
- **Don't** forget to handle NaN values in statistical tests (use `nan_policy="raise"` in scipy functions)
- **Don't** hardcode group/timepoint names - read from config or metadata
- **Don't** reuse column names across different contexts (e.g., "group" for both actual groups and pseudo-groups in CMH across-time mode)