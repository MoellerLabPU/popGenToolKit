---
applyTo: "**"
---

# Environment Setup

## Loading the Conda Environment

**Before running any Python script, test, or shell command**, load the module and activate the environment:

```bash
module load anaconda3/2025.12
conda activate alleleflux
```

Or run a one-off command without activating:

```bash
module load anaconda3/2025.12
conda run -n alleleflux python -m pytest tests/ -v
```

**Python executable:** `/home/su2806/.conda/envs/alleleflux/bin/python`

Without this environment, `python`, `pytest`, `pysam`, `snakemake`, and all other dependencies will not be found. The system Python (`/usr/bin/python`) does not have any project packages installed.

## Running Tests

```bash
module load anaconda3/2025.12

# All tests
conda run -n alleleflux python -m pytest tests/ -v --tb=short

# Single module
conda run -n alleleflux python -m pytest tests/analysis/test_profile_mags.py -v

# With coverage
conda run -n alleleflux python -m pytest tests/ --cov=alleleflux --cov-report=html
```

## Data Paths

Reference data and BAM/FASTA files live on cluster scratch storage:

```
/scratch/gpfs/AMOELLER/sidd/
```

Do not assume these paths are accessible from local machines.
