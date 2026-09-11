---
applyTo: "alleleflux/smk_workflow/**/*.smk,alleleflux/smk_workflow/**/Snakefile"
---

# Snakemake Workflow Conventions

## Architecture

The pipeline is a unified Snakefile with checkpoint-based DAG resolution:

```
alleleflux/smk_workflow/alleleflux_pipeline/
├── Snakefile              # Main entry: includes all modules, defines get_final_pipeline_outputs()
├── shared/
│   ├── common.smk         # Config parsing, resource helpers, MAG eligibility functions
│   └── dynamic_targets.smk # Checkpoint-aware target generation functions
└── rules/                 # 13 rule modules (one per pipeline stage)
```

## Key Patterns

### Resource Management

All rules use centralized resource helpers from `shared/common.smk`:
```python
threads: get_threads("rule_name")
resources:
    mem_mb=get_mem_mb("rule_name"),
    time=get_time("rule_name"),
```

Per-rule overrides come from `config["resources_override"]`.

### Rule Structure

Rules call AlleleFlux CLI commands (not Python directly):
```python
rule example:
    input: ...
    output: ...
    threads: get_threads("example")
    resources:
        mem_mb=get_mem_mb("example"),
        time=get_time("example"),
    params:
        log_level=config.get("log_level", "INFO"),
    shell:
        """
        alleleflux-command \
            --input {input.file} \
            --output {output.result} \
            --cpus {threads} \
            --log-level {params.log_level}
        """
```

### Checkpoints

Two checkpoint stages control DAG resolution:
1. **`eligibility_table`** — QC checkpoint after Step 1 (determines which MAGs proceed)
2. **`preprocessing_eligibility_*`** — Preprocessing filter checkpoint before Step 2 scoring

Target generation functions in `dynamic_targets.smk` read checkpoint output files to determine which MAGs are eligible for each analysis.

### Wildcard Constraints

Defined in `shared/common.smk` from config values. Wildcards: `{groups}`, `{timepoints}`, `{test_type}`, `{focus_tp}`, `{mag}`, `{sample}`.

### Config Variables

Global variables set in `shared/common.smk` and available to all rule files:
- `DATA_TYPE` — `"single"` or `"longitudinal"`
- `OUTDIR` — root output directory
- `USE_EXISTING_PROFILES` — skip profiling if True
- `PROFILES_DIR` — profiles directory path
- `timepoints_labels`, `groups_labels` — list of combination labels
- `all_focus_timepoints` — extracted focus timepoints

### Stable Inputs

Reference files (FASTA, Prodigal, MAG mapping) are wrapped with `ancient()` to prevent re-runs on timestamp changes.

## After Refactoring: Keeping Rules Clean

After any architectural change to a rule, verify:

- Every `params:` entry is referenced in the `shell:` block (and vice versa).
- Every `shell:` flag maps to a real, currently-used CLI argument in the target script — check the script's `argparse` definition.
- Wildcard-dependent `params` use a `lambda wildcards:` — bare `{wildcards.X}` inside a `params:` string value is **not** expanded by Snakemake.
- `data_type`-conditional flags (e.g. `--timepoint` only meaningful for `longitudinal`) should be conditionally passed via a lambda param and made `required=False` in the script, with validation in the branch that needs them.
- Functions defined in a `.smk` file (e.g. `_cache_paths_for_combination`) are actually called by at least one rule input/param — orphaned helpers silently survive but never run.

## When Adding a New Rule

1. Create a new `.smk` file in `rules/`
2. Use `get_threads()` / `get_mem_mb()` / `get_time()` for resources
3. Call the CLI command in `shell:` block (add the CLI entry point to `pyproject.toml` first)
4. Include the new rule file in `Snakefile`
5. If the rule depends on MAG eligibility, add target generation in `dynamic_targets.smk`
6. Add resource defaults to `config.template.yml`

## Reference

- Config template: [`config.template.yml`](../../config.template.yml)
- Test configs: [`config_test.yml`](../../config_test.yml), [`config_test_permuted.yml`](../../config_test_permuted.yml)
