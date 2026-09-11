# Design: Permuted Metadata Generation as a Snakemake Rule

**Date:** 2026-06-10
**Branch:** gene-scores
**Status:** Implemented in `1ae4571` (`rules/permute_metadata.smk`)

## Problem

Permuted (null) runs currently generate their per-comparison-pair metadata
sheets in a Python `subprocess.run` loop inside `run_permutations`
([`workflow.py`](../../alleleflux/workflow.py)). For each seed, for each
group pair, the orchestrator shells out to `alleleflux-permute-metadata` and
writes `perm_<seed>/permuted_metadata/permuted_metadata_<t>_<c>.tsv` **before**
Snakemake runs.

Three consequences:

1. **Dry-run side effects.** The loop is plain Python, outside Snakemake's
   dry-run awareness, so `alleleflux run -c <perm_config> -n` still writes real
   TSVs *and* persists a leaf config (`config_perm_<seed>.yml`) into the output
   tree. A dry-run should preview, not litter.
2. **No provenance / no skip-if-exists.** The sheets are not DAG artifacts.
   They are regenerated unconditionally and are invisible to Snakemake's
   up-to-date logic.
3. **Bespoke orchestration code.** ~15 lines of subprocess loop that duplicates
   what a rule would express declaratively.

## Goal

Move per-pair sheet generation into a tracked Snakemake rule that lives inside
each per-seed leaf run. Gains: dry-run safety, provenance + skip-if-exists, and
deletion of the subprocess loop.

**Non-goal:** parallelizing seeds. Seeds remain sequential per-leaf
`execute_workflow` calls (the per-leaf model chosen to avoid multiplying the
`eligibility_table` checkpoint by a `{seed}` wildcard). The rule parallelizes
only the cheap within-seed pair sheets, which is incidental — the value here is
hygiene, not speed.

## Current Architecture (baseline)

- **Orchestrator** (`run_permutations`): per seed → per pair → subprocess
  permute → write leaf config with `permutation.permuted_metadata_dir` set →
  `execute_workflow(leaf, working_dir=seed_dir)`.
- **Consumers** (5 sites across 4 rule files: `eligibility.smk`,
  `allele_analysis.smk`, `significance_within_group.smk` ×2,
  `significance_between_groups.smk`) read the sheet via
  `params.permuted_metadata = permuted_metadata_flag(wildcards.groups)`, which
  returns the CLI flag `--permuted_metadata <dir>/permuted_metadata_<groups>.tsv`.
  **The sheet is not declared as a Snakemake `input`** — it is an external file.
- **Leaf detection** (`execute_workflow`): a config is a leaf (skip
  orchestration) when `permutation.permuted_metadata_dir` is set.
- **BYO mode**: user supplies `permuted_metadata_dir` pointing at pre-made
  sheets; no generation, no orchestration.

## Target Architecture

### Three modes, gated by config

| Mode | Config signature | Behavior |
|---|---|---|
| Orchestrate | `enabled`, no `seed`, no `permuted_metadata_dir` | `run_permutations` fans out per seed |
| Generate leaf | `permutation.seed` (int) set | `permute_metadata` rule builds sheets; consumers depend on them |
| BYO | `permuted_metadata_dir` set, no `seed` | consumers read provided sheets; rule not wired in |

### New rule — `rules/permute_metadata.smk`

```python
rule permute_metadata:
    input:
        metadata=config["input"]["metadata_path"],
    output:
        sheet=os.path.join(OUTDIR, "permuted_metadata", "permuted_metadata_{groups}.tsv"),
    params:
        unit=PERMUTATION.get("unit", "subjectID"),
        seed=PERMUTATION_SEED,
        groups_args=lambda wc: wc.groups.replace("_", " "),
        block_flag=f"--block {PERMUTATION['block']}" if PERMUTATION.get("block") else "",
        swaps_flag=f"--swaps {PERMUTATION['swaps']}" if PERMUTATION.get("swaps") is not None else "",
    resources:
        mem_mb=get_mem_mb("permute_metadata"),
        time=get_time("permute_metadata"),
        runtime=get_runtime("permute_metadata"),
    shell:
        """
        alleleflux-permute-metadata \
            --input {input.metadata} \
            --output {output.sheet} \
            --unit {params.unit} \
            --groups {params.groups_args} \
            --seed {params.seed} \
            {params.block_flag} {params.swaps_flag}
        """
```

The original metadata is declared as `input` so the rule re-runs if the source
sheet changes. Output lands in `OUTDIR/permuted_metadata/` — identical path to
today (for a leaf, `OUTDIR == seed_dir`), so consumers are unaffected.

### `common.smk` additions (the gate — the conceptually heavy part)

```python
PERMUTATION = config.get("permutation", {})
PERMUTATION_SEED = PERMUTATION.get("seed")                    # None unless a generate-leaf
GENERATE_PERMUTED = bool(PERMUTATION.get("enabled") and PERMUTATION_SEED is not None)

def permuted_metadata_path(groups_label):
    """Absolute path to a pair's permuted sheet, generate-mode or BYO."""
    base = (os.path.join(OUTDIR, "permuted_metadata") if GENERATE_PERMUTED
            else PERMUTATION.get("permuted_metadata_dir", ""))
    return os.path.join(base, f"permuted_metadata_{groups_label}.tsv")

def permuted_metadata_input(groups_label):
    """Sheet path as a rule dependency — ONLY in generate mode (else [] so BYO
    consumers don't depend on a rule that won't run)."""
    return [permuted_metadata_path(groups_label)] if GENERATE_PERMUTED else []
```

`permuted_metadata_flag(groups_label)` keeps its signature and return contract
(`--permuted_metadata <path>` or `""`), now sourcing the path from
`permuted_metadata_path`.

### Consumer wiring (5 sites)

Add one `input` entry per consuming rule (params CLI flag unchanged):

```python
input:
    ...,
    permuted_md=lambda wildcards: permuted_metadata_input(wildcards.groups),
```

In BYO mode `permuted_metadata_input` returns `[]`, so behavior is byte-for-byte
unchanged for those runs.

### `workflow.py` — `run_permutations` changes

- **Delete** the inner `for treatment, control in pairs: subprocess.run(...)`
  loop.
- Leaf config now carries `permutation.seed = seed` (singular) plus the
  generation params (`unit`/`block`/`swaps`) instead of pre-pointing
  `permuted_metadata_dir`. `output.root_dir = seed_dir` and the
  `run_within_group_tests` default stay.
- **Leaf detection** in `execute_workflow` updates to treat *either* a `seed` or
  a `permuted_metadata_dir` as a leaf:

  ```python
  if (permutation.get("enabled")
          and permutation.get("seed") is None
          and not permutation.get("permuted_metadata_dir")):
      return run_permutations(...)
  ```

- **Dry-run cleanliness**: on `dry_run`, write the leaf config to a temporary
  directory and use it as the working dir, so nothing lands in the output tree.
  On a real run, persist the leaf config to `seed_dir` and use
  `working_dir=seed_dir` (the isolation fix already merged). TSVs are never
  written by Python in either case — the rule owns them.

### Resources

Add a small default for `permute_metadata` (cheap: shuffles a ~3k-row TSV).
Modest mem/time; relies on `get_mem_mb`/`get_time`/`get_runtime` defaults or a
single `resources_override` entry.

## Data Flow

```
orchestrating config (enabled, seeds: [...])
  └─ run_permutations: for each seed →
       write leaf config (seed set)  ──▶  execute_workflow(leaf, wd=seed_dir)
                                              └─ Snakemake DAG:
                                                   permute_metadata{groups}  (NEW)
                                                        │ output sheet
                                                        ▼
                                                   eligibility_table (checkpoint)
                                                   allele_analysis
                                                   significance_within / _between
```

## Error Handling

- Generate mode without a seed is impossible by construction
  (`GENERATE_PERMUTED` requires `seed is not None`).
- A failed `permute_metadata` job fails its leaf; the orchestrator's existing
  break-on-first-failure logic is unchanged.
- BYO with a missing sheet fails in the consumer exactly as today.

## Testing

1. **Unit** (`tests/preprocessing/test_permute_metadata.py`): unchanged — the
   `permute_metadata` *function* is untouched; only its caller moves.
2. **Equivalence (key correctness test)**: for a fixed seed, the sheet produced
   by the rule must be byte-identical to the sheet the old subprocess loop
   produced (same function, same seed → same RNG draws). Extend
   `tests/regression/test_permutation_equivalence.py`.
3. **DAG/dry-run smoke (new)**: assert (a) `permute_metadata` appears in the DAG
   when `seed` is set, (b) the sheet is an `input` to `eligibility_table`, and
   (c) a dry-run writes nothing into the output tree.
4. **Checkpoint-input verification (flagged risk)**: the sheet becomes an input
   to the `eligibility_table` checkpoint. Per the project's search-before-claim
   rule on Snakemake internals, this interaction is **verified by dry-run**
   during implementation, not assumed.

## Risk

Low-to-medium. Mechanical edits (5 one-line input wirings) dominate the line
count; the real care is ~30 lines (the BYO gate + orchestrator trim). Every
risky interaction (checkpoint input, BYO preservation, dry-run cleanliness) is
dry-run-verifiable before trusting it.

## Files Touched

**New (1):** `rules/permute_metadata.smk`

**Modified (8):** `Snakefile` (include), `shared/common.smk` (gate + helpers),
`rules/eligibility.smk`, `rules/allele_analysis.smk`,
`rules/significance_within_group.smk` (×2 sites),
`rules/significance_between_groups.smk`, `workflow.py` (orchestrator trim +
leaf detection + dry-run guard), `config.template.yml` (docs).

**Tests:** extend `test_permutation_equivalence.py`; add a DAG/dry-run smoke
test.

## Backward Compatibility

- BYO mode (`permuted_metadata_dir`) preserved unchanged.
- Old persisted leaf configs (with `permuted_metadata_dir`) still resolve via
  the BYO path.
- `config.template.yml` updated to document the seed-based generation flow.
