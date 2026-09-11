# Artifact Reuse and Null (Permutation) Runs

This page explains how AlleleFlux reuses expensive intermediate artifacts from a prior
run, and how that same machinery powers null (permutation) runs. It covers the two reuse
levels (`profiles_path` vs `reuse_from`), when to use each, how to run a permutation null,
and how to supply your own pre-made permuted metadata sheets.

---

## Overview

A full AlleleFlux run computes three **group-independent** stages before any group
comparison happens:

```
profiles  →  QC  →  allele_freq_cache   →   [group-dependent tail]
   │           │           │                  eligibility → Stage 2
   └─ Step 1   └─ breadth  └─ per-(MAG,        → statistical tests → scores
      pileup      coverage    timepoint)
                              frequency
                              parquet
```

The first three stages do not depend on *which* groups you compare — a mouse's pileup,
its QC breadth, and its per-position allele frequencies are properties of the sample, not
of the comparison (see [Allele Frequency Cache Architecture](allele_freq_cache_architecture.md)).
They are also, by far, the most expensive stages.

AlleleFlux exposes **two reuse levels** so a second run can skip re-computing what it can
safely inherit:

| Config key | Reuses | Rebuilds |
|------------|--------|----------|
| `input.profiles_path` | profiles **only** | QC, allele_freq_cache, and the whole tail |
| `input.reuse_from` | profiles **+ QC + allele_freq_cache** | only the group-dependent tail |

`reuse_from` subsumes `profiles_path`: when you set `reuse_from`, profiles default to
`<reuse_from>/profiles` automatically. An explicit `profiles_path` still wins if you set
both (an escape hatch for split-storage layouts).

---

## When to use which

The dividing line is a single question: **does my change invalidate QC?**

QC (breadth, coverage) and the allele-frequency cache both depend on the
`quality_control` thresholds and `profiling` parameters. The raw profiles (Step 1 pileup)
do not — they are just per-position base counts.

### Use `profiles_path` when...

You changed a **QC / coverage / breadth / profiling** parameter and need QC and the cache
rebuilt, but the underlying pileup is unchanged:

```yaml
input:
  profiles_path: /path/to/prior_run/longitudinal/profiles
quality_control:
  breadth_threshold: 0.2     # changed from 0.1 → QC + cache must rebuild
```

This reuses the pileup (expensive, threshold-independent) and recomputes everything that
depends on the new threshold.

### Use `reuse_from` when...

The underlying data is unchanged and you only want a **different comparison or different
statistics** — or you are running a **null/permutation** (below). Point it at the prior
run's *data-type output directory* (the `longitudinal/` or `single_timepoint/` subdir, not
the root):

```yaml
input:
  reuse_from: /path/to/prior_run/longitudinal
```

Everything group-independent is read straight from the prior run; only
eligibility → Stage 2 → stats → scores re-runs on the new settings.

:::{tip}
**The decision rule.**
Changed a QC/coverage/breadth threshold? → `profiles_path` (reuse pileup, rebuild the rest).
Same data, different groups/stats — or doing permutations? → `reuse_from` (reuse the whole
group-independent stack).
:::

---

## `reuse_from` is a standalone feature — you do *not* need permutation

A common misconception: `reuse_from` is only for null runs. **It is not.** `reuse_from`
works entirely on its own. The dependency is one-directional:

- A **permutation** run *requires* `reuse_from` (a null that rebuilt the cache from scratch
  each time would defeat the purpose).
- But `reuse_from` *requires nothing* — it is a general artifact-reuse knob.

Concrete standalone uses:

- **Add a group comparison** to an existing run without re-profiling. Add the new pair to
  `analysis.groups_combinations`, set `reuse_from` to the old run, point `output.root_dir`
  somewhere new, and only the new comparison's tail computes.
- **Re-run with different statistics** (toggle `use_lmm`, change `p_value_threshold`,
  adjust preprocessing) on the same cached data.
- **Recover a crashed run** whose profiles/QC/cache completed but whose stats did not.

If the reuse directory is missing an expected subdir (`QC/` or `allele_freq_cache/`),
AlleleFlux **warns and rebuilds** that stage rather than failing — so pointing `reuse_from`
at a run that only got as far as profiles gracefully degrades to profiles-only reuse.

---

## Null (permutation) runs

A null run estimates how large an AlleleFlux score looks **by chance** by shuffling group
labels among samples and re-running the group-dependent tail. Because the science of the
shuffle lives entirely in the `group` column, AlleleFlux reuses the real run's
profiles/QC/cache and **relabels in memory** — it never re-profiles.

### The three modes, in plain language

A permutation run is always in one of three modes. In practice you only *choose* between
two of them — let AlleleFlux make the permuted sheets for you, or hand it your own. The
third mode is just what the top-level run does behind the scenes.

- **Orchestrate — the conductor.** This is your top-level permutation config (`enabled: true`
  with `n_permutations` or `seeds`). It runs **no analysis itself**. It loops over your
  seeds and, for each one, spins off an independent child run. You never ask for "orchestrate"
  explicitly — it is simply what happens when you turn permutation on and let AlleleFlux
  generate the sheets.

- **Generate — the default worker.** Each child the orchestrator spins off is a *generate*
  run. It builds its **own** permuted sheets (one per comparison pair, for that one seed)
  as a normal pipeline step, then runs the tail. This is the path you get automatically:
  fresh, reproducible sheets with zero manual work. Because the generation is a tracked
  pipeline step, a dry run (`-n`) shows the work and writes **nothing**.

- **BYO (bring-your-own) — the manual override.** If you already have permuted sheets, you
  skip both the conductor and the generation: point `permuted_metadata_dir` at your folder
  and AlleleFlux runs the tail directly on your files. No seeds, no shuffling — your sheets
  are used exactly as provided.

| | Orchestrate | Generate | BYO |
|---|---|---|---|
| **What you set** | `enabled` + `seeds`/`n_permutations` | *(created automatically, one per seed)* | `enabled` + `permuted_metadata_dir` |
| **Who makes the sheets** | — *(delegates to Generate)* | AlleleFlux, from the seed | **you**, beforehand |
| **Runs the analysis?** | no — spawns children | yes | yes |
| **Reproducible from a seed?** | yes | yes | up to you |

:::{tip}
**Which do I pick?** Want AlleleFlux to do everything? Set `enabled` + `n_permutations`
(orchestrate, which fans out into generate runs for you). Already have sheets from another
tool or a fixed design? Set `permuted_metadata_dir` (BYO). You never configure "generate"
directly — it is the worker the orchestrator creates for each seed.
:::

### How it works

```
real run  ──reuse_from──►  permutation orchestrator
                                │
                                ├─ seed 1 ─► permute labels per pair ─► run tail ─► perm_1/
                                ├─ seed 2 ─► permute labels per pair ─► run tail ─► perm_2/
                                └─ ...                                              ...
```

For each seed, the orchestrator writes a per-seed *leaf* config and runs it. Inside that
leaf run, a built-in `permute_metadata` pipeline step produces one permuted metadata sheet
**per comparison pair** (only that pair's labels are shuffled, the rest of the sheet is
untouched), and the tail runs on those sheets. Because the sheets are produced as a tracked
pipeline step rather than up front, a dry run (`-n`) plans the work without writing any
sheets, and a re-run regenerates a sheet only if its seed or the source metadata changed.
Outputs land at:

```
<output.root_dir>/<output_subdir>/perm_<seed>/<data_type>/...
```

### Minimal config

```yaml
input:
  # ... usual inputs ...
  reuse_from: /path/to/real_run/longitudinal   # REQUIRED for permutation

permutation:
  enabled: true
  unit: subjectID          # the independent unit whose label is swapped as a whole
  block: replicate         # keep swaps balanced within this column (optional)
  n_permutations: 100      # one full tail run per seed
  output_subdir: permuted  # outputs under <root_dir>/permuted/perm_<seed>/
```

The shuffling unit (`unit`) is swapped as a whole — for a longitudinal study, swapping
`subjectID` moves all of that subject's timepoints together. `block` preserves balance
within a cohort/batch. Use `n_permutations: N` for seeds `1..N`, or give an explicit
`seeds: [...]` list (which overrides `n_permutations`).

:::{note}
On permutation runs, within-group (across-time) tests are **suppressed by default** (a
divergence null only needs the between-group comparison). Re-enable them with
`analysis.run_within_group_tests: true`.
:::

---

## Bring your own permuted metadata

If you already have permuted metadata sheets — generated by another tool, a fixed design,
or a reproducibility archive — you can **skip the generation step entirely** and have
AlleleFlux run the tail directly on your files.

Drop your sheets in one folder, named by the convention
`permuted_metadata_<treatment>_<control>.tsv`, and point `permuted_metadata_dir` at it:

```yaml
permutation:
  enabled: true
  permuted_metadata_dir: /path/to/my_sheets
  #   /path/to/my_sheets/permuted_metadata_1D_AL.tsv
  #   /path/to/my_sheets/permuted_metadata_2D_AL.tsv
  #   ...
```

This is the same convention the auto-generation orchestrator uses internally, so it is the
one supported bring-your-own path. Setting `permuted_metadata_dir` tells AlleleFlux "the
sheets are ready" — it skips the `alleleflux-permute-metadata` generation step (the
`unit`/`block`/`swaps`/`seeds` knobs are ignored) and runs the group-dependent tail directly
on your data. `reuse_from` is still required (the whole point is to reuse the real run's
cache).

:::{important}
The lookup is **by exact filename**, not a directory scan. For each pair in
`groups_combinations`, AlleleFlux constructs `permuted_metadata_<treatment>_<control>.tsv`
and reads that file. A pair whose file is **missing** raises a `FileNotFoundError` when its
sheet is loaded — so a forgotten sheet fails loudly rather than silently running unpermuted.
Make sure every pair has a correctly-named sheet.
:::

### Generating sheets manually

To pre-make sheets matching the directory convention, call the CLI per pair:

```bash
alleleflux-permute-metadata \
  --input  /path/to/real_metadata.tsv \
  --output /path/to/my_sheets/permuted_metadata_1D_AL.tsv \
  --unit   subjectID \
  --block  replicate \
  --groups 1D AL \
  --seed   1
```

Only the `1D` and `AL` rows are shuffled (among themselves); every other group is left
untouched. Repeat per comparison pair.

---

## Config key reference

| Key | Where | Description |
|-----|-------|-------------|
| `input.profiles_path` | optional | Reuse an existing `profiles/` dir (pileup only). |
| `input.reuse_from` | optional | Reuse a prior run's profiles + QC + allele_freq_cache. Required when `permutation.enabled`. |
| `permutation.enabled` | `false` | Turn the run into a null/permutation orchestration. |
| `permutation.unit` | `subjectID` | The independent unit whose `group` label is swapped as a whole. |
| `permutation.block` | *(none)* | Column to keep swaps balanced within (e.g. cohort/replicate). |
| `permutation.swaps` | *(half)* | Number of label swaps per permutation; default is half the eligible units. |
| `permutation.n_permutations` | `1` | Number of seeds (`1..N`), one full tail run each. |
| `permutation.seeds` | *(none)* | Explicit seed list; overrides `n_permutations`. |
| `permutation.output_subdir` | `permuted` | Subdir under `output.root_dir` for permuted outputs. |
| `permutation.permuted_metadata_dir` | *(none)* | Bring-your-own: folder of `permuted_metadata_<treatment>_<control>.tsv` sheets (skips generation). |
