# End-to-end regression test

This directory hosts the AlleleFlux pipeline regression test — the test that
catches changes to pipeline behaviour by running the bundled example dataset
end-to-end and comparing its outputs to a committed golden snapshot. The golden
is captured from **`main`** (the trusted baseline), and the bundled scenarios
scope the comparison to the **statistical outputs** within a float32-aware
tolerance — see [What the e2e checks](#what-the-e2e-checks-scope--tolerance).

It complements (does not replace) the unit tests in `tests/<module>/`. Unit
tests catch broken functions; this regression test catches broken
**outputs** — including silent numerical drift, dropped CLI flags, and
mis-wired Snakemake rules.

The test runs in two flavours — once for the **longitudinal** data type and
once for the **single** data type — so both code paths get coverage.

## Commands & what to expect

> **Run the e2e through the conda env, not a bare python path.** The test
> spawns `alleleflux` as a *subprocess*, resolved from `PATH`. If you launch
> pytest via `/…/envs/<env>/bin/python -m pytest`, the env's `bin/` is **not**
> on `PATH` and the subprocess grabs `~/.local/bin/alleleflux` (system Python,
> often missing deps like `questionary`) → the pipeline crashes before it
> runs. Use `conda run -n <env> --no-capture-output python -m pytest …` or an
> activated env so `alleleflux` resolves to the env's copy.

| Command | What it does | Expect |
|---------|--------------|--------|
| `pytest tests/ -m "not regression"` | Fast unit loop (module tests + 18 comparator tests), no pipeline | All pass, seconds |
| `pytest tests/regression/test_compare.py -v` | The 18 comparator unit tests only | `18 passed`, ~3 s |
| `conda run -n alleleflux-dev --no-capture-output python -m pytest tests/regression/ -v` | Longitudinal pipeline + **scoped, float32-tolerant** diff vs the main-derived golden | `PASS` if the branch's statistical outputs match `main` within float32 tolerance; ~60 s |
| `… --regression-scenario single` | Same, for the single-timepoint data type | `PASS`/`FAIL` per scenario |
| `… --regression-scenario all` | Both scenarios in sequence | both `PASS` |
| `pytest tests/regression/ --update-golden [--regression-scenario all]` | Re-snapshot the **current branch** as the golden, then **skip** the compare | `SKIPPED` with "Golden snapshot refreshed …" |
| `tests/regression/diff_branches.sh --capture-golden` | Re-snapshot **`main`** as the golden (drift correction) | "Copied tracked files into …" |
| `tests/regression/diff_branches.sh main HEAD [scenario]` | Live `main`-vs-`HEAD` diff, no golden mutation | empty punch-list = output-neutral |

A `FAIL` from the scoped e2e is a *real* signal: the branch's **statistical**
results diverge from `main` by more than float32 noise. The assertion message
names each differing file and the first ten mismatched rows (`__actual` vs
`__golden`).

## What the e2e checks: scope & tolerance

A feature branch legitimately changes pipeline *structure* — dropped columns,
re-formatted or renamed files, disabled optional outputs — while leaving the
*science* unchanged. To verify equivalence to `main` without drowning in that
intended structural drift, the bundled scenarios scope the diff in
`conftest.py`'s `SCENARIOS` (via `compare_only` / `allow_golden_only` /
`exclude_substrings`) and relax the float tolerance (`rtol` / `atol`).  The
shared values live in `_scope.py` so the live cross-branch tool
(`diff_branches.sh` / `_diff_outputs.py`) applies the identical contract.

**Compared strictly** (`compare_only`) — the structurally-stable statistical
outputs, which are also the integration point where any upstream numerical
change surfaces:

| Path prefix | Why it's checked |
|-------------|------------------|
| `significance_tests/` | raw per-site statistical test outputs |
| `p_value_summary/` | summarised p / q-values |
| `scores/intermediate/` | per-MAG scores (identical 1:1 across branches) |
| `scores/processed/combined/` | taxonomic aggregation (shared taxa levels) |
| `preprocessing_eligibility/`, `eligibility_table` | QC / filter decisions |

**Ignored** (intended divergence — neither compared nor flagged MISSING):
`profiles/` (the `mapq_scores` column was dropped), `allele_analysis/` (now
parquet, not tsv.gz), `QC/` & `inputMetadata/` (renamed group-independent),
`outlier_genes/` (disabled by config), `scores/processed/gene_scores_*`, and
any taxonomic levels the branch's `taxa_score_levels` config omits (handled by
`allow_golden_only`, so main's extra levels are not phantom-MISSING).

**Carved out per-scenario** (`exclude_substrings` — substring match anywhere in
the path, applied *after* `compare_only`, so it can drop subtrees that live
*inside* an included prefix):

| Scenario | Excluded | Why |
|----------|----------|-----|
| `single` | `two_sample_paired` | Deterministic but float32-sensitive (verified: same-branch reruns are bit-identical; paired-test code is identical across branches). single's paired test runs Wilcoxon on **raw** frequencies (longitudinal uses `diff_means`); the branch's float32 values differ ~1e-7 from the float64 golden, flipping a near-tie **rank** in the discrete small-n Wilcoxon → a discrete p-jump (e.g. `0.375↔0.625`) the float tolerance can't absorb, cascading into q-values. Scientifically null. |
| `single` | `preprocessed` | The `*_allele_frequency_preprocessed` intermediates (under `significance_tests/preprocessed_*/`) carry metadata columns the branch's refactor prunes → column-set mismatch. Intermediate inputs, not statistical outputs. |
| `longitudinal` | *(none)* | Its paired test (pre-vs-post) is deterministic and its preprocessed files match — both stay in scope for full coverage. |

> Note: `significance_tests/` is a *broad* prefix — it contains the test
> outputs (`single_sample_*`, `two_sample_*`) **and** the `preprocessed_*`
> intermediate inputs.  `compare_only` sweeps the whole subtree; the
> `exclude_substrings` carve-outs are how single drops the parts that aren't
> comparable.

**Float tolerance** — `rtol=1e-5, atol=1e-3` (vs the comparator's strict
`1e-9` / `1e-12` default). The branch stores allele frequencies as **float32**
(~1e-7 precision — a deliberate memory optimization in `_allele_freq_common.py`),
so the float64 golden can never match to 1e-9. These bounds sit ~4 orders above
the observed benign drift (max ~1.3e-4 on a q-value) yet still catch any
scientifically meaningful change (a p-value moving 0.01, a frequency moving
0.001).

> **Comparing every file at full strictness instead.** The scope and tolerance
> live in the scenario bundle, not in `compare_trees` itself — call
> `compare_trees(actual, golden)` with no `include_prefixes` / `rtol` / `atol`
> to compare the whole tree at the strict `1e-9` default. Useful when the
> golden is captured from the *same* branch (e.g. via `--update-golden`) so no
> structural or precision drift is expected.

## Files

| File | Purpose |
|------|---------|
| `test_pipeline_e2e.py` | The e2e test. Runs `alleleflux run` and diffs outputs against the committed golden. Parametrized over `longitudinal` and `single` scenarios. |
| `test_permutation_equivalence.py` | Proves reuse+relabel == full rebuild for permuted (null) runs. No committed golden — computes both sides live (~3 pipeline runs/scenario). Shares the same `scenario` fixture, so it's parametrized over `--regression-scenario` too (default `longitudinal` locally; CI runs `all`). |
| `compare.py`           | File comparator (TSV / TSV.gz / parquet / JSON), tolerant of row order and tiny float drift. |
| `_diff_outputs.py`     | Thin CLI wrapper around `compare.compare_trees`, used by `diff_branches.sh`. |
| `test_compare.py`      | Eighteen unit tests pinning the comparator's contract — row/column-order tolerance, sign-folding, the non-deterministic-diagnostic-column skip (`*_warnings` / `*_converged_LMM`), the scope filters (`include_prefixes` / `allow_golden_only`), and tolerance threading (rapid feedback if `compare.py` ever drifts). |
| `conftest.py`          | Pytest options (`--update-golden`, `--regression-scenario`, `--regression-config`). |
| `diff_branches.sh`     | Bash tool for branch-vs-branch comparison and main-derived golden capture. See "Which command for which task" below. |
| `golden/`              | Committed snapshot of expected outputs. Subdirectories per scenario: `golden/longitudinal/` and `golden/single/`. |

### Golden provenance notes

The golden payload is a *snapshot*, so an intentional change to an output's
**name** breaks the test even when the **values** are unchanged — the
comparator reports the golden-side path as `MISSING` and never gets to compare
contents. Record such renames here.

* **`p_value_summary_*` filenames (`178ed5d`, 2026-06-18).** `p_value_summary`
  outputs gained the group pair in the filename
  (`p_value_summary_{test}_{tp}.tsv` → `p_value_summary_{test}_{tp}-{groups}.tsv`)
  as part of the group-pair conflation fix. The golden dated from `4b6b6ab`
  and still carried the old names, so the e2e failed with three `MISSING`
  entries for `longitudinal`.

  The golden files were **renamed in place, not regenerated**. That is the
  correct treatment here because the conflation bug only triggered with more
  than one group pair sharing a timepoint, and both example datasets define
  exactly one (`treatment`/`control`) — so the recorded values were never
  affected by the bug. Verified after renaming: same schema, same row count,
  and `min_p_value` / `q_value` agree to ≤1.3e-4, inside the scenario
  tolerance (`STAT_RTOL=1e-5`, `STAT_ATOL=1e-3`) that already absorbs the
  float32-vs-float64 drift described above.

  Had the values genuinely changed, the right move would have been
  `--update-golden` (or `diff_branches.sh --capture-golden`), not a rename.

## Which command for which task

The regression machinery has four common use cases.  Each maps to exactly one
command, with no overlapping intent:

| Want to ... | Run | Approximate runtime |
|-------------|-----|---------------------|
| Daily check: "did my current branch drift from the committed golden?" | `pytest tests/regression/ -v` | ~50 s per scenario |
| Snapshot the **current branch's** outputs as the new golden | `pytest tests/regression/ --update-golden` | ~50 s per scenario |
| Snapshot **main's** outputs as the new golden (drift correction) | `tests/regression/diff_branches.sh --capture-golden` | ~3 min per scenario |
| Ad-hoc "does my branch match main right now?" (no snapshot mutation) | `tests/regression/diff_branches.sh main HEAD` | ~5 min per scenario |

`pytest --update-golden` always snapshots from the branch you're on — useful
when you've intentionally changed outputs and want the new shape to become the
baseline.  When you want to *reset* the golden to match what main produces
(e.g. after merging main into your branch), use `diff_branches.sh
--capture-golden` instead.  The pytest test never touches main.

## Live vs snapshot: `pytest` and `diff_branches.sh` are not interchangeable

`pytest tests/regression/ --regression-scenario <X>` and
`tests/regression/diff_branches.sh main HEAD <X>` look like they answer the
same question.  They don't.  Both use the same comparator under the hood
(`compare.py`); the difference is **what they compare**.

| Axis | `diff_branches.sh main HEAD <scenario>` | `pytest --regression-scenario <scenario>` |
|------|------------------------------------------|--------------------------------------------|
| Baseline source | **Built live** — worktrees `main`, runs the pipeline | **Frozen snapshot** — reads `tests/regression/golden/` as-is |
| Candidate source | **Built live** from the **committed** HEAD | **Built live** from your **working tree** |
| Sees uncommitted changes? | ❌ Committed HEAD only | ✅ Yes — working tree is what runs |
| Pipeline runs | Two (baseline + candidate) | One (candidate only) |
| Runtime | ~5 min per scenario | ~50 s per scenario |
| Conda envs used | Dedicated `alleleflux-main` + `alleleflux-head` | Whichever env is currently active |
| Mutates committed state | Nothing | Nothing (unless `--update-golden`) |
| Output on mismatch | Punch-list to stdout | Pytest FAIL with same punch-list |

### When they agree, and when they don't

After this exact sequence — regenerate data → `--capture-golden` → commit
your code → run pytest — both commands should report the **same set of
differing files**, because both effectively ask "does HEAD match main on the
current data?"

They disagree in three predictable situations:

| Situation | `diff_branches.sh main HEAD` sees | `pytest` sees |
|-----------|-----------------------------------|---------------|
| Uncommitted changes in your working tree | The state at the committed HEAD | The state with your uncommitted changes layered in |
| Golden was captured a while ago, main has since moved | Whatever main is **right now** | Whatever main was **at capture time** |
| Example data on disk has changed since golden capture | Whatever data is on disk now (overlaid into both worktrees) | Whatever data is on disk now (vs. golden that reflects different inputs) |

### Mental model

- **`diff_branches.sh`** is a *direct, ephemeral* comparison.  Computes both
  sides fresh every time, so it always reflects "main as it is right now vs
  HEAD as it is right now."  Slower but self-consistent — no snapshot to
  go stale.
- **`pytest`** is a *snapshot-based* comparison.  The baseline is **frozen**,
  so its accuracy depends entirely on how recent the golden is.  Faster and
  CI-friendly — no git acrobatics required.

That's why the from-scratch recipe captures golden from main right before
running pytest: it brings the snapshot up to date so pytest's frozen
baseline momentarily matches what `diff_branches.sh main HEAD` would
compute live.  The two start drifting apart again as soon as either main
moves or the working tree changes.

### Pick by what you want to know

| Question | Use |
|----------|-----|
| "Does my branch's current uncommitted code match what main produces?" | `pytest` (after a fresh `--capture-golden`) |
| "Did my last commit drift from main behaviorally?" | `diff_branches.sh main HEAD` |
| "Is CI still green against the committed expectations?" | `pytest` (cheap, fast, golden-driven) |
| "I just merged main into my branch — am I clean against the new main?" | `diff_branches.sh --capture-golden` then `pytest` |

### Why the two golden-capture flags both exist

`pytest --update-golden` and `diff_branches.sh --capture-golden` both write
files into `tests/regression/golden/`, but they answer different questions
about whose code produced the bytes.

| Flag | Lives in | Snapshots from | When to use |
|------|----------|----------------|-------------|
| `--update-golden` | pytest (conftest.py) | **Current branch's** outputs | You made intentional changes — your branch IS the new truth |
| `--capture-golden` | diff_branches.sh | **`main`'s** outputs (built live) | You want golden to reflect `main` — for drift correction or baselining a new branch |

`test_pipeline_e2e.py` only **mentions** `--capture-golden` in its docstring
as a cross-reference pointer — the flag itself is implemented in
`diff_branches.sh`.  Two distinct mechanisms for two distinct intents; no
duplication.

## Full end-to-end validation (from scratch) — ≈ 15 min

When you want to confirm the **entire** testing infrastructure is healthy —
generator → BAM script → pipeline → comparator → golden snapshot — run this
recipe.  It walks the full loop for both scenarios.

**What this catches that `pytest tests/regression/` alone does not:**

* Drift in `generate_synthetic_data.py` (gene placement, allele model, config emission).
* Drift in `generate_bams.sh` (wgsim flags, sample discovery from metadata).
* Stale committed bundled data that no longer matches what the generator emits.
* Mismatch between the generator's configs and what the pipeline currently expects.

**Prerequisites:** conda env loaded, plus `bwa` and `wgsim` on PATH (install
once via `conda install -c bioconda bwa wgsim` in the `alleleflux` env if
absent).

### 0. Activate the env and confirm tooling

```bash
module load anaconda3/2025.12
conda activate alleleflux
for cmd in alleleflux bwa wgsim samtools python; do command -v "$cmd" || echo "MISSING: $cmd"; done
```

All five must resolve.

**One-time bootstrap (only the first time you run this recipe):**
`diff_branches.sh` runs the baseline and candidate refs inside two
dedicated conda envs so it never touches your daily `alleleflux` or
`alleleflux-dev` envs (which may have live SLURM jobs).  Create the
dedicated envs once by cloning `alleleflux`:

```bash
conda create --name alleleflux-main --clone alleleflux
conda create --name alleleflux-head --clone alleleflux
```

You can confirm both exist with `conda env list | grep alleleflux-`.
Re-cloning is only needed if a system-level dep (samtools, bwa, …)
changes in `alleleflux`; the script handles per-ref Python code via
`pip install -e .` automatically.

### 1. Regenerate the LONGITUDINAL dataset from scratch

```bash
cd docs/source/examples

# Wipe generator-owned artifacts; keep the hand-curated _main configs.
rm -rf example_data_longitudinal/reference \
       example_data_longitudinal/metadata \
       example_data_longitudinal/profiles \
       example_data_longitudinal/bams \
       example_data_longitudinal/significant_sites \
       example_data_longitudinal/output \
       example_data_longitudinal/output_bams \
       example_data_longitudinal/.snakemake \
       example_data_longitudinal/config_generated_*.yml \
       example_data_longitudinal/config_with_bams_longitudinal.yml \
       example_data_longitudinal/README.md

python generate_synthetic_data.py \
    --output_dir example_data_longitudinal \
    --data_type longitudinal --seed 42
```

Sanity checks:

```bash
wc -l example_data_longitudinal/metadata/sample_metadata.tsv   # 9 lines (1 header + 8 samples)
ls example_data_longitudinal/profiles/ | wc -l                  # 8 (one dir per sample)
ls example_data_longitudinal/config_*.yml                       # 4 configs
#   config_example_main.yml    (kept — hand-curated)
#   config_generated_longitudinal.yml
#   config_with_bams_longitudinal.yml
#   config_with_bams_main.yml  (kept — hand-curated)
```

### 2. Generate the BAMs for longitudinal

```bash
bash generate_bams.sh --data-dir example_data_longitudinal
ls example_data_longitudinal/bams/*.bam | wc -l   # 8
ls example_data_longitudinal/bams/*.bai | wc -l   # 8
```

### 3. Capture the golden snapshot **from `main`** (the trusted baseline)

The "truth" the regression test compares against should come from the
trusted ref (`main`), not from the branch you're validating.  Capturing
the golden from main means a passing test on your branch actually proves
"my branch matches main's outputs", not just "my branch matches itself".

```bash
cd ../../..   # back to repo root
tests/regression/diff_branches.sh --capture-golden          # scenario = longitudinal (default)
```

Under the hood this worktrees `main`, overlays the current repo's freshly
regenerated `example_data_longitudinal/` into the worktree (so both this
step and step 4 see bit-identical inputs), runs the pipeline against the
`_main` config, and copies the tracked outputs into
`tests/regression/golden/longitudinal/`.

Expect: `Copying baseline outputs into .../golden/longitudinal ...`
followed by `Copied tracked files into ...`.

> Don't `git add` the regenerated golden yet.  This step is part of the
> from-scratch *validation*; whether to commit the snapshot is a separate
> decision (covered in the tutorial below under "Refresh the golden after
> an intentional change").

### 4. Run the regression test against the main-derived golden

```bash
pytest tests/regression/ --regression-scenario longitudinal -v
```

Expected outcomes:

| Result | What it means |
|--------|---------------|
| PASSED ✅ | Your branch produces the same outputs as `main` on the fresh data. Infrastructure healthy, no behavioural drift on this scenario. |
| FAILED ❌ | Your branch's outputs differ from `main`'s on identical data — either an intended change you'll commit later, or an unintended regression worth chasing. The failure message lists the differing files. |

### 5. Repeat steps 1–4 for the SINGLE scenario

```bash
cd docs/source/examples

rm -rf example_data_single/reference \
       example_data_single/metadata \
       example_data_single/profiles \
       example_data_single/bams \
       example_data_single/significant_sites \
       example_data_single/output \
       example_data_single/output_bams \
       example_data_single/.snakemake \
       example_data_single/config_generated_*.yml \
       example_data_single/config_with_bams_single.yml \
       example_data_single/README.md

python generate_synthetic_data.py \
    --output_dir example_data_single \
    --data_type single --seed 42

bash generate_bams.sh --data-dir example_data_single

cd ../../..
# Capture golden from main for the single scenario.
# `--capture-golden` ignores CANDIDATE_REF, but the script's positional API
# still requires it as a placeholder before the SCENARIO arg — pass HEAD.
tests/regression/diff_branches.sh --capture-golden main HEAD single

# Run the test against the main-derived golden.
pytest tests/regression/ --regression-scenario single -v
```

### 6. Canary — both scenarios in one shot

```bash
pytest tests/regression/ --regression-scenario all -v
```

Both PASSED → infrastructure is healthy end-to-end. ✅

### 7. (Optional) main-vs-HEAD diff without touching the golden

Steps 3–4 already validate "HEAD matches main" via the captured golden.
If you'd rather see the diff *directly* — without overwriting the
committed snapshot in `tests/regression/golden/` — run:

```bash
tests/regression/diff_branches.sh main HEAD longitudinal
tests/regression/diff_branches.sh main HEAD single
```

This worktrees both refs, overlays the current branch's example data
into both worktrees (so neither side benefits from data drift), runs the
pipeline twice, and prints a punch-list of differing files.  An empty
punch-list means the refactor is output-neutral.  Same comparison as
steps 3–4 give you, just without the golden round-trip.

> The single-scenario baseline uses `config_with_bams_main.yml` with the
> pre-refactor list-of-list `groups_combinations` format that `main` still
> reads.  If you change the generator's emitted config in a way that
> affects pipeline outputs, update both `_main` configs in parallel so the
> cross-branch comparison stays meaningful — see "Maintaining the golden"
> below.

## Tutorial — your first ten minutes

This walkthrough takes you from a fresh checkout to confidently running and
updating the regression test. About 10 minutes start to finish.

### 1. Set up the environment (once)

The test invokes the `alleleflux` CLI via subprocess, so the conda
environment must be loaded:

```bash
module load anaconda3/2025.12
conda activate alleleflux
```

Confirm the CLI is available:

```bash
which alleleflux
# /home/<user>/.conda/envs/alleleflux/bin/alleleflux
```

### 2. Run the fast unit-test loop (sanity check)

This excludes the slow regression test and runs in a few seconds:

```bash
pytest tests/ -m "not regression" -v
```

You should see all unit tests pass (292 module unit tests + 18 comparator
tests). If any fail here, fix them before touching the regression test — the
comparator's own unit tests must be green first.

### 3. Run the regression test (the real thing)

Default runs the `longitudinal` scenario:

```bash
pytest tests/regression/ -v
```

What happens under the hood:

1. The test wipes any leftover `output_bams/` and `.snakemake/` from the
   committed `docs/source/examples/example_data_<scenario>/` directory.
2. It invokes `alleleflux run --config <bundled config>` from that directory.
   The generated configs use absolute paths, so outputs land at
   `example_data_<scenario>/output_bams/<data_type>/` regardless of cwd.
3. Once the pipeline completes, the test walks the committed golden snapshot
   at `tests/regression/golden/<scenario>/` and compares every file to
   the produced output.
4. On match: PASS in green. On mismatch: FAIL with the file path and the
   first ten mismatched rows.

Approximate runtime: ~50 seconds on 4 cores per scenario.

> **Note:** the test runs the pipeline **in place** inside the committed
> example directory. `output_bams/`, `output/`, and `config_runtime_*.yml`
> inside `example_data_*/` are gitignored so they don't appear in
> `git status` after a test run. The test wipes them before each run, so
> consecutive invocations always start from a clean slate.

### 4. Run a specific scenario or both

```bash
# Just the longitudinal data type (default)
pytest tests/regression/ -v

# Just the single data type
pytest tests/regression/ --regression-scenario single -v

# Both scenarios in sequence
pytest tests/regression/ --regression-scenario all -v
```

### 5. Simulate a real workflow — break something on purpose

Make a deliberate change that affects pipeline outputs. For example, edit the
`p_value_threshold` in
`docs/source/examples/example_data_longitudinal/config_with_bams_longitudinal.yml`:

```yaml
statistics:
  p_value_threshold: 0.10   # was 0.05
```

Re-run the regression test:

```bash
pytest tests/regression/ -v
```

It should now FAIL — the assertion message will name the score files and
outlier files that differ, with sample diffs. **This is the mechanism that
catches LLM-introduced regressions in real life.**

Undo the change (`git checkout -- docs/`) and re-run; the test should go back
to passing.

### 6. Refresh the golden after an intentional change

When you make a change that *should* alter outputs (a real bug fix, a new
feature), refresh the snapshot:

```bash
# Refresh the longitudinal golden
pytest tests/regression/ --update-golden -v

# Refresh the single golden
pytest tests/regression/ --update-golden --regression-scenario single -v

# Refresh both at once
pytest tests/regression/ --update-golden --regression-scenario all -v
```

This re-runs the pipeline, copies the new outputs into the golden tree, and
skips the comparison. Inspect what changed:

```bash
git diff --stat tests/regression/golden/
git diff tests/regression/golden/<file-of-interest>.tsv
```

If the diff is what you expected → commit it alongside the code change. If
something unexpected changed → investigate before committing.

### 7. Regenerate the example dataset itself

When the inputs (synthetic reference, profiles, BAMs) need to change — or
after pulling a branch that updated `generate_synthetic_data.py` or
`generate_bams.sh` — re-run both generators **for each data type**.

The exact commands that produced the currently-committed bundled datasets
(reproducible byte-for-byte under `--seed 42`):

```bash
# Load env once
module load anaconda3/2025.12
conda activate alleleflux

cd docs/source/examples

# --- Longitudinal dataset → example_data_longitudinal/ -----------------
python generate_synthetic_data.py \
    --output_dir example_data_longitudinal \
    --data_type longitudinal \
    --seed 42
bash generate_bams.sh --data-dir example_data_longitudinal

# --- Single-timepoint dataset → example_data_single/ -------------------
python generate_synthetic_data.py \
    --output_dir example_data_single \
    --data_type single \
    --seed 42
bash generate_bams.sh --data-dir example_data_single
```

Each generator pass writes, into its output dir:

| Subdir / file | Origin | Content |
|---------------|--------|---------|
| `reference/combined_mags.fasta` | Python generator | 2 MAGs × 2 contigs each |
| `reference/prodigal_genes.fna`  | Python generator | Gene annotations in Prodigal header format |
| `reference/mag_mapping.tsv`     | Python generator | contig → MAG_ID mapping |
| `reference/gtdbtk_taxonomy.tsv` | Python generator | Mock GTDB taxonomy |
| `metadata/sample_metadata.tsv`      | Python generator | 8-row sheet, paths point at `profiles/` |
| `metadata/sample_metadata_bams.tsv` | Python generator | 8-row sheet, paths point at `bams/` |
| `profiles/{sample}/*.tsv.gz`    | Python generator | One gzipped per-sample profile per MAG |
| `significant_sites/`            | Python generator | Mock significant sites for viz tests |
| `config_generated_<data_type>.yml`  | Python generator | Profiles-mode config |
| `config_with_bams_<data_type>.yml`  | Python generator | BAM-mode config |
| `README.md`                      | Python generator | Minimal pointer file |
| `bams/`                         | `generate_bams.sh` | 8 sorted+indexed BAMs |

The bundled configs **ARE the generator's output** — there is no separate
hand-curated layer. To change the defaults, edit `generate_synthetic_data.py`'s
`write_config()` function and re-run the generators.

Customisation knobs (see `python generate_synthetic_data.py --help`):

* `--seed`           — change the RNG; everything downstream shifts byte-wise
* `--num_mags`       — default 2
* `--num_samples`    — default 8 (must be divisible by groups × timepoints)
* `--coverage_min / --coverage_max` — per-position coverage range, default 20–80
* `--genes_per_contig` — default 3
* `--contig_length_min / --contig_length_max` — default 800–2000 bp

After regeneration, refresh the goldens:

```bash
cd ../../..   # back to repo root
pytest tests/regression/ --update-golden --regression-scenario all -v
```

`generate_bams.sh` discovers samples from `metadata/sample_metadata.tsv` (not
a hardcoded list), so it works for any dataset the Python generator produces
— longitudinal, single, or a custom seed/size combo.

### 8. Compare two git branches without committing

For one-shot "does this branch break anything?" checks:

```bash
tests/regression/diff_branches.sh main HEAD
```

This spins up worktrees, runs the pipeline on both refs, and prints a
punch-list of files that differ. No golden mutation, no commit. Useful during
PR review.

### 9. Reading a failure message

When `pytest tests/regression/` fails, you get something like:

```
FAILED tests/regression/test_pipeline_e2e.py::test_pipeline_matches_golden[longitudinal]
   3 file(s) differ from the golden snapshot:

   scores/processed/combined/MAG/scores_two_sample_paired-pre_post-treatment_control-MAGs.tsv: 7 mismatched rows
       MAG_ID  score_two_sample_paired_tTest__actual ... score_two_sample_paired_tTest__golden ...
       ...

   significance_tests/two_sample_paired_pre_post-treatment_control/MAG_001_two_sample_paired.tsv.gz: ...
```

The `__actual` / `__golden` suffix tells you which side has what. The file
path is relative to the pipeline's output root, so it's directly greppable in
the working directory.

## Reference — running commands

```bash
# Fast unit tests only (default workflow)
pytest tests/ -m "not regression"

# Just the regression test (longitudinal by default)
pytest tests/regression/ -v

# Select a scenario
pytest tests/regression/ --regression-scenario {longitudinal|single|all}

# Override the config path (rare — useful for ad-hoc comparisons)
pytest tests/regression/ --regression-config /path/to/some_config.yml

# Refresh the golden snapshot
pytest tests/regression/ --update-golden [--regression-scenario all]
```

The test depends on the `alleleflux` CLI being importable in the active conda
environment. Load the env first:

```bash
module load anaconda3/2025.12
conda activate alleleflux
```

## Scenarios

Each scenario is a (data type, dataset directory, config, golden subdir)
bundle, declared in `conftest.py`:

| Scenario | Data type | Dataset dir | Config | Golden subdir |
|----------|-----------|-------------|--------|---------------|
| `longitudinal` | longitudinal | `docs/source/examples/example_data_longitudinal` | `config_with_bams_longitudinal.yml` | `golden/longitudinal` |
| `single`       | single       | `docs/source/examples/example_data_single` | `config_with_bams_single.yml` | `golden/single` |

Both scenarios go through the full BAM-based pipeline (profile → allele_freq
cache → analysis → preprocessing → significance tests → scoring), so each
exercises the maximum amount of pipeline code.

## Maintaining the golden

* When the example dataset itself changes (new MAGs, new samples, new
  reference) you MUST regenerate the snapshot — the contract is "these inputs
  produce these outputs."

* Keep the snapshot small. It is checked into git; the current per-scenario
  footprint is on the order of tens to a hundred KB. If you add larger
  fixtures, switch to git-LFS.

* The `_main` config variants (`config_example_main.yml`,
  `config_with_bams_main.yml`) live in **both** `example_data_longitudinal/`
  and `example_data_single/` as frozen, hand-curated artifacts for the
  pre-refactor `groups_combinations` list-of-list format that the `main`
  branch still reads.  They are **not** regenerated by
  `generate_synthetic_data.py`.  They are **not** read by the pytest test,
  but `diff_branches.sh` uses `config_with_bams_main.yml` as the
  baseline-side config when comparing main → HEAD.  Whenever the
  generator's emitted configs change in a way that affects pipeline
  outputs, update both `_main` files in parallel so the cross-branch
  comparison stays apples-to-apples.

## What the comparator tolerates

| Difference                       | Treated as |
|----------------------------------|------------|
| Row order                        | OK (sorted on stable identity keys, incl. `period` / `test_type` / `group_analyzed` / `source_file`) |
| Column order                     | OK (set-level check; columns reordered before comparison) |
| Float drift within `rtol` / `atol` | OK (default `1e-9` / `1e-12`; the bundled scenarios loosen to `1e-5` / `1e-3` — see below) |
| `NaN` vs `NaN`                   | OK |
| Sign flip on allele-frequency diff columns (`*_change`, `*_diff_mean`, `*_diff`) | OK — compared on absolute value, see below |
| Non-deterministic LMM diagnostic columns (`*_warnings`, `*_converged_LMM`) | OK — *value* skipped (column still required present); statsmodels convergence noise, see below |
| Files outside `compare_only` scope | Ignored (not compared, not flagged) — see [scope](#what-the-e2e-checks-scope--tolerance) |
| Golden-only files under `allow_golden_only` | Ignored (intended absence, e.g. omitted taxa levels) |
| Missing / extra columns (in-scope) | **FAIL** |
| Different row count (in-scope)   | **FAIL** |
| Any non-float value mismatch     | **FAIL** |
| Float drift above tolerance      | **FAIL** |

Two tolerance regimes apply depending on whose golden you're diffing against:

* **Strict default (`rtol=1e-9, atol=1e-12`)** — the comparator's built-in
  bound. Right when the golden and the run under test share float precision
  (e.g. a same-branch `--update-golden` snapshot). Looser bounds would hide
  subtle regressions.
* **float32-aware (`rtol=1e-5, atol=1e-3`)** — what the bundled scenarios pass
  in, because the branch stores frequencies as float32 against a float64
  `main`-derived golden. Still ~4 orders tighter than any scientifically
  meaningful change. Set in `conftest.py`'s `SCENARIOS`; threaded through
  `compare_trees → compare_file → compare_tables`.

### Why sign-insensitive on allele-frequency diffs

The pipeline currently has an upstream non-determinism: the sign of pre-vs-post
allele-frequency differences depends on sample iteration order inside the
multiprocessing pool, so running the same code on the same inputs can produce
`A=+0.5, C=-0.5` on one run and `A=-0.5, C=+0.5` on the next. Until that root
cause is fixed, the comparator compares the absolute value of these columns.

If you fix the determinism, empty `SIGN_INSENSITIVE_SUBSTRINGS` in `_scope.py`
(where the column-level comparison rules now live, alongside the scope and
tolerance contract) so the comparator is strict again.

### Why skip the LMM diagnostic columns

When LMM is enabled, the per-fit diagnostic columns (`*_warnings`,
`*_n_warnings`, `*_converged_LMM`) are **not reproducible** run-to-run: the
statsmodels optimizer emits a run-dependent number of convergence warnings — and
can flip the `converged` boolean — on near-degenerate positions (e.g. p≈1.0
perfect ties), while the *scientific* outputs (p-value, coefficient, t-value) on
those same rows are identical. The comparator therefore skips these columns'
values (they are still required to be *present* — a dropped column still fails
the schema gate). The list lives in `_scope.py` as `IGNORE_VALUE_SUBSTRINGS`.

## When the test is the wrong tool

* Establishing **whether new numbers are scientifically correct** — only
  *you* can do that. The snapshot remembers correctness; it doesn't establish
  it.
* Catching bugs that only appear on large real datasets (e.g. DRIDO-scale
  sample sizes). For that, build a third scenario from a 1-MAG DRIDO subset
  and wire it through `conftest.py`'s `SCENARIOS` dict.
