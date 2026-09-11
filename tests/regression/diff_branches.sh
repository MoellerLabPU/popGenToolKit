#!/usr/bin/env bash
#
# diff_branches.sh — run the AlleleFlux example pipeline on two git refs and
# compare their outputs.
#
# The motivating workflow: confirm that the development branch (typically
# `HEAD`/`gene-scores`) produces the same scientific outputs as the trusted
# baseline (typically `main`), on the SAME input data.  Architectural
# refactors that don't change the numbers should pass; bug fixes or
# behaviour changes show up as a punch-list of differing files.
#
# Relationship to test_pipeline_e2e.py
# -------------------------------------
# This script and `tests/regression/test_pipeline_e2e.py` are siblings, not
# duplicates.  They share the comparator (`compare.py`) and the golden-
# payload helper (`_copy_golden_payload`), but they answer different
# questions about pipeline behaviour.
#
# * test_pipeline_e2e.py — fast, snapshot-based check.  Runs the pipeline
#   on the working tree (uncommitted changes ARE visible) ONCE, then diffs
#   against the frozen golden at `tests/regression/golden/<scenario>/`.
#   ~50 s per scenario.  CI-friendly; the daily inner-loop tool.
#
# * This script — slow, live cross-branch comparison.  Worktrees two git
#   refs (default `main` vs `HEAD`), installs each ref's code into its own
#   conda env, runs the pipeline TWICE, then diffs the two output trees
#   directly — no golden round-trip.  Only sees committed code
#   (uncommitted changes are INVISIBLE).  ~5 min per scenario.  The
#   canonical "is my refactor output-neutral vs main?" test.
#
# The two golden-capture flags answer different questions and live in
# different files:
#
# * `--update-golden` (pytest, conftest.py) snapshots from the CURRENT
#   branch's outputs.  Use after intentional output changes.
# * `--capture-golden` (this script) snapshots from `main`'s outputs.
#   Use to drift-correct the golden against the trusted baseline.
#
# See tests/regression/README.md § "Live vs snapshot" for the full
# side-by-side comparison and decision guide.
#
# Usage
# -----
#     tests/regression/diff_branches.sh [BASELINE_REF [CANDIDATE_REF [SCENARIO]]]
#
# Defaults:
#   BASELINE_REF   = main
#   CANDIDATE_REF  = HEAD
#   SCENARIO       = longitudinal  (alternative: single)
#
# How it works
# ------------
# 1. Worktree the BASELINE_REF at /tmp/alleleflux-diff-baseline.
# 2. Worktree the CANDIDATE_REF at /tmp/alleleflux-diff-candidate.
# 3. **Overlay the current branch's example_data_<scenario>/ into both
#    worktrees.** This is critical: branches diverge in both code AND data,
#    but a meaningful comparison needs the same inputs on both sides, so we
#    overwrite each worktree's example_data_<scenario>/ with the *current*
#    repo's copy.  Whatever was there before (a different data layout, an
#    older synthetic dataset) gets replaced.
# 4. Install each worktree into its own dedicated conda environment:
#       * baseline  -> `alleleflux-main` env
#       * candidate -> `alleleflux-head` env
#    These envs are SEPARATE from the user's daily `alleleflux` and
#    `alleleflux-dev` envs, so SLURM jobs that use those envs are never
#    disturbed by this script.  Create the envs once before first run:
#
#       conda create --name alleleflux-main --clone alleleflux
#       conda create --name alleleflux-head --clone alleleflux
#
#    The clones give samtools / bwa / pandas / etc. a healthy starting
#    point; the per-ref `pip install -e .` then overlays each ref's code.
# 5. Run the pipeline in each worktree's example_data_<scenario>/ directory:
#       * baseline  uses `config_with_bams_main.yml` (old-format groups syntax)
#       * candidate uses `config_with_bams_<scenario>.yml` (new dict syntax)
#    Both run the BAMs flow (full pipeline including profiling).
# 6. Parse each config's `output.root_dir` to find where it wrote.
# 7. Call `python -m tests.regression._diff_outputs` to diff every file
#    under the data_type subdir and print a punch-list.
# 8. On success, restore both dedicated envs to the candidate ref's code so
#    they're left in a known, consistent state for the next invocation.
#
# Limitations
# -----------
# * The two dedicated envs (`alleleflux-main`, `alleleflux-head`) must exist
#   before the script runs.  See step 4 for the one-time `conda create`
#   commands.
# * `pip install -e .` into the dedicated envs IS destructive to those envs'
#   prior state.  The user's `alleleflux` and `alleleflux-dev` envs are
#   never touched, so SLURM jobs depending on those are safe.
# * The script reinstalls the candidate ref into `alleleflux-head` at the
#   end so that env is left with HEAD's code; if the script crashes mid-way
#   one of the dedicated envs may hold a stale install.  Rerun the script
#   or `conda run -n alleleflux-head pip install -e .` from the repo root.
# * Pass `--keep-worktrees` to preserve /tmp/alleleflux-diff-* for inspection.
#
# Examples
# --------
#     # Default: compare current HEAD to main, longitudinal scenario
#     tests/regression/diff_branches.sh
#
#     # Compare a feature branch to a tagged release
#     tests/regression/diff_branches.sh v0.1.3 feature/cmh-scores
#
#     # Single-timepoint scenario
#     tests/regression/diff_branches.sh main HEAD single
#
#     # Override config names per side (e.g. comparing two future-format branches)
#     BASELINE_CONFIG=config_with_bams_longitudinal.yml \
#         tests/regression/diff_branches.sh dev-branch HEAD longitudinal

set -euo pipefail

# --- Arg parsing -----------------------------------------------------------
KEEP_WORKTREES=0
CAPTURE_GOLDEN=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --keep-worktrees) KEEP_WORKTREES=1 ;;
        --capture-golden)
            # Run ONLY the baseline side and snapshot its outputs into
            # tests/regression/golden/<scenario>/.  No candidate side,
            # no diff.  This is what ``pytest --update-golden`` delegates
            # to so the regression test's contract is "match what the
            # baseline (typically main) produces."
            CAPTURE_GOLDEN=1
            ;;
        --help|-h)
            sed -n '/^# Usage/,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

BASELINE_REF="${POSITIONAL[0]:-main}"
CANDIDATE_REF="${POSITIONAL[1]:-HEAD}"
SCENARIO="${POSITIONAL[2]:-longitudinal}"

case "$SCENARIO" in
    longitudinal)
        DATA_DIR_NAME="example_data_longitudinal"
        PIPELINE_DATA_DIR="longitudinal"
        GOLDEN_SUBDIR="longitudinal"
        BASELINE_CONFIG_DEFAULT="config_with_bams_main.yml"
        CANDIDATE_CONFIG_DEFAULT="config_with_bams_longitudinal.yml"
        ;;
    single)
        DATA_DIR_NAME="example_data_single"
        # The pipeline writes the single data_type under "single_timepoint"
        # (common.smk), not "single" — must match or BASELINE_DATA_OUT points at
        # a nonexistent dir and the golden capture / diff silently sees nothing.
        PIPELINE_DATA_DIR="single_timepoint"
        # Golden dir name is keyed by SCENARIO, decoupled from the pipeline's
        # internal subdir, and must match conftest.SCENARIOS["single"]["golden_subdir"].
        GOLDEN_SUBDIR="single"
        BASELINE_CONFIG_DEFAULT="config_with_bams_main.yml"
        CANDIDATE_CONFIG_DEFAULT="config_with_bams_single.yml"
        ;;
    *)
        echo "ERROR: unknown scenario '$SCENARIO' (expected: longitudinal, single)" >&2
        exit 2
        ;;
esac

BASELINE_CONFIG="${BASELINE_CONFIG:-$BASELINE_CONFIG_DEFAULT}"
CANDIDATE_CONFIG="${CANDIDATE_CONFIG:-$CANDIDATE_CONFIG_DEFAULT}"

BASELINE_ENV="${BASELINE_ENV:-alleleflux-main}"
CANDIDATE_ENV="${CANDIDATE_ENV:-alleleflux-head}"

# --- Safety guard: never target the user's daily envs ----------------------
# Each `pip install -e .` rewrites the TARGET env's editable MAPPING to a /tmp
# worktree (absolute path, last-writer-wins, persistent).  The daily
# `alleleflux` / `alleleflux-dev` envs back live SLURM jobs and must never be
# repointed.  Refuse to run if either side resolves to a protected env — this
# makes "dedicated throwaway envs only" an enforced invariant, not a comment.
PROTECTED_ENVS=("alleleflux" "alleleflux-dev")
for _candidate_env in "$BASELINE_ENV" "$CANDIDATE_ENV"; do
    for _protected in "${PROTECTED_ENVS[@]}"; do
        if [[ "$_candidate_env" == "$_protected" ]]; then
            echo "ERROR: refusing to install into protected env '$_candidate_env'." >&2
            echo "       This script must target dedicated throwaway envs only" >&2
            echo "       (default: alleleflux-main / alleleflux-head). Repoint" >&2
            echo "       BASELINE_ENV / CANDIDATE_ENV to a non-protected env." >&2
            exit 2
        fi
    done
done

REPO_ROOT="$(git rev-parse --show-toplevel)"
TMP_BASELINE="/tmp/alleleflux-diff-baseline"
TMP_CANDIDATE="/tmp/alleleflux-diff-candidate"

# --- Env bootstrap ---------------------------------------------------------
# Make sure conda is on PATH for non-interactive shells.
module load anaconda3/2025.12 2>/dev/null || true

# --- Pre-flight: dedicated envs must already exist -------------------------
# Fail early with an actionable message instead of a cryptic conda error
# mid-run.  Only the envs this invocation will actually install into are
# required (capture-golden mode never touches the candidate env).  Capture
# `conda env list` into a variable first, then grep a here-string — piping
# straight into `grep -q` under `set -o pipefail` risks a SIGPIPE on conda
# that would be misread as "env missing".
assert_env_exists() {
    local env_name="$1"
    local env_list
    env_list="$(conda env list)"
    if ! grep -qE "^${env_name}[[:space:]]" <<<"$env_list"; then
        echo "ERROR: conda env '$env_name' not found. Create it once with:" >&2
        echo "       conda create --name $env_name --clone alleleflux" >&2
        exit 2
    fi
}
assert_env_exists "$BASELINE_ENV"
if [[ "$CAPTURE_GOLDEN" -ne 1 ]]; then
    assert_env_exists "$CANDIDATE_ENV"
fi

# --- Cleanup ---------------------------------------------------------------
cleanup_worktrees() {
    if [[ "$KEEP_WORKTREES" -eq 1 ]]; then
        echo "[cleanup] --keep-worktrees set; leaving $TMP_BASELINE and $TMP_CANDIDATE"
        return 0
    fi
    git -C "$REPO_ROOT" worktree remove -f "$TMP_BASELINE" 2>/dev/null || true
    git -C "$REPO_ROOT" worktree remove -f "$TMP_CANDIDATE" 2>/dev/null || true
}

# Track which dedicated envs we've started mutating, so the exit trap only
# restores the ones we actually touched.  Armed (set to 1) at each install
# site, BEFORE the install runs.
RESTORE_BASELINE=0
RESTORE_CANDIDATE=0

# Single exit handler.  Restores any dedicated env we installed into back to
# $REPO_ROOT FIRST, then removes the worktrees.  Crucially this runs on ANY
# exit — success, error, or Ctrl-C — so an interrupted run can never leave a
# dedicated env's editable install pointing at a /tmp worktree that cleanup
# is about to delete (the original "hijacked + vanished" failure mode).
# $? is preserved across the trap: nothing here calls `exit`, and every
# command is `|| true`, so the script's real exit code survives.
on_exit() {
    if [[ "$RESTORE_BASELINE" -eq 1 ]]; then
        echo "[cleanup] Restoring $BASELINE_ENV -> $REPO_ROOT"
        ( cd "$REPO_ROOT" && conda run -n "$BASELINE_ENV" pip install -e . >/dev/null 2>&1 ) || true
    fi
    if [[ "$RESTORE_CANDIDATE" -eq 1 ]]; then
        echo "[cleanup] Restoring $CANDIDATE_ENV -> $REPO_ROOT"
        ( cd "$REPO_ROOT" && conda run -n "$CANDIDATE_ENV" pip install -e . >/dev/null 2>&1 ) || true
    fi
    cleanup_worktrees
}
trap on_exit EXIT

# --- Helpers ---------------------------------------------------------------

# Overlay the CURRENT branch's example_data_<scenario>/ directory into a
# worktree.  This is what makes the comparison fair: both branches end up
# running against bit-identical input data, so any output diff is purely a
# function of code differences.
overlay_example_data() {
    local worktree="$1"
    local label="$2"

    local src="$REPO_ROOT/docs/source/examples/$DATA_DIR_NAME"
    local src_abs
    src_abs=$(cd "$src" && pwd)   # the absolute path embedded in the generator's configs
    local dst_parent="$worktree/docs/source/examples"
    local dst="$dst_parent/$DATA_DIR_NAME"
    local dst_abs="$dst"          # destination is already absolute (worktree path is absolute)

    if [[ ! -d "$src" ]]; then
        echo "[$label] ERROR: source data dir missing: $src" >&2
        exit 1
    fi
    mkdir -p "$dst_parent"
    rm -rf "$dst"
    cp -r "$src" "$dst"

    # The generator embeds absolute paths into the configs ("output.root_dir"
    # and every "input.*_path").  After copying, those still point at the
    # ORIGINAL repo, so both worktrees would read/write to the SAME location
    # and contaminate each other.  Rewrite each copied YAML so its absolute
    # paths reference the worktree's copy of the data instead.
    # Skip the _main configs — their paths are already relative (legacy
    # hand-curated artifacts), so the rewrite would be a no-op there anyway.
    local cfg
    for cfg in "$dst"/config_*.yml; do
        [[ -f "$cfg" ]] || continue
        # In-place substitution: src_abs -> dst_abs
        sed -i.bak "s|${src_abs}|${dst_abs}|g" "$cfg"
        rm -f "$cfg.bak"
    done
    echo "[$label] Overlaid example data: $src_abs -> $dst_abs (configs path-rewritten)"
}

# Parse output.root_dir from a config file, returning an absolute path.
# Relative paths in the config resolve against the config's own directory.
output_root_from_config() {
    local config_path="$1"
    python -c "
import sys, yaml
from pathlib import Path
cfg_path = Path(sys.argv[1]).resolve()
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)
root = Path(cfg['output']['root_dir'])
if not root.is_absolute():
    root = (cfg_path.parent / root).resolve()
print(root)
" "$config_path"
}

# Run the pipeline in one worktree's example_data dir.
run_pipeline() {
    local worktree="$1"
    local config_name="$2"
    local env_name="$3"
    local label="$4"

    local example_dir="$worktree/docs/source/examples/$DATA_DIR_NAME"
    local config_path="$example_dir/$config_name"
    if [[ ! -f "$config_path" ]]; then
        echo "[$label] ERROR: config not found: $config_path" >&2
        exit 1
    fi

    echo "[$label] Installing alleleflux from $worktree into env=$env_name"
    (cd "$worktree" && conda run -n "$env_name" pip install -e . >/dev/null)

    echo "[$label] Running pipeline: alleleflux run --config $config_name"
    (cd "$example_dir" \
        && rm -rf output_bams output example_output_bams example_output .snakemake \
        && conda run -n "$env_name" alleleflux run --config "$config_name" --threads 4)
}

# --- Run -------------------------------------------------------------------

if [[ "$CAPTURE_GOLDEN" -eq 1 ]]; then
    GOLDEN_DIR="$REPO_ROOT/tests/regression/golden/$GOLDEN_SUBDIR"
    echo "=========================================================================="
    echo "Capturing golden snapshot - scenario=$SCENARIO"
    echo "  Baseline:  $BASELINE_REF  (config: $BASELINE_CONFIG, env: $BASELINE_ENV)"
    echo "  Data:      overlaying current repo's $DATA_DIR_NAME/ into worktree"
    echo "  Target:    $GOLDEN_DIR"
    echo "=========================================================================="
else
    echo "=========================================================================="
    echo "Comparing pipeline outputs - scenario=$SCENARIO"
    echo "  Baseline:  $BASELINE_REF  (config: $BASELINE_CONFIG, env: $BASELINE_ENV)"
    echo "  Candidate: $CANDIDATE_REF (config: $CANDIDATE_CONFIG, env: $CANDIDATE_ENV)"
    echo "  Data:      overlaying current repo's $DATA_DIR_NAME/ into both worktrees"
    echo "=========================================================================="
fi

cleanup_worktrees
git -C "$REPO_ROOT" worktree add "$TMP_BASELINE" "$BASELINE_REF"
overlay_example_data "$TMP_BASELINE"  "baseline"
RESTORE_BASELINE=1   # arm restore: we're about to mutate $BASELINE_ENV
run_pipeline       "$TMP_BASELINE"  "$BASELINE_CONFIG"  "$BASELINE_ENV"  "baseline"

BASELINE_OUT_ROOT=$(
    output_root_from_config "$TMP_BASELINE/docs/source/examples/$DATA_DIR_NAME/$BASELINE_CONFIG"
)
BASELINE_DATA_OUT="$BASELINE_OUT_ROOT/$PIPELINE_DATA_DIR"

if [[ "$CAPTURE_GOLDEN" -eq 1 ]]; then
    # --- Capture-only mode: copy baseline outputs into the golden snapshot dir.
    # Mirrors what tests/regression/test_pipeline_e2e.py::_copy_golden_payload
    # does — same allow-list of suffixes, same exclusions — so the golden
    # captured here matches the shape the regression test expects.
    echo
    echo "Copying baseline outputs into $GOLDEN_DIR ..."
    rm -rf "$GOLDEN_DIR"
    mkdir -p "$GOLDEN_DIR"
    conda run -n "$BASELINE_ENV" python -c "
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '$REPO_ROOT')
from tests.regression.test_pipeline_e2e import _copy_golden_payload

src = Path('$BASELINE_DATA_OUT')
dst = Path('$GOLDEN_DIR')
_copy_golden_payload(src, dst)
print(f'Copied tracked files into {dst}')
"
    CAPTURE_EXIT=$?
    # Dedicated-env restore to $REPO_ROOT is handled by the on_exit trap.
    exit $CAPTURE_EXIT
fi

# --- Compare mode: also run candidate side, then diff. ---------------------
git -C "$REPO_ROOT" worktree add "$TMP_CANDIDATE" "$CANDIDATE_REF"
overlay_example_data "$TMP_CANDIDATE" "candidate"
RESTORE_CANDIDATE=1   # arm restore: we're about to mutate $CANDIDATE_ENV
run_pipeline       "$TMP_CANDIDATE" "$CANDIDATE_CONFIG" "$CANDIDATE_ENV" "candidate"

CANDIDATE_OUT_ROOT=$(
    output_root_from_config "$TMP_CANDIDATE/docs/source/examples/$DATA_DIR_NAME/$CANDIDATE_CONFIG"
)
CANDIDATE_DATA_OUT="$CANDIDATE_OUT_ROOT/$PIPELINE_DATA_DIR"

echo
echo "Baseline outputs : $BASELINE_DATA_OUT"
echo "Candidate outputs: $CANDIDATE_DATA_OUT"
echo
echo "Diffing outputs ..."
# Run from the repo root so `python -m tests.regression._diff_outputs` can
# import the `tests` package regardless of the caller's CWD.  The package may
# be untracked (it lives only in this checkout, not in the worktrees), and the
# conda env's interpreter resolves `-m` against the CWD — without this cd the
# import fails with `No module named 'tests.regression'`, which conda reports
# as exit 127.  The `|| DIFF_EXIT=$?` keeps `set -e` from aborting on a
# non-zero diff: "files differ" (exit 1) is an expected result we still want to
# report and restore the envs after, not a script crash.
DIFF_EXIT=0
( cd "$REPO_ROOT" \
    && conda run -n "$CANDIDATE_ENV" python -m tests.regression._diff_outputs \
        --scenario  "$SCENARIO" \
        --baseline  "$BASELINE_DATA_OUT" \
        --candidate "$CANDIDATE_DATA_OUT" ) || DIFF_EXIT=$?

# --- Restore ---------------------------------------------------------------
# Dedicated-env restore to $REPO_ROOT is handled by the on_exit trap, which
# fires on ANY exit (success, error, Ctrl-C).  A mid-run crash therefore can
# never leave a dedicated env's editable install pointing at a /tmp worktree
# that cleanup is about to delete — the original hijack-and-vanish bug.

exit $DIFF_EXIT
