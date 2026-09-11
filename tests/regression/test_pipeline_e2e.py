"""
End-to-end regression test for the AlleleFlux pipeline.

Runs the bundled example dataset through the full Snakemake workflow and
compares every output file to a committed golden snapshot.  Catches any
change in pipeline behaviour that affects downstream data — including
silent numerical drift that unit tests cannot see.

Relationship to ``diff_branches.sh``
------------------------------------
This file and ``tests/regression/diff_branches.sh`` are siblings, not
duplicates.  They share the comparator (``compare.py``) and the golden-
payload helper (``_copy_golden_payload``), but they answer different
questions about pipeline behaviour.

* **This file (``test_pipeline_e2e.py``)** — fast, snapshot-based check.
  Runs the pipeline on the **working tree** (uncommitted changes are
  visible) ONCE, then diffs against the **frozen** golden snapshot at
  ``tests/regression/golden/<scenario>/``.  ~50 s per scenario.
  CI-friendly; the daily inner-loop tool.

* **``diff_branches.sh``** — slow, live cross-branch comparison.  Git-
  worktrees two refs (default ``main`` vs ``HEAD``), installs each ref's
  code into its own conda env, runs the pipeline TWICE, then diffs the
  two output trees directly — no golden round-trip.  Only sees committed
  code (uncommitted changes are invisible).  ~5 min per scenario.  The
  canonical "is my refactor output-neutral vs main?" test.

The two golden-capture flags answer different questions and live in
different files:

* ``--update-golden`` (this file, conftest.py) snapshots from the
  **CURRENT branch's** outputs.  Use after intentional output changes.
* ``--capture-golden`` (``diff_branches.sh``) snapshots from **``main``'s**
  outputs.  Use to drift-correct the golden against the trusted baseline.

See ``tests/regression/README.md`` § "Live vs snapshot" for the full
side-by-side comparison and decision guide.

Two scenarios are parametrized over via ``conftest.SCENARIOS``:

  * ``longitudinal`` — pre/post timepoint design, ``example_data_longitudinal/``
  * ``single``       — single-timepoint design, ``example_data_single/``

Both run the **BAM mode** (``config_with_bams_<dt>.yml``) which exercises
the full pipeline from BAM profiling onward.  The profiles-mode configs
(``config_generated_<dt>.yml``) exist as the tutorial's "Option A — skip
profiling" fast path; they are NOT exercised by this test today.  If you
want them covered, add the corresponding entries to ``conftest.SCENARIOS``.

Run in place
------------
The test runs the pipeline **directly inside the committed
``example_data_<scenario>/`` directory** (no tmp staging).  This matches
the generator's design choice of writing absolute paths into the configs,
so the pipeline writes outputs at the location ``output.root_dir``
specifies — which is ``<example_data_dir>/output_bams/<data_type>/``.

The test wipes ``output_bams/`` and ``.snakemake/`` from the example dir
before each run so consecutive runs always start clean.  Both directories
are gitignored, so the dirty committed dir is transient.

Run with:

    # default — longitudinal only
    pytest tests/regression/ -v

    # single only
    pytest tests/regression/ --regression-scenario single -v

    # both
    pytest tests/regression/ --regression-scenario all -v

Regenerate the snapshot after an intentional change:

    pytest tests/regression/ --update-golden --regression-scenario all -v

The test is marked ``regression`` so it can be deselected from the fast
unit-test loop:

    pytest tests/ -m "not regression"
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.regression.compare import compare_trees
from tests.regression.conftest import SCENARIOS


logger = logging.getLogger(__name__)


# Files we intentionally do NOT track in the snapshot — Snakemake bookkeeping
# and per-run metadata that varies even with deterministic inputs.
_EXCLUDED_SUFFIXES = (".log", ".snakemake_timestamp")
_EXCLUDED_PREFIXES = ("config_runtime_",)
_TRACKED_SUFFIXES = (".tsv", ".tsv.gz", ".parquet", ".json")


# Directory names the pipeline writes inside the example_data_* dir that we
# must wipe before each run (Snakemake reuses outputs based on timestamps,
# so a leftover from a previous run could silently masquerade as a fresh one).
_TRANSIENT_OUTPUT_DIRS = (
    "output_bams",   # written by config_with_bams_<dt>.yml
    "output",        # written by config_generated_<dt>.yml (profiles mode)
    ".snakemake",    # Snakemake's own bookkeeping
)


def _resolve_config_path(scenario: str, override: Path | None) -> Path:
    """Pick the config the pipeline run will use for this scenario.

    Precedence:
      1. ``--regression-config`` override, when supplied.
      2. The scenario's bundled config (config_name inside example_data_dir).
    """
    if override is not None:
        return override
    bundle = SCENARIOS[scenario]
    return Path(bundle["example_data_dir"]) / bundle["config_name"]


def _resolve_output_root(config_path: Path) -> Path:
    """Parse the config to find where the pipeline will write outputs.

    The generator embeds ``output.root_dir`` as an absolute path, so the
    pipeline writes there regardless of cwd.  We read it once up front so the
    test can both wipe the prior run's output AND know where to look for the
    new one.
    """
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    root_dir = Path(cfg["output"]["root_dir"])
    if not root_dir.is_absolute():
        # Resolve against the config's own directory if it happens to be
        # relative (a possible future generator change).
        root_dir = (config_path.parent / root_dir).resolve()
    return root_dir


def _copy_golden_payload(src: Path, dst: Path) -> None:
    """Copy the subset of pipeline outputs we track in the snapshot."""
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        if f.name.startswith(_EXCLUDED_PREFIXES):
            continue
        if f.name.endswith(_EXCLUDED_SUFFIXES):
            continue
        if not f.name.endswith(_TRACKED_SUFFIXES):
            continue
        rel = f.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def _wipe_transient_outputs(example_dir: Path, output_root: Path) -> None:
    """Remove anything the pipeline might have left from a previous run."""
    # output.root_dir may live anywhere the config points; clear it explicitly.
    if output_root.exists():
        shutil.rmtree(output_root)
    # Also sweep the conventional transient locations inside the example dir
    # in case a different config (e.g. the profiles-mode one) wrote there.
    for name in _TRANSIENT_OUTPUT_DIRS:
        p = example_dir / name
        if p.exists():
            shutil.rmtree(p)


@pytest.mark.regression
def test_pipeline_matches_golden(
    scenario,
    regression_config_override,
    golden_root,
    update_golden,
):
    """Run one scenario of the example pipeline and diff against its golden.

    Two modes, picked via ``--update-golden``:

    * **Normal mode** (default): runs the pipeline on the CURRENT branch
      using the committed
      ``docs/source/examples/example_data_<scenario>/`` directory, then
      diffs its outputs against the committed golden snapshot at
      ``tests/regression/golden/<scenario>/``.
    * **Update mode** (``--update-golden``): snapshots the CURRENT branch's
      outputs into the golden directory.  Use this when your branch
      intentionally changes pipeline outputs and you want those outputs to
      become the new baseline.

      To refresh the golden from ``main`` instead (e.g. to verify your
      branch matches main), run
      ``tests/regression/diff_branches.sh --capture-golden`` directly —
      it worktrees main, overlays the current example data, and snapshots.

    The pipeline must be installed in the active conda environment
    (``pip install -e .`` from the repo root, which is what the alleleflux
    env already provides).
    """
    bundle = SCENARIOS[scenario]
    config_path = _resolve_config_path(scenario, regression_config_override)
    assert config_path.exists(), f"Regression config not found: {config_path}"

    example_dir = config_path.parent
    output_root = _resolve_output_root(config_path)
    pipeline_data_dir = output_root / bundle["pipeline_data_dir"]

    # Clean slate — wipe any output_bams/, output/, .snakemake/ left behind by
    # a previous run.  Without this, Snakemake would reuse those outputs and
    # the test would validate stale data instead of a fresh run.
    _wipe_transient_outputs(example_dir, output_root)

    cmd = [
        "alleleflux", "run",
        "--config", str(config_path),
        "--threads", "4",
    ]
    logger.info(
        "Running pipeline for scenario=%s: %s (cwd=%s)",
        scenario, " ".join(cmd), example_dir,
    )
    result = subprocess.run(
        cmd, cwd=example_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"Pipeline failed (scenario={scenario}):\n"
            f"  exit code: {result.returncode}\n"
            f"  stdout (tail):\n{result.stdout[-2000:]}\n"
            f"  stderr (tail):\n{result.stderr[-2000:]}"
        )

    assert pipeline_data_dir.exists(), (
        f"Pipeline data dir missing for scenario '{scenario}': {pipeline_data_dir}\n"
        f"Expected the pipeline to write outputs under {output_root}/."
    )

    golden_dir = golden_root / bundle["golden_subdir"]

    # --update-golden mode: this is NOT a check, it is a snapshot refresh
    # FROM the current branch.  Blow away the old golden, repopulate from
    # this run, and skip (so the run is not also reported as a pass/fail
    # against itself).
    if update_golden:
        if golden_dir.exists():
            shutil.rmtree(golden_dir)
        golden_dir.mkdir(parents=True)
        _copy_golden_payload(pipeline_data_dir, golden_dir)
        pytest.skip(
            f"Golden snapshot refreshed at {golden_dir} from CURRENT branch.  "
            f"For a main-derived snapshot, run "
            f"`tests/regression/diff_branches.sh --capture-golden` instead."
        )

    assert golden_dir.exists(), (
        f"Golden snapshot missing for scenario '{scenario}' at {golden_dir}. "
        f"Run `pytest tests/regression/ --update-golden --regression-scenario {scenario}` to create it."
    )

    # The golden is captured from main; scope the diff to the scientifically
    # meaningful, structurally-stable outputs so intended branch divergence
    # (dropped columns, renamed/re-formatted files, disabled optional outputs)
    # does not drown the real signal.  See SCENARIOS in conftest for rationale.
    # float32 frequencies on this branch vs float64 in the main-derived golden
    # mean ~1e-7 rounding is expected; bundle rtol/atol relax the strict default
    # accordingly (see SCENARIOS).  Falls back to the strict comparator defaults
    # when a scenario does not set them.
    tol = {k: bundle[k] for k in ("rtol", "atol") if k in bundle}
    failures = compare_trees(
        pipeline_data_dir,
        golden_dir,
        include_prefixes=bundle.get("compare_only"),
        allow_golden_only=bundle.get("allow_golden_only"),
        exclude_substrings=bundle.get("exclude_substrings"),
        **tol,
    )
    if failures:
        joined = "\n\n".join(failures)
        pytest.fail(
            f"[scenario={scenario}] {len(failures)} file(s) differ from the "
            f"golden snapshot:\n\n{joined}"
        )
