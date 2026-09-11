"""
Pytest configuration for the end-to-end regression test.

Adds three custom CLI options:

  --update-golden        Re-run the pipeline on the example dataset and overwrite
                         the golden snapshot in tests/regression/golden/. Use after
                         an intentional change to pipeline outputs.

  --regression-scenario  Which scenario(s) to run.  One of
                         ``longitudinal`` (default), ``single``, or ``all``.
                         Each scenario is a (data type, dataset directory,
                         config, golden subdir) bundle — see ``SCENARIOS``
                         below.

  --regression-config    Optional explicit override for the config path used in
                         the run.  Rare — primarily for ad-hoc comparisons.
                         When given, it replaces the scenario's default config.

The ``SCENARIOS`` mapping is the single source of truth for what the test
exercises.  Add a new entry to introduce a new scenario; ``test_pipeline_e2e``
parametrizes over the entries selected by ``--regression-scenario``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Scope + tolerance for branch-vs-main diffs, shared with the live cross-branch
# tool (_diff_outputs.py) so both apply the identical contract.  See _scope.py.
from tests.regression._scope import (
    STAT_ALLOW_GOLDEN_ONLY as _STAT_ALLOW_GOLDEN_ONLY,
    STAT_ATOL as _STAT_ATOL,
    STAT_COMPARE_ONLY as _STAT_COMPARE_ONLY,
    STAT_RTOL as _STAT_RTOL,
    exclude_substrings_for as _exclude_substrings_for,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_ROOT = REPO_ROOT / "tests/regression/golden"
EXAMPLES_DIR = REPO_ROOT / "docs/source/examples"


# Single source of truth for which scenarios the regression test covers.
# Each entry resolves to a fully-qualified bundle: where the inputs live,
# which config drives the run, where the pipeline writes its outputs, and
# where the committed golden snapshot lives.  The compare_only / allow_golden_only
# / rtol / atol knobs come from _scope.py (imported above) so the live
# cross-branch tool applies the identical contract.
SCENARIOS: dict[str, dict] = {
    "longitudinal": {
        "data_type": "longitudinal",
        "example_data_dir": str(EXAMPLES_DIR / "example_data_longitudinal"),
        "config_name": "config_with_bams_longitudinal.yml",
        "pipeline_data_dir": "longitudinal",
        "golden_subdir": "longitudinal",
        "compare_only": _STAT_COMPARE_ONLY,
        "allow_golden_only": _STAT_ALLOW_GOLDEN_ONLY,
        "exclude_substrings": _exclude_substrings_for("longitudinal"),
        "rtol": _STAT_RTOL,
        "atol": _STAT_ATOL,
    },
    "single": {
        "data_type": "single",
        "example_data_dir": str(EXAMPLES_DIR / "example_data_single"),
        "config_name": "config_with_bams_single.yml",
        # The pipeline writes the single data_type under "single_timepoint"
        # (common.smk: DATA_TYPE=="single" -> OUTDIR/single_timepoint), NOT
        # "single".  Must match or the test can't find the outputs.
        "pipeline_data_dir": "single_timepoint",
        "golden_subdir": "single",
        "compare_only": _STAT_COMPARE_ONLY,
        "allow_golden_only": _STAT_ALLOW_GOLDEN_ONLY,
        # two_sample_paired is float32-sensitive on single (Wilcoxon on raw freqs
        # → discrete rank flips vs the float64 golden) and the preprocessed
        # intermediates are column-pruned — carve both out (see _scope.py).
        "exclude_substrings": _exclude_substrings_for("single"),
        "rtol": _STAT_RTOL,
        "atol": _STAT_ATOL,
    },
}


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Overwrite the regression golden snapshot with the pipeline's "
             "current outputs.  Only use after an intentional output change.",
    )
    parser.addoption(
        "--regression-scenario",
        action="store",
        default="longitudinal",
        choices=[*SCENARIOS.keys(), "all"],
        help="Which scenario(s) to run.  ``all`` runs every entry in SCENARIOS.",
    )
    parser.addoption(
        "--regression-config",
        action="store",
        default=None,
        help="Optional override of the config path for the chosen scenario.  "
             "Defaults to the config bundled in the scenario's example_data_* dir.",
    )


def _selected_scenarios(config) -> list[str]:
    """Resolve --regression-scenario to a list of scenario names."""
    val = config.getoption("--regression-scenario")
    if val == "all":
        return list(SCENARIOS.keys())
    return [val]


def pytest_generate_tests(metafunc):
    """Parametrize the regression test over the requested scenarios.

    Tests that take a ``scenario`` parameter get one invocation per selected
    scenario.  Other tests are unaffected.
    """
    if "scenario" in metafunc.fixturenames:
        scenarios = _selected_scenarios(metafunc.config)
        metafunc.parametrize("scenario", scenarios, ids=scenarios)


@pytest.fixture(scope="session")
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture
def regression_config_override(request) -> Path | None:
    """Optional explicit config-path override.  None when not set."""
    val = request.config.getoption("--regression-config")
    return Path(val).resolve() if val else None


@pytest.fixture(scope="session")
def golden_root() -> Path:
    return GOLDEN_ROOT
