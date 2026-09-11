"""Unit tests for the Snakemake eligibility helper ``_get_mags_by_eligibility``.

The helper lives in a ``.smk`` file whose module body references Snakemake
globals (``config``, ``checkpoints``, ``workflow``) and directives
(``wildcard_constraints``) that are not valid Python outside a running
workflow, so the module cannot be imported directly.  Instead we slice the
pure ``_get_mags_by_eligibility`` function out of the source by indentation
and exec it in a namespace that supplies its only free globals
(``os``, ``pd``, ``OUTDIR``).

These tests pin the ``between_only`` eligibility type, which exists so that
allele-analysis targets exclude MAGs eligible *only* for single-sample
(within-group) tests when ``run_within_group_tests`` is False — see
``generate_allele_analysis_targets`` in ``dynamic_targets.smk``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

COMMON_SMK = (
    Path(__file__).resolve().parents[1]
    / "alleleflux"
    / "smk_workflow"
    / "alleleflux_pipeline"
    / "shared"
    / "common.smk"
)


def _load_eligibility_helper(outdir):
    """Extract ``_get_mags_by_eligibility`` from common.smk and bind OUTDIR.

    The full ``.smk`` file contains Snakemake directives (``wildcard_constraints``
    etc.) that are not valid Python, so we slice out just the target function by
    indentation before parsing it.
    """
    lines = COMMON_SMK.read_text().splitlines()
    start = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("def _get_mags_by_eligibility")
    )
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        # Function ends at the next top-level (column-0, non-blank) statement.
        if ln.strip() and not ln[0].isspace():
            break
        end += 1
    func_src = "\n".join(lines[start:end])
    namespace = {"os": os, "pd": pd, "OUTDIR": str(outdir)}
    exec(compile(func_src, str(COMMON_SMK), "exec"), namespace)
    return namespace["_get_mags_by_eligibility"]


@pytest.fixture
def eligibility_table(tmp_path):
    """Write a synthetic eligibility table and return (outdir, helper).

    MAG layout:
      - mag_both:   unpaired + paired eligible (a real between-group MAG)
      - mag_unpair: unpaired only
      - mag_within: eligible ONLY via a single_sample_eligible_* column
      - mag_none:   eligible for nothing
    """
    timepoints, groups = "pre_post", "treatment_control"
    df = pd.DataFrame(
        {
            "MAG_ID": ["mag_both", "mag_unpair", "mag_within", "mag_none"],
            "unpaired_test_eligible": [True, True, False, False],
            "paired_test_eligible": [True, False, False, False],
            "single_sample_eligible_treatment": [False, False, True, False],
            "single_sample_eligible_control": [False, False, False, False],
        }
    )
    df.to_csv(
        tmp_path / f"eligibility_table_{timepoints}-{groups}.tsv",
        sep="\t",
        index=False,
    )
    helper = _load_eligibility_helper(tmp_path)
    return helper, timepoints, groups


class TestBetweenOnlyEligibility:
    def test_between_only_excludes_within_group_only_mags(self, eligibility_table):
        helper, tp, gr = eligibility_table
        result = helper(tp, gr, eligibility_type="between_only")
        assert set(result) == {"mag_both", "mag_unpair"}
        assert "mag_within" not in result

    def test_all_includes_within_group_only_mags(self, eligibility_table):
        helper, tp, gr = eligibility_table
        result = helper(tp, gr, eligibility_type="all")
        # "all" is the union including single-sample columns.
        assert set(result) == {"mag_both", "mag_unpair", "mag_within"}

    def test_between_only_is_subset_of_all(self, eligibility_table):
        helper, tp, gr = eligibility_table
        between = set(helper(tp, gr, eligibility_type="between_only"))
        every = set(helper(tp, gr, eligibility_type="all"))
        assert between <= every
        # The difference is exactly the within-group-only MAG.
        assert every - between == {"mag_within"}

    def test_unknown_type_lists_between_only(self, eligibility_table):
        helper, tp, gr = eligibility_table
        with pytest.raises(ValueError, match="between_only"):
            helper(tp, gr, eligibility_type="bogus")
