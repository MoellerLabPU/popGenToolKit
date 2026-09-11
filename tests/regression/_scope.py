"""Shared comparison scope + float tolerance for branch-vs-main regression.

Both regression tools diff a candidate branch against a ``main``-derived
reference, and both must apply the **same** contract or they disagree:

* ``test_pipeline_e2e.py`` (via ``conftest.SCENARIOS``) — snapshot-based.
* ``_diff_outputs.py`` (via ``diff_branches.sh``) — live cross-branch.

This module is that single source of truth for the whole comparison contract:
the file-selection scope (``STAT_COMPARE_ONLY`` / ``STAT_ALLOW_GOLDEN_ONLY`` /
the per-scenario ``exclude_substrings``), the float tolerances, AND the
column-level rules (``SIGN_INSENSITIVE_SUBSTRINGS`` / ``IGNORE_VALUE_SUBSTRINGS``).
``compare.py`` imports the column-rules from here rather than hard-coding
AlleleFlux column names, so its mechanism stays generic and all AlleleFlux
comparison knowledge sits in one file.  It lives here — not in ``conftest.py``
(which imports pytest, and ``compare`` / ``_diff_outputs`` must not) — and this
module itself imports nothing beyond ``__future__``, so every importer stays
clean.

The goal: verify the branch produces the same *science* as main while ignoring
intentional structural drift.  Rationale per prefix / list below.
"""

from __future__ import annotations

# --- What to compare ------------------------------------------------------
# Structurally-stable statistical outputs — identical filenames/format on both
# branches, and the integration point where any upstream numerical change
# surfaces:
#   * significance_tests/, p_value_summary/  — raw + summarised stats.
#   * scores/intermediate/                   — per-MAG scores; identical 1:1.
#   * scores/processed/combined/             — taxonomic aggregation; main
#     computes all 7 levels, the branch only the ``taxa_score_levels`` subset,
#     so the extra golden levels are allowed-missing (shared levels strict).
#   * preprocessing_eligibility/, eligibility_table — QC/filter decisions; 1:1.
STAT_COMPARE_ONLY = [
    "significance_tests",
    "p_value_summary",
    "scores/intermediate",
    "scores/processed/combined",
    "preprocessing_eligibility",
    "eligibility_table",
]

# Golden (main) files the branch intentionally does not emit — not MISSING:
#   * scores/processed/combined/<level>/ for taxa levels omitted by config.
# Excluded entirely (not even in COMPARE_ONLY): profiles/ (mapq_scores column
# dropped), allele_analysis/ (tsv.gz -> parquet, files dropped), QC/ +
# inputMetadata/ (renamed group-independent), outlier_genes/
# (use_outlier_detection=False), scores/processed/gene_scores_*
# (use_gene_scores=False).
STAT_ALLOW_GOLDEN_ONLY = ["scores/processed/combined"]

# --- How loosely to compare floats ----------------------------------------
# The golden is float64 (main), but the branch stores allele frequencies as
# float32 (~1e-7 precision — a deliberate memory optimization in
# _allele_freq_common.py), so the strict comparator default (rtol=1e-9) is
# unachievable by construction.  These bounds sit ~4 orders above the observed
# benign drift (max ~1.3e-4 on a q_value) yet still catch any scientifically
# meaningful change (a p-value moving 0.01, a frequency moving 0.001).  Tighten
# back toward the defaults if the golden and the branch are ever brought to the
# same float precision.
STAT_RTOL = 1e-5
STAT_ATOL = 1e-3

# --- How to compare specific columns --------------------------------------
# These two lists are the column-level half of the contract: they tell the
# comparator how to treat named columns whose raw values are non-deterministic
# run-to-run.  They live here (not hard-coded in compare.py) so all AlleleFlux
# comparison knowledge sits in one place; compare.py imports them and both the
# pytest e2e and the live ``_diff_outputs`` tool apply the identical rules.

# Substrings identifying allele-frequency *derivative* columns, compared on
# ABSOLUTE value.  The pipeline has an upstream non-determinism that flips the
# sign of pre-vs-post differences depending on sample iteration order, so a
# column can be (+0.5, -0.5) on one run and (-0.5, +0.5) on the next.  Magnitude
# must still match — this is sign-insensitive, NOT magnitude-insensitive.  When
# the root cause is fixed, empty this list and the columns go back to signed.
SIGN_INSENSITIVE_SUBSTRINGS = (
    "_change",
    "_diff_mean",
    "_diff",
)

# Substrings identifying NON-DETERMINISTIC diagnostic columns whose *values*
# vary run-to-run and so must not gate equivalence.  The statsmodels LMM
# optimizer emits a run-dependent number of convergence warnings — and can flip
# the per-fit ``converged`` boolean — on near-degenerate positions (e.g. p≈1.0
# perfect ties), while the scientific outputs (p-value, coefficient, t-value) on
# those very rows are identical.  These columns are still required to be PRESENT
# (the comparator's schema gate catches a dropped/renamed column); only their
# *values* are skipped.  ``"_warnings"`` also matches ``"_n_warnings"`` as a
# substring, so one token covers both the warning text and the count.
IGNORE_VALUE_SUBSTRINGS = (
    "_warnings",       # {NUC}_warnings (text) and {NUC}_n_warnings (count)
    "_converged_LMM",  # per-fit convergence flag
)

# --- Per-scenario carve-outs ----------------------------------------------
# Substrings matched anywhere in a golden file's relative path, applied AFTER
# STAT_COMPARE_ONLY.  Used to drop non-comparable subtrees that fall *inside* a
# broad include prefix (e.g. the preprocessed intermediates live under
# ``significance_tests/preprocessed_*/``).  Keyed by scenario name.
#
#   single:
#     * two_sample_paired — DETERMINISTIC but float32-sensitive (verified: two
#       same-branch runs are bit-identical; the paired-test code is identical
#       across branches — empty git diff — so only the input differs).  The
#       single paired test runs Wilcoxon on RAW frequencies (longitudinal uses
#       diff_means).  The branch stores frequencies as float32, so its values
#       differ from the float64 golden by ~1e-7, which flips a near-tie RANK in
#       the discrete small-n Wilcoxon → the p-value jumps a discrete step
#       (e.g. 0.375↔0.625) that the float tolerance can't absorb, cascading into
#       q-values.  Scientifically null (a non-significant p stays non-significant).
#       longitudinal's paired test uses diff_means (continuous, no rank flips),
#       so it is unaffected and stays in scope.
#     * preprocessed — the *_allele_frequency_preprocessed intermediate inputs
#       carry metadata columns (mapq_scores, coverage, breadth, …) the branch's
#       refactor prunes, so they column-set-mismatch.  Intermediate, not output.
#   longitudinal: none — its paired test (pre-vs-post) is deterministic and its
#     preprocessed files match, so both stay in scope for full coverage.
SCENARIO_EXCLUDE_SUBSTRINGS = {
    "longitudinal": [],
    "single": ["two_sample_paired", "preprocessed"],
}


def exclude_substrings_for(scenario: str) -> list[str]:
    """Return the substring carve-outs for ``scenario`` (empty if it has none)."""
    return SCENARIO_EXCLUDE_SUBSTRINGS.get(scenario, [])
