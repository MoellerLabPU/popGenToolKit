#!/usr/bin/env python3
"""
Unit tests for preprocess_between_groups site filtering.
=======================================================

These tests pin the *behaviour* of the between-groups site filter so that the
memory refactor of :func:`filter_sites_parallel` (shipping minimal per-nucleotide
arrays to workers instead of whole per-site DataFrames) is provably
behaviour-preserving.

A site is REMOVED only when ALL four nucleotides are non-significant
(``t_p >= alpha``); it is KEPT as soon as any nucleotide is significant.

Sites are engineered so the outcome is unambiguous without predicting exact
scipy p-values:
- all-identical (e.g. all-zero) columns -> ``np.all(d == 0)`` short-circuit -> p == 1 -> removed
- <= 3 samples -> guard returns p == 1 -> removed
- one nucleotide with a large, consistent split -> p << 0.05 -> kept
"""

import numpy as np
import pandas as pd

from alleleflux.scripts.preprocessing.preprocess_between_groups import (
    NUCLEOTIDES,
    filter_sites_parallel,
    process_group,
)

# A column whose top-half vs bottom-half split is large and consistent enough
# that the paired t-test is unambiguously significant (p << 0.05), while the
# paired differences keep a non-zero variance (so scipy does not return NaN).
SIGNIFICANT = [12.0, 10.0, 10.0, 8.0, 1.0, 0.0, 0.0, -1.0]
# Same signal with one extra value at the median, which the odd-length branch
# drops (closest-to-median), recovering SIGNIFICANT.
SIGNIFICANT_ODD = [12.0, 10.0, 10.0, 8.0, 5.0, 1.0, 0.0, 0.0, -1.0]


def _make_df(sites, value_suffix):
    """Build a long-form per-site DataFrame.

    ``sites`` maps (contig, position) -> the values for the A nucleotide; the
    other three nucleotides are all-zero (non-significant). ``value_suffix`` is
    ``"_diff_mean"`` for longitudinal data and ``""`` for single.
    """
    rows = []
    for (contig, position), a_vals in sites.items():
        n = len(a_vals)
        for i in range(n):
            row = {"contig": contig, "position": position}
            row[f"A_frequency{value_suffix}"] = a_vals[i]
            row[f"T_frequency{value_suffix}"] = 0.0
            row[f"G_frequency{value_suffix}"] = 0.0
            row[f"C_frequency{value_suffix}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# --- New worker contract (the RED test) -----------------------------------


def test_process_group_operates_on_per_nucleotide_arrays():
    """process_group consumes a {nucleotide: ndarray} mapping, not a DataFrame.

    The caller resolves columns (data_type) up front and ships only the arrays,
    so the worker no longer needs the DataFrame or data_type.
    """
    site_values = {
        "A_frequency": np.array(SIGNIFICANT),
        "T_frequency": np.zeros(8),
        "G_frequency": np.zeros(8),
        "C_frequency": np.zeros(8),
    }

    pvals = process_group(site_values, filter_type="t-test")

    # A: strong, consistent split -> significant
    assert pvals["A_frequency"][0] < 0.05
    # all-identical columns -> short-circuit p == 1 (non-significant)
    for nuc in ("T_frequency", "G_frequency", "C_frequency"):
        assert pvals[nuc][0] == 1


# --- End-to-end equivalence safety net ------------------------------------


def test_filter_removes_only_all_nonsignificant_sites_longitudinal():
    sites = {
        ("c1", 1): [0.0] * 8,        # all-zero -> removed
        ("c1", 2): SIGNIFICANT,      # A significant -> kept
        ("c1", 3): [5.0, 5.0, 5.0],  # <= 3 samples -> removed
        ("c1", 4): SIGNIFICANT_ODD,  # odd-length, significant after median drop -> kept
    }
    df = _make_df(sites, value_suffix="_diff_mean")
    grouped = df.groupby(["contig", "position"], dropna=False)

    removed = filter_sites_parallel(
        grouped, alpha=0.05, filter_type="t-test", cpus=1, data_type="longitudinal"
    )

    assert set(removed) == {("c1", 1), ("c1", 3)}


def test_filter_removes_only_all_nonsignificant_sites_single():
    sites = {
        ("c1", 1): [0.0] * 8,    # all-zero -> removed
        ("c1", 2): SIGNIFICANT,  # A significant -> kept
    }
    df = _make_df(sites, value_suffix="")
    grouped = df.groupby(["contig", "position"], dropna=False)

    removed = filter_sites_parallel(
        grouped, alpha=0.05, filter_type="t-test", cpus=1, data_type="single"
    )

    assert set(removed) == {("c1", 1)}


def test_nucleotides_constant_unchanged():
    """Guards the column-name contract the parent relies on when extracting arrays."""
    assert NUCLEOTIDES == [
        "A_frequency",
        "T_frequency",
        "G_frequency",
        "C_frequency",
    ]
