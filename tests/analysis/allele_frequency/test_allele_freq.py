#!/usr/bin/env python3
"""Unit tests for allele_freq.py Stage-2 memory optimizations.

Covers two exact (output-preserving) memory optimizations added to handle the
heaviest DRIDO MAGs (e.g. MRGM_0841 at 5mo_16mo, ~1.36B rows) that OOM-killed
the allele_analysis rule even at ~1 TB:

1. ``calculate_allele_frequency_changes`` pre-filters both timepoint frames to
   the subjectID intersection before the merge.  Rows for subjects present at
   only one timepoint cannot survive an inner join keyed on subjectID, so
   dropping them pre-merge is exact and roughly halves the hash-join input on
   cross-sectional cohorts.

2. ``load_cache_files`` does not read the ``total_coverage`` column for
   longitudinal data unless archival output is requested — it is only consumed
   by the (default-off) archival diff table, so carrying it through load + merge
   is dead weight.
"""

import os
import tempfile
import unittest

import pandas as pd

from alleleflux.scripts.analysis.allele_frequency._allele_freq_common import (
    CACHE_COLUMNS_LONGITUDINAL,
    NUCLEOTIDES,
)
from alleleflux.scripts.analysis.allele_frequency.allele_freq import (
    calculate_allele_frequency_changes,
    load_cache_files,
)


def _long_row(time, subjectID, group, freqs, contig="c1", gene="g1",
              position=10, replicate="r1", total_coverage=100):
    """One per-(sample, position) longitudinal allele-frequency row.

    ``freqs`` is a 4-tuple of (A, T, G, C) frequency values matching the
    NUCLEOTIDES column order.
    """
    row = {
        "time": time,
        "subjectID": subjectID,
        "group": group,
        "contig": contig,
        "gene_id": gene,
        "position": position,
        "replicate": replicate,
        "total_coverage": total_coverage,
    }
    for nuc, val in zip(NUCLEOTIDES, freqs):
        row[nuc] = val
    return row


def _make_longitudinal_df(rows, categorical=True):
    """Build a Stage-2 input frame mirroring main()'s post-load dtypes."""
    df = pd.DataFrame(rows)
    if categorical:
        for col in ("group", "time", "subjectID", "replicate"):
            df[col] = df[col].astype("category")
    return df


class TestCalculateAlleleFrequencyChangesPrefilter(unittest.TestCase):
    """Pre-merge subjectID-intersection filter must preserve exact output."""

    def test_singleton_subjects_excluded_and_diffs_correct(self):
        # S1 is present at both timepoints (a true pair); S2 only at t1 and S3
        # only at t2 are cross-sectional singletons that must not appear.
        rows = [
            _long_row("t1", "S1", "A", (0.1, 0.2, 0.3, 0.4)),
            _long_row("t2", "S1", "A", (0.5, 0.2, 0.3, 0.0)),
            _long_row("t1", "S2", "A", (0.9, 0.0, 0.1, 0.0)),
            _long_row("t2", "S3", "A", (0.0, 0.0, 0.5, 0.5)),
        ]
        df = _make_longitudinal_df(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = calculate_allele_frequency_changes(df, tmp, "MAG_X")

        # Only the paired subject survives the inner join.
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["subjectID"], "S1")

        # Diff convention: {nuc}_diff = tp2 - tp1.
        self.assertAlmostEqual(out.iloc[0]["A_frequency_diff"], 0.4, places=5)
        self.assertAlmostEqual(out.iloc[0]["T_frequency_diff"], 0.0, places=5)
        self.assertAlmostEqual(out.iloc[0]["G_frequency_diff"], 0.0, places=5)
        self.assertAlmostEqual(out.iloc[0]["C_frequency_diff"], -0.4, places=5)

    def test_all_shared_subjects_preserved(self):
        # When every subject is paired, the prefilter must be a no-op: all
        # subjects appear in the output.
        rows = [
            _long_row("t1", "S1", "A", (0.1, 0.2, 0.3, 0.4)),
            _long_row("t2", "S1", "A", (0.2, 0.2, 0.3, 0.3)),
            _long_row("t1", "S2", "A", (0.0, 0.5, 0.5, 0.0), position=20),
            _long_row("t2", "S2", "A", (0.0, 0.4, 0.6, 0.0), position=20),
        ]
        df = _make_longitudinal_df(rows)

        with tempfile.TemporaryDirectory() as tmp:
            out = calculate_allele_frequency_changes(df, tmp, "MAG_X")

        self.assertEqual(set(out["subjectID"]), {"S1", "S2"})


class TestLoadCacheFilesTotalCoverage(unittest.TestCase):
    """total_coverage is read only when archival output needs it."""

    def _write_cache(self, path, time):
        rows = [
            _long_row(time, "S1", "A", (0.1, 0.2, 0.3, 0.4)),
            _long_row(time, "S2", "A", (0.2, 0.2, 0.3, 0.3), position=20),
        ]
        df = pd.DataFrame(rows)
        # Add the remaining cache columns (raw counts + sample_id) so the file
        # carries the full cache schema the loader may select from.
        df["A"], df["C"], df["G"], df["T"] = 10, 20, 30, 40
        df["sample_id"] = df["subjectID"] + "_" + str(time)
        df = df[[c for c in CACHE_COLUMNS_LONGITUDINAL if c in df.columns]]
        df.to_parquet(path, engine="pyarrow", index=False)

    def test_longitudinal_default_drops_total_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "t1.parquet")
            p2 = os.path.join(tmp, "t2.parquet")
            self._write_cache(p1, "t1")
            self._write_cache(p2, "t2")

            df = load_cache_files([p1, p2], "longitudinal")

        self.assertNotIn("total_coverage", df.columns)
        # Frequency columns and the merge keys must still be present.
        for col in NUCLEOTIDES + ["subjectID", "group", "time"]:
            self.assertIn(col, df.columns)

    def test_longitudinal_archival_keeps_total_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "t1.parquet")
            p2 = os.path.join(tmp, "t2.parquet")
            self._write_cache(p1, "t1")
            self._write_cache(p2, "t2")

            df = load_cache_files([p1, p2], "longitudinal", save_archival=True)

        self.assertIn("total_coverage", df.columns)


if __name__ == "__main__":
    unittest.main()
