#!/usr/bin/env python3
"""
Unit Tests for P-Value Summary Script
====================================

This module contains comprehensive unit tests for the p_value_summary.py script,
which processes and consolidates AlleleFlux test results with FDR-BH correction.

Test Coverage:
- File discovery functionality
- P-value extraction and processing
- MAG ID extraction from filenames
- FDR-BH correction application
- Error handling and edge cases
- Integration with the main workflow

Usage:
    python -m unittest tests.test_p_value_summary
    python -m unittest discover tests -p "test_p_value_summary.py"
"""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from statsmodels.stats.multitest import multipletests

from alleleflux.scripts.summary.p_value_summary import (
    extract_mag_id_from_filepath,
    find_test_files,
    process_results_file,
)


class TestExtractMagIdFromFilepath(unittest.TestCase):
    """Test the extract_mag_id_from_filepath function."""

    def test_extract_mag_id_standard_format(self):
        """Test MAG ID extraction from standard filename format."""
        test_cases = [
            ("MAG001_lmm_across_time_results.tsv.gz", "lmm_across_time", "MAG001"),
            ("SAMPLE_123_cmh_results.tsv.gz", "cmh", "SAMPLE_123"),
            (
                "complex.mag.id_two_sample_paired_output.tsv.gz",
                "two_sample_paired",
                "complex.mag.id",
            ),
        ]

        for filename, test_type, expected_mag_id in test_cases:
            filepath = Path(filename)
            result = extract_mag_id_from_filepath(filepath, test_type)
            self.assertEqual(result, expected_mag_id)

    def test_extract_mag_id_no_test_type_in_filename(self):
        """Test behavior when test type is not found in filename."""
        filepath = Path("MAG001_different_test_results.tsv.gz")
        test_type = "lmm"

        # Should return None when pattern not found
        result = extract_mag_id_from_filepath(filepath, test_type)
        self.assertIsNone(result)


class TestProcessResultsFile(unittest.TestCase):
    """Test the process_results_file function."""

    def setUp(self):
        """Set up test data for processing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create mock data
        self.test_data = pd.DataFrame(
            {
                "mag_id": ["MAG001", "MAG001", "MAG001"],
                "contig": ["contig1", "contig1", "contig2"],
                "position": [100, 200, 300],
                "gene_id": ["gene1", "gene1", "gene2"],
                "p_value_test1": [0.01, 0.05, 0.001],
                "p_value_test2": [0.02, 0.08, 0.003],
                "other_column": ["A", "B", "C"],
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_process_results_file_success(self):
        """Test successful processing of a results file."""
        # Create test file
        test_file = self.temp_path / "MAG001_lmm_results.tsv.gz"
        self.test_data.to_csv(test_file, sep="\t", compression="gzip", index=False)

        result = process_results_file(test_file, "lmm", "pre_post")

        # Check that results are returned
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        # Check that the result key follows the new format
        expected_key = "lmm_pre_post"
        self.assertIn(expected_key, result)

        # Check that each sub-test has a DataFrame with the expected columns
        for key, df in result.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn("min_p_value", df.columns)
            self.assertIn("mag_id", df.columns)
            self.assertIn("source_file", df.columns)
            self.assertIn("period", df.columns)
            self.assertIn("test_type", df.columns)

    def test_process_results_file_no_p_value_columns(self):
        """Test behavior when no p-value columns are found."""
        # Create test data without p-value columns
        data_no_pvals = pd.DataFrame(
            {
                "contig": ["contig1"],
                "position": [100],
                "gene_id": ["gene1"],
                "other_column": ["A"],
            }
        )

        test_file = self.temp_path / "no_pvals_results.tsv.gz"
        data_no_pvals.to_csv(test_file, sep="\t", compression="gzip", index=False)

        # Mock extract_relevant_columns to return empty dict
        with patch(
            "alleleflux.scripts.summary.p_value_summary.extract_relevant_columns"
        ) as mock_extract:
            mock_extract.return_value = {}
            result = process_results_file(test_file, "lmm", "pre_post")
            self.assertEqual(result, {})

    def test_process_results_file_cmh_with_time_column(self):
        """Test CMH processing assigns group_analyzed from 'time' column when present."""
        cmh_df = pd.DataFrame(
            {
                "mag_id": ["MAG002", "MAG002", "MAG002"],
                "contig": ["c1", "c1", "c2"],
                "position": [10, 20, 30],
                "gene_id": ["g1", "g1", "g2"],
                "time": ["t0", "t1", "t2"],
                "p_value_testA": [0.05, 0.01, 0.2],
            }
        )
        test_file = self.temp_path / "MAG002_cmh_results.tsv.gz"
        cmh_df.to_csv(test_file, sep="\t", compression="gzip", index=False)

        result = process_results_file(test_file, "cmh", "pre_post")
        expected_key = "cmh_pre_post"
        self.assertIn(expected_key, result)
        df = result[expected_key]
        self.assertIn("group_analyzed", df.columns)
        # Ensure group_analyzed matches the original time column values
        # Because p_value columns collapse into min_p_value and rows replicate across sub-tests,
        # we check that the set of values equals the expected set.
        self.assertSetEqual(set(df["group_analyzed"].unique()), {"t0", "t1", "t2"})

    def test_process_results_file_min_p_value_na_dropped(self):
        """Rows with NA min_p_value should be dropped with a warning, not raise."""
        df_with_na = pd.DataFrame(
            {
                "mag_id": ["MAG003", "MAG003"],
                "contig": ["c1", "c1"],
                "position": [1, 2],
                "gene_id": ["g1", "g2"],
                # One valid row, one row of all NAs for p-value cols -> NA min_p_value
                "p_value_testA": [0.05, None],
                "p_value_testB": [0.10, None],
            }
        )
        test_file = self.temp_path / "MAG003_lmm_results.tsv.gz"
        df_with_na.to_csv(test_file, sep="\t", compression="gzip", index=False)
        result = process_results_file(test_file, "lmm", "pre_post")
        key = "lmm_pre_post"
        self.assertIn(key, result)
        df = result[key]
        # Only one non-NA row should remain
        self.assertEqual(
            len(df), 2
        )  # two sub-tests (testA,testB) each from same surviving original row
        self.assertTrue(df["min_p_value"].notna().all())


class TestFindTestFiles(unittest.TestCase):
    """Test that find_test_files selects the correct comparison directories.

    Regression coverage for the group-pair conflation bug: the script used to
    glob ``{test_type}_{timepoint}-*`` which swallowed *every* group-pair sharing
    a timepoint, so each comparison's summary pooled all pairs (e.g. 20_AL's file
    was byte-identical to 1D_AL's). The discovery must be restricted to the
    requested ``groups_label`` so a comparison only sees its own pair.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Two group pairs at the SAME timepoint, plus a different timepoint.
        self.layout = {
            "two_sample_unpaired_5mo_10mo-20_AL": "MAG1_two_sample_unpaired.tsv.gz",
            "two_sample_unpaired_5mo_10mo-1D_AL": "MAG1_two_sample_unpaired.tsv.gz",
            "two_sample_unpaired_5mo_22mo-20_AL": "MAG1_two_sample_unpaired.tsv.gz",
        }
        for dirname, fname in self.layout.items():
            d = self.root / dirname
            d.mkdir(parents=True)
            (d / fname).write_text("stub")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_restricts_to_requested_group_pair(self):
        """With a groups_label, only the matching pair's dir is discovered."""
        result = find_test_files(
            self.root, ["two_sample_unpaired"], "5mo_10mo", groups_label="20_AL"
        )
        found = result["two_sample_unpaired"]
        self.assertEqual(len(found), 1)
        # The single hit must come from the 20_AL dir, not 1D_AL.
        self.assertIn("5mo_10mo-20_AL", str(found[0]))
        self.assertFalse(any("1D_AL" in str(p) for p in found))

    def test_does_not_match_other_timepoints(self):
        """The timepoint is pinned: 5mo_22mo dirs are never returned for 5mo_10mo."""
        result = find_test_files(
            self.root, ["two_sample_unpaired"], "5mo_10mo", groups_label="20_AL"
        )
        found = result["two_sample_unpaired"]
        self.assertFalse(any("5mo_22mo" in str(p) for p in found))

    def test_groups_label_is_required(self):
        """groups_label is mandatory: there is no all-pairs fallback to conflate on."""
        with self.assertRaises(TypeError):
            find_test_files(self.root, ["two_sample_unpaired"], "5mo_10mo")


class TestOutputFilenameIncludesGroups(unittest.TestCase):
    """The written summary filename must carry the group-pair label.

    Now that each summary is one comparison, the filename should be
    self-describing (e.g. ``p_value_summary_two_sample_unpaired_pre_post-fat_control.tsv``)
    rather than only the dir encoding the pair. Kept in lockstep with the rule
    output, dynamic_targets, and dnds_analysis input patterns.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_dir = self.root / "significance_tests"
        comp_dir = self.input_dir / "two_sample_unpaired_pre_post-fat_control"
        comp_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "contig": ["c1", "c1"],
                "position": [100, 200],
                "gene_id": ["g1", "g1"],
                "p_value_tTest": [0.01, 0.5],
            }
        )
        df.to_csv(
            comp_dir / "MAG1_two_sample_unpaired.tsv.gz",
            sep="\t",
            compression="gzip",
            index=False,
        )
        self.outdir = self.root / "out"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_output_filename_contains_group_label(self):
        from alleleflux.scripts.summary import p_value_summary as pvs

        argv = [
            "alleleflux-p-value-summary",
            "--input-dir", str(self.input_dir),
            "--timepoints", "pre_post",
            "--groups", "fat_control",
            "--test-types", "two_sample_unpaired",
            "--outdir", str(self.outdir),
        ]
        with patch("sys.argv", argv):
            pvs.main()

        expected = self.outdir / "p_value_summary_two_sample_unpaired_pre_post-fat_control.tsv"
        written = list(self.outdir.glob("*.tsv"))
        self.assertEqual(
            [p.name for p in written],
            [expected.name],
            f"Expected {expected.name}, got {[p.name for p in written]}",
        )


class TestFdrCorrection(unittest.TestCase):
    """Test FDR-BH correction functionality."""

    def test_fdr_correction_calculation(self):
        """Test that FDR correction is calculated correctly."""
        # Create test p-values
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        alpha = 0.05

        # Calculate expected q-values using statsmodels
        _, expected_q_values, _, _ = multipletests(
            p_values, method="fdr_bh", alpha=alpha
        )

        # Test that our expected values match statsmodels output
        self.assertEqual(len(expected_q_values), len(p_values))
        self.assertTrue(all(q >= p for q, p in zip(expected_q_values, p_values)))

    def test_fdr_correction_edge_cases(self):
        """Test FDR correction with edge cases."""
        # Test with all significant p-values
        all_sig = [0.001, 0.002, 0.003]
        _, q_values_sig, _, _ = multipletests(all_sig, method="fdr_bh", alpha=0.05)
        self.assertTrue(all(q < 0.05 for q in q_values_sig))

        # Test with all non-significant p-values
        all_nonsig = [0.9, 0.95, 0.99]
        _, q_values_nonsig, _, _ = multipletests(
            all_nonsig, method="fdr_bh", alpha=0.05
        )
        self.assertTrue(all(q > 0.05 for q in q_values_nonsig))


if __name__ == "__main__":
    unittest.main()
