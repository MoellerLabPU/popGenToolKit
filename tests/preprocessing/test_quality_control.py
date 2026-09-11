#!/usr/bin/env python3

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from alleleflux.scripts.preprocessing.quality_control import (
    add_subject_count_per_group,
    check_timepoints,
    count_paired_replicates,
    init_worker,
    process_mag_files,
)
from alleleflux.scripts.utilities.qc_metrics import aggregate_mag_stats


class TestQualityControl(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample metadata for testing
        self.metadata_dict = {
            "sample1": {
                "group": "treatment",
                "subjectID": "subject1",
                "replicate": "rep1",
                "time": "T1",
            },
            "sample2": {
                "group": "control",
                "subjectID": "subject1",
                "replicate": "rep1",
                "time": "T2",
            },
            "sample3": {
                "group": "treatment",
                "subjectID": "subject2",
                "replicate": "rep2",
                "time": "T1",
            },
            "sample4": {
                "group": "control",
                "subjectID": "subject2",
                "replicate": "rep2",
                "time": "T2",
            },
        }

        # MAG size dictionary
        self.mag_size_dict = {
            "MAG001": 1000000,  # 1M bp
            "MAG002": 2000000,  # 2M bp
        }

        # Contig length dictionary
        self.contig_length_dict = {
            "contig1": 500000,
            "contig2": 300000,
            "contig3": 200000,
        }

        # Contig to MAG mapping
        self.contig_to_mag = {
            "contig1": "MAG001",
            "contig2": "MAG001",
            "contig3": "MAG002",
        }

        # Pre-inverted MAG -> contigs index (mirrors what main() builds)
        from collections import defaultdict
        mag_to_contigs = defaultdict(set)
        for contig, mid in self.contig_to_mag.items():
            mag_to_contigs[mid].add(contig)

        # Initialize worker globals
        init_worker(
            self.metadata_dict,
            self.mag_size_dict,
            self.contig_length_dict,
            self.contig_to_mag,
            mag_to_contigs,
        )

    def create_temp_profile(self, data):
        """Create a temporary profile file with given data."""
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv")
        df = pd.DataFrame(data)
        df.to_csv(temp_file.name, sep="\t", index=False)
        temp_file.close()
        return temp_file.name


class TestProcessMagFiles(TestQualityControl):

    def test_normal_coverage_calculation(self):
        """Test normal coverage calculation with valid data."""
        # Create test profile data
        profile_data = {
            "contig": ["contig1", "contig1", "contig2", "contig2"],
            "position": [1, 2, 1, 2],
            "gene_id": ["gene1", "gene2", "gene3", "gene4"],
            "total_coverage": [10.0, 20.0, 5.0, 15.0],
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            args = ("sample1", profile_file, "MAG001", 0.1, 1.0)
            result = process_mag_files(args)

            # Check basic structure
            self.assertEqual(result["sample_id"], "sample1")
            self.assertEqual(result["MAG_ID"], "MAG001")
            self.assertEqual(result["genome_size"], 1000000)

            # Check coverage calculations
            self.assertEqual(
                result["breadth"], 4 / 1000000
            )  # 4 positions with coverage >= 1
            self.assertEqual(
                result["average_coverage"], 50.0 / 1000000
            )  # sum of coverage / genome size
            self.assertEqual(
                result["median_coverage"], 12.5
            )  # median of [10, 20, 5, 15]

            # Check median including zeros
            expected_median_with_zeros = np.median(
                np.concatenate([[10.0, 20.0, 5.0, 15.0], np.zeros(1000000 - 4)])
            )
            self.assertEqual(
                result["median_coverage_including_zeros"], expected_median_with_zeros
            )

            # Verify that either threshold passed or a failure reason is provided
            self.assertTrue(
                result["breadth_threshold_passed"] or result["breadth_fail_reason"]
            )

        finally:
            os.unlink(profile_file)

    def test_zero_coverage_warning(self):
        """Test that zero coverage values trigger a warning."""
        profile_data = {
            "contig": ["contig1", "contig1"],
            "position": [1, 2],
            "gene_id": ["gene1", "gene2"],
            "total_coverage": [10.0, 0.0],  # One zero coverage
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            with patch(
                "alleleflux.scripts.preprocessing.quality_control.logger"
            ) as mock_logger:
                args = ("sample1", profile_file, "MAG001", 0.1, 1.0)
                result = process_mag_files(args)

                # Check that warning was logged
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                self.assertIn("positions with zero coverage", warning_call)

                # Check that median_coverage excludes zeros
                self.assertEqual(
                    result["median_coverage"], 10.0
                )  # Only the non-zero value

        finally:
            os.unlink(profile_file)

    def test_mag_filtering(self):
        """Test MAG filtering functionality."""
        profile_data = {
            "contig": ["contig1", "contig2", "contig3"],  # contig3 belongs to MAG002
            "position": [1, 1, 1],
            "gene_id": ["gene1", "gene2", "gene3"],
            "total_coverage": [10.0, 20.0, 30.0],
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            with patch("alleleflux.scripts.utilities.qc_metrics.logger") as mock_logger:
                args = (
                    "sample1",
                    profile_file,
                    "MAG001",
                    0.1,
                    1.0,
                )  # Processing MAG001
                result = process_mag_files(args)

                # Should warn about filtering out contig3
                mock_logger.warning.assert_called()
                warning_call = mock_logger.warning.call_args[0][0]
                self.assertIn("1 rows not belonging to MAG MAG001", warning_call)

                # Should only include contig1 and contig2 in calculations
                self.assertEqual(result["median_coverage"], 15.0)  # median of [10, 20]

        finally:
            os.unlink(profile_file)

    def test_invalid_mag_size(self):
        """Test handling of invalid MAG sizes."""
        profile_data = {
            "contig": ["contig1"],
            "position": [1],
            "gene_id": ["gene1"],
            "total_coverage": [10.0],
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            args = ("sample1", profile_file, "INVALID_MAG", 0.1, 1.0)
            result = process_mag_files(args)

            # Should fail with appropriate error message
            self.assertFalse(result["breadth_threshold_passed"])
            self.assertIn(
                "MAG size for INVALID_MAG not found", result["breadth_fail_reason"]
            )

        finally:
            os.unlink(profile_file)

    def test_length_weighted_coverage(self):
        """Test length-weighted coverage calculation."""
        profile_data = {
            "contig": ["contig1", "contig1", "contig2", "contig2"],
            "position": [1, 2, 1, 2],
            "gene_id": ["gene1", "gene2", "gene3", "gene4"],
            "total_coverage": [
                10.0,
                20.0,
                30.0,
                40.0,
            ],  # contig1: 30 total, contig2: 70 total
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            args = ("sample1", profile_file, "MAG001", 0.1, 1.0)
            result = process_mag_files(args)

            # contig1: mean_coverage = 30/500000, contig2: mean_coverage = 70/300000
            # length_weighted = (30 + 70) / mag_size = 100 / 1000000
            expected_lw_coverage = (30 + 70) / 1000000
            self.assertAlmostEqual(
                result["length_weighted_coverage"], expected_lw_coverage
            )

        finally:
            os.unlink(profile_file)


class TestCheckTimepoints(TestQualityControl):

    def test_single_data_type(self):
        """Test single data type skips timepoint checks."""
        df = pd.DataFrame(
            {
                "subjectID": ["subject1", "subject2"],
                "coverage_threshold_passed": [True, False],
            }
        )

        result = check_timepoints(df, "single")

        # Should match coverage_threshold_passed
        self.assertTrue(result.loc[0, "two_timepoints_passed"])
        self.assertFalse(result.loc[1, "two_timepoints_passed"])

    def test_longitudinal_valid_timepoints(self):
        """Test longitudinal data with valid timepoints."""
        df = pd.DataFrame(
            {
                "subjectID": ["subject1", "subject1", "subject2", "subject2"],
                "time": ["T1", "T2", "T1", "T2"],
                "coverage_threshold_passed": [True, True, True, True],
            }
        )

        result = check_timepoints(df, "longitudinal")

        # All should pass since both subjects have 2 timepoints
        self.assertTrue(all(result["two_timepoints_passed"]))

    def test_longitudinal_missing_timepoint(self):
        """Test longitudinal data with missing timepoints."""
        df = pd.DataFrame(
            {
                "subjectID": [
                    "subject1",
                    "subject1",
                    "subject2",
                ],  # subject2 missing T2
                "time": ["T1", "T2", "T1"],
                "coverage_threshold_passed": [True, True, True],
            }
        )

        result = check_timepoints(df, "longitudinal")

        # Only subject1 should pass (has both timepoints)
        subject1_mask = result["subjectID"] == "subject1"
        subject2_mask = result["subjectID"] == "subject2"

        self.assertTrue(all(result.loc[subject1_mask, "two_timepoints_passed"]))
        self.assertFalse(any(result.loc[subject2_mask, "two_timepoints_passed"]))

    def test_too_many_timepoints(self):
        """Test error when more than 2 timepoints found."""
        df = pd.DataFrame(
            {
                "subjectID": ["subject1", "subject1", "subject1"],
                "time": ["T1", "T2", "T3"],  # 3 timepoints
                "coverage_threshold_passed": [True, True, True],
            }
        )

        with self.assertRaises(ValueError) as context:
            check_timepoints(df, "longitudinal")

        self.assertIn("More than 2 unique timepoints", str(context.exception))


class TestSubjectCounting(TestQualityControl):

    def test_add_subject_count_per_group(self):
        """Test subject and replicate counting per group."""
        df = pd.DataFrame(
            {
                "two_timepoints_passed": [True, True, True, False],
                "group": ["treatment", "treatment", "control", "control"],
                "subjectID": ["subject1", "subject2", "subject1", "subject3"],
                "replicate": ["rep1", "rep2", "rep1", "rep3"],
            }
        )

        result = add_subject_count_per_group(df)

        # Treatment group: 2 subjects, 2 replicates
        treatment_mask = (result["group"] == "treatment") & result[
            "two_timepoints_passed"
        ]
        self.assertTrue(all(result.loc[treatment_mask, "subjects_per_group"] == 2))
        self.assertTrue(all(result.loc[treatment_mask, "replicates_per_group"] == 2))

        # Control group: 1 subject, 1 replicate (subject3 failed timepoints)
        control_mask = (result["group"] == "control") & result["two_timepoints_passed"]
        self.assertTrue(all(result.loc[control_mask, "subjects_per_group"] == 1))
        self.assertTrue(all(result.loc[control_mask, "replicates_per_group"] == 1))

        # Failed samples should have NaN
        failed_mask = ~result["two_timepoints_passed"]
        self.assertTrue(all(result.loc[failed_mask, "subjects_per_group"].isna()))

    def test_count_paired_replicates(self):
        """Test paired replicate counting."""
        df = pd.DataFrame(
            {
                "two_timepoints_passed": [True, True, True, True, True],
                "group": ["treatment", "control", "treatment", "control", "treatment"],
                "replicate": [
                    "rep1",
                    "rep1",
                    "rep2",
                    "rep2",
                    "rep3",
                ],  # rep1 and rep2 are paired
            }
        )

        result = count_paired_replicates(df)

        # rep1 and rep2 appear in both groups, so 2 paired replicates per group
        paired_mask = result["replicate"].isin(["rep1", "rep2"])
        self.assertTrue(
            all(result.loc[paired_mask, "paired_replicates_per_group"] == 2)
        )

        # rep3 only in treatment, should be NaN
        unpaired_mask = result["replicate"] == "rep3"
        self.assertTrue(
            all(result.loc[unpaired_mask, "paired_replicates_per_group"].isna())
        )


class TestAggregation(TestQualityControl):

    def test_aggregate_mag_stats_group(self):
        """Test MAG statistics aggregation by group."""
        df_results = pd.DataFrame(
            {
                "sample_id": [
                    "sample1",
                    "sample2",
                    "sample3",
                    "sample4",
                ],  # Add missing column
                "breadth_threshold_passed": [True, True, True, True],
                "group": ["treatment", "treatment", "control", "control"],
                "breadth": [0.8, 0.9, 0.7, 0.6],
                "average_coverage": [50.0, 60.0, 40.0, 30.0],
                "median_coverage": [45.0, 55.0, 35.0, 25.0],
                "median_coverage_including_zeros": [0.1, 0.2, 0.05, 0.03],
                "coverage_std": [15.0, 20.0, 12.0, 8.0],
                "coverage_std_including_zeros": [25.0, 30.0, 20.0, 15.0],
                "length_weighted_coverage": [48.0, 58.0, 38.0, 28.0],
                "subjects_per_group": [2, 2, 2, 2],
                "replicates_per_group": [2, 2, 2, 2],
                "paired_replicates_per_group": [1, 1, 1, 1],
            }
        )

        result = aggregate_mag_stats(df_results, "MAG001", ["group"], overall=False)

        # Should have 2 rows (treatment and control)
        self.assertEqual(len(result), 2)
        self.assertIn("MAG_ID", result.columns)
        self.assertTrue(all(result["MAG_ID"] == "MAG001"))

        # Check treatment group means
        treatment_row = result[result["group"] == "treatment"].iloc[0]
        self.assertAlmostEqual(treatment_row["breadth_mean"], 0.85)  # (0.8 + 0.9) / 2
        self.assertAlmostEqual(
            treatment_row["average_coverage_mean"], 55.0
        )  # (50 + 60) / 2

    def test_aggregate_includes_all_metrics(self):
        """Test that all metrics including breadth_genome are aggregated."""
        # Create sample data with breadth_genome field (from positions-based QC)
        df_results = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2"],
                "coverage_threshold_passed": [True, True],
                "group": ["treatment", "treatment"],
                "breadth": [0.8, 0.9],
                "breadth_genome": [
                    0.0001,
                    0.0002,
                ],  # Should be included in aggregation
                "average_coverage": [50.0, 60.0],
                "median_coverage": [45.0, 55.0],
                "median_coverage_including_zeros": [0.1, 0.2],
                "coverage_std": [15.0, 20.0],
                "coverage_std_including_zeros": [25.0, 30.0],
                "subjects_per_group": [2, 2],
                "replicates_per_group": [2, 2],
                "paired_replicates_per_group": [1, 1],
            }
        )

        # Test group aggregation
        result = aggregate_mag_stats(df_results, "MAG001", ["group"], overall=False)

        # Both breadth metrics should be aggregated
        self.assertIn("breadth_mean", result.columns)
        self.assertIn("breadth_genome_mean", result.columns)

        # Verify breadth_genome aggregation
        self.assertAlmostEqual(result.iloc[0]["breadth_genome_mean"], 0.00015)

        # Test overall aggregation
        result_overall = aggregate_mag_stats(df_results, "MAG001", [], overall=True)

        # Both breadth metrics should be in overall results too
        self.assertIn("breadth_mean", result_overall.columns)
        self.assertIn("breadth_genome_mean", result_overall.columns)

    def test_aggregate_mag_stats_overall(self):
        """Test overall MAG statistics aggregation."""
        df_results = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2"],  # Add missing column
                "breadth_threshold_passed": [True, True],
                "breadth": [0.8, 0.6],
                "average_coverage": [50.0, 30.0],
                "median_coverage": [45.0, 25.0],
                "median_coverage_including_zeros": [0.1, 0.03],
                "coverage_std": [15.0, 8.0],
                "coverage_std_including_zeros": [25.0, 15.0],
                "length_weighted_coverage": [48.0, 28.0],
                "subjects_per_group": [2, 2],
                "replicates_per_group": [2, 2],
                "paired_replicates_per_group": [1, 1],
            }
        )

        result = aggregate_mag_stats(df_results, "MAG001", [], overall=True)

        # Should have 1 row with _mean suffix columns
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["MAG_ID"], "MAG001")
        self.assertEqual(result.iloc[0]["num_samples"], 2)
        self.assertAlmostEqual(result.iloc[0]["breadth_mean"], 0.7)  # (0.8 + 0.6) / 2
        self.assertAlmostEqual(
            result.iloc[0]["average_coverage_mean"], 40.0
        )  # (50 + 30) / 2

    def test_aggregate_empty_results(self):
        """Test aggregation with empty results."""
        df_results = pd.DataFrame()

        result = aggregate_mag_stats(df_results, "MAG001", ["group"], overall=False)

        # Should return None for empty input
        self.assertIsNone(result)


class TestEdgeCases(TestQualityControl):

    def test_empty_profile_after_filtering(self):
        """Test handling when profile is empty after MAG filtering."""
        profile_data = {
            "contig": ["contig3"],  # Only contig3, which belongs to MAG002
            "position": [1],
            "gene_id": ["gene1"],
            "total_coverage": [10.0],
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            args = ("sample1", profile_file, "MAG001", 0.1, 1.0)  # Processing MAG001
            result = process_mag_files(args)

            # Should fail because no rows remain after filtering
            self.assertFalse(result["breadth_threshold_passed"])
            self.assertIn(
                "No rows remain after filtering", result["breadth_fail_reason"]
            )

        finally:
            os.unlink(profile_file)

    def test_all_zero_coverage(self):
        """Test profile with all zero coverage values."""
        profile_data = {
            "contig": ["contig1", "contig1"],
            "position": [1, 2],
            "gene_id": ["gene1", "gene2"],
            "total_coverage": [0.0, 0.0],
        }

        profile_file = self.create_temp_profile(profile_data)

        try:
            args = ("sample1", profile_file, "MAG001", 0.1, 1.0)
            result = process_mag_files(args)

            # median_coverage should be NaN (no positive coverage)
            self.assertTrue(np.isnan(result["median_coverage"]))
            self.assertTrue(np.isnan(result["coverage_std"]))

            # But median_coverage_including_zeros should be 0.0
            self.assertEqual(result["median_coverage_including_zeros"], 0.0)

        finally:
            os.unlink(profile_file)

    def test_no_valid_subjects(self):
        """Test subject counting with no valid subjects."""
        df = pd.DataFrame(
            {
                "two_timepoints_passed": [False, False],
                "group": ["treatment", "control"],
                "subjectID": ["subject1", "subject2"],
                "replicate": ["rep1", "rep2"],
            }
        )

        result = add_subject_count_per_group(df)

        # All should be NaN since no subjects passed
        self.assertTrue(all(result["subjects_per_group"].isna()))
        self.assertTrue(all(result["replicates_per_group"].isna()))

    def test_single_data_type_aggregation(self):
        """Test that aggregation works correctly when data_type='single'."""
        # Create sample data without 'time' column (single timepoint)
        df_results = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2", "sample3"],
                "breadth_threshold_passed": [True, True, True],
                "group": ["treatment", "treatment", "control"],
                "breadth": [0.8, 0.9, 0.7],
                "average_coverage": [50.0, 60.0, 40.0],
                "median_coverage": [45.0, 55.0, 35.0],
                "median_coverage_including_zeros": [0.1, 0.2, 0.05],
                "coverage_std": [15.0, 20.0, 12.0],
                "coverage_std_including_zeros": [25.0, 30.0, 20.0],
                "length_weighted_coverage": [48.0, 58.0, 38.0],
                "subjects_per_group": [1, 1, 1],
                "replicates_per_group": [1, 1, 1],
                "paired_replicates_per_group": [1, 1, 1],
                # Note: NO 'time' column for single data type
            }
        )

        # Test group summary
        group_summary = aggregate_mag_stats(
            df_results, "MAG001", ["group"], overall=False
        )
        self.assertIsNotNone(group_summary)
        self.assertEqual(len(group_summary), 2)  # treatment and control
        # Verify _mean suffix is added to metric columns
        self.assertIn("breadth_mean", group_summary.columns)
        self.assertIn("average_coverage_mean", group_summary.columns)

        # Test overall summary (should work)
        overall_summary = aggregate_mag_stats(df_results, "MAG001", [], overall=True)
        self.assertIsNotNone(overall_summary)
        self.assertEqual(len(overall_summary), 1)

        # Test group-time summary (should return None since no 'time' column)
        # This simulates the check: if "time" in df_results.columns
        if "time" in df_results.columns:
            group_time_summary = aggregate_mag_stats(
                df_results, "MAG001", ["group", "time"], overall=False
            )
        else:
            group_time_summary = None

        self.assertIsNone(group_time_summary)  # Should be None for single data type

    def test_single_data_type_end_to_end(self):
        """Test complete flow with single data type from check_timepoints to aggregation."""
        # Start with data that simulates single data type processing
        df_results = pd.DataFrame(
            {
                "sample_id": ["sample1", "sample2"],
                "subjectID": ["subject1", "subject2"],
                "group": ["treatment", "control"],
                "replicate": ["rep1", "rep2"],
                "coverage_threshold_passed": [True, True],
                "breadth": [0.8, 0.7],
                "average_coverage": [50.0, 40.0],
                "median_coverage": [45.0, 35.0],
                "median_coverage_including_zeros": [0.1, 0.05],
                "coverage_std": [15.0, 12.0],
                "coverage_std_including_zeros": [25.0, 20.0],
                "length_weighted_coverage": [48.0, 38.0],
                # Note: NO 'time' column for single data type
            }
        )

        # Process through check_timepoints with data_type='single'
        df_after_timepoints = check_timepoints(df_results, "single")

        # Should have two_timepoints_passed matching coverage_threshold_passed
        self.assertTrue(
            all(
                df_after_timepoints["two_timepoints_passed"]
                == df_after_timepoints["coverage_threshold_passed"]
            )
        )

        # Process through subject counting
        df_with_counts = add_subject_count_per_group(df_after_timepoints)

        # Process through replicate counting
        df_final = count_paired_replicates(df_with_counts)

        # Test aggregation (simulating the logic in process_mag)
        group_summary = aggregate_mag_stats(
            df_final, "MAG001", ["group"], overall=False
        )
        overall_summary = aggregate_mag_stats(df_final, "MAG001", [], overall=True)

        # Group-time summary should be None (no time column)
        group_time_summary = None
        if "time" in df_final.columns:
            group_time_summary = aggregate_mag_stats(
                df_final, "MAG001", ["group", "time"], overall=False
            )

        # Verify results
        self.assertIsNotNone(group_summary)
        self.assertIsNotNone(overall_summary)
        self.assertIsNone(group_time_summary)  # Should be None for single data type

        # Verify group summary structure
        self.assertEqual(len(group_summary), 2)  # treatment and control
        self.assertTrue(all(group_summary["MAG_ID"] == "MAG001"))
        self.assertIn("breadth_mean", group_summary.columns)
        self.assertIn("average_coverage_mean", group_summary.columns)

        # Verify overall summary structure
        self.assertEqual(len(overall_summary), 1)
        self.assertEqual(overall_summary.iloc[0]["MAG_ID"], "MAG001")


if __name__ == "__main__":
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
