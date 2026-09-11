#!/usr/bin/env python3
"""
Test suite for allele frequency trajectory plotting.

Focuses on the per-replicate combined line plots and the shared helpers they reuse,
plus regression coverage ensuring the pooled (all-subjects) plot is unaffected.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless rendering for tests

import pandas as pd

from alleleflux.scripts.visualization.plot_allele_trajectory import (
    apply_min_samples_filter,
    compute_group_means,
    plot_combined,
    plot_combined_per_replicate,
    plot_group_distributions,
)


def make_long_df(
    replicate_frequencies,
    days=(0, 1),
    value_col="min_p_value",
    mag_id="MAGTEST",
):
    """
    Build a minimal long-format frequency table.

    Args:
        replicate_frequencies: dict of {replicate: {group: [freq_per_subject, ...]}}.
                               Each frequency is held constant across days for that subject,
                               which makes expected means trivial to assert.
        days: day values assigned to every subject.
        value_col: name of the ranking column.
        mag_id: MAG identifier written into the table.

    Returns:
        pd.DataFrame in the long format expected by the plotting entrypoints.
    """
    rows = []
    subject_counter = 0
    for replicate, groups in replicate_frequencies.items():
        for group, freqs in groups.items():
            for freq in freqs:
                subject_counter += 1
                subject = f"S{subject_counter}"
                for day in days:
                    rows.append(
                        {
                            "mag_id": mag_id,
                            "contig": "contig_1",
                            "position": 100,
                            "anchor_allele": "A",
                            value_col: 0.001,
                            "sample_id": f"{subject}_d{day}",
                            "frequency": freq,
                            "group": group,
                            "subjectID": subject,
                            "day": day,
                            "replicate": replicate,
                        }
                    )
    return pd.DataFrame(rows)


class TestApplyMinSamplesFilter(unittest.TestCase):
    """The sparse-point filter extracted from get_plotting_data."""

    def setUp(self):
        self.mean_df = pd.DataFrame(
            {
                "day": [0, 1, 2],
                "group": ["fat", "fat", "fat"],
                "contig": ["c", "c", "c"],
                "position": [1, 1, 1],
                "anchor_allele": ["A", "A", "A"],
                "frequency": [0.1, 0.2, 0.3],
                "valid_count": [1, 2, 3],
            }
        )

    def test_threshold_of_one_is_a_no_op(self):
        """A threshold of 1 (or 0) must not drop anything."""
        result = apply_min_samples_filter(self.mean_df, "day", 1)
        pd.testing.assert_frame_equal(result, self.mean_df)

    def test_drops_points_below_threshold(self):
        """Points backed by fewer than N samples are removed."""
        result = apply_min_samples_filter(self.mean_df, "day", 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(result["valid_count"]), [2, 3])

    def test_returns_empty_when_all_dropped(self):
        """An impossible threshold yields an empty frame rather than an error."""
        result = apply_min_samples_filter(self.mean_df, "day", 99)
        self.assertTrue(result.empty)


class TestValidationPhase(unittest.TestCase):
    """Guards at the top of plot_group_distributions, before any expensive work."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.df = make_long_df({1: {"fat": [0.2], "control": [0.8]}})

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run(self, df, **kwargs):
        return plot_group_distributions(
            df=df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["line"],
            per_site=False,
            output_dir=self.temp_dir,
            output_format="png",
            **kwargs,
        )

    def test_duplicate_rows_raise(self):
        """A repeated (site, sample) row is a data-integrity error, not something to plot.

        Without this guard the duplicate silently skews the group mean and inflates
        valid_count, which can then defeat min_samples_per_bin.
        """
        dup_df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)

        with self.assertRaises(ValueError) as ctx:
            self._run(dup_df)

        self.assertIn("Duplicate entries found", str(ctx.exception))

    def test_clean_input_does_not_raise(self):
        """The guard must not fire on well-formed input."""
        self._run(self.df)
        self.assertTrue(
            (Path(self.temp_dir) / "MAGTEST_combined_nALL_day_line.png").exists()
        )


class TestPlotCombinedBackwardCompatibility(unittest.TestCase):
    """The pooled plot must be untouched by the per-replicate additions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_filename_has_no_suffix(self):
        """Without label_suffix the historical filename is preserved exactly."""
        df = make_long_df({1: {"fat": [0.2], "control": [0.8]}})
        mean_df = compute_group_means(df, "day")

        plot_combined(
            mean_df, "day", "10", "min_p_value", self.temp_dir, "png", "line", "MAGTEST"
        )

        expected = Path(self.temp_dir) / "MAGTEST_combined_n10_day_line.png"
        self.assertTrue(expected.exists(), f"expected {expected.name} to be written")

    def test_label_suffix_appends_before_extension(self):
        """The suffix lands before the extension, not after."""
        df = make_long_df({1: {"fat": [0.2], "control": [0.8]}})
        mean_df = compute_group_means(df, "day")

        plot_combined(
            mean_df,
            "day",
            "10",
            "min_p_value",
            self.temp_dir,
            "png",
            "line",
            "MAGTEST",
            label_suffix="_rep7",
        )

        expected = Path(self.temp_dir) / "MAGTEST_combined_n10_day_line_rep7.png"
        self.assertTrue(expected.exists(), f"expected {expected.name} to be written")


class TestPlotCombinedPerReplicate(unittest.TestCase):
    """One combined line plot per replicate, averaged within the replicate."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Replicate 1 and 2 have deliberately different frequencies so that a pooled mean
        # (0.5 everywhere) is distinguishable from correct within-replicate means.
        self.df = make_long_df(
            {
                1: {"fat": [0.1, 0.3], "control": [0.1, 0.3]},
                2: {"fat": [0.7, 0.9], "control": [0.7, 0.9]},
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_writes_one_file_per_replicate(self):
        """Each replicate gets its own file in a by_replicate/ subdirectory."""
        plot_combined_per_replicate(
            self.df, "day", "10", "min_p_value", self.temp_dir, "png", mag_id="MAGTEST"
        )

        outdir = Path(self.temp_dir) / "by_replicate"
        written = sorted(p.name for p in outdir.glob("*.png"))
        self.assertEqual(
            written,
            [
                "MAGTEST_combined_n10_day_line_rep1.png",
                "MAGTEST_combined_n10_day_line_rep2.png",
            ],
        )

    def test_means_are_computed_within_replicate_not_pooled(self):
        """The core guarantee: replicate 1 plots 0.2, replicate 2 plots 0.8, never 0.5."""
        with patch(
            "alleleflux.scripts.visualization.plot_allele_trajectory.plot_combined"
        ) as mock_plot:
            plot_combined_per_replicate(
                self.df,
                "day",
                "10",
                "min_p_value",
                self.temp_dir,
                "png",
                mag_id="MAGTEST",
            )

        self.assertEqual(mock_plot.call_count, 2)

        by_suffix = {
            call.kwargs["label_suffix"]: call.args[0] for call in mock_plot.call_args_list
        }

        rep1_freqs = set(by_suffix["_rep1"]["frequency"].round(6))
        rep2_freqs = set(by_suffix["_rep2"]["frequency"].round(6))

        self.assertEqual(rep1_freqs, {0.2}, "replicate 1 mean should be (0.1+0.3)/2")
        self.assertEqual(rep2_freqs, {0.8}, "replicate 2 mean should be (0.7+0.9)/2")
        self.assertNotIn(0.5, rep1_freqs | rep2_freqs, "pooled mean must not appear")

    def test_group_order_and_xlim_shared_across_replicates(self):
        """Every figure gets the same group ordering and the same x-limits."""
        with patch(
            "alleleflux.scripts.visualization.plot_allele_trajectory.plot_combined"
        ) as mock_plot:
            plot_combined_per_replicate(
                self.df,
                "day",
                "10",
                "min_p_value",
                self.temp_dir,
                "png",
                mag_id="MAGTEST",
            )

        group_orders = [c.kwargs["group_order"] for c in mock_plot.call_args_list]
        xlims = [c.kwargs["xlim"] for c in mock_plot.call_args_list]

        self.assertEqual(group_orders, [["control", "fat"], ["control", "fat"]])
        self.assertEqual(len(set(xlims)), 1, "all replicates must share one x-range")
        left, right = xlims[0]
        self.assertLess(left, 0, "x-limits should pad below the minimum day")
        self.assertGreater(right, 1, "x-limits should pad above the maximum day")

    def test_share_x_disabled_leaves_limits_unset(self):
        """With share_x=False each figure autoscales."""
        with patch(
            "alleleflux.scripts.visualization.plot_allele_trajectory.plot_combined"
        ) as mock_plot:
            plot_combined_per_replicate(
                self.df,
                "day",
                "10",
                "min_p_value",
                self.temp_dir,
                "png",
                mag_id="MAGTEST",
                share_x=False,
            )

        self.assertTrue(all(c.kwargs["xlim"] is None for c in mock_plot.call_args_list))

    def test_min_samples_filter_applies_within_replicate(self):
        """A group with a single subject in a replicate is dropped at threshold 2."""
        df = make_long_df(
            {
                1: {"fat": [0.1, 0.3], "control": [0.5]},  # control has one subject
                2: {"fat": [0.7, 0.9], "control": [0.5, 0.5]},
            }
        )

        with patch(
            "alleleflux.scripts.visualization.plot_allele_trajectory.plot_combined"
        ) as mock_plot:
            plot_combined_per_replicate(
                df,
                "day",
                "10",
                "min_p_value",
                self.temp_dir,
                "png",
                mag_id="MAGTEST",
                min_samples_per_bin=2,
            )

        by_suffix = {
            call.kwargs["label_suffix"]: call.args[0] for call in mock_plot.call_args_list
        }

        self.assertEqual(
            set(by_suffix["_rep1"]["group"]),
            {"fat"},
            "single-subject control group should be filtered out of replicate 1",
        )
        self.assertEqual(
            set(by_suffix["_rep2"]["group"]),
            {"control", "fat"},
            "replicate 2 has two subjects per group and should keep both",
        )

    def test_missing_replicate_column_raises(self):
        """A missing replicate column is a hard error, not a silent skip."""
        df = self.df.drop(columns=["replicate"])

        with self.assertRaises(ValueError) as ctx:
            plot_combined_per_replicate(
                df, "day", "10", "min_p_value", self.temp_dir, "png", mag_id="MAGTEST"
            )

        self.assertIn("replicate", str(ctx.exception))

    def test_empty_dataframe_returns_without_writing(self):
        """An empty input warns and writes nothing rather than raising."""
        plot_combined_per_replicate(
            self.df.iloc[:0],
            "day",
            "10",
            "min_p_value",
            self.temp_dir,
            "png",
            mag_id="MAGTEST",
        )
        self.assertFalse((Path(self.temp_dir) / "by_replicate").exists())


class TestPlotGroupDistributionsIntegration(unittest.TestCase):
    """End-to-end behaviour through the orchestrator."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.df = make_long_df(
            {
                1: {"fat": [0.1, 0.3], "control": [0.1, 0.3]},
                2: {"fat": [0.7, 0.9], "control": [0.7, 0.9]},
            }
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pooled_and_per_replicate_plots_both_written(self):
        """Enabling the flag adds replicate figures without removing the pooled one."""
        plot_group_distributions(
            df=self.df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["line"],
            per_site=False,
            output_dir=self.temp_dir,
            output_format="png",
            combined_per_replicate=True,
        )

        pooled = Path(self.temp_dir) / "MAGTEST_combined_nALL_day_line.png"
        rep_dir = Path(self.temp_dir) / "by_replicate"

        self.assertTrue(pooled.exists(), "pooled combined plot should still be written")
        self.assertEqual(
            sorted(p.name for p in rep_dir.glob("*.png")),
            [
                "MAGTEST_combined_nALL_day_line_rep1.png",
                "MAGTEST_combined_nALL_day_line_rep2.png",
            ],
        )

    def test_disabled_by_default(self):
        """Without the flag, no by_replicate directory is created."""
        plot_group_distributions(
            df=self.df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["line"],
            per_site=False,
            output_dir=self.temp_dir,
            output_format="png",
        )

        self.assertFalse((Path(self.temp_dir) / "by_replicate").exists())

    def test_missing_replicate_column_fails_early(self):
        """The orchestrator validates before doing any expensive filtering."""
        df = self.df.drop(columns=["replicate"])

        with self.assertRaises(ValueError) as ctx:
            plot_group_distributions(
                df=df,
                value_col="min_p_value",
                n_line="all",
                n_dist="all",
                x_col="day",
                plot_types=["line"],
                per_site=False,
                output_dir=self.temp_dir,
                output_format="png",
                combined_per_replicate=True,
            )

        self.assertIn("replicate", str(ctx.exception))

    def test_box_plots_do_not_trigger_replicate_figures(self):
        """Per-replicate output is scoped to line plots only."""
        plot_group_distributions(
            df=self.df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["box"],
            per_site=False,
            output_dir=self.temp_dir,
            output_format="png",
            combined_per_replicate=True,
        )

        self.assertFalse((Path(self.temp_dir) / "by_replicate").exists())


class TestBinWidthFilenameTag(unittest.TestCase):
    """Binned outputs must carry the bin width in the filename (e.g. _bin10d)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Days spanning two 10-day bins (0-9 and 10-19) so binned line plots
        # have two x-points and the binning path is genuinely exercised.
        self.df = make_long_df(
            {
                1: {"fat": [0.1, 0.3], "control": [0.1, 0.3]},
                2: {"fat": [0.7, 0.9], "control": [0.7, 0.9]},
            },
            days=(0, 12),
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_binned_filenames_carry_bin_tag(self):
        """Combined, per-replicate, and per-site filenames all embed the width."""
        plot_group_distributions(
            df=self.df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["line"],
            per_site=True,
            output_dir=self.temp_dir,
            output_format="png",
            bin_width_days=10,
            combined_per_replicate=True,
        )

        pooled = Path(self.temp_dir) / "MAGTEST_combined_nALL_bin_midpoint_line_bin10d.png"
        self.assertTrue(pooled.exists(), f"expected {pooled.name} to be written")

        rep_dir = Path(self.temp_dir) / "by_replicate"
        self.assertEqual(
            sorted(p.name for p in rep_dir.glob("*.png")),
            [
                "MAGTEST_combined_nALL_bin_midpoint_line_bin10d_rep1.png",
                "MAGTEST_combined_nALL_bin_midpoint_line_bin10d_rep2.png",
            ],
        )

        site_files = list((Path(self.temp_dir) / "single_sites").glob("*.png"))
        self.assertTrue(site_files, "expected per-site plots to be written")
        for f in site_files:
            self.assertIn("_bin10d", f.name, f"{f.name} lacks the bin tag")

    def test_unbinned_filenames_have_no_tag(self):
        """Without binning, filenames keep their historical form (no _bin tag)."""
        plot_group_distributions(
            df=self.df,
            value_col="min_p_value",
            n_line="all",
            n_dist="all",
            x_col="day",
            plot_types=["line"],
            per_site=False,
            output_dir=self.temp_dir,
            output_format="png",
        )

        pooled = Path(self.temp_dir) / "MAGTEST_combined_nALL_day_line.png"
        self.assertTrue(pooled.exists(), f"expected {pooled.name} to be written")


class TestCliMultipleBinWidths(unittest.TestCase):
    """--bin_width_days accepts several widths; each gets an isolated DataFrame."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_tsv = Path(self.temp_dir) / "long.tsv"
        make_long_df(
            {1: {"fat": [0.1, 0.3], "control": [0.1, 0.3]}}, days=(0, 12)
        ).to_csv(self.input_tsv, sep="\t", index=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_main(self, extra_argv):
        """Invoke main() with a stubbed orchestrator; return its recorded calls."""
        argv = [
            "alleleflux-plot-trajectories",
            "--input_file",
            str(self.input_tsv),
            "--x_col",
            "day",
            "--output_dir",
            self.temp_dir,
        ] + extra_argv
        with patch(
            "alleleflux.scripts.visualization.plot_allele_trajectory.plot_group_distributions"
        ) as mock_plot, patch("sys.argv", argv):
            from alleleflux.scripts.visualization.plot_allele_trajectory import main

            main()
        return mock_plot.call_args_list

    def test_one_orchestrator_call_per_width(self):
        calls = self._run_main(["--bin_width_days", "10", "20"])
        self.assertEqual([c.kwargs["bin_width_days"] for c in calls], [10, 20])

    def test_each_width_gets_its_own_dataframe_copy(self):
        """The orchestrator mutates and row-filters its input (binning columns,
        initial-frequency filter), so widths sharing one object would corrupt
        each other's site selection."""
        calls = self._run_main(["--bin_width_days", "10", "20"])
        df_ids = [id(c.kwargs["df"]) for c in calls]
        self.assertEqual(len(set(df_ids)), len(df_ids), "widths share a DataFrame object")

    def test_single_width_still_works(self):
        calls = self._run_main(["--bin_width_days", "10"])
        self.assertEqual([c.kwargs["bin_width_days"] for c in calls], [10])

    def test_no_width_passes_none(self):
        calls = self._run_main([])
        self.assertEqual([c.kwargs["bin_width_days"] for c in calls], [None])


if __name__ == "__main__":
    unittest.main()
