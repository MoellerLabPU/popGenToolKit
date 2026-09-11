"""Tests for the pair-comparison engine and genome-level aggregation.

Dense arrays here are built exactly like ``profile_io.dense_contig_counts`` builds
them: ``(contig_length, 4)`` uint16 in profile order A,C,G,T, zeros where no reads
landed.  The real worked-example rows (SLG1007 vs SLG1102) reappear as one contig
so the engine-level counts can be checked by hand.
"""
import unittest

import numpy as np

from alleleflux.scripts.analysis.ani.engine import (
    PAIR_COLUMNS,
    SNP_LOCATION_COLUMNS,
    compare_all_pairs,
    compare_pair_on_contig,
)
from alleleflux.scripts.analysis.ani.null_model import build_error_model

MODEL = build_error_model(min_base_quality=30, fdr=1e-6)
MIN_FREQ = 0.05


def _dense(rows, length):
    """rows: {position: (A, C, G, T)} -> a (length, 4) uint16 array with gaps at 0."""
    arr = np.zeros((length, 4), dtype=np.uint16)
    for pos, counts in rows.items():
        arr[pos] = counts
    return arr


class TestComparePairOnContig(unittest.TestCase):
    def test_positions_below_min_cov_in_either_sample_are_not_compared(self):
        a = _dense({0: (10, 0, 0, 0), 1: (4, 0, 0, 0)}, 3)
        b = _dense({0: (10, 0, 0, 0), 1: (10, 0, 0, 0)}, 3)
        got = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=5)
        # Position 0 qualifies; position 1 is 4x in sample A; position 2 is empty.
        self.assertEqual(got["compared"], 1)
        # Covered in either: positions 0 and 1 (B has 10x at position 1).
        self.assertEqual(got["covered_either"], 2)
        self.assertEqual(got["consensus_snps"], 0)

    def test_counts_consensus_and_population_snps(self):
        """The two real worked-example rows plus one boring identical position."""
        a = _dense({0: (2, 0, 28, 0), 1: (0, 9, 0, 10), 2: (20, 0, 0, 0)}, 3)
        b = _dense({0: (5, 0, 1, 0), 1: (0, 8, 0, 6), 2: (20, 0, 0, 0)}, 3)
        got = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=5)
        self.assertEqual(got["compared"], 3)
        # Position 0 is a fixed difference; position 1 differs only at the majority.
        self.assertEqual(got["consensus_snps"], 2)
        self.assertEqual(got["population_snps"], 1)

    def test_empty_overlap_is_reported_not_crashed(self):
        a = _dense({0: (10, 0, 0, 0)}, 2)
        b = _dense({1: (10, 0, 0, 0)}, 2)
        got = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=5)
        self.assertEqual(got["compared"], 0)
        self.assertEqual(got["consensus_snps"], 0)
        self.assertIsNone(got["snp_positions"])

    def test_min_cov_one_turns_single_reads_into_alleles(self):
        """The toggle, and why its default is 5: at min_cov 1 two lone sequencing
        errors face each other and become a 'fixed difference' (the spec section 1.7
        disaster); at the default the position is simply not compared."""
        a = _dense({0: (1, 0, 0, 0)}, 2)
        b = _dense({0: (0, 0, 0, 1)}, 2)
        loose = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=1)
        self.assertEqual(loose["compared"], 1)
        self.assertEqual(loose["population_snps"], 1)
        strict = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=5)
        self.assertEqual(strict["compared"], 0)

    def test_want_locations_returns_flagged_positions_with_counts(self):
        a = _dense({0: (2, 0, 28, 0), 2: (20, 0, 0, 0)}, 3)
        b = _dense({0: (5, 0, 1, 0), 2: (20, 0, 0, 0)}, 3)
        got = compare_pair_on_contig(a, b, MODEL, MIN_FREQ, min_cov=5, want_locations=True)
        found = got["snp_positions"]
        self.assertIsNotNone(found)
        np.testing.assert_array_equal(found["position"], [0])
        self.assertTrue(bool(found["population_SNP"][0]))
        np.testing.assert_array_equal(found["counts_1"][0], [2, 0, 28, 0])
        self.assertEqual(int(found["coverage_2"][0]), 6)


class TestCompareAllPairs(unittest.TestCase):
    def setUp(self):
        self.contig_lengths = {"c1": 3, "c2": 2}
        clean = {
            "c1": _dense({0: (20, 0, 0, 0), 1: (20, 0, 0, 0), 2: (20, 0, 0, 0)}, 3),
            "c2": _dense({0: (20, 0, 0, 0), 1: (20, 0, 0, 0)}, 2),
        }
        # S2 carries ONE fixed difference on c1 position 0 (pure T vs pure A).
        differing = {
            "c1": _dense({0: (0, 0, 0, 20), 1: (20, 0, 0, 0), 2: (20, 0, 0, 0)}, 3),
            "c2": _dense({0: (20, 0, 0, 0), 1: (20, 0, 0, 0)}, 2),
        }
        self.dense = {"S1": clean, "S2": differing, "S3": clean}

    def test_metrics_sum_across_contigs(self):
        table, _ = compare_all_pairs(
            self.dense, self.contig_lengths, [("S1", "S2")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5, mag_id="MAG_X",
        )
        self.assertEqual(list(table.columns), PAIR_COLUMNS)
        row = table.iloc[0]
        self.assertEqual(int(row["compared_bases_count"]), 5)   # 3 on c1 + 2 on c2
        self.assertEqual(int(row["consensus_SNPs"]), 1)
        self.assertEqual(int(row["population_SNPs"]), 1)
        self.assertAlmostEqual(row["conANI"], 4 / 5)
        self.assertAlmostEqual(row["popANI"], 4 / 5)
        self.assertAlmostEqual(row["percent_genome_compared"], 1.0)
        self.assertAlmostEqual(row["coverage_overlap"], 1.0)
        # The gate that produced these numbers travels with them.
        self.assertEqual(int(row["min_cov"]), 5)
        self.assertEqual(row["MAG_ID"], "MAG_X")

    def test_identical_samples_score_one(self):
        table, _ = compare_all_pairs(
            self.dense, self.contig_lengths, [("S1", "S3")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5,
        )
        self.assertAlmostEqual(table.iloc[0]["conANI"], 1.0)
        self.assertAlmostEqual(table.iloc[0]["popANI"], 1.0)

    def test_zero_overlap_yields_nan_not_a_flattering_one(self):
        """No compared positions -> ANI is undefined. NaN, never 1.0."""
        empty = {"c1": np.zeros((3, 4), np.uint16), "c2": np.zeros((2, 4), np.uint16)}
        dense = {"S1": self.dense["S1"], "S4": empty}
        table, _ = compare_all_pairs(
            dense, self.contig_lengths, [("S1", "S4")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5,
        )
        row = table.iloc[0]
        self.assertEqual(int(row["compared_bases_count"]), 0)
        self.assertTrue(np.isnan(row["conANI"]))
        self.assertTrue(np.isnan(row["popANI"]))
        # S1 IS deep at 5 positions, S4 at none: overlap is a defined, honest 0.0
        # ("our deep regions never coincide"), not NaN.
        self.assertAlmostEqual(row["coverage_overlap"], 0.0)

    def test_two_empty_samples_have_undefined_overlap(self):
        """Only when NEITHER sample reaches min_cov anywhere is overlap undefined."""
        empty_a = {"c1": np.zeros((3, 4), np.uint16), "c2": np.zeros((2, 4), np.uint16)}
        empty_b = {"c1": np.zeros((3, 4), np.uint16), "c2": np.zeros((2, 4), np.uint16)}
        table, _ = compare_all_pairs(
            {"S4": empty_a, "S5": empty_b}, self.contig_lengths, [("S4", "S5")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5,
        )
        row = table.iloc[0]
        self.assertEqual(int(row["compared_bases_count"]), 0)
        self.assertTrue(np.isnan(row["coverage_overlap"]))
        self.assertTrue(np.isnan(row["popANI"]))

    def test_popani_never_below_conani(self):
        table, _ = compare_all_pairs(
            self.dense, self.contig_lengths,
            [("S1", "S2"), ("S1", "S3"), ("S2", "S3")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5,
        )
        valid = table.dropna(subset=["conANI", "popANI"])
        self.assertTrue(bool((valid["popANI"] >= valid["conANI"]).all()))

    def test_snp_locations_recorded_only_for_requested_pairs(self):
        table, locations = compare_all_pairs(
            self.dense, self.contig_lengths, [("S1", "S2"), ("S1", "S3")],
            MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5, mag_id="MAG_X",
            snp_location_pairs={("S1", "S2")},
        )
        self.assertEqual(len(table), 2)
        self.assertEqual(list(locations.columns), SNP_LOCATION_COLUMNS)
        self.assertEqual(len(locations), 1)                      # only the fixed diff
        row = locations.iloc[0]
        self.assertEqual((row["sample1"], row["sample2"]), ("S1", "S2"))
        self.assertEqual((row["contig"], int(row["position"])), ("c1", 0))
        self.assertTrue(bool(row["population_SNP"]))
        self.assertEqual(int(row["A_1"]), 20)                    # S1 is pure A there
        self.assertEqual(int(row["T_2"]), 20)                    # S2 is pure T
        self.assertEqual(int(row["coverage_1"]), 20)

    def test_missing_contig_array_is_a_loud_error(self):
        """profile_io guarantees every contig an array (zeros if uncovered), so a
        missing key means the caller assembled the inputs wrong -- not skippable."""
        broken = {"S1": {"c1": self.dense["S1"]["c1"]}, "S2": self.dense["S2"]}
        with self.assertRaises(KeyError):
            compare_all_pairs(
                broken, self.contig_lengths, [("S1", "S2")],
                MODEL, min_freq=MIN_FREQ, min_cov=5, genome_length=5,
            )


if __name__ == "__main__":
    unittest.main()
