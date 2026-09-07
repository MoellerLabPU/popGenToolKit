"""Tests for per-mouse strain-background calls from the pairwise ANI table.

The rows fed in mimic what the engine emits: ``sample1 < sample2`` by ID, which
is NOT chronological -- orientation by time is one of the things under test.
"""
import unittest

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.ani.strain_turnover import (
    alleles_present_at,
    call_transitions,
    scan_for_new_alleles,
)

# Q30 / FDR 1e-6: threshold is 3 reads for coverages 5..99 (see null_model docs).
MODEL = build_error_model(min_base_quality=30, fdr=1e-6)


def _pair_row(s1, s2, subj, t1, t2, popani, pgc, conani=None, group="40"):
    """One engine-style pair row; conANI defaults to popANI when not given."""
    return {
        "MAG_ID": "MRGM_0841", "sample1": s1, "sample2": s2,
        "subjectID_1": subj, "subjectID_2": subj,
        "time_1": t1, "time_2": t2, "group_1": group, "group_2": group,
        "replicate_1": "1", "replicate_2": "1",
        "popANI": popani, "conANI": popani if conani is None else conani,
        "percent_genome_compared": pgc,
        "compared_bases_count": int(pgc * 4_000_000), "min_cov": 5,
    }


class TestCallTransitions(unittest.TestCase):
    TRANSITIONS = [("5mo", "22mo")]

    def _call(self, rows, **overrides):
        kwargs = dict(min_compared=0.1, pop_threshold=0.99999, con_threshold=0.999)
        kwargs.update(overrides)
        return call_transitions(pd.DataFrame(rows), self.TRANSITIONS, **kwargs)

    def test_orients_by_time_not_by_sample_id(self):
        # "Abe" sorts before "Zed" so the engine put the 22mo sample in sample1.
        got = self._call([_pair_row("Abe", "Zed", "m1", "22mo", "5mo", 0.999999, 0.8)])
        self.assertEqual(got.iloc[0]["sample_t1"], "Zed")      # the 5mo sample
        self.assertEqual(got.iloc[0]["sample_t2"], "Abe")
        self.assertEqual(got.iloc[0]["transition"], "5mo_22mo")

    def test_four_backgrounds(self):
        got = self._call([
            _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.9999990, 0.8, conani=0.9995),  # stable
            _pair_row("B1", "B2", "m2", "5mo", "22mo", 0.9999950, 0.8, conani=0.9985),  # dominant only
            _pair_row("C1", "C2", "m3", "5mo", "22mo", 0.9999800, 0.8, conani=0.9995),  # replacement only
            _pair_row("D1", "D2", "m4", "5mo", "22mo", 0.9900000, 0.8, conani=0.9800),  # both
        ])
        self.assertEqual(list(got["background"]), [
            "stable", "dominant_strain_change", "strain_replacement",
            "strain_replacement+dominant_strain_change"])
        self.assertEqual(list(got["strain_replacement"]), [False, False, True, True])
        self.assertEqual(list(got["dominant_strain_change"]), [False, True, False, True])
        # popANI >= conANI always, so the shift is never negative.
        self.assertTrue(bool((got["frequency_shift"] >= 0).all()))

    def test_threshold_is_strict_less_than(self):
        # Exactly AT the threshold is NOT a change (>= threshold means same strain).
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", 0.99999, 0.8, conani=0.999)])
        self.assertEqual(got.iloc[0]["background"], "stable")

    def test_undetermined_leaves_both_flags_blank(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", 0.5, 0.01)])
        self.assertEqual(got.iloc[0]["background"], "undetermined")
        self.assertTrue(pd.isna(got.iloc[0]["strain_replacement"]))
        self.assertTrue(pd.isna(got.iloc[0]["dominant_strain_change"]))

    def test_min_compared_is_configurable(self):
        row = _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.99, 0.05, conani=0.99)
        self.assertEqual(self._call([row]).iloc[0]["background"], "undetermined")  # 5% < 10%
        self.assertEqual(self._call([row], min_compared=0.05).iloc[0]["background"],
                         "strain_replacement+dominant_strain_change")            # 5% >= 5%

    def test_nan_ani_is_undetermined(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", np.nan, 0.0)])
        self.assertEqual(got.iloc[0]["background"], "undetermined")

    def test_non_transition_and_cross_subject_rows_are_excluded(self):
        rows = [
            _pair_row("A1", "A2", "m1", "5mo", "28mo", 0.9999990, 0.8),   # not a configured transition
            _pair_row("A1", "B1", "m1", "5mo", "22mo", 0.9999990, 0.8),   # two different mice
        ]
        rows[1]["subjectID_2"] = "m2"
        self.assertEqual(len(self._call(rows)), 0)

    def test_empty_input_keeps_the_schema(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "28mo", 0.9999990, 0.8)])
        for col in ("transition", "sample_t1", "sample_t2", "strain_replacement",
                    "dominant_strain_change", "frequency_shift", "background"):
            self.assertIn(col, got.columns)

    def test_rejects_degenerate_transition(self):
        with self.assertRaises(ValueError):
            call_transitions(pd.DataFrame([_pair_row("A", "B", "m", "5mo", "22mo", 1.0, 1.0)]),
                             [("5mo", "5mo")], 0.1, 0.99999, 0.999)


def _dense(rows, length):
    """Dense (length, 4) uint16 counts from {position: (A, C, G, T)}; unlisted rows are zero."""
    arr = np.zeros((length, 4), dtype=np.uint16)
    for pos, counts in rows.items():
        arr[pos] = counts
    return arr


class TestScanForNewAlleles(unittest.TestCase):
    """One mouse, baseline (t1) vs later (t2) dense counts; min_freq 5 %, min_cov 5."""

    def _scan(self, t1, t2, length=1):
        return scan_for_new_alleles(_dense({0: t1}, length), _dense({0: t2}, length), MODEL, 0.05, 5)

    def test_new_minor_allele_without_majority_flip(self):
        """The case no SNP table can show: A stays consensus, C appears at 20 %."""
        got = self._scan((30, 0, 0, 0), (24, 6, 0, 0), length=2)
        self.assertEqual(list(got["position"]), [0])
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["t1_reads"]), [0])
        self.assertEqual(list(got["t1_threshold"]), [int(MODEL[30])])
        self.assertEqual(list(got["t1_evidence"]), ["absent"])
        self.assertEqual(list(got["fully_replaced"]), [False])
        self.assertEqual(list(got["t2_consensus"]), [False])          # A still leads at t2
        self.assertEqual(list(got["t2_reads"]), [6])
        self.assertEqual(list(got["t2_threshold"]), [int(MODEL[30])])
        self.assertAlmostEqual(float(got["freq_t1"][0]), 0.0)
        self.assertAlmostEqual(float(got["freq_t2"][0]), 0.2)

    def test_full_sweep_is_flagged(self):
        got = self._scan((30, 0, 0, 0), (0, 28, 0, 0))
        self.assertEqual(list(got["fully_replaced"]), [True])
        self.assertEqual(list(got["t1_evidence"]), ["absent"])

    def test_majority_flip_with_old_allele_still_present_is_not_fully_replaced(self):
        # C takes over but A (5/30 = 17 %) is still present at t2 -> shared allele.
        got = self._scan((30, 0, 0, 0), (5, 25, 0, 0))
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["fully_replaced"]), [False])
        self.assertEqual(list(got["t2_consensus"]), [True])           # ...but C IS the majority now

    def test_thin_t1_majority_is_not_reported_as_new(self):
        # A has 2 of 5 reads at t1 (bar 3): fails presence but IS the consensus -> credible.
        got = self._scan((2, 1, 1, 1), (30, 0, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_below_min_cov_positions_are_never_scanned(self):
        got = self._scan((3, 0, 0, 0), (10, 10, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_allele_already_present_at_t1_is_not_new(self):
        # C had 5 of 25 at t1 (bar 3, 20 %) -> present -> credible -> not new.
        got = self._scan((20, 5, 0, 0), (18, 9, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_one_read_short_at_t1_is_below_detection(self):
        """t1 has 1 C read of 30 (bar 3): not credible, reported, and labelled so."""
        got = self._scan((29, 1, 0, 0), (24, 6, 0, 0))
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["t1_reads"]), [1])
        self.assertEqual(list(got["t1_threshold"]), [3])
        self.assertEqual(list(got["t1_evidence"]), ["below_detection"])

    def test_under_five_percent_at_t1_is_below_detection(self):
        # 4 of 100 passes the bar (4 at 100x) but fails the 5 % floor.
        got = self._scan((96, 4, 0, 0), (80, 20, 0, 0))
        self.assertEqual(list(got["t1_evidence"]), ["below_detection"])
        self.assertEqual(list(got["t1_reads"]), [4])

    def test_raw_rows_ride_along(self):
        got = self._scan((30, 0, 0, 0), (24, 6, 0, 0))
        self.assertEqual(got["counts_t1_rows"].tolist(), [[30, 0, 0, 0]])
        self.assertEqual(got["counts_t2_rows"].tolist(), [[24, 6, 0, 0]])
        self.assertEqual((list(got["coverage_t1"]), list(got["coverage_t2"])), ([30], [30]))

    def test_empty_result_has_every_key(self):
        got = self._scan((30, 0, 0, 0), (30, 0, 0, 0))
        for key in ("position", "base", "t1_reads", "t1_threshold", "t1_evidence", "freq_t1",
                    "t2_reads", "t2_threshold", "freq_t2", "t2_consensus", "fully_replaced", "counts_t1_rows", "coverage_t1", "counts_t2_rows", "coverage_t2"):
            self.assertEqual(len(got[key]), 0, key)


class TestAllelesPresentAt(unittest.TestCase):
    def test_presence_lookup_at_named_positions(self):
        dense = _dense({0: (30, 0, 0, 0), 2: (24, 6, 0, 0)}, 3)
        got = alleles_present_at(dense, np.array([0, 1, 2]), MODEL, 0.05)
        self.assertEqual(got.shape, (3, 4))
        self.assertTrue(bool(got[0, 0]))          # A present at pos 0
        self.assertFalse(got[1].any())            # zero coverage -> nothing present
        self.assertTrue(bool(got[2, 1]))          # C (6/30) present at pos 2
        self.assertTrue(bool(got[2, 0]))


if __name__ == "__main__":
    unittest.main()
