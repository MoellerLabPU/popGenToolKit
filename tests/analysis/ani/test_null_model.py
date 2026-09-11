"""Tests for the sequencing-error null model (``alleleflux.scripts.analysis.ani.null_model``).

The model answers "how many reads must support an allele before it is more than
sequencing error?" as a lookup ``model[coverage] -> minimum read count``.

Every hard-coded threshold below was recomputed independently with
``scipy.stats.binom`` *before* the implementation was written (2026-08-28), so a
failure here means the implementation drifted, not the expectation.
"""
import unittest

import numpy as np

from alleleflux.scripts.analysis.ani.null_model import MAX_COVERAGE, build_error_model


class TestBuildErrorModel(unittest.TestCase):
    def test_known_thresholds_at_q30(self):
        """At Q30 (p = 0.001) with fdr = 1e-6 the thresholds are fixed, known numbers.

        Worked check at coverage 30: 3 * P[Binomial(30, 0.001/3) >= k] is
        3.0e-2 for k=1, 1.4e-4 for k=2 and 4.5e-7 for k=3 -- the first value
        below 1e-6 -- so an allele needs 3 reads.
        """
        model = build_error_model(min_base_quality=30, fdr=1e-6)
        self.assertEqual(int(model[5]), 3)
        self.assertEqual(int(model[10]), 3)
        self.assertEqual(int(model[30]), 3)
        self.assertEqual(int(model[100]), 4)
        self.assertEqual(int(model[200]), 5)
        self.assertEqual(int(model[1000]), 7)
        self.assertEqual(int(model[5000]), 12)

    def test_shallow_positions_cannot_support_a_minor_allele(self):
        """At 1x the bar (2 reads) is above the depth, so nothing can be called
        present; at 2x only a base carrying both reads clears it.  Either way no
        *minor* allele is ever credible this shallow, which is the point."""
        model = build_error_model(min_base_quality=30, fdr=1e-6)
        self.assertEqual(int(model[1]), 2)
        self.assertEqual(int(model[2]), 2)

    def test_lower_base_quality_demands_more_reads(self):
        """A looser quality floor means more errors, so the bar must rise."""
        q30 = build_error_model(min_base_quality=30, fdr=1e-6)
        q20 = build_error_model(min_base_quality=20, fdr=1e-6)
        self.assertGreater(int(q20[5]), int(q30[5]))
        self.assertGreater(int(q20[100]), int(q30[100]))
        # Pin the Q20 values too: p = 0.01 is ten times Q30's error rate.
        self.assertEqual(int(q20[5]), 4)
        self.assertEqual(int(q20[100]), 7)

    def test_looser_fdr_demands_fewer_reads(self):
        strict = build_error_model(min_base_quality=30, fdr=1e-6)
        loose = build_error_model(min_base_quality=30, fdr=1e-4)
        self.assertLessEqual(int(loose[100]), int(strict[100]))
        self.assertEqual(int(loose[5]), 2)
        self.assertEqual(int(loose[100]), 3)

    def test_monotonic_in_coverage(self):
        """More reads means more expected errors, so the bar never falls.

        (True of the computed model; inStrain's simulated table has 2,312 dips
        from Monte-Carlo noise, one reason we compute instead of copying it.)
        """
        model = build_error_model()
        self.assertTrue(bool(np.all(np.diff(model[1:]) >= 0)))

    def test_shape_dtype_and_placeholder(self):
        model = build_error_model()
        self.assertEqual(len(model), MAX_COVERAGE + 1)
        self.assertEqual(model.dtype, np.int32)
        # Index 0 is a never-used placeholder: a position with no reads is never
        # looked up because callers gate on min_cov first.
        self.assertEqual(int(model[0]), 1)

    def test_custom_max_coverage_is_a_prefix_of_the_default(self):
        short = build_error_model(max_coverage=50)
        full = build_error_model()
        self.assertEqual(len(short), 51)
        np.testing.assert_array_equal(short, full[:51])

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            build_error_model(fdr=0.0)
        with self.assertRaises(ValueError):
            build_error_model(fdr=1.0)
        with self.assertRaises(ValueError):
            build_error_model(fdr=-1e-6)
        with self.assertRaises(ValueError):
            build_error_model(min_base_quality=0)
        with self.assertRaises(ValueError):
            build_error_model(max_coverage=0)


if __name__ == "__main__":
    unittest.main()
