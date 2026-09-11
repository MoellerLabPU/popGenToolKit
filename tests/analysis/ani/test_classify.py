"""Tests for per-position consensus / allele-presence primitives and the pair classifier.

Counts are ``(n_positions, 4)`` arrays in AlleleFlux's profile order ``A, C, G, T``.

The rule under test (see ``classify.py`` for the rationale):

* consensus set  = every base tied at the maximum count
* credible set   = consensus set  +  every base clearing the null model AND min_freq
* consensus SNP  <=> the two samples' consensus sets are disjoint
* population SNP <=> the two samples' credible sets are disjoint

Two oracles live in this file.  ``_oracle_classify`` is a slow, literal Python port
of OUR rule, used for brute-force agreement.  ``_instrain_classify`` is a port of
inStrain's row-wise rule, used to prove we never call a SNP inStrain would not
(we differ only by being more careful: ties and shared non-top alleles).
"""
import unittest

import numpy as np

from alleleflux.scripts.analysis.ani.classify import (
    BASES,
    classify_positions,
    consensus_mask,
    credible_alleles,
    presence_matrix,
)
from alleleflux.scripts.analysis.ani.null_model import build_error_model

MODEL = build_error_model(min_base_quality=30, fdr=1e-6)
MIN_FREQ = 0.05

# Real rows from SLG1007 vs SLG1102, MAG SLG1007_DASTool_bins_35 (profile order A,C,G,T).
REAL_FIXED_1 = [2, 0, 28, 0]   # k141_17312:13581, 30x, majority G
REAL_FIXED_2 = [5, 0, 1, 0]    # same position,    6x, majority A
REAL_MIXED_1 = [0, 9, 0, 10]   # k141_105179:250,  19x, C and T both real
REAL_MIXED_2 = [0, 8, 0, 6]    # same position,    14x


def _cov(rows):
    return np.asarray(rows).sum(axis=1)


def _classify(c1, c2):
    c1 = np.atleast_2d(np.asarray(c1))
    c2 = np.atleast_2d(np.asarray(c2))
    return classify_positions(c1, _cov(c1), c2, _cov(c2), MODEL, MIN_FREQ)


# --------------------------------------------------------------------------- oracles

def _oracle_consensus(counts):
    top = max(counts)
    return {i for i in range(4) if counts[i] == top and top > 0}


def _oracle_credible(counts, coverage):
    credible = set(_oracle_consensus(counts))
    threshold = MODEL[min(coverage, len(MODEL) - 1)]
    for i in range(4):
        if coverage > 0 and counts[i] >= threshold and counts[i] / coverage >= MIN_FREQ:
            credible.add(i)
    return credible


def _oracle_classify(c1, cov1, c2, cov2):
    """Literal, set-based port of OUR rule."""
    consensus_snp = _oracle_consensus(c1).isdisjoint(_oracle_consensus(c2))
    population_snp = _oracle_credible(c1, cov1).isdisjoint(_oracle_credible(c2, cov2))
    return consensus_snp, population_snp


def _instrain_is_present(count, coverage):
    threshold = MODEL[min(coverage, len(MODEL) - 1)]
    return count >= threshold and count / coverage >= MIN_FREQ


def _instrain_classify(c1, cov1, c2, cov2):
    """Row-wise port of inStrain's calc_con_snps / call_pop_snps (argmax tie-break,
    shared-allele rescue only via the single top runner-up)."""
    con1, con2 = int(np.argmax(c1)), int(np.argmax(c2))
    if con1 == con2:
        return False, False
    if _instrain_is_present(c2[con1], cov2) or _instrain_is_present(c1[con2], cov1):
        return True, False
    present1 = [i for i in range(4) if _instrain_is_present(c1[i], cov1)]
    present2 = [i for i in range(4) if _instrain_is_present(c2[i], cov2)]
    if len(present1) > 1 and len(present2) > 1:
        var1 = int(np.argmax([v if i != con1 else -1 for i, v in enumerate(c1)]))
        var2 = int(np.argmax([v if i != con2 else -1 for i, v in enumerate(c2)]))
        if var1 == var2:
            return True, False
    return True, True


def _random_pairs(seed=0):
    """Random count vectors paired every-with-every: shallow uniform ones (where ties
    and the error model bind) plus deeper skewed ones (where minors are credible)."""
    rng = np.random.default_rng(seed)
    vectors = []
    for total in (5, 6, 7, 8):
        vectors.extend(rng.multinomial(total, [0.25] * 4, size=300))
    for total in (20, 60):
        vectors.extend(rng.multinomial(total, [0.6, 0.3, 0.07, 0.03], size=200))
    vectors = np.array(vectors)
    left = np.repeat(vectors, 8, axis=0)
    right = np.tile(vectors, (8, 1))
    return left, right


# ------------------------------------------------------------------------ primitives

class TestPrimitives(unittest.TestCase):
    def test_base_order_is_the_profile_order(self):
        """No reordering anywhere: the set-based rule never depends on column order."""
        self.assertEqual(BASES, ("A", "C", "G", "T"))

    def test_consensus_is_the_most_common_base(self):
        counts = np.array([[10, 2, 0, 0], [0, 0, 1, 9]])
        np.testing.assert_array_equal(
            consensus_mask(counts),
            [[True, False, False, False], [False, False, False, True]],
        )

    def test_consensus_tie_keeps_every_tied_base(self):
        """3 A + 3 G at 6x: neither is 'the' majority, so both are consensus.  inStrain's
        argmax would pick one by column order; keeping both means a tie can never
        manufacture a consensus SNP against a sample carrying either base."""
        counts = np.array([[3, 0, 3, 0]])
        np.testing.assert_array_equal(consensus_mask(counts), [[True, False, True, False]])

    def test_consensus_of_an_empty_row_is_nothing(self):
        counts = np.array([[0, 0, 0, 0]])
        np.testing.assert_array_equal(consensus_mask(counts), [[False] * 4])

    def test_presence_requires_both_count_and_frequency(self):
        """At 30x the model needs 3 reads; the 5% floor needs 2.  Both must pass."""
        counts = np.array([[27, 2, 1, 0], [24, 3, 3, 0]])
        present = presence_matrix(counts, _cov(counts), MODEL, MIN_FREQ)
        # Row 0: C has 2 reads (6.7%, clears the floor, not the 3-read bar); G has 1.
        np.testing.assert_array_equal(present[0], [True, False, False, False])
        # Row 1: 3 reads is 10% of 30 -- clears both bars.
        np.testing.assert_array_equal(present[1], [True, True, True, False])

    def test_frequency_floor_binds_at_high_coverage(self):
        """At 200x the model needs 5 reads but 5% needs 10, so 6 reads fails."""
        counts = np.array([[194, 6, 0, 0]])
        present = presence_matrix(counts, _cov(counts), MODEL, MIN_FREQ)
        np.testing.assert_array_equal(present[0], [True, False, False, False])

    def test_frequency_floor_is_inclusive(self):
        """Exactly 5% passes (>=), matching the read-count test's >=."""
        counts = np.array([[95, 5, 0, 0]])  # 100x: model needs 4, floor needs 5
        present = presence_matrix(counts, _cov(counts), MODEL, MIN_FREQ)
        self.assertTrue(bool(present[0, 1]))

    def test_coverage_above_table_is_clipped_not_crashing(self):
        counts = np.array([[50_000, 500, 0, 0]])
        present = presence_matrix(counts, _cov(counts), MODEL, MIN_FREQ)
        self.assertTrue(bool(present[0, 0]))
        self.assertFalse(bool(present[0, 1]))  # 500 / 50,500 is below 5%

    def test_zero_coverage_row_is_simply_absent(self):
        """Callers gate on min_cov, but a zero row must never divide by zero."""
        counts = np.array([[0, 0, 0, 0]])
        present = presence_matrix(counts, np.array([0]), MODEL, MIN_FREQ)
        np.testing.assert_array_equal(present[0], [False] * 4)

    def test_credible_alleles_always_include_the_consensus(self):
        """(2,1,1,1) at 5x: nothing clears the 3-read bar, but the majority base is
        still the best evidence of what is there, so it stays credible."""
        counts = np.array([[2, 1, 1, 1]])
        credible = credible_alleles(counts, _cov(counts), MODEL, MIN_FREQ)
        np.testing.assert_array_equal(credible[0], [True, False, False, False])

    def test_credible_alleles_add_present_minors_to_the_consensus(self):
        counts = np.array([[24, 3, 3, 0]])  # 30x
        credible = credible_alleles(counts, _cov(counts), MODEL, MIN_FREQ)
        np.testing.assert_array_equal(credible[0], [True, True, True, False])


# ------------------------------------------------------------------------ classifier

class TestClassifyPositions(unittest.TestCase):
    def test_identical_samples_produce_no_snps(self):
        con, pop = _classify([20, 0, 0, 0], [20, 0, 0, 0])
        self.assertFalse(bool(con[0]))
        self.assertFalse(bool(pop[0]))

    def test_real_fixed_difference_is_both_consensus_and_population_snp(self):
        """G-dominant at 30x vs A-dominant at 6x.  A in sample 1 is 2 reads (needs 3),
        G in sample 2 is 1 read (needs 3): no credible allele is shared."""
        con, pop = _classify(REAL_FIXED_1, REAL_FIXED_2)
        self.assertTrue(bool(con[0]))
        self.assertTrue(bool(pop[0]))

    def test_real_shared_alleles_rescue_a_consensus_difference(self):
        """Both samples carry C and T; only the majority differs -> mixed population."""
        con, pop = _classify(REAL_MIXED_1, REAL_MIXED_2)
        self.assertTrue(bool(con[0]))
        self.assertFalse(bool(pop[0]))

    def test_shared_minor_allele_rescues(self):
        con, pop = _classify([12, 6, 0, 0], [6, 12, 0, 0])
        self.assertTrue(bool(con[0]))
        self.assertFalse(bool(pop[0]))

    def test_tie_against_either_tied_base_is_not_a_consensus_snp(self):
        """A/G tie vs pure A, and vs pure G: the sets overlap, so no SNP of any kind.
        inStrain would call one of these a consensus SNP purely by column order."""
        for other in ([20, 0, 0, 0], [0, 0, 20, 0]):
            con, pop = _classify([3, 0, 3, 0], other)
            self.assertFalse(bool(con[0]), other)
            self.assertFalse(bool(pop[0]), other)

    def test_tie_against_a_third_base_is_a_fixed_difference(self):
        con, pop = _classify([3, 0, 3, 0], [0, 20, 0, 0])
        self.assertTrue(bool(con[0]))
        self.assertTrue(bool(pop[0]))

    def test_shared_third_allele_is_not_a_population_snp(self):
        """Deliberate difference from inStrain, which rescues only via the single top
        runner-up: sample 1 = A>C>G, sample 2 = T>G>C.  Runner-ups differ (C vs G) so
        inStrain calls a fixed difference, yet both samples credibly carry C AND G."""
        con, pop = _classify([60, 30, 10, 0], [0, 5, 40, 55])
        self.assertTrue(bool(con[0]))
        self.assertFalse(bool(pop[0]))
        # ...and the inStrain port really does disagree, so the test is not vacuous.
        self.assertEqual(_instrain_classify([60, 30, 10, 0], 100, [0, 5, 40, 55], 100), (True, True))

    def test_rejects_misaligned_inputs(self):
        with self.assertRaises(ValueError):
            classify_positions(
                np.array([[20, 0, 0, 0], [20, 0, 0, 0]]), np.array([20, 20]),
                np.array([[20, 0, 0, 0]]), np.array([20]),
                MODEL, MIN_FREQ,
            )
        with self.assertRaises(ValueError):
            classify_positions(
                np.array([[20, 0, 0]]), np.array([20]),
                np.array([[20, 0, 0]]), np.array([20]),
                MODEL, MIN_FREQ,
            )

    def test_brute_force_agreement_with_literal_oracle(self):
        """Every random pair must match the slow set-based port of the same rule, and
        a population SNP must always also be a consensus SNP (consensus is a subset of
        credible, so this is guaranteed by construction -- verify it anyway)."""
        left, right = _random_pairs()
        cov_left, cov_right = _cov(left), _cov(right)
        con, pop = classify_positions(left, cov_left, right, cov_right, MODEL, MIN_FREQ)

        self.assertTrue(bool(np.all(~pop | con)), "population SNP without consensus SNP")
        for i in range(len(left)):
            exp_con, exp_pop = _oracle_classify(left[i], int(cov_left[i]), right[i], int(cov_right[i]))
            self.assertEqual(bool(con[i]), exp_con, msg=f"consensus row {i}: {left[i]} vs {right[i]}")
            self.assertEqual(bool(pop[i]), exp_pop, msg=f"population row {i}: {left[i]} vs {right[i]}")

    def test_never_less_conservative_than_instrain(self):
        """Wherever we differ from inStrain's rule we call FEWER SNPs, never more:
        ties can only remove consensus SNPs, and shared non-top alleles can only
        remove population SNPs.  So our calls must be a subset of inStrain's."""
        left, right = _random_pairs(seed=1)
        cov_left, cov_right = _cov(left), _cov(right)
        con, pop = classify_positions(left, cov_left, right, cov_right, MODEL, MIN_FREQ)

        n_con_removed = n_pop_removed = 0
        for i in range(len(left)):
            their_con, their_pop = _instrain_classify(left[i], int(cov_left[i]), right[i], int(cov_right[i]))
            self.assertFalse(bool(con[i]) and not their_con, msg=f"row {i}: extra consensus SNP {left[i]} vs {right[i]}")
            self.assertFalse(bool(pop[i]) and not their_pop, msg=f"row {i}: extra population SNP {left[i]} vs {right[i]}")
            n_con_removed += int(their_con and not con[i])
            n_pop_removed += int(their_pop and not pop[i])
        # Shallow uniform vectors tie often, so the batch must contain real divergences.
        self.assertGreater(n_con_removed, 0)
        self.assertGreater(n_pop_removed, 0)


if __name__ == "__main__":
    unittest.main()
