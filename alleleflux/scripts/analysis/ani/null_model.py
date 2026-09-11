"""Sequencing-error null model: how many reads must support an allele to believe it?

At a position with C reads, a base seen in only a few reads may be a real minor
allele or may be sequencing error.  This module answers "how few is too few?" as a
lookup array ``model[coverage] -> minimum read count``.

The model is computed analytically -- nothing is shipped or read from disk.  With a
per-base error probability p (from the Phred floor applied during profiling,
p = 10 ** (-Q / 10)), a wrong call is assumed to land on any of the three other
bases with equal chance, so the probability that ONE specific wrong base reaches k
or more reads at coverage C is P[Binomial(C, p / 3) >= k], and the probability that
ANY of the three does is three times that:

    P(k, C) = 3 * P[Binomial(C, p / 3) >= k]

The threshold is the smallest k for which P(k, C) < fdr.

Worked example (Q30 -> p = 0.001, fdr = 1e-6, coverage 30):

    k = 1  ->  3.0e-02   error produces this at ~3% of positions      too common
    k = 2  ->  1.4e-04   ~1 position in 7,000 (~430 per 3 Mb genome)  too common
    k = 3  ->  4.5e-07   below 1e-6 (~1 per genome)                   accepted

so ``model[30] == 3``: an allele needs at least 3 reads (and, separately, at least
``min_freq`` of the reads -- see ``classify.presence_matrix``) to count as present.

Note this is a per-position false-positive probability, not a Benjamini-Hochberg
FDR: at 1e-6 it permits roughly 3 false alleles per 3 Mb genome per sample.

inStrain answers the same question with a shipped Monte-Carlo table that is valid
for Q30 only, is not monotonic in coverage, and is parsed one read too leniently
upstream (see docs/superpowers/instrain-nullmodel-report/).  Computing the value is
exact, works for any base-quality floor, and takes ~150 ms.

Library module: imported by the rest of the ani package and by tests; never run
directly, hence no ``if __name__ == "__main__"`` block.  The one runnable entry
point of this feature is the ``alleleflux-pairwise-ani`` command (pairwise_ani.py).

Method credit: Olm et al. 2021, Nat Biotechnol, doi:10.1038/s41587-020-00797-0.
"""

import logging

import numpy as np
from scipy.stats import binom

logger = logging.getLogger(__name__)

# Coverage values above this are clipped to it when looking up a threshold.  Harmless
# in practice: above a few hundred reads the min_freq floor (5% of coverage) demands
# far more reads than the error model does, so the clipped row never decides a call.
# The array costs ~40 KB.
MAX_COVERAGE = 10_000

# Largest allele read-count considered (the width of the probability grid).  The
# largest realistic threshold is ~67 (Q20 at 10,000x), so 100 leaves headroom;
# exceeding it is logged rather than silently clipped.
_MAX_K = 100


def build_error_model(min_base_quality=30, fdr=1e-6, max_coverage=MAX_COVERAGE):
    """Return ``model`` where ``model[coverage]`` is the minimum read count for an
    allele to be considered present at that coverage.

    Parameters
    ----------
    min_base_quality : int
        The Phred floor applied when the profiles were built (samtools ``--min-BQ``).
        Sets the assumed per-base error probability ``10 ** (-Q / 10)``.
    fdr : float
        Tolerated per-position probability that error alone explains the allele.
    max_coverage : int
        Highest coverage tabulated; higher coverages reuse this row.

    Returns
    -------
    numpy.ndarray
        int32 array of length ``max_coverage + 1``, indexed directly by coverage.
        Index 0 is a placeholder (1) and is never used, since callers gate positions
        on ``min_cov`` first.

    Notes
    -----
    At 1x the threshold (2 reads) exceeds the depth, so nothing can be called
    present; at 2x only a base carrying both reads clears it.  Either way no
    *minor* allele is ever credible that shallowly, which is the intended behaviour
    and the reason ``min_cov`` defaults to 5.
    """
    if not 0.0 < fdr < 1.0:
        raise ValueError(f"fdr must be strictly between 0 and 1, got {fdr}")
    if min_base_quality <= 0:
        raise ValueError(f"min_base_quality must be positive, got {min_base_quality}")
    if max_coverage < 1:
        raise ValueError(f"max_coverage must be at least 1, got {max_coverage}")

    error_rate = 10.0 ** (-min_base_quality / 10.0)

    # One (coverage, k) grid evaluated in a single vectorised call: a column of
    # depths against a row of candidate read counts.  binom.sf(k - 1, C, p) is
    # P[X > k - 1] == P[X >= k]; the factor of 3 accounts for the error landing on
    # any one of the three wrong bases.  Both axes are 0-based here (row i is
    # coverage i + 1, column j is k = j + 1); the conversions happen below.
    ks = np.arange(1, _MAX_K + 1)
    coverages = np.arange(1, max_coverage + 1)[:, None]
    probabilities = 3.0 * binom.sf(ks[None, :] - 1, coverages, error_rate / 3.0)

    below_fdr = probabilities < fdr
    found = below_fdr.any(axis=1)
    # argmax on a boolean row returns the position of the first True -- a 0-based
    # column index, so ks[...] converts it back to a read count.  Rows with no True
    # also return 0; `found` masks those out so they get the cap, not "1 read".
    first_below = below_fdr.argmax(axis=1)
    thresholds = np.where(found, ks[first_below], ks[-1])

    if not found.all():
        n_capped = int((~found).sum())
        logger.warning(
            f"Error model hit the k cap ({_MAX_K}) at {n_capped} coverage values; "
            f"thresholds there are clipped. Check min_base_quality={min_base_quality}."
        )

    # Shift by one so the array is indexed by coverage itself: model[30] is the
    # threshold at 30x, no arithmetic at the call site.
    model = np.empty(max_coverage + 1, dtype=np.int32)
    model[0] = 1  # placeholder; coverage-0 positions never reach a lookup
    model[1:] = thresholds

    logger.info(
        f"Error model built: Q{min_base_quality} (p={error_rate:.2g}), fdr={fdr:.0e}; "
        f"thresholds 5x->{model[5] if max_coverage >= 5 else 'n/a'}, "
        f"30x->{model[30] if max_coverage >= 30 else 'n/a'}, "
        f"100x->{model[100] if max_coverage >= 100 else 'n/a'}"
    )
    return model
