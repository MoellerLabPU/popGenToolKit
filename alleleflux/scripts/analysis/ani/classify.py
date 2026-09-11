"""Per-position primitives for comparing two samples at one genome position.

Everything here is pure NumPy over ``(n_positions, 4)`` count arrays in AlleleFlux's
profile order ``A, C, G, T``; no I/O.  Callers pass positions that already clear the
``min_cov`` gate in both samples (see ``engine.py``); nothing here re-checks depth.

The inputs, concretely
----------------------
``counts`` is one sample's pileup tally: one row per genome position, four columns
saying how many of that sample's reads reported A, C, G or T there.  Rows come
straight from the A, C, G, T columns of the sample's profile file
(``{sample}_{MAG}_profiled.tsv.gz``).  A real line from SLG1007's profile::

    contig        position  ref  total_coverage  A  C  G  T  N
    k141_102792   0         G    4               2  0  1  1  0

    its counts row:  [2, 0, 1, 1]   "2 reads say A, 0 say C, 1 says G, 1 says T"

``counts1`` and ``counts2`` are the SAME genome positions for two different
samples, row-aligned by the engine: row i of both arrays is one (contig, position),
seen through each sample's reads.  ``coverage`` is the row sum A + C + G + T (the
profile's ``total_coverage`` also counts N calls, so it is NOT used).  ``model``
is ``null_model.build_error_model``'s lookup (coverage -> minimum reads).
``min_freq`` is the population floor on allele frequency -- see
``presence_matrix`` for what it means and why it exists.

The rule
--------
For each sample at each position:

* the **consensus set** is every base tied at the maximum count -- normally one
  base, two on an exact tie;
* a base is **present** when it clears BOTH bars: at least ``model[coverage]``
  reads (sequencing error could not plausibly produce it -- ``null_model.py``) and
  at least ``min_freq`` of the reads;
* the **credible set** is the consensus set plus every present base.

Comparing two samples:

* **consensus SNP**  <=>  the consensus sets are disjoint (the majorities differ);
* **population SNP** <=>  the credible sets are disjoint (they share NO credible
  allele -- a fixed difference between the two populations).

Because the consensus set is a subset of the credible set, a population SNP is
always also a consensus SNP, so ``popANI >= conANI`` holds by construction.

Why sets, not inStrain's argmax
-------------------------------
inStrain picks a single majority base with ``argmax`` (ties broken by column order)
and rescues a consensus difference only if one majority is present in the other
sample or the two *top* runner-ups coincide.  Two consequences: an exact tie (3 A +
3 G at 6x) against a sample that is pure A becomes a "consensus SNP" by column
order, and a shared *third* allele is missed.  Sets fix both: a tied position is
compared as the pair of bases it is, and "share any credible allele" is tested
literally.  Wherever the two rules differ we call fewer SNPs, never more
(``tests/analysis/ani/test_classify.py::test_never_less_conservative_than_instrain``).

Worked example (real position k141_17312:13581, SLG1007 vs SLG1102; order A,C,G,T)::

    sample 1  [2, 0, 28, 0]  30x   consensus {G}   present {G}          credible {G}
    sample 2  [5, 0,  1, 0]   6x   consensus {A}   present {A}          credible {A}

    A in sample 1 is 2 reads (needs 3 at 30x); G in sample 2 is 1 read (needs 3 at
    6x).  Consensus sets disjoint -> consensus SNP.  Credible sets disjoint ->
    population SNP: one of only 3 fixed differences in 1.65 Mb compared.

    sample 1  [0, 9, 0, 10]  19x   consensus {T}   present {C, T}       credible {C, T}
    sample 2  [0, 8, 0,  6]  14x   consensus {C}   present {C, T}       credible {C, T}

    Majorities differ -> consensus SNP, but both carry C and T -> NOT a population
    SNP: the same mixture in different proportions.

Library module: imported by the rest of the ani package and by tests; never run
directly, hence no ``if __name__ == "__main__"`` block.  The one runnable entry
point of this feature is the ``alleleflux-pairwise-ani`` command (pairwise_ani.py).

Known edge case
---------------
A majority that itself fails the error model (``[2, 1, 1, 1]`` at 5x: 2 reads,
bar is 3) is still credible -- it is the best evidence of what is there.  Such
positions are rare at ``min_cov`` 5 and are the reason ``min_cov`` is not lower.
"""

import numpy as np
import numpy.typing as npt

# Readable aliases for the two array shapes that flow through this module:
# counts are (n_positions, 4) integers; every verdict/mask is (n_positions, ...) bools.
CountArray = npt.ArrayLike  # anything coercible to an (n, 4) int array
BoolMatrix = npt.NDArray[np.bool_]  # (n, 4) mask, one column per base
BoolVector = npt.NDArray[np.bool_]  # (n,) verdicts, one per position

# AlleleFlux's profile column order.  The set-based rule never depends on this
# order, so profiles are used as stored -- no reordering anywhere in the feature.
BASES = ("A", "C", "G", "T")


def _as_counts(counts: CountArray) -> np.ndarray:
    # Accept lists/tuples in tests but hand every caller a real ndarray.
    counts = np.asarray(counts)
    # Shape is a contract, not a hint: a transposed or 1-D array would silently
    # produce garbage verdicts downstream, so fail loudly here instead.
    if counts.ndim != 2 or counts.shape[1] != len(BASES):
        raise ValueError(
            f"counts must be an (n_positions, {len(BASES)}) array in {BASES} order, "
            f"got shape {counts.shape}"
        )
    return counts


def consensus_mask(counts: CountArray) -> BoolMatrix:
    """Boolean ``(n_positions, 4)``: which bases are tied for the most reads?

    Usually exactly one True per row; two (or more) on an exact tie.  A row with no
    reads has no consensus at all rather than "every base", which is why ``top > 0``
    is part of the test.

    Parameters
    ----------
    counts : (n_positions, 4) int array
        One sample's read tallies ``[A, C, G, T]``, one genome position per row
        (see the module docstring for where these rows come from).

    Example: ``[[10, 2, 0, 0]] -> [[True, False, False, False]]``;
    ``[[3, 0, 3, 0]] -> [[True, False, True, False]]``.
    """
    counts = _as_counts(counts)
    # Row-wise maximum, kept as a column (keepdims) so it broadcasts back against
    # all four columns in the comparison below.
    top = counts.max(axis=1, keepdims=True)
    # "== top" marks every base tied at the maximum -- that IS the tie handling.
    # "top > 0" stops an all-zero row from claiming all four bases as consensus.
    return (counts == top) & (top > 0)


def presence_matrix(
    counts: CountArray,
    coverage: npt.ArrayLike,
    model: npt.NDArray[np.int32],
    min_freq: float,
) -> BoolMatrix:
    """Boolean ``(n_positions, 4)``: is each base credibly present?

    A base passes only if it clears BOTH the error-model read count and the
    ``min_freq`` fraction.  Below roughly 70x the error model is the binding
    constraint; above it, the frequency floor is.

    Parameters
    ----------
    counts : (n_positions, 4) int array
        One sample's read tallies ``[A, C, G, T]``, one row per position.
    coverage : (n_positions,) int array
        Usable reads per position: the row sum A + C + G + T (N excluded).
    model : int array, from ``null_model.build_error_model``
        Lookup ``model[coverage] -> minimum reads`` for an allele to be more
        than sequencing error at that depth.
    min_freq : float
        The population floor: the minimum FRACTION of a position's reads an
        allele must make up to count as present (0.05 = 5%).  A second,
        independent bar because the error model weakens with depth: at 1000x it
        asks for just 7 reads (0.7% of the pile), a level that cross-sample
        contamination, index hopping or mis-mapped reads from a related strain
        can reach even though true sequencing error cannot.  The floor says
        "anything under 5% of the reads is not something we will call part of
        this population, however deep the pile".

    Example (coverage 30, Q30 model needing 3 reads, min_freq 0.05):
    ``[27, 2, 1, 0] -> [True, False, False, False]``; the 2-read C is 6.7% of reads,
    clearing the frequency floor but not the error model.
    """
    counts = _as_counts(counts)
    coverage = np.asarray(coverage)
    # Clip so unusually deep positions reuse the deepest tabulated threshold rather
    # than raising IndexError (harmless: the 5% floor dominates at such depths).
    # model[...] is one lookup per position; [:, None] turns the result into a
    # column so one threshold per row broadcasts against that row's four counts.
    thresholds = model[np.clip(coverage, 0, len(model) - 1)][:, None]
    depth = coverage[:, None]
    # Zero-coverage rows never reach here in production (callers gate on min_cov),
    # but guard the division so a stray one is "nothing present", not a crash.
    frequencies = np.divide(
        counts, depth, out=np.zeros(counts.shape, dtype=np.float64), where=depth > 0
    )
    # Both bars are inclusive (>=): an allele at exactly the threshold count or
    # exactly min_freq passes.  Pinned by tests; keep the two comparisons matched.
    return (counts >= thresholds) & (frequencies >= min_freq)


def credible_alleles(
    counts: CountArray,
    coverage: npt.ArrayLike,
    model: npt.NDArray[np.int32],
    min_freq: float,
) -> BoolMatrix:
    """Boolean ``(n_positions, 4)``: consensus bases plus present minor alleles.

    This is the set popANI reasons about: two samples are "the same" at a position
    if their credible sets overlap at all.

    Parameters are identical to ``presence_matrix``.
    """
    # Union, not intersection: the majority base counts even when it is too shallow
    # to clear the error model ([2,1,1,1] at 5x) -- it is still the best evidence of
    # what is there.  Present minors extend the set beyond the majority.
    return consensus_mask(counts) | presence_matrix(counts, coverage, model, min_freq)


def classify_positions(
    counts1: CountArray,
    coverage1: npt.ArrayLike,
    counts2: CountArray,
    coverage2: npt.ArrayLike,
    model: npt.NDArray[np.int32],
    min_freq: float,
) -> tuple[BoolVector, BoolVector]:
    """Classify each aligned position as a consensus SNP and/or a population SNP.

    Positions must already be restricted to those with at least ``min_cov`` reads
    in BOTH samples -- the engine does that gating before calling here.

    Parameters
    ----------
    counts1, counts2 : (n_positions, 4) int arrays
        The two samples' read tallies for the SAME genome positions, row-aligned:
        row i of ``counts1`` and row i of ``counts2`` are one (contig, position),
        seen through each sample's reads as ``[A, C, G, T]`` counts.
    coverage1, coverage2 : (n_positions,) int arrays
        Row sums of the corresponding counts (A + C + G + T, excluding N) --
        passed separately only because the caller has already computed them.
    model : int array, from ``null_model.build_error_model``
        Coverage -> minimum-reads lookup for allele presence.
    min_freq : float
        Population floor for allele presence -- meaning and rationale in
        ``presence_matrix``.

    Returns
    -------
    (consensus_snp, population_snp) : tuple of boolean arrays, length n_positions
        ``population_snp`` implies ``consensus_snp`` at every position.
    """
    counts1 = _as_counts(counts1)
    counts2 = _as_counts(counts2)
    coverage1 = np.asarray(coverage1)
    coverage2 = np.asarray(coverage2)
    if not (len(counts1) == len(counts2) == len(coverage1) == len(coverage2)):
        raise ValueError(
            "classify_positions requires aligned inputs: got "
            f"{len(counts1)}, {len(coverage1)}, {len(counts2)}, {len(coverage2)}"
        )

    # Each verdict is one set-intersection test, vectorised: AND the two samples'
    # (n, 4) masks -- True only where a base is in BOTH sets -- then any(axis=1)
    # asks "is there at least one shared base at this position?".
    shared_consensus = (consensus_mask(counts1) & consensus_mask(counts2)).any(axis=1)
    shared_credible = (
        credible_alleles(counts1, coverage1, model, min_freq)
        & credible_alleles(counts2, coverage2, model, min_freq)
    ).any(axis=1)

    # A SNP is the ABSENCE of overlap.  Because consensus is a subset of credible,
    # shared_consensus implies shared_credible, hence population_snp implies
    # consensus_snp -- no extra enforcement needed.
    consensus_snp = ~shared_consensus
    population_snp = ~shared_credible
    return consensus_snp, population_snp
