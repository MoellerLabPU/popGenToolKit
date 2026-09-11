"""Comparing every sample pair of one MAG, contig by contig.

The work is organised contig-first, pair-second: each contig's dense count arrays
(from ``profile_io.dense_contig_counts``) are compared for every requested pair
while they are hot in cache, and per-pair tallies accumulate across contigs.  At
the end the totals become genome-level ANI -- a sum over contigs, nothing fancier.

Definitions (denominators count positions, never reads):

* **compared**       -- positions with >= ``min_cov`` reads in BOTH samples.  The
  denominator of both ANIs; the gate is part of the metric definition, which is
  why its value is stamped into every output row.
* **covered_either** -- positions reaching ``min_cov`` in at least one sample;
  ``coverage_overlap = compared / covered_either`` (a descriptive column, not part
  of either ANI; computed genome-wide directly rather than inStrain's per-scaffold
  weighted mean -- documented difference).
* **conANI** = (compared - consensus_SNPs) / compared
* **popANI** = (compared - population_SNPs) / compared

When ZERO positions are compared both ANIs are NaN, never 1.0 -- a perfect score
over no data is exactly the failure ``percent_genome_compared`` exists to expose.

How this module is driven
-------------------------
This is a library module: it has no ``if __name__ == "__main__"`` block because
there is nothing to run directly -- it exports functions.  The runnable entry
point is the ``alleleflux-pairwise-ani`` command (``pairwise_ani.py``), whose
``main()`` loads the data via ``profile_io``, builds the error model, and calls
``compare_all_pairs`` -- by hand you would do the same from a Python session.

Two INDEPENDENT outputs come back: the pair table (the scoreboard -- compared
bases, SNP counts, both ANIs -- computed in full for EVERY pair, always) and the
SNP-locations table (per-position detail rows, gathered ONLY for the pairs named
in ``snp_location_pairs``).  Leaving that set empty changes nothing about the
pair table; it only leaves the locations table empty.

The inputs, concretely
----------------------
``dense_by_sample`` is ``{sample_id -> {contig -> (length, 4) uint16 array}}``,
i.e. one ``profile_io.dense_contig_counts`` result per QC-passing sample.  Every
sample dict carries an array for EVERY contig of the MAG (zeros if uncovered), so
a missing key is a caller bug and raises.

Worked example (the real pair SLG1007 vs SLG1102 on MAG SLG1007_DASTool_bins_35,
min_cov 5, Q30 analytic model; verified 2026-08-27):  1,649,007 positions compared
(54.8% of the 3,007,590 bp genome), 212 consensus SNPs and 29 population SNPs ->
conANI 0.999871, popANI 0.9999824.  For orientation against the investigation's
references: inStrain's argmax rule on the same arrays gives 242 consensus SNPs --
all 30 removed positions are exact ties -- and 29 population SNPs under the
documented thresholds (the spec's "3" was the lenient off-by-one table).  Note the
strict model puts this pair BELOW the 99.999% same-strain line that the lenient
rule had it above -- the 11 -> 2 flip documented in the spec, a consequence of the
decided default, not a bug.
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

from alleleflux.scripts.analysis.ani.classify import BASES, classify_positions

logger = logging.getLogger(__name__)

# One row per sample pair.  min_cov travels with the numbers it shaped, so no
# table can ever be read without knowing which gate produced it.
PAIR_COLUMNS = [
    "MAG_ID", "sample1", "sample2",
    "compared_bases_count", "percent_genome_compared", "coverage_overlap",
    "consensus_SNPs", "population_SNPs", "conANI", "popANI", "min_cov",
]

# One row per flagged position of a requested pair, with BOTH samples' counts --
# the raw material for the future strain-turnover / de-novo work.
SNP_LOCATION_COLUMNS = [
    "MAG_ID", "sample1", "sample2", "contig", "position",
    "consensus_SNP", "population_SNP",
    "A_1", "C_1", "G_1", "T_1", "coverage_1",
    "A_2", "C_2", "G_2", "T_2", "coverage_2",
]


def compare_pair_on_contig(
    counts_a: npt.NDArray[np.uint16],
    counts_b: npt.NDArray[np.uint16],
    model: npt.NDArray[np.int32],
    min_freq: float,
    min_cov: int,
    want_locations: bool = False,
) -> dict:
    """Compare two samples across one contig's dense count arrays.

    Parameters
    ----------
    counts_a, counts_b : (contig_length, 4) uint16 arrays
        The two samples' dense read tallies for the SAME contig, row i = position
        i, columns in A,C,G,T order, zeros where no reads landed.
    model : int array, from ``null_model.build_error_model``
        Coverage -> minimum-reads lookup for allele presence.
    min_freq : float
        Population floor for allele presence (see ``classify.presence_matrix``).
    min_cov : int
        Reads required in BOTH samples for a position to be compared (the user
        toggle; 1 disables the gate beyond "has any reads").
    want_locations : bool
        Also return per-position rows for every flagged SNP (kept off for most
        pairs: storing them for all pairs made a 7.7 GB table in inStrain).

    Returns
    -------
    dict
        ``compared, covered_either, consensus_snps, population_snps`` (ints) and
        ``snp_positions`` (None, or a dict of aligned arrays when
        ``want_locations`` found something).
    """
    # uint16 cells sum into int64 so even a pathologically deep contig cannot
    # overflow the row sums.  Row sum == per-position coverage (A+C+G+T, no N --
    # there are only four columns, so N is structurally excluded).
    coverage_a = counts_a.sum(axis=1, dtype=np.int64)
    coverage_b = counts_b.sum(axis=1, dtype=np.int64)

    # The position gate.  deep_a[i] answers "does sample A have >= min_cov reads
    # at position i?" -- one boolean per contig position.
    deep_a = coverage_a >= min_cov
    deep_b = coverage_b >= min_cov
    # AND: compared only when BOTH samples clear the gate.  This mask IS the
    # denominator of both ANIs for this contig.
    compared_mask = deep_a & deep_b

    result = {
        # True sums as 1, so .sum() over a boolean mask = "how many positions".
        "compared": int(compared_mask.sum()),
        # OR: deep in at least one sample -- the denominator of coverage_overlap.
        "covered_either": int((deep_a | deep_b).sum()),
        # SNP counters stay 0 unless the classifier below finds differences.
        "consensus_snps": 0,
        "population_snps": 0,
        # Filled only when want_locations finds flagged positions.
        "snp_positions": None,
    }
    if result["compared"] == 0:
        # Legitimate outcome for shallow samples; the pair-level NaN handling
        # lives in compare_all_pairs.
        return result

    # Slice down to the compared positions once; everything after is aligned to
    # this index.  astype(int64) because the classifier does integer math and
    # uint16 would overflow inside its comparisons at extreme (clipped) depths.
    # flatnonzero turns the mask into the surviving row indices, e.g.
    # [True, False, True] -> [0, 2].  Everything below stays aligned to it.
    positions = np.flatnonzero(compared_mask)
    sub_a = counts_a[positions].astype(np.int64)
    sub_b = counts_b[positions].astype(np.int64)
    # Coverages sliced with the SAME index remain row-aligned with the counts.
    cov_a = coverage_a[positions]
    cov_b = coverage_b[positions]

    # One boolean verdict per compared position, from the set-based rule.
    consensus_snp, population_snp = classify_positions(
        sub_a, cov_a, sub_b, cov_b, model, min_freq
    )
    # Booleans sum as 0/1: these are the contig's SNP counts.  Note they are
    # tallied BEFORE want_locations is even consulted -- counting SNPs and
    # recording their addresses are independent operations, which is why an
    # empty snp_location_pairs never affects any ANI number.
    result["consensus_snps"] = int(consensus_snp.sum())
    result["population_snps"] = int(population_snp.sum())

    if want_locations:
        # Keep only flagged positions -- a handful per contig -- with both
        # samples' evidence attached, so downstream never re-reads profiles.
        # OR: record a position if it is EITHER kind of SNP (population implies
        # consensus, so in practice this is the consensus-SNP set).
        flagged = np.flatnonzero(consensus_snp | population_snp)
        if len(flagged):
            result["snp_positions"] = {
                # Double indirection on purpose: `flagged` indexes rows of the
                # CLASSIFIER's input, positions[flagged] maps those rows back to
                # real contig coordinates.
                "position": positions[flagged],
                "consensus_SNP": consensus_snp[flagged],
                "population_SNP": population_snp[flagged],
                "counts_1": sub_a[flagged],
                "coverage_1": cov_a[flagged],
                "counts_2": sub_b[flagged],
                "coverage_2": cov_b[flagged],
            }
    return result


def compare_all_pairs(
    dense_by_sample: dict[str, dict[str, np.ndarray]],
    contig_lengths: dict[str, int],
    pairs: list[tuple[str, str]],
    model: npt.NDArray[np.int32],
    min_freq: float,
    min_cov: int,
    genome_length: int,
    mag_id: str = "",
    snp_location_pairs: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every requested sample pair and roll totals up to genome level.

    Parameters
    ----------
    dense_by_sample : dict
        ``sample_id -> {contig -> (length, 4) array}`` -- one
        ``profile_io.dense_contig_counts`` result per sample.  Every sample must
        carry every contig in ``contig_lengths`` (profile_io guarantees it);
        a missing key raises KeyError rather than silently skipping data.
    contig_lengths : dict
        ``contig -> length``; iteration order fixes the contig processing order.
    pairs : list of (sample1, sample2)
        Which pairs to compare (the CLI enumerates them, sorted, sample1 < sample2).
    genome_length : int
        Sum of this MAG's contig lengths; denominator of percent_genome_compared.
    snp_location_pairs : set, optional
        The subset of ``pairs`` whose flagged positions are returned row-by-row.
        FILLED BY THE CALLER: the CLI builds it from ``--store_snp_locations``
        (default ``within_subject`` = same-mouse pairs only; ``all`` was a 7.7 GB
        table in inStrain; ``none`` = empty set).  An empty set is the off
        switch, not an error: membership tests are simply False for every pair,
        so no location rows are gathered -- the pair table is unaffected.

    Returns
    -------
    (pair_table, snp_locations) : tuple of DataFrames
        ``pair_table`` has PAIR_COLUMNS, one row per pair (NaN ANIs when nothing
        was compared); ``snp_locations`` has SNP_LOCATION_COLUMNS, empty when no
        pair was requested or nothing was flagged.
    """
    # Normalise None -> empty set so the membership test below just works.
    # (`pair in set()` is always False -- an empty set means "record locations
    # for nobody", while every pair still gets its full scoreboard row.)
    snp_location_pairs = snp_location_pairs or set()

    # One accumulator per pair; the keys deliberately mirror
    # compare_pair_on_contig's counters so results sum straight in.
    totals = {
        pair: {"compared": 0, "covered_either": 0, "consensus_snps": 0, "population_snps": 0}
        for pair in pairs
    }
    # Small per-(contig, pair) frames of flagged positions, concatenated ONCE at
    # the end -- growing one big frame row-by-row is quadratic, this is not.
    location_rows = []

    # Contig-outer, pair-inner: one contig's arrays stay hot in cache while every
    # pair uses them, instead of every pair re-touching every contig.
    for contig in contig_lengths:
        for pair in pairs:
            sample_a, sample_b = pair
            # Two-level lookup: dense_by_sample[SAMPLE][CONTIG] -> (length, 4)
            # array.  profile_io.dense_contig_counts built each sample's inner
            # {contig -> array} dict; the CLI stacked them per sample.  ANI is a
            # PAIR metric, so we pull the SAME contig from TWO different
            # samples' dicts and compare the arrays row by row.
            # Direct indexing on purpose: profile_io gives every sample an array
            # for every contig (zeros if uncovered), so a missing key is a caller
            # bug and must raise, not skip (silently dropping a contig would
            # deflate `compared` and inflate ANI).
            counts_a = dense_by_sample[sample_a][contig]
            counts_b = dense_by_sample[sample_b][contig]

            # Only requested pairs pay for gathering per-position rows.  With an
            # empty set this is False every time -- the counters above still run.
            want = pair in snp_location_pairs
            contig_result = compare_pair_on_contig(
                counts_a, counts_b, model, min_freq, min_cov, want_locations=want
            )
            # Fold this contig's four counters into the pair's genome totals.
            for key in ("compared", "covered_either", "consensus_snps", "population_snps"):
                totals[pair][key] += contig_result[key]

            found = contig_result["snp_positions"]
            if want and found is not None:
                # Assemble this contig's flagged rows immediately (small frames)
                # rather than holding raw arrays until the end.
                frame = pd.DataFrame({
                    "MAG_ID": mag_id,
                    "sample1": sample_a,
                    "sample2": sample_b,
                    "contig": contig,
                    "position": found["position"],
                    "consensus_SNP": found["consensus_SNP"],
                    "population_SNP": found["population_SNP"],
                    "coverage_1": found["coverage_1"],
                    "coverage_2": found["coverage_2"],
                })
                # Split the (n, 4) count blocks into flat per-base columns
                # (A_1, C_1, ...) so the output TSV is plain rectangular data.
                for index, base in enumerate(BASES):
                    frame[f"{base}_1"] = found["counts_1"][:, index]
                    frame[f"{base}_2"] = found["counts_2"][:, index]
                # Fix the column order once here; concat preserves it.
                location_rows.append(frame[SNP_LOCATION_COLUMNS])

    # Totals -> one genome-level row per pair.
    records = []
    for sample_a, sample_b in pairs:
        # Unpack this pair's genome-wide sums into readable names.
        pair_totals = totals[(sample_a, sample_b)]
        compared = pair_totals["compared"]
        either = pair_totals["covered_either"]
        consensus = pair_totals["consensus_snps"]
        population = pair_totals["population_snps"]

        # With nothing compared an ANI is undefined: NaN, never a flattering 1.0.
        con_ani = (compared - consensus) / compared if compared else np.nan
        pop_ani = (compared - population) / compared if compared else np.nan

        records.append({
            "MAG_ID": mag_id,
            "sample1": sample_a,
            "sample2": sample_b,
            "compared_bases_count": compared,
            # The honesty column: how much genome the verdict rests on.
            "percent_genome_compared": compared / genome_length if genome_length else np.nan,
            # 0.0 = the two samples' deep regions never coincide (defined!);
            # NaN only when NEITHER sample is deep anywhere (either == 0).
            "coverage_overlap": compared / either if either else np.nan,
            "consensus_SNPs": consensus,
            "population_SNPs": population,
            "conANI": con_ani,
            "popANI": pop_ani,
            "min_cov": min_cov,
        })

    # columns= enforces schema AND order even when records is empty, so the CLI
    # can always write a header-only file for a MAG with too few samples.
    pair_table = pd.DataFrame(records, columns=PAIR_COLUMNS)
    # Nothing gathered -> an EMPTY frame that still carries the full schema, so
    # downstream writers always emit a header instead of a zero-column mystery.
    snp_locations = (
        pd.concat(location_rows, ignore_index=True)
        if location_rows
        else pd.DataFrame(columns=SNP_LOCATION_COLUMNS)
    )

    if len(pair_table):
        logger.info(
            f"MAG {mag_id}: {len(pairs)} pairs compared; median compared bases "
            f"{pair_table['compared_bases_count'].median():,.0f}; "
            f"{len(snp_locations)} SNP-location rows"
        )
    else:
        logger.warning(f"MAG {mag_id}: no pairs to compare")
    return pair_table, snp_locations
