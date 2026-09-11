"""Loading AlleleFlux profiles for the pairwise-ANI comparison.

Three jobs, in the order the CLI uses them:

1. **Which samples?**  ``qc_passing_samples`` reuses AlleleFlux's own QC verdict
   (``coverage_threshold_passed`` in the per-MAG ``*_QC.tsv``) rather than inventing
   a new sample filter, unioning across the per-timepoint QC files.
2. **Where is each profile?**  ``profile_path`` mirrors the layout the profiling
   rule writes: ``{profiles_dir}/{sample}/{sample}_{MAG}_profiled.tsv.gz``.
3. **Into what shape?**  ``load_profile`` + ``dense_contig_counts`` turn the sparse
   profile rows into one dense ``(contig_length, 4)`` array per contig, the shape
   ``classify.classify_positions`` consumes; ``contig_lengths_for_mag`` supplies the
   array sizes from the reference (``.fai`` preferred).

The inputs, concretely
----------------------
A profile file has one row per COVERED genome position (0-based) of one sample x
one MAG.  A real line from SLG1007's profile, and where its numbers land::

    contig        position  ref  total_coverage  A  C  G  T  N
    k141_102792   0         G    4               2  0  1  1  0

    dense["k141_102792"][0] == [2, 0, 1, 1]      (uint16, order A,C,G,T)
    coverage there          == 4                 (A+C+G+T; the file's
                                                  total_coverage column also counts
                                                  N calls and is deliberately unused)

Two file-format facts drive everything here.  Profiles contain **only covered
positions** -- an absent row means zero coverage, not missing data -- so the dense
arrays start at zero and rows are scattered into them.  And positions are
**0-based** (``profile_mags.py`` converts from mpileup's 1-based), so a position is
a direct array index.  Base columns stay in the file's own ``A, C, G, T`` order:
the set-based classifier never depends on column order, so nothing is reordered
anywhere in this feature.

Library module: imported by the rest of the ani package and by tests; never run
directly, hence no ``if __name__ == "__main__"`` block.  The one runnable entry
point of this feature is the ``alleleflux-pairwise-ani`` command (pairwise_ani.py).

Older profiles (pre-samtools-optimisation)
carry an extra ``mapq_scores`` column; ``usecols`` makes the loader indifferent
to it.
"""

import logging
import os

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.classify import BASES
from alleleflux.scripts.utilities.utilities import build_contig_length_index

logger = logging.getLogger(__name__)

# Read only what the comparison needs: ref_base, total_coverage, N, mapq_scores and
# gene_id are all unused.  usecols also makes the loader indifferent to the extra
# mapq_scores column in older profiles -- and raises if a needed column is absent.
_PROFILE_USECOLS = ["contig", "position", *BASES]
_PROFILE_DTYPES = {
    "contig": str,
    "position": "int64",
    **{base: "int32" for base in BASES},
}

# Metadata carried through to the output tables.  Everything is read as str:
# DRIDO's group labels are 20/40 and subject ids are plain numbers, and letting
# pandas sniff them as int64 makes every later string comparison match nothing.
_QC_META_COLUMNS = ["sample_id", "subjectID", "group", "time", "replicate"]
# genome_size and breadth ride along (numeric, NOT str) purely so the CLI can
# cross-check its own reference-derived genome length and profile-derived breadth
# against the values QC recorded -- catching "QC ran against a different
# FASTA/mapping" or "profile changed since QC" drift loudly instead of silently.
_QC_KEEP_COLUMNS = _QC_META_COLUMNS + ["genome_size", "breadth"]
_QC_DTYPES = {column: str for column in _QC_META_COLUMNS}

# Dense arrays use uint16 (2 bytes/cell): memory scales as genome_length x 4 x
# n_samples, ~2.2 GB for a 3 Mb MAG across 90 samples.  Counts above 65,535 are
# clipped -- presence thresholds are two digits at most, so a clip can never flip
# a verdict; it only nudges a frequency by <0.1% at absurd depth.
_DENSE_DTYPE = np.uint16
_DENSE_MAX = np.iinfo(_DENSE_DTYPE).max


def profile_path(profiles_dir: str, sample_id: str, mag_id: str) -> str:
    """Path to one sample's profile for one MAG.

    Mirrors the profiling rule's output layout -- change them together.

    Example: ``profile_path("/profiles", "SLG1007", "bins_35")`` ->
    ``/profiles/SLG1007/SLG1007_bins_35_profiled.tsv.gz``.
    """
    return os.path.join(profiles_dir, sample_id, f"{sample_id}_{mag_id}_profiled.tsv.gz")


def qc_passing_samples(qc_files: list[str], mag_id: str) -> pd.DataFrame:
    """Samples that cleared AlleleFlux QC for this MAG, unioned across QC files.

    One QC file exists per MAG **per timepoint combination** and lists EVERY
    sample of that combination (passing or not).  Combinations overlap on shared
    timepoints, so the same sample can be judged in several files; the union keeps
    a sample wherever it passed at least once, collapsed to one row per
    ``sample_id``.

    Worked example (combinations pre_end and pre_post share "pre")::

        QC/QC_pre_end/MAG_X_QC.tsv    60 rows, 30 pass   (pre and end samples)
        QC/QC_pre_post/MAG_X_QC.tsv   60 rows, 29 pass   (pre and post samples)

        qc_passing_samples([qc_pre_end, qc_pre_post], "MAG_X")
        -> 43 rows: 59 passing rows minus 16 "pre" samples that passed in both
           files (their duplicate rows are identical, so first-wins loses nothing)

    Note the caller chooses the scope: pass ONE file and the roster is exactly
    that combination's samples; pass all files (what the Snakemake rule does) and
    the roster spans every timepoint, which is what within-subject cross-timepoint
    pairs need.

    Parameters
    ----------
    qc_files : list of str
        Per-MAG QC TSVs (``{mag}_QC.tsv``), one per timepoint combination --
        quality_control.py's real output files.
    mag_id : str
        Only used for log/error context; the files are already per-MAG.

    Returns
    -------
    pandas.DataFrame
        One row per QC-passing sample with the metadata columns present in the
        files (``sample_id, subjectID, group, time, replicate`` as str, plus
        ``genome_size`` numeric for the CLI's reference cross-check).
        MAY BE EMPTY: a low-abundance MAG where no sample passes QC is a
        legitimate outcome -- the caller writes header-only outputs, not an error.

    Raises
    ------
    ValueError
        If a QC file lacks ``coverage_threshold_passed`` (malformed, or
        quality_control.py did not finish), or if no QC files are supplied at all.
    """
    frames = []
    for qc_file in qc_files:
        # Force the label columns to str on the way in (_QC_DTYPES note above);
        # columns not named there, like the bool verdict, keep pandas' inference.
        table = pd.read_csv(qc_file, sep="\t", dtype=_QC_DTYPES)
        if "coverage_threshold_passed" not in table.columns:
            raise ValueError(
                f"QC file {qc_file} has no 'coverage_threshold_passed' column; "
                "was quality_control.py run to completion?"
            )
        # The verdict column arrives as real bools (pandas parses TSV True/False);
        # keep only rows that passed.
        passing = table[table["coverage_threshold_passed"] == True]  # noqa: E712
        # Keep whatever metadata columns this file has -- 'time' is absent for
        # single-timepoint data, and the outputs simply omit what was never there.
        available = [c for c in _QC_KEEP_COLUMNS if c in passing.columns]
        frames.append(passing[available].copy())

    if not frames:
        raise ValueError(f"No QC files supplied for MAG {mag_id}")

    # A sample judged usable at several timepoint combinations appears in several
    # files; keep its first row. Safe because group/subject/time come from the one
    # input metadata sheet, so duplicate rows carry identical values.
    combined = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["sample_id"])
        .reset_index(drop=True)
    )
    if combined.empty:
        logger.warning(
            f"MAG {mag_id}: no sample passed QC in any of {len(qc_files)} QC file(s)"
        )
    else:
        logger.info(f"MAG {mag_id}: {len(combined)} samples passed QC")
    return combined


def contig_lengths_for_mag(
    fasta_path: str, mag_mapping_path: str, mag_id: str
) -> dict[str, int]:
    """Contig -> length for the contigs belonging to one MAG.

    A thin per-MAG wrapper around the shared
    ``utilities.build_contig_length_index``, which prefers the samtools ``.fai``
    index beside the FASTA (milliseconds) over parsing the 0.5 GB reference
    (seconds, repeated by every per-MAG job).  What this wrapper adds is the
    ANI-specific contract: subset to ONE MAG's contigs, and fail loudly when the
    mapping and the reference disagree -- dense arrays sized from wrong lengths
    would silently corrupt every comparison downstream.

    Parameters
    ----------
    fasta_path : str
        The reference FASTA (config ``input.fasta_path``); ``{fasta_path}.fai``
        is used by the shared helper when present.
    mag_mapping_path : str
        TSV with columns ``mag_id, contig_id`` (config ``input.mag_mapping_path``).
    mag_id : str
        Which MAG's contigs to return.

    Raises
    ------
    ValueError
        If the mapping lacks its two required columns, names no contigs for this
        MAG, or names contigs the reference does not contain (mapping and FASTA
        out of sync).
    """
    # dtype=str: MAG and contig ids are labels; a numeric-looking id must not
    # silently become int64 and then fail every string comparison.
    mapping = pd.read_csv(mag_mapping_path, sep="\t", dtype=str)
    missing_columns = {"mag_id", "contig_id"} - set(mapping.columns)
    if missing_columns:
        raise ValueError(
            f"MAG mapping {mag_mapping_path} is missing columns "
            f"{sorted(missing_columns)}; found {list(mapping.columns)}"
        )

    # The mapping covers the whole assembly (all MAGs); keep this MAG's contigs.
    wanted = set(mapping.loc[mapping["mag_id"] == mag_id, "contig_id"])
    if not wanted:
        raise ValueError(f"MAG {mag_id} has no contigs listed in {mag_mapping_path}")

    # The shared helper does the actual length reading (.fai fast path when
    # present).  It returns lengths for every mapped contig genome-wide; that is
    # cheap (the index is ~1.5 MB) and keeps ONE implementation serving QC,
    # positions-QC and ANI alike.
    all_lengths = build_contig_length_index(fasta_path, mag_mapping_path)

    # Iterate the REFERENCE-derived dict (built in .fai order), not `wanted`:
    # `wanted` is a set of strings, and Python randomises string hashing per
    # process, so iterating it would give a different contig order on every run
    # -- and the engine's contig-outer loop would emit SNP-location rows in a
    # different order each time. This keeps the output reproducible.
    lengths = {
        contig: int(length) for contig, length in all_lengths.items() if contig in wanted
    }

    # Every mapped contig must exist in the reference; anything left over means
    # the mapping and the FASTA are from different assemblies.
    absent = wanted - set(lengths)
    if absent:
        raise ValueError(
            f"MAG {mag_id}: {len(absent)} contig(s) from the mapping are not in "
            f"{fasta_path}, e.g. {sorted(absent)[:3]}. Mapping and reference disagree."
        )
    return lengths


def load_profile(path: str, include_gene_id: bool = False) -> pd.DataFrame:
    """Read one profile file into ``contig, position, A, C, G, T, coverage``.

    ``coverage`` is computed here as A + C + G + T -- NOT the file's
    ``total_coverage`` column, which also counts N calls and would silently
    disagree with inStrain-style coverage semantics.

    Parameters
    ----------
    path : str
        A ``{sample}_{MAG}_profiled.tsv.gz`` file (see the module docstring for
        the row format).
    include_gene_id : bool
        Also return the profile's ``gene_id`` column (str, "" where blank).
        Off for the ANI comparison; on for candidate annotation downstream.

    Raises
    ------
    ValueError
        If a required column is absent (pandas raises on ``usecols``) or the same
        ``(contig, position)`` appears twice -- both mean a corrupt profile.
    """
    # usecols prunes the unused columns, tolerates the legacy mapq_scores one, and
    # raises ValueError if a required column is missing (the fail-loud we want).
    # gene_id is read only on request: the ANI comparison never needs it, the
    # strain-turnover tools do (for candidate annotation).
    usecols = _PROFILE_USECOLS + (["gene_id"] if include_gene_id else [])
    dtypes = {**_PROFILE_DTYPES, **({"gene_id": str} if include_gene_id else {})}
    table = pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtypes)

    # Two rows for one (contig, position) would double-count reads downstream --
    # that is a corrupt profile to report, not something to dedupe quietly.
    if table.duplicated(subset=["contig", "position"]).any():
        offenders = table[table.duplicated(subset=["contig", "position"], keep=False)]
        raise ValueError(
            f"Duplicate (contig, position) rows in {path}:\n"
            f"{offenders.head(10).to_string(index=False)}"
        )

    # Fix the column order regardless of file layout, then .copy() so the coverage
    # assignment below writes to a real frame, not a view (SettingWithCopyWarning).
    kept = ["contig", "position", *BASES] + (["gene_id"] if include_gene_id else [])
    table = table[kept].copy()
    # A + C + G + T only: N calls are deliberately excluded (module docstring).
    table["coverage"] = table[list(BASES)].sum(axis=1).astype("int32")
    if include_gene_id:
        # Blank gene_id (intergenic positions) reads as NaN; normalise to "".
        table["gene_id"] = table["gene_id"].fillna("")
        # gene_id goes LAST so the ANI columns sit at the same offsets whether or
        # not the optional column was requested.
        table = table[["contig", "position", *BASES, "coverage", "gene_id"]]
    return table


def dense_contig_counts(
    profile_df: pd.DataFrame, contig_lengths: dict[str, int]
) -> dict[str, np.ndarray]:
    """Scatter one sample's profile rows into dense ``(contig_length, 4)`` arrays.

    Positions absent from the profile stay zero, which is exactly what absence
    means.  EVERY contig in ``contig_lengths`` gets an array -- even one this
    sample never covered -- so all samples share one position index per contig and
    the engine can compare arrays blindly.

    Parameters
    ----------
    profile_df : pandas.DataFrame
        Output of ``load_profile`` (0-based positions, counts in A,C,G,T order).
    contig_lengths : dict
        ``contig -> length`` for this MAG, from ``contig_lengths_for_mag``.

    Raises
    ------
    ValueError
        If the profile names a contig outside ``contig_lengths`` (profile and MAG
        mapping out of sync) or a position beyond its contig's length (profile and
        reference out of sync).  Both are contract violations, not data to drop.
    """
    # Pre-allocate EVERY contig at zero: zero is the true value for uncovered
    # positions, and identical shapes across samples let the engine compare blindly.
    dense = {
        contig: np.zeros((length, len(BASES)), dtype=_DENSE_DTYPE)
        for contig, length in contig_lengths.items()
    }

    # sort=False: iterate contigs in file order, no pointless re-sort.
    # observed=True: if a caller ever hands us categorical contigs, don't explode
    # into every unobserved category (the regional_contrast bug of 2026-07).
    for contig, rows in profile_df.groupby("contig", sort=False, observed=True):
        if contig not in dense:
            raise ValueError(
                f"Profile contains contig {contig!r} that is not in this MAG's "
                f"mapping ({len(contig_lengths)} contigs). Profile and mapping disagree."
            )
        length = contig_lengths[contig]
        # 0-based positions ARE array indices; bounds-check before scattering
        # (initial= keeps max/min safe even on an empty slice).
        positions = rows["position"].to_numpy()
        if positions.max(initial=-1) >= length or positions.min(initial=0) < 0:
            raise ValueError(
                f"Profile positions for contig {contig} fall outside its length "
                f"{length}: min={positions.min()}, max={positions.max()}. "
                "Profile and reference FASTA disagree."
            )
        counts = rows[list(BASES)].to_numpy()
        # Saturate instead of overflowing on the way down to uint16 (see the
        # _DENSE_DTYPE note at the top of the module).
        np.clip(counts, 0, _DENSE_MAX, out=counts)
        # Fancy-index scatter: one vectorised write per contig instead of a Python
        # loop over ~3M rows.
        dense[contig][positions] = counts.astype(_DENSE_DTYPE)

    return dense
