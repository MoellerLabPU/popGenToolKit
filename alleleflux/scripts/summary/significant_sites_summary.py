#!/usr/bin/env python3
"""
Significant-Sites Summary for AlleleFlux runs
=============================================

After an AlleleFlux run finishes, the pipeline writes one ``p_value_summary`` table
per *comparison* (a timepoint pair crossed with a group pair). Each table has one row
per genomic site per statistical test, including an FDR-corrected ``q_value``.

Those tables are huge (tens of millions of rows, multiple GB each) and answer questions
only after you slice them. This helper boils them down to the two things you usually want
right after a run:

    1. "In each test and group, which MAGs have how many sites with q < 0.05?"
    2. "What gene is each of those significant sites in?"

It is **generic**: it does not hard-code Drido (or any) timepoint / group / test names.
It discovers everything from the directory layout and the file columns.

--------------------------------------------------------------------------------
Input layout (what AlleleFlux produces)
--------------------------------------------------------------------------------
You point ``--input-dir`` at a run's ``p_value_summary/`` directory. Inside it, every
*comparison* gets its own subdirectory, and each subdirectory holds one TSV per test
family::

    p_value_summary/
      5mo_22mo-40_AL/                                          <- one comparison
        p_value_summary_single_sample_5mo_22mo-40_AL.tsv       <- single-sample tests
        p_value_summary_two_sample_unpaired_5mo_22mo-40_AL.tsv <- two-sample test
      5mo_22mo-2D_1D/
        ...

The group pair (``40_AL``) appears in BOTH the subdirectory name (``5mo_22mo-40_AL``) and
the filename suffix (``...-40_AL.tsv``); the ``period`` column inside the file is just the
timepoints (``5mo_22mo``). We read the comparison and group pair from the directory name.

Each TSV has columns like::

    period  mag_id  contig  position  gene_id  test_type  [group_analyzed]  min_p_value  source_file  q_value

``group_analyzed`` is present for single-sample tests (e.g. ``1D``, ``AL``) and absent for
two-sample tests. We handle both transparently.

--------------------------------------------------------------------------------
Outputs (three TSVs in --outdir)
--------------------------------------------------------------------------------
1. ``<prefix>_mag_counts_long.tsv``  -- one row per (comparison, test, group, MAG):

       comparison      period   group_pair  test_type             group_analyzed  mag_id     n_sig_sites
       5mo_22mo-40_AL  5mo_22mo  40_AL       single_sample_tTest   1D              MRGM_1020  37

2. ``<prefix>_mag_counts_pivot.tsv`` -- the same numbers as a scannable matrix:
   rows = MAG, columns = each (comparison | test | group) combination, cells = counts,
   blanks filled with 0. Great for eyeballing which MAG / comparison is "hot".

       mag_id     5mo_22mo-40_AL|single_sample_tTest|1D   5mo_22mo-40_AL|two_sample_unpaired_tTest   ...
       MRGM_1020  37                                      4                                          ...

3. ``<prefix>_sig_sites.tsv``        -- one row per significant site, with its gene:

       comparison      period   group_pair  test_type            group_analyzed  mag_id     contig             position  gene_id                q_value
       5mo_22mo-40_AL  5mo_22mo  40_AL       single_sample_tTest  1D              MRGM_1020  MRGM_1020_contig_1  44746     MRGM_1020_contig_1_42  0.0031

--------------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------------
    python significant_sites_summary.py \\
        --input-dir /path/to/run/longitudinal/p_value_summary \\
        --outdir    /path/to/run/longitudinal/significant_sites_summary \\
        --q-threshold 0.05 \\
        --cpus 8

--------------------------------------------------------------------------------
Why it does not run out of memory
--------------------------------------------------------------------------------
A single input file can be ~7 GB / ~42M rows, but the q < 0.05 subset is tiny. Each file
is read in *chunks*; every chunk is filtered down to its significant rows immediately, and
only those survivors are kept. Peak memory per worker is therefore a few tens of MB, no
matter how large the file. Files are processed in parallel across ``--cpus`` workers.
"""

import argparse
import functools
import logging
import multiprocessing
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging

logger = logging.getLogger(__name__)


class DataIntegrityError(ValueError):
    """A p_value_summary file violates an assumption we refuse to summarize past.

    Subclasses ``ValueError`` so callers can catch it as one, but is distinct enough that the
    per-file worker's broad ``except Exception`` (which drops unreadable files as ``None``)
    can let it propagate instead -- a corrupt-data error must STOP the run, not be silently
    skipped.
    """

# --------------------------------------------------------------------------------------
# Column contract
# --------------------------------------------------------------------------------------
# Columns we try to read from each p_value_summary TSV. Anything in OPTIONAL_COLS that is
# missing from a given file (e.g. two-sample files have no `group_analyzed`) is simply
# skipped -- we never assume it exists. Restricting `usecols` to just these keeps the
# chunked read fast and lean (we ignore min_p_value, source_file, etc.).
REQUIRED_COLS = ["mag_id", "q_value"]
# `min_p_value` is read (in addition to q_value) so the per-cell stats can count sites by
# RAW p as well as by FDR-corrected q -- the heatmap's "% of sites p<0.05" panel uses raw p,
# the "min BH-corrected p-value" panel uses q. It is NOT added to the per-site detail table.
OPTIONAL_COLS = [
    "period", "contig", "position", "gene_id", "test_type", "group_analyzed", "min_p_value",
]
WANTED_COLS = REQUIRED_COLS + OPTIONAL_COLS

# Final column order for the per-site detail table (output 3). Columns the source file
# lacked are filled with "" before writing so every run produces the same schema.
SIG_SITE_COLS = [
    "comparison",
    "period",
    "group_pair",
    "test_type",
    "group_analyzed",
    "mag_id",
    "contig",
    "position",
    "gene_id",
    "q_value",
]

# Dimensions that define one "test and group" cell. A site that is significant in two
# different comparisons is counted once in EACH -- we never pool across comparisons.
GROUP_DIMS = ["comparison", "period", "group_pair", "test_type", "group_analyzed"]

# One row per heatmap cell = (comparison, test, group, MAG). Unlike the count tables above,
# this covers EVERY MAG with >=1 tested site -- including MAGs with zero significant sites
# (the dim rows the reference figure still shows). Feeds both heatmap panels:
#   pct_sig_p / pct_sig_q -> "% of sites significant"; min_q_value (+ stars) -> "min BH p".
CELL_STATS_COLS = [
    "comparison", "period", "group_pair", "test_type", "group_analyzed", "mag_id",
    "n_total_sites", "n_sig_p", "n_sig_q", "pct_sig_p", "pct_sig_q",
    "min_p_value", "min_q_value",
]

# Per-cell aggregate columns BEFORE percentages are derived (what harvest accumulates and
# returns; build_mag_cell_stats_long adds pct_sig_p / pct_sig_q on top).
CELL_AGG_KEYS = ["comparison", "period", "group_pair", "test_type", "group_analyzed", "mag_id"]
CELL_AGG_COLS = CELL_AGG_KEYS + [
    "n_total_sites", "n_sig_p", "n_sig_q", "min_p_value", "min_q_value",
]

# Rows per read chunk. Bounds each worker's peak memory to ~tens of MB regardless of file
# size (multi-GB p_value_summary files are read incrementally, not all at once). Internal
# constant -- the default holds across every dataset we've run; change here if a future
# file is large enough to need it.
CHUNKSIZE = 1_000_000


def parse_comparison(file_path: Path) -> tuple[str, str]:
    """
    Pull the comparison label and group pair out of a result file's parent directory name.

    AlleleFlux names each comparison directory ``<timepoints>-<group_pair>``. We split on
    the FIRST hyphen: timepoints never contain a hyphen (they look like ``5mo``, ``10mo``,
    ``5mo_22mo``), but group pairs are joined with an underscore (``40_AL``), so the first
    hyphen is an unambiguous boundary. The rollup always discovers files by rglob-ing the
    ``p_value_summary/`` tree, so a file's parent is always its comparison dir.

    Examples
    --------
    >>> parse_comparison(Path("p_value_summary/5mo_22mo-40_AL/p_value_summary_single_sample_5mo_22mo-40_AL.tsv"))
    ('5mo_22mo-40_AL', '40_AL')
    >>> parse_comparison(Path("p_value_summary/T1-fat_control/p_value_summary_two_sample_unpaired_T1-fat_control.tsv"))
    ('T1-fat_control', 'fat_control')

    Returns
    -------
    (comparison, group_pair) : tuple[str, str]
        ``comparison`` is the full directory name; ``group_pair`` is everything after its
        first hyphen, or "" if the directory name has none.
    """
    comparison = file_path.parent.name
    group_pair = comparison.split("-", 1)[1] if "-" in comparison else ""
    return comparison, group_pair


def across_time_suffix_for(file_path: Path) -> str:
    """Return ``'_across_time'`` for an across-time test family, else ``''``.

    AlleleFlux's ``p_value_summary`` writes ``lmm`` and ``lmm_across_time`` (and ``cmh`` /
    ``cmh_across_time``) to SEPARATE files, but stamps the same collapsed ``test_type``
    value (``'LMM'`` / ``'CMH'``) inside both. Once the rollup concatenates files, the two
    blur together. The filename is the only reliable tell -- ``p_value_summary_lmm_across_time_...``
    vs ``p_value_summary_lmm_...`` -- so we use it to suffix the test_type and keep them apart.

    The substring check is safe here: no timepoint or group label in AlleleFlux outputs
    contains ``across_time`` (timepoints look like ``T0_T21``/``5mo_10mo``, groups like
    ``antibiotic_control``).

    Examples
    --------
    >>> across_time_suffix_for(Path("x/p_value_summary_lmm_across_time_T0_T21.tsv"))
    '_across_time'
    >>> across_time_suffix_for(Path("x/p_value_summary_lmm_T0_T21.tsv"))
    ''
    """
    return "_across_time" if "across_time" in file_path.name else ""


def _assert_no_na_values(chunk: pd.DataFrame, file_path: Path) -> None:
    """Fail loud if ``q_value`` or ``min_p_value`` has any NA in this chunk.

    A NA in either column silently corrupts the stats: ``q_value < thr`` and
    ``min_p_value < thr`` are both False for NA (so the site is miscounted as NON-significant
    in the numerator), while ``n_total_sites`` uses ``size`` and still counts it in the
    denominator -- deflating the percentage and hiding the problem. ``p_value_summary.py``
    already drops NA ``min_p_value`` rows upstream, so any NA here means the source is wrong;
    we stop rather than emit confidently-wrong numbers. Checked per chunk so we fail on the
    first bad chunk with the offending file named.
    """
    for col in ("q_value", "min_p_value"):
        if col in chunk.columns:
            n_na = int(chunk[col].isna().sum())
            if n_na:
                raise DataIntegrityError(
                    f"{file_path.name}: found {n_na} NA value(s) in column '{col}'. "
                    f"q_value/min_p_value must be non-NA (upstream p_value_summary.py drops "
                    f"NA rows) -- investigate the source file before summarizing."
                )


def _aggregate_chunk_cell_stats(
    chunk: pd.DataFrame, q_threshold: float, p_threshold: float
) -> pd.DataFrame:
    """Per-(period, test_type, group_analyzed, mag_id) partial aggregate for ONE chunk.

    Unlike the significant-site harvest, this folds in EVERY row -- significant or not -- so
    the heatmap can show a total-site denominator (for "% significant") and a min q-value for
    every MAG, including MAGs with zero significant sites. We aggregate per chunk and merge
    the partials at the end of the file, so peak memory stays at a few hundred cells, never
    the whole file. Counts are summed and min-values are min-combined across partials.

    ``group_analyzed`` / ``period`` are normalized to "" when the source lacks them so the
    grouping keys line up across test families (two-sample tests have no group_analyzed).
    """
    work = chunk.copy()
    if "group_analyzed" not in work.columns:
        work["group_analyzed"] = ""
    else:
        work["group_analyzed"] = work["group_analyzed"].fillna("")
    if "period" not in work.columns:
        work["period"] = ""
    # Boolean significance flags; .sum() over them gives the per-cell significant counts.
    work["_sig_q"] = work["q_value"] < q_threshold
    work["_sig_p"] = work["min_p_value"] < p_threshold
    keys = ["period", "test_type", "group_analyzed", "mag_id"]
    return (
        work.groupby(keys, dropna=False)
        .agg(
            n_total_sites=("q_value", "size"),
            n_sig_q=("_sig_q", "sum"),
            n_sig_p=("_sig_p", "sum"),
            min_q_value=("q_value", "min"),
            min_p_value=("min_p_value", "min"),
        )
        .reset_index()
    )


def harvest_significant_sites(
    file_path: Path, q_threshold: float, p_threshold: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Read ONE p_value_summary TSV in chunks; return its q<threshold rows AND its tested combos.

    This is the per-file worker that runs in the multiprocessing pool. It never holds the
    whole file in memory: each ~1M-row chunk is filtered to significant rows on the spot,
    and only those (usually a handful) are accumulated.

    The significant DataFrame is annotated with the comparison/group_pair parsed from the
    directory name, a normalized ``group_analyzed`` (empty string when the source lacks the
    column, e.g. two-sample tests), and a whitespace-stripped ``gene_id`` (empty ⇒
    intergenic site, kept as "").

    Parameters
    ----------
    file_path : Path
        One ``p_value_summary_*.tsv`` file.
    q_threshold : float
        Keep rows with ``q_value < q_threshold`` (also defines ``n_sig_q`` in cell stats).
    p_threshold : float
        Raw-p significance cutoff for ``n_sig_p`` in the cell stats (heatmap "% of sites
        p<0.05" panel). Does not affect the significant-site detail/count tables.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
        ``(sig, universe, cell_stats)`` where ``sig`` is the significant rows
        (columns = SIG_SITE_COLS, possibly empty); ``universe`` lists every
        ``(comparison, test_type, group_analyzed)`` combo present in the file regardless of
        significance (so the pivot keeps a 0-column for a tested-but-empty combo); and
        ``cell_stats`` (columns = CELL_AGG_COLS) holds per-(comparison, test, group, MAG)
        totals, significant-site counts (by p and by q), and min p/q over ALL rows -- one row
        per MAG even when it has zero significant sites. Returns ``None`` if the file is
        unusable (missing ``q_value`` column, or a read error); ``None`` is logged and dropped
        by the orchestrator.
    """
    comparison, group_pair = parse_comparison(file_path)
    try:
        # Peek at just the header so we know which optional columns this file actually has.
        header_cols = pd.read_csv(file_path, sep="\t", nrows=0).columns.tolist()
    except Exception as e:  # unreadable / truncated file
        logger.error(f"Could not read header of {file_path}: {e}")
        return None

    missing_required = [c for c in REQUIRED_COLS if c not in header_cols]
    if missing_required:
        logger.warning(
            f"Skipping {file_path}: missing required column(s) {missing_required}."
        )
        return None

    usecols = [c for c in WANTED_COLS if c in header_cols]
    has_ga = "group_analyzed" in usecols
    suffix = across_time_suffix_for(file_path)

    # Collect the significant slice of each chunk; concat once at the end (cheap, the
    # surviving rows are few). Separately, track every (test_type, group_analyzed) the file
    # contains -- even rows that are NOT significant -- so the pivot can keep a 0-column for
    # a combo that was tested but had no q<threshold site.
    # Cell stats need both a test label and raw p; without either we can still produce the
    # significant-site tables, just not the per-cell heatmap aggregates for this file.
    can_cell = "test_type" in usecols and "min_p_value" in usecols

    kept_chunks: list[pd.DataFrame] = []
    seen_combos: set[tuple] = set()
    cell_partials: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(
            file_path,
            sep="\t",
            usecols=usecols,
            chunksize=CHUNKSIZE,
            low_memory=False,
        ):
            # Refuse to summarize past corrupt p-values -- fail loud, do not silently miscount.
            _assert_no_na_values(chunk, file_path)
            if "test_type" in chunk.columns:
                cols = ["test_type"] + (["group_analyzed"] if has_ga else [])
                for rec in chunk[cols].drop_duplicates().itertuples(index=False, name=None):
                    tt = rec[0]
                    ga = rec[1] if has_ga else ""
                    seen_combos.add((str(tt), "" if pd.isna(ga) else str(ga)))
            hits = chunk[chunk["q_value"] < q_threshold]
            if not hits.empty:
                kept_chunks.append(hits)
            # Fold this chunk's ALL-rows aggregate into the per-cell stats (same single pass).
            if can_cell:
                cell_partials.append(
                    _aggregate_chunk_cell_stats(chunk, q_threshold, p_threshold)
                )
    except DataIntegrityError:
        # Corrupt data (NA p/q): STOP the run -- never swallow this as a skippable file.
        raise
    except Exception as e:
        logger.error(f"Error while streaming {file_path}: {e}")
        return None

    if kept_chunks:
        sig = pd.concat(kept_chunks, ignore_index=True)
    else:
        # No significant sites in this file -- return an empty, correctly-typed frame so
        # the schema is uniform downstream.
        sig = pd.DataFrame(columns=usecols)

    # --- Annotate with directory-derived and normalized columns -----------------------
    sig["comparison"] = comparison
    sig["group_pair"] = group_pair

    # `group_analyzed` may be absent (two-sample tests) -> normalize to "" so every row
    # has the column and grouping keys line up across test families.
    if "group_analyzed" not in sig.columns:
        sig["group_analyzed"] = ""
    else:
        sig["group_analyzed"] = sig["group_analyzed"].fillna("")

    # gene_id arrives with trailing spaces in some AlleleFlux outputs; "" == intergenic.
    if "gene_id" in sig.columns:
        sig["gene_id"] = sig["gene_id"].fillna("").astype(str).str.strip()

    # Keep LMM/CMH distinct from their across_time variants, which p_value_summary collapses
    # to the same test_type value but writes to a separate file (see across_time_suffix_for).
    if suffix and "test_type" in sig.columns and not sig.empty:
        tt = sig["test_type"].astype(str)
        sig["test_type"] = tt.where(tt.str.endswith(suffix), tt + suffix)

    # Guarantee every expected column exists (fill blanks for any the source lacked),
    # then return them in canonical order.
    for col in SIG_SITE_COLS:
        if col not in sig.columns:
            sig[col] = ""

    # Universe of tested (comparison, test_type, group_analyzed) combos, with the same
    # across_time relabel applied so it lines up with the significant rows' labels.
    universe = pd.DataFrame(
        [
            (
                comparison,
                tt + suffix if (suffix and not tt.endswith(suffix)) else tt,
                ga,
            )
            for tt, ga in seen_combos
        ],
        columns=["comparison", "test_type", "group_analyzed"],
    ).drop_duplicates(ignore_index=True)

    # --- Per-cell stats: merge the chunk partials, annotate, relabel across_time ----------
    if cell_partials:
        cell_keys = ["period", "test_type", "group_analyzed", "mag_id"]
        cell = (
            pd.concat(cell_partials, ignore_index=True)
            .groupby(cell_keys, dropna=False)
            .agg(
                n_total_sites=("n_total_sites", "sum"),
                n_sig_q=("n_sig_q", "sum"),
                n_sig_p=("n_sig_p", "sum"),
                min_q_value=("min_q_value", "min"),
                min_p_value=("min_p_value", "min"),
            )
            .reset_index()
        )
        cell["comparison"] = comparison
        cell["group_pair"] = group_pair
        # Same across_time relabel as the sig rows, so cell labels line up with everything else.
        if suffix:
            tt = cell["test_type"].astype(str)
            cell["test_type"] = tt.where(tt.str.endswith(suffix), tt + suffix)
        cell = cell[CELL_AGG_COLS]
    else:
        cell = pd.DataFrame(columns=CELL_AGG_COLS)

    return sig[SIG_SITE_COLS], universe, cell


def build_mag_counts_long(sig_sites: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the per-site table into per-(comparison, test, group, MAG) significant counts.

    This is output 1a: one tidy row per cell with an ``n_sig_sites`` column. "Keep separate
    per comparison and test" lives here -- we group by every dimension in GROUP_DIMS plus
    ``mag_id`` and never merge across them.

    Example
    -------
    Input (per-site) rows for MAG ``MRGM_1020`` in one cell -> a single output row with
    ``n_sig_sites = <number of those rows>``.
    """
    if sig_sites.empty:
        return pd.DataFrame(columns=GROUP_DIMS + ["mag_id", "n_sig_sites"])

    long_counts = (
        sig_sites.groupby(GROUP_DIMS + ["mag_id"], dropna=False)
        .size()
        .reset_index(name="n_sig_sites")
        .sort_values(GROUP_DIMS + ["n_sig_sites"], ascending=[True] * len(GROUP_DIMS) + [False])
        .reset_index(drop=True)
    )
    return long_counts


def _pivot_col_label(comparison, test_type, group_analyzed) -> str:
    """Join the non-empty dimensions into one flat pivot column header.

    e.g. ``5mo_22mo-40_AL|single_sample_tTest|1D`` or, with no group_analyzed,
    ``5mo_22mo-40_AL|two_sample_unpaired_tTest``.
    """
    return "|".join(str(v) for v in (comparison, test_type, group_analyzed) if str(v) != "")


def build_mag_counts_pivot(
    long_counts: pd.DataFrame, universe: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Reshape the long counts into a scannable MAG x (comparison|test|group) matrix.

    Output 1b. Cells are the q<threshold counts; a MAG with no significant sites in a given
    column shows 0. When ``universe`` (the set of tested ``(comparison, test_type,
    group_analyzed)`` combos) is given, every tested combo is kept as a column even if NO
    MAG had a significant site there -- so the matrix shows a 0-column rather than silently
    omitting a test that was run. A flat ``|``-joined header per cell loads cleanly into a
    spreadsheet (vs an awkward multi-index pivot).
    """
    if long_counts.empty:
        pivot = pd.DataFrame(columns=["mag_id"])
    else:
        labeled = long_counts.copy()
        labeled["col_label"] = labeled.apply(
            lambda r: _pivot_col_label(r["comparison"], r["test_type"], r["group_analyzed"]),
            axis=1,
        )
        pivot = (
            labeled.pivot_table(
                index="mag_id",
                columns="col_label",
                values="n_sig_sites",
                aggfunc="sum",
                fill_value=0,
            )
            .reset_index()
        )
        # pivot_table names the column axis "col_label"; drop it for a clean TSV.
        pivot.columns.name = None

    # Keep tested-but-all-zero combos as explicit 0-columns (not silently dropped).
    if universe is not None and not universe.empty:
        for lbl in {
            _pivot_col_label(r.comparison, r.test_type, r.group_analyzed)
            for r in universe.itertuples(index=False)
        }:
            if lbl and lbl not in pivot.columns:
                pivot[lbl] = 0

    # Deterministic, scannable column order: mag_id first, then labels sorted.
    label_cols = sorted(c for c in pivot.columns if c != "mag_id")
    return pivot[["mag_id"] + label_cols]


def build_mag_cell_stats_long(cell_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Finish the per-cell stats: merge across files and derive the two significance percentages.

    ``harvest_significant_sites`` returns one cell-stats frame per file; the orchestrator
    concatenates them. A heatmap cell ``(comparison, test, group, MAG)`` lives in exactly one
    file, so the groupby here is normally a no-op, but it is defensive: if a cell were ever
    split across files the counts sum and the min p/q min-combine correctly.

    Adds ``pct_sig_p`` / ``pct_sig_q`` = 100 * n_sig / n_total. A cell with ``n_total == 0``
    (no tested sites) yields 0% rather than NaN/inf. Output columns = CELL_STATS_COLS.
    """
    if cell_stats is None or cell_stats.empty:
        return pd.DataFrame(columns=CELL_STATS_COLS)

    agg = (
        cell_stats.groupby(CELL_AGG_KEYS, dropna=False)
        .agg(
            n_total_sites=("n_total_sites", "sum"),
            n_sig_p=("n_sig_p", "sum"),
            n_sig_q=("n_sig_q", "sum"),
            min_p_value=("min_p_value", "min"),
            min_q_value=("min_q_value", "min"),
        )
        .reset_index()
    )

    # Guard against divide-by-zero: a 0-total cell becomes 0%, not NaN/inf.
    safe_total = agg["n_total_sites"].where(agg["n_total_sites"] > 0)
    agg["pct_sig_p"] = (100.0 * agg["n_sig_p"] / safe_total).fillna(0.0)
    agg["pct_sig_q"] = (100.0 * agg["n_sig_q"] / safe_total).fillna(0.0)

    agg = agg.sort_values(
        ["comparison", "test_type", "group_analyzed", "pct_sig_q"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return agg[CELL_STATS_COLS]


def drop_empty_cells(cell_long: pd.DataFrame, mode: str = "none") -> pd.DataFrame:
    """
    Optionally prune comparisons and/or test-columns that have NO significant site anywhere.

    "Significant" here is q-based (``n_sig_q``), matching how the rest of AlleleFlux defines
    significance. Modes:
      * ``none``        -> return unchanged (matches the reference figure, which shows all).
      * ``comparisons`` -> drop every row whose ``comparison`` summed to zero significant sites.
      * ``tests``       -> drop every row whose ``(test_type, group_analyzed)`` summed to zero.
      * ``both``        -> apply both prunings.

    Pruning uses ``groupby(...).transform("sum")`` so the decision is made on the WHOLE group
    (all MAGs in that comparison / test), not row-by-row.
    """
    if mode == "none" or cell_long.empty:
        return cell_long
    out = cell_long
    if mode in ("comparisons", "both"):
        keep = out.groupby("comparison")["n_sig_q"].transform("sum") > 0
        out = out[keep]
    if mode in ("tests", "both"):
        keep = out.groupby(["test_type", "group_analyzed"], dropna=False)["n_sig_q"].transform("sum") > 0
        out = out[keep]
    return out.reset_index(drop=True)


def round_up_the_significant_sites(
    input_dir: Path, q_threshold: float, cpus: int, p_threshold: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrator: discover every p_value_summary file, harvest significant sites in
    parallel, and stitch the results into one per-site DataFrame.

    Returns ``(all_sites, universe, cell_stats)`` -- the concatenated per-site table (possibly
    empty), the deduped set of tested ``(comparison, test_type, group_analyzed)`` combos (for
    keeping 0-columns in the pivot), and the concatenated per-cell stats (one row per
    comparison/test/group/MAG, including zero-significant MAGs; columns = CELL_AGG_COLS).
    Raises SystemExit(1) if no input files are found at all -- that almost always means
    ``--input-dir`` is wrong.
    """
    # Resolve to an absolute path FIRST: when called with a relative dir (e.g. "." from
    # inside a comparison folder), rglob would otherwise yield relative paths whose
    # ``.parent.name`` is "" -- blanking the comparison/group_pair columns.
    input_dir = input_dir.resolve()
    files = sorted(input_dir.rglob("p_value_summary_*.tsv"))
    if not files:
        logger.error(
            f"No 'p_value_summary_*.tsv' files found under {input_dir}. "
            f"Point --input-dir at a run's p_value_summary/ directory."
        )
        sys.exit(1)
    logger.info(f"Found {len(files)} p_value_summary file(s) under {input_dir}.")

    worker = functools.partial(
        harvest_significant_sites, q_threshold=q_threshold, p_threshold=p_threshold
    )

    sig_frames: list[pd.DataFrame] = []
    combo_frames: list[pd.DataFrame] = []
    cell_frames: list[pd.DataFrame] = []
    # imap_unordered streams results as workers finish; tqdm shows file-level progress.
    with multiprocessing.Pool(processes=cpus) as pool:
        for result in tqdm(
            pool.imap_unordered(worker, files),
            total=len(files),
            desc="Scanning p_value_summary files",
            unit="file",
        ):
            if result is None:
                continue
            sig_df, universe_df, cell_df = result
            if sig_df is not None and not sig_df.empty:
                sig_frames.append(sig_df)
            if universe_df is not None and not universe_df.empty:
                combo_frames.append(universe_df)
            if cell_df is not None and not cell_df.empty:
                cell_frames.append(cell_df)

    universe = (
        pd.concat(combo_frames, ignore_index=True).drop_duplicates(ignore_index=True)
        if combo_frames
        else pd.DataFrame(columns=["comparison", "test_type", "group_analyzed"])
    )
    cell_stats = (
        pd.concat(cell_frames, ignore_index=True)
        if cell_frames
        else pd.DataFrame(columns=CELL_AGG_COLS)
    )

    if not sig_frames:
        logger.warning(
            f"No sites with q_value < {q_threshold} found in any file. "
            f"Writing empty (header-only) site table; pivot keeps tested combos as 0-columns."
        )
        return pd.DataFrame(columns=SIG_SITE_COLS), universe, cell_stats

    all_sites = pd.concat(sig_frames, ignore_index=True)
    logger.info(
        f"Collected {len(all_sites):,} significant site-rows (q < {q_threshold}) "
        f"across {all_sites['mag_id'].nunique()} MAG(s); "
        f"{len(universe)} tested test_type×group combo(s); "
        f"{len(cell_stats):,} per-cell stat rows."
    )
    return all_sites, universe, cell_stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize AlleleFlux p_value_summary tables: per-(test, group) MAG counts of "
            "q<0.05 sites (long + pivot) and a per-site table annotated with its gene."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="A run's p_value_summary/ directory (searched recursively).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory to write the three output TSVs (created if absent).",
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=0.05,
        help="Significance cutoff applied to the q_value column.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help=(
            "Raw-p cutoff for the cell-stats 'n_sig_p'/'pct_sig_p' columns (the heatmap "
            "'%% of sites p<0.05' panel). Does not affect the significant-site tables."
        ),
    )
    parser.add_argument(
        "--drop-empty",
        choices=["none", "comparisons", "tests", "both"],
        default="none",
        help=(
            "Prune the cell-stats table: drop comparisons and/or test-columns with NO "
            "significant (q<threshold) site anywhere. 'none' keeps everything (matches the "
            "reference figure)."
        ),
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel workers (one file per worker at a time).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="significant_sites",
        help="Filename prefix for the three output tables.",
    )
    args = parser.parse_args()

    # Package logging (matches every other alleleflux CLI): idempotent, root-logger-safe.
    setup_logging()

    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # 1. Discover + harvest significant sites, the tested-combo universe, AND the per-cell
    #    stats (totals/sig-counts/min-p/min-q for every MAG) in a single parallel pass.
    sig_sites, universe, cell_stats = round_up_the_significant_sites(
        args.input_dir, args.q_threshold, args.cpus, p_threshold=args.p_threshold
    )

    # 2. Derive the count tables (universe keeps 0-columns for tested-but-empty combos) and
    #    the per-cell heatmap table (percentages + optional pruning of all-empty cells).
    long_counts = build_mag_counts_long(sig_sites)
    pivot_counts = build_mag_counts_pivot(long_counts, universe=universe)
    cell_long = drop_empty_cells(build_mag_cell_stats_long(cell_stats), mode=args.drop_empty)

    # 3. Write all four outputs.
    args.outdir.mkdir(parents=True, exist_ok=True)
    long_path = args.outdir / f"{args.prefix}_mag_counts_long.tsv"
    pivot_path = args.outdir / f"{args.prefix}_mag_counts_pivot.tsv"
    sites_path = args.outdir / f"{args.prefix}_sig_sites.tsv"
    cell_path = args.outdir / f"{args.prefix}_mag_cell_stats_long.tsv"

    long_counts.to_csv(long_path, sep="\t", index=False)
    pivot_counts.to_csv(pivot_path, sep="\t", index=False)
    sig_sites.to_csv(sites_path, sep="\t", index=False)
    cell_long.to_csv(cell_path, sep="\t", index=False)

    logger.info(f"Wrote MAG counts (long):  {long_path}  ({len(long_counts):,} rows)")
    logger.info(f"Wrote MAG counts (pivot): {pivot_path}  ({len(pivot_counts):,} MAG rows)")
    logger.info(f"Wrote per-site detail:    {sites_path}  ({len(sig_sites):,} sites)")
    logger.info(f"Wrote per-cell stats:     {cell_path}  ({len(cell_long):,} cell rows)")
    logger.info("Done.")


if __name__ == "__main__":
    main()
