"""
File-by-file comparison helpers for the AlleleFlux end-to-end regression test.

Each public ``compare_*`` function returns ``None`` when the two files match
and raises ``AssertionError`` with a helpful, bounded diff message when they
do not.  Comparators are deliberately tolerant of:

  * Row ordering (parallel pipelines do not guarantee order).
  * Tiny float drift from BLAS / threading non-determinism.
  * NaN equality (``NaN == NaN`` is treated as a match here).

What is NOT tolerated:

  * Schema changes (column added / removed / renamed).
  * Row count changes.
  * Differences in non-float values.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Column-level comparison rules (which columns are sign-insensitive, which are
# non-deterministic diagnostics skipped on value) live in _scope.py alongside
# the rest of the AlleleFlux comparison contract — imported here so this
# mechanism carries no hard-coded AlleleFlux column names.
from tests.regression._scope import (
    IGNORE_VALUE_SUBSTRINGS,
    SIGN_INSENSITIVE_SUBSTRINGS,
)

logger = logging.getLogger(__name__)

# How many mismatched rows to print before truncating.  Keeps assertion
# messages readable when a whole column has drifted.
MAX_DIFF_ROWS = 10

# Default float tolerances.  Tight enough to flag any real numerical change,
# loose enough to ignore BLAS / threading noise.  Override per-column if a
# specific output needs different treatment.
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12


def _is_ignored_value_column(col: str) -> bool:
    """Return True for non-deterministic diagnostic columns to skip on value.

    Matches any column whose name contains a token in
    ``IGNORE_VALUE_SUBSTRINGS`` (defined in _scope.py — the LMM
    convergence-warning and converged-flag columns).  Such columns must still
    EXIST in both files — the schema gate enforces that — but their cell values
    are not compared, because the statsmodels optimizer produces a run-dependent
    warning count / converged flag on near-degenerate fits while the p-value
    stays identical.

    Parameters
    ----------
    col : str
        Column name to test.

    Returns
    -------
    bool

    Examples
    --------
    >>> _is_ignored_value_column("A_n_warnings")
    True
    >>> _is_ignored_value_column("A_converged_LMM")
    True
    >>> _is_ignored_value_column("A_p_value_LMM")   # the science — still compared
    False
    """
    return any(s in col for s in IGNORE_VALUE_SUBSTRINGS)


def _is_sign_insensitive(col: str) -> bool:
    """Return True for columns whose sign is unstable across pipeline runs.

    A column matches if its name contains any of ``SIGN_INSENSITIVE_SUBSTRINGS``
    (defined in _scope.py: ``_change``, ``_diff_mean``, ``_diff``).  Matching
    columns are compared on magnitude only — see _scope.py for the upstream
    root cause.

    Parameters
    ----------
    col : str
        Column name to test.

    Returns
    -------
    bool

    Examples
    --------
    >>> _is_sign_insensitive("A_frequency_diff_mean")
    True
    >>> _is_sign_insensitive("A_change")
    True
    >>> _is_sign_insensitive("A_frequency")   # plain frequency — sign matters
    False
    """
    return any(s in col for s in SIGN_INSENSITIVE_SUBSTRINGS)


def _read_table(path: Path) -> pd.DataFrame:
    """Read a tabular file as a DataFrame, picking the loader by extension.

    This is deliberately a tiny, fail-loud reader rather than a reuse of the
    production ``load_allele_freq_inputs`` helper: a regression oracle should
    stay independent of the code it judges, and should raise on any file type
    it does not explicitly understand instead of silently guessing.

    Parameters
    ----------
    path : Path
        File to read.  Must end in ``.parquet``, ``.tsv``, or ``.tsv.gz``.

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    ValueError
        If the extension is not a recognised tabular format.

    Examples
    --------
    >>> _read_table(Path("scores.tsv.gz"))        # doctest: +SKIP
    >>> _read_table(Path("cache.parquet"))        # doctest: +SKIP
    >>> _read_table(Path("notes.txt"))            # doctest: +SKIP
    ValueError: Unsupported tabular extension: notes.txt
    """
    s = str(path)
    if s.endswith(".parquet"):
        return pd.read_parquet(path)
    # pandas auto-detects gzip from the ``.gz`` suffix, so one read_csv call
    # covers both ``.tsv`` and ``.tsv.gz``.  low_memory=False forces a single
    # dtype-inference pass over the whole column (no mixed-dtype chunk surprises).
    if s.endswith(".tsv.gz") or s.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    # Unknown extension: refuse rather than guess.  A silent wrong-format read
    # would let a real regression slip through as a "pass".
    raise ValueError(f"Unsupported tabular extension: {path}")


def _sort_key_columns(df: pd.DataFrame) -> list[str]:
    """Pick stable identity columns to sort by so two unordered files line up.

    Parallel pipeline stages emit rows in nondeterministic order, so before
    a row-for-row comparison both files must be sorted into a canonical order.
    This picks the identity-like columns to sort on.

    Preference order matches columns that appear across most AlleleFlux
    outputs.  Falls back to up to three non-float columns if none of the
    preferred keys are present (float columns make poor sort keys — they are
    the values we expect to drift).

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose columns are inspected (values are not read).

    Returns
    -------
    list of str
        Column names to pass to ``DataFrame.sort_values``.

    Examples
    --------
    >>> _sort_key_columns(pd.DataFrame(columns=["contig", "position", "A_change"]))
    ['contig', 'position']
    >>> _sort_key_columns(pd.DataFrame(columns=["weird_id", "score"]))  # no preferred key
    ['weird_id', 'score']
    """
    preferred = [
        "period", "mag_id", "contig", "position", "gene_id", "sample_id",
        "group", "group_analyzed", "time", "nucleotide", "test", "test_type",
        "feature", "rank", "taxon", "subjectID", "replicate", "source_file",
    ]
    keys = [c for c in preferred if c in df.columns]
    if keys:
        return keys
    # No known identity column — fall back to whatever non-float columns exist
    # (capped at three to keep the sort cheap on wide tables).
    non_float = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    return non_float[:3] if non_float else list(df.columns[:1])


def _format_diff(diffs: pd.DataFrame, file_label: str) -> str:
    """Render a bounded mismatch report for an assertion message.

    Truncates to ``MAX_DIFF_ROWS`` rows so that a fully-drifted column does
    not bury the assertion message under thousands of lines.

    Parameters
    ----------
    diffs : pandas.DataFrame
        The mismatched rows, typically with ``__actual`` / ``__golden``
        column suffixes from the side-by-side join in ``compare_tables``.
    file_label : str
        Human-readable file name to prefix the message with.

    Returns
    -------
    str
        A multi-line message: a count, the first ``MAX_DIFF_ROWS`` rows, and
        an "... and N more" footer when truncated.

    Examples
    --------
    >>> d = pd.DataFrame({"A__actual": [0.5], "A__golden": [0.6]})
    >>> print(_format_diff(d, "scores.tsv"))   # doctest: +NORMALIZE_WHITESPACE
    scores.tsv: 1 mismatched rows
     A__actual  A__golden
           0.5        0.6
    """
    head = diffs.head(MAX_DIFF_ROWS).to_string(index=False)
    extra = ""
    if len(diffs) > MAX_DIFF_ROWS:
        extra = f"\n... and {len(diffs) - MAX_DIFF_ROWS:,} more mismatched rows"
    return f"{file_label}: {len(diffs):,} mismatched rows\n{head}{extra}"


def compare_tables(
    actual: Path,
    golden: Path,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Compare two tabular files (TSV / TSV.gz / parquet) row-for-row.

    The comparison is intentionally tolerant of cosmetic differences that vary
    between pipeline runs and strict about anything that signals a real change:

    Tolerated
        * Column *order* (compared as a set, then realigned).
        * Row *order* (both files sorted by identity columns first).
        * Tiny float drift within ``rtol`` / ``atol`` (BLAS / threading noise).
        * ``NaN`` in the same cell of both files.
        * Sign flips on derivative columns (see ``_is_sign_insensitive``).

    Caught
        * Column added / removed / renamed (the column *set* differs).
        * Row count differences.
        * Any float beyond tolerance, or any non-float value differing.

    Parameters
    ----------
    actual : Path
        File produced by the pipeline run under test.
    golden : Path
        Known-good reference file defining the expected contract.
    rtol, atol : float
        Relative / absolute float tolerances forwarded to ``numpy.isclose``.

    Returns
    -------
    None
        Returns silently when the files match.

    Raises
    ------
    AssertionError
        On any column-set, row-count, or value mismatch.  The message is
        bounded to ``MAX_DIFF_ROWS`` rows.

    Examples
    --------
    >>> compare_tables(Path("run/scores.tsv"), Path("golden/scores.tsv"))  # doctest: +SKIP
    # returns None — files match within tolerance

    A golden value of 0.5 vs an actual of 0.5000000001 passes (float noise),
    but 0.5 vs 0.6 raises AssertionError.  On an ``A_change`` column, 0.5 vs
    -0.5 *passes* (sign-insensitive) while 0.5 vs 0.6 still fails.
    """
    a = _read_table(actual)
    g = _read_table(golden)

    rel = actual.name

    # --- Schema gate: the column SET must match exactly --------------------
    # Order-tolerant on purpose: column order is not a semantic property of
    # these outputs and varies run-to-run.  A differing set, however, means a
    # column was added / removed / renamed — a real schema regression.
    if set(a.columns) != set(g.columns):
        only_actual = sorted(set(a.columns) - set(g.columns))
        only_golden = sorted(set(g.columns) - set(a.columns))
        raise AssertionError(
            f"{rel}: column-set mismatch.\n"
            f"  actual only: {only_actual}\n"
            f"  golden only: {only_golden}"
        )

    # Sets matched — now impose golden's column ORDER on ``a`` so every
    # later positional / column-name operation can assume the two frames are
    # laid out identically.
    a = a[list(g.columns)]

    # --- Shape gate: same number of rows -----------------------------------
    if len(a) != len(g):
        raise AssertionError(
            f"{rel}: row count mismatch — actual={len(a):,}, golden={len(g):,}"
        )

    if len(a) == 0:
        return  # both empty, trivially equal

    # --- Canonicalise row order --------------------------------------------
    # kind="stable" keeps rows that tie on the sort keys in their original
    # relative order (the default quicksort does not), so ties resolve the
    # same way in both files.  reset_index(drop=True) then stamps a fresh
    # 0..N-1 index on each frame — critical, because the value comparisons
    # below rely on row i of ``a`` lining up with row i of ``g``.  Without the
    # reset, pandas would align by the shuffled original labels and silently
    # undo the sort.
    sort_keys = _sort_key_columns(g)
    a_sorted = a.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    g_sorted = g.sort_values(sort_keys, kind="stable").reset_index(drop=True)

    # --- Value comparison, column by column --------------------------------
    # diff_mask[i] becomes True if ANY column disagrees in row i.
    diff_mask = pd.Series(False, index=a_sorted.index)
    for col in g_sorted.columns:
        # Non-deterministic diagnostic columns (LMM convergence warnings /
        # converged flag) are present in the schema but their values are not
        # reproducible run-to-run — skip the value check, keep the schema gate.
        if _is_ignored_value_column(col):
            continue
        a_col = a_sorted[col]
        g_col = g_sorted[col]
        # is_float_dtype is the fork in the road: float columns get the
        # tolerant numeric path, everything else gets exact equality.  We use
        # the pandas predicate (not ``dtype == float``) so float32 / nullable
        # Float64 columns also take the tolerant path — see project memory
        # comparisons docs.
        if pd.api.types.is_float_dtype(g_col):
            a_vals = a_col.to_numpy(dtype=float)
            g_vals = g_col.to_numpy(dtype=float)
            if _is_sign_insensitive(col):
                # Fold sign away for derivative columns whose +/- is unstable
                # upstream — magnitude must still match, so this is NOT
                # magnitude-insensitive.  See SIGN_INSENSITIVE_SUBSTRINGS (_scope.py).
                a_vals = np.abs(a_vals)
                g_vals = np.abs(g_vals)
            # equal_nan=True so a NaN in the same cell of both files matches.
            mismatch = ~np.isclose(
                a_vals, g_vals, rtol=rtol, atol=atol, equal_nan=True,
            )
        else:
            # Non-float (strings, ints, categoricals): exact inequality, but
            # forgive the both-NaN case (NaN != NaN is True in raw pandas).
            both_na = a_col.isna() & g_col.isna()
            mismatch = (a_col != g_col).to_numpy() & ~both_na.to_numpy()
        diff_mask |= mismatch

    if diff_mask.any():
        # Build a side-by-side report: mask every matching cell to NA so the
        # join keeps only the offending values, suffix the two sources, then
        # drop columns that are all-NA (i.e. columns that never mismatched).
        joined = a_sorted.where(diff_mask, other=pd.NA).join(
            g_sorted.where(diff_mask, other=pd.NA), lsuffix="__actual", rsuffix="__golden"
        )
        diffs = joined.loc[diff_mask].dropna(how="all", axis=1)
        raise AssertionError(_format_diff(diffs, rel))


def compare_json(actual: Path, golden: Path) -> None:
    """Compare two JSON files as parsed Python objects.

    Comparing the *parsed* objects (not the raw text) makes the check
    insensitive to dict key order and whitespace — only the semantic content
    matters.

    Parameters
    ----------
    actual, golden : Path
        JSON files to compare.

    Returns
    -------
    None
        Returns silently when the parsed objects are equal.

    Raises
    ------
    AssertionError
        If the parsed objects differ.

    Examples
    --------
    >>> # '{"a": 1, "b": 2}' vs '{"b": 2, "a": 1}'  -> match (key order ignored)
    >>> # '{"a": 1}'         vs '{"a": 2}'          -> AssertionError
    >>> compare_json(Path("run/meta.json"), Path("golden/meta.json"))  # doctest: +SKIP
    """
    a = json.loads(actual.read_text())
    g = json.loads(golden.read_text())
    if a != g:
        raise AssertionError(
            f"{actual.name}: JSON content mismatch.\n"
            f"  actual: {a}\n"
            f"  golden: {g}"
        )


def compare_file(
    actual: Path,
    golden: Path,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Route a single file to the right comparator based on its extension.

    Dispatch is keyed off the *golden* file's extension (golden defines the
    contract).  Tabular files go to ``compare_tables``, ``.json`` to
    ``compare_json``, and anything else falls back to a byte-for-byte compare.

    Parameters
    ----------
    actual, golden : Path
        The pair of files to compare.
    rtol, atol : float
        Float tolerances forwarded to ``compare_tables`` (ignored for JSON /
        binary files).  Default to the strict module constants; loosen when the
        golden and the run under test use different float precisions (e.g. a
        float64 golden vs a float32 branch).

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        Propagated from the chosen comparator, or raised directly when the
        byte-for-byte fallback finds differing content.

    Examples
    --------
    >>> compare_file(Path("run/scores.tsv.gz"), Path("golden/scores.tsv.gz"))  # doctest: +SKIP
    >>> compare_file(Path("run/summary.json"), Path("golden/summary.json"))    # doctest: +SKIP
    >>> compare_file(Path("run/plot.png"), Path("golden/plot.png"))            # doctest: +SKIP
    """
    s = str(golden)
    if s.endswith(".parquet") or s.endswith(".tsv") or s.endswith(".tsv.gz"):
        compare_tables(actual, golden, rtol=rtol, atol=atol)
    elif s.endswith(".json"):
        compare_json(actual, golden)
    else:
        # Last resort: byte-for-byte compare.  Used only for file types we
        # have not added a structured comparator for.  Open through gzip when
        # the suffix calls for it so we compare decompressed payloads, not the
        # (timestamp-bearing) gzip container bytes.
        opener = gzip.open if s.endswith(".gz") else open
        with opener(actual, "rb") as fa, opener(golden, "rb") as fg:
            if fa.read() != fg.read():
                raise AssertionError(f"{actual.name}: binary content differs")


def _rel_matches_prefix(rel: Path, prefixes: tuple[str, ...]) -> bool:
    """True if ``rel`` (as a POSIX string) starts with any of ``prefixes``.

    Matching is plain ``str.startswith`` on the forward-slash relative path,
    which deliberately serves two shapes at once:

      * **Directory prefixes** — ``"significance_tests"`` matches every file
        under ``significance_tests/...``.
      * **Filename prefixes** — ``"eligibility_table"`` matches the top-level
        file ``eligibility_table_pre_post-treatment_control.tsv``.

    Parameters
    ----------
    rel : Path
        Path relative to the tree root.
    prefixes : tuple of str
        Prefixes to test against ``rel.as_posix()``.

    Returns
    -------
    bool
    """
    s = rel.as_posix()
    return any(s.startswith(p) for p in prefixes)


def compare_trees(
    actual_root: Path,
    golden_root: Path,
    include_prefixes: Iterable[str] | None = None,
    allow_golden_only: Iterable[str] | None = None,
    exclude_substrings: Iterable[str] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> list[str]:
    """Compare every file under ``golden_root`` against its twin in ``actual_root``.

    This is the top-level entry point for the regression test.  Unlike the
    ``compare_*`` helpers (which raise on the first problem), this collects
    ALL failures so a single run reports every drifted file at once.

    The golden tree defines the contract: a golden file with no actual twin
    is reported as ``MISSING``, but an *extra* file present only in
    ``actual_root`` is tolerated (newly-added outputs do not fail old goldens).

    Scoping for branch-vs-branch comparison
    ---------------------------------------
    When the golden is captured from a *different* branch (e.g. ``main``) than
    the run under test, some divergence is intended — columns dropped, files
    renamed or re-formatted, optional outputs disabled.  Two optional filters
    let the caller compare only the scientifically-meaningful, structurally
    stable outputs while ignoring the known-intentional differences:

      * ``include_prefixes`` — an allowlist.  When given, only golden files
        whose relative path matches one of the prefixes are considered; every
        other golden file is skipped entirely (neither compared nor reported).
      * ``allow_golden_only`` — within the swept files, golden files whose
        relative path matches one of these prefixes are NOT flagged ``MISSING``
        when absent from ``actual_root``.  Use for outputs the run under test
        deliberately does not emit (e.g. taxonomic levels the branch's
        ``taxa_score_levels`` config omits).  A *shared* file under such a
        prefix is still compared strictly.

    Both prefix lists are matched with :func:`_rel_matches_prefix`
    (``str.startswith`` on the POSIX relative path).

    Parameters
    ----------
    actual_root : Path
        Root of the directory tree produced by the run under test.
    golden_root : Path
        Root of the reference tree.  Every file beneath it is checked,
        subject to ``include_prefixes``.
    include_prefixes : iterable of str, optional
        Allowlist of path prefixes to compare.  ``None`` (default) compares
        every golden file — the original behaviour.
    allow_golden_only : iterable of str, optional
        Path prefixes under which a golden file missing from ``actual_root``
        is an expected absence, not a ``MISSING`` failure.
    exclude_substrings : iterable of str, optional
        Substrings matched anywhere in a golden file's POSIX relative path; any
        match is skipped entirely (not compared, not flagged), even when it
        falls under an ``include_prefixes`` prefix.  Substring (not prefix) so a
        token like ``two_sample_paired`` also catches it mid-path in the
        ``p_value_summary`` filenames.  Use to carve non-comparable subtrees out
        of a broad include prefix (e.g. drop the ``preprocessed`` intermediate
        inputs and the non-deterministic ``two_sample_paired`` test from the
        ``significance_tests`` prefix for the single scenario).
    rtol, atol : float
        Float tolerances forwarded to every per-file comparison.  Default to
        the strict module constants (``DEFAULT_RTOL`` / ``DEFAULT_ATOL``).
        Loosen when the golden was captured at a different float precision than
        the run under test -- e.g. a float64 ``main`` golden vs a branch that
        stores frequencies as float32, where ~1e-7 rounding is expected and
        ``rtol=1e-9`` is unachievable by construction.

    Returns
    -------
    list of str
        Human-readable failure messages; an empty list means everything
        matched.  Each entry is one of: a ``MISSING: <rel>`` line, an
        assertion message from a comparator, or a ``comparator error`` line
        when a comparator itself blew up (e.g. unreadable file).

    Examples
    --------
    >>> fails = compare_trees(Path("run_output"), Path("golden_output"))  # doctest: +SKIP
    >>> # Compare only the statistical outputs against a main-derived golden:
    >>> fails = compare_trees(                                            # doctest: +SKIP
    ...     Path("run_output"), Path("golden_output"),
    ...     include_prefixes=["significance_tests", "p_value_summary"],
    ... )
    >>> if fails:                                                         # doctest: +SKIP
    ...     raise AssertionError("\\n".join(fails))
    """
    failures: list[str] = []
    inc = tuple(include_prefixes) if include_prefixes is not None else None
    allow = tuple(allow_golden_only) if allow_golden_only else ()
    excl = tuple(exclude_substrings) if exclude_substrings else ()
    # Sorted for deterministic, readable failure ordering.
    golden_files = sorted(
        p for p in golden_root.rglob("*") if p.is_file()
    )
    for g in golden_files:
        rel = g.relative_to(golden_root)
        # Allowlist gate: outside the requested scope -> skip silently.
        if inc is not None and not _rel_matches_prefix(rel, inc):
            continue
        # Exclude gate: carve non-comparable subtrees out of a broad include
        # prefix (substring match anywhere in the path).
        if excl and any(s in rel.as_posix() for s in excl):
            continue
        a = actual_root / rel
        if not a.exists():
            # Intended absence (e.g. a taxa level this branch omits) is not a
            # failure; a genuine drop elsewhere still is.
            if _rel_matches_prefix(rel, allow):
                continue
            failures.append(f"MISSING: {rel}")
            continue
        try:
            compare_file(a, g, rtol=rtol, atol=atol)
        except AssertionError as e:
            # Expected "files differ" signal — record and keep going.
            failures.append(str(e))
        except Exception as e:
            # Unexpected blow-up in a comparator (corrupt file, bad dtype...).
            # Capture it as a failure rather than aborting the whole sweep.
            failures.append(f"{rel}: comparator error ({type(e).__name__}: {e})")
    return failures
