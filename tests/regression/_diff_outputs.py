"""
CLI wrapper around ``tests.regression.compare.compare_trees`` for use by
``diff_branches.sh``.

Replaces the previous inline ``python -c "..."`` block inside the bash script
— factoring it out into a real file gets us proper argparse, no string
escaping, and a stack trace if anything goes wrong.

Usage:
    python -m tests.regression._diff_outputs \\
        --baseline /tmp/.../output_root/<data_type> \\
        --candidate /tmp/.../output_root/<data_type>

By default the comparison is **scoped and float32-tolerant** — identical to the
pytest regression test — so it answers "does the candidate produce the same
*science* as the baseline?" while ignoring intended structural divergence
(dropped columns, parquet vs tsv.gz, renamed dirs, disabled outputs).  Pass
``--full-strict`` to instead diff the ENTIRE tree at the strict comparator
default; that surfaces every structural and float32 difference too.

Exit code 0  → all in-scope baseline files matched by candidate (within tol).
Exit code 1  → at least one file differed; failure list printed to stdout.
Exit code 2  → a baseline/candidate path was not a directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tests.regression._scope import (
    STAT_ALLOW_GOLDEN_ONLY,
    STAT_ATOL,
    STAT_COMPARE_ONLY,
    STAT_RTOL,
    exclude_substrings_for,
)
from tests.regression.compare import compare_trees


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Directory containing the baseline pipeline output (the 'golden' side).",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Directory containing the candidate pipeline output (the side under test).",
    )
    parser.add_argument(
        "--full-strict",
        action="store_true",
        help="Compare the ENTIRE output tree at the strict comparator default "
             "instead of the default scoped, float32-tolerant statistical "
             "comparison.  Surfaces intended structural divergence (dropped "
             "columns, parquet vs tsv.gz, renamed dirs, disabled outputs) too.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Scenario name (e.g. 'single', 'longitudinal') whose per-scenario "
             "carve-outs to apply on top of the shared scope — e.g. single "
             "drops its non-deterministic two_sample_paired test and the "
             "column-pruned preprocessed intermediates.  See _scope.py.",
    )
    args = parser.parse_args()

    for label, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.is_dir():
            print(f"ERROR: {label} path is not a directory: {path}", file=sys.stderr)
            return 2

    # baseline is the golden side; candidate is the side under test.  Default to
    # the same scoped/float32-tolerant contract as the pytest e2e (see _scope.py)
    # so a same-science refactor reports clean; --full-strict opts into the raw
    # whole-tree diff.
    if args.full_strict:
        fails = compare_trees(args.candidate, args.baseline)
    else:
        fails = compare_trees(
            args.candidate,
            args.baseline,
            include_prefixes=STAT_COMPARE_ONLY,
            allow_golden_only=STAT_ALLOW_GOLDEN_ONLY,
            exclude_substrings=(
                exclude_substrings_for(args.scenario) if args.scenario else None
            ),
            rtol=STAT_RTOL,
            atol=STAT_ATOL,
        )
    if not fails:
        print("OK — all baseline files matched by candidate.")
        return 0

    print(f"{len(fails)} file(s) differ:")
    for f in fails:
        print()
        print(f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
