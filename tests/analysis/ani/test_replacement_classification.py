"""Tests for rolling per-mouse strain-background calls up to MAGs.

Every MAG-level row carries two blocks: counts over MICE, and counts over
REPLICATES (a replicate "changed" if ANY of its mice changed).  With no
replicate column in the metadata replicate == mouse and the blocks agree.
"""
import unittest

import pandas as pd

from alleleflux.scripts.analysis.ani.replacement_classification import METRICS, classify_mags


def _call_row(mag, mouse, group, transition, replacement, dominant, replicate=None):
    """One turnover-table row (Task 6 schema).  ``None`` for a flag means undetermined (pd.NA)."""
    return {
        "MAG_ID": mag, "subjectID": mouse, "group": group,
        "replicate": replicate or mouse, "transition": transition,
        "strain_replacement": pd.NA if replacement is None else replacement,
        "dominant_strain_change": pd.NA if dominant is None else dominant,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    for col in METRICS:
        df[col] = df[col].astype("boolean")
    return df


class TestMouseBlock(unittest.TestCase):
    ROWS = [
        # MAG_A, group 40, 3 mice: replacement T/T/F -> majority; dominant F/F/F -> none.
        _call_row("MAG_A", "m1", "40", "5mo_22mo", True, False),
        _call_row("MAG_A", "m2", "40", "5mo_22mo", True, False),
        _call_row("MAG_A", "m3", "40", "5mo_22mo", False, False),
        # MAG_B, group 40, 2 called mice (T/F) + 1 undetermined -> denominator 2, no majority.
        _call_row("MAG_B", "m4", "40", "5mo_22mo", True, True),
        _call_row("MAG_B", "m5", "40", "5mo_22mo", False, False),
        _call_row("MAG_B", "m6", "40", "5mo_22mo", None, None),
        # MAG_B, group 20, everything undetermined -> no verdict of any kind.
        _call_row("MAG_B", "m7", "20", "5mo_22mo", None, None),
        # MAG_C, group 40, 2 mice both changed -> all_changed.
        _call_row("MAG_C", "m8", "40", "5mo_22mo", True, True),
        _call_row("MAG_C", "m9", "40", "5mo_22mo", True, False),
    ]

    def _mags(self, metric):
        both = classify_mags(_frame(self.ROWS))
        return both[both.metric == metric].set_index(["MAG_ID", "group"])

    def test_majority_uses_only_called_mice(self):
        got = self._mags("strain_replacement")
        a = got.loc[("MAG_A", "40")]
        self.assertEqual((a.n_mice_with_call, a.n_mice_undetermined, a.n_mice_changed), (3, 0, 2))
        self.assertTrue(a.majority_mice_changed and a.any_mouse_changed)
        self.assertFalse(a.all_mice_changed or a.no_mouse_changed)
        b = got.loc[("MAG_B", "40")]
        self.assertEqual((b.n_mice_with_call, b.n_mice_undetermined, b.n_mice_changed), (2, 1, 1))
        self.assertFalse(b.majority_mice_changed)      # 1 of 2 is NOT a majority
        self.assertTrue(b.any_mouse_changed)

    def test_all_changed(self):
        got = self._mags("strain_replacement")
        self.assertTrue(got.loc[("MAG_C", "40")].all_mice_changed)
        self.assertFalse(got.loc[("MAG_A", "40")].all_mice_changed)   # 2 of 3
        # dominant: MAG_C is T/F -> not all
        self.assertFalse(self._mags("dominant_strain_change").loc[("MAG_C", "40")].all_mice_changed)

    def test_no_change_and_all_changed_need_at_least_one_called_mouse(self):
        got = self._mags("dominant_strain_change")
        self.assertTrue(got.loc[("MAG_A", "40")].no_mouse_changed)
        b20 = got.loc[("MAG_B", "20")]
        self.assertEqual(b20.n_mice_with_call, 0)
        for col in ("majority_mice_changed", "any_mouse_changed", "all_mice_changed", "no_mouse_changed"):
            self.assertFalse(b20[col], col)            # zero evidence is not a verdict

    def test_metric_column_names_the_evidence(self):
        got = classify_mags(_frame(self.ROWS))
        self.assertEqual(got.metric.value_counts().to_dict(),
                         {"strain_replacement": 4, "dominant_strain_change": 4})
        self.assertEqual(list(got.columns[:4]), ["MAG_ID", "group", "transition", "metric"])

    def test_groups_are_never_pooled(self):
        got = self._mags("strain_replacement").reset_index()
        self.assertEqual(sorted(zip(got.MAG_ID, got.group)),
                         [("MAG_A", "40"), ("MAG_B", "20"), ("MAG_B", "40"), ("MAG_C", "40")])



class TestReplicateBlock(unittest.TestCase):
    ROWS = [
        # cage c1: m1 changed, m2 not  -> the cage counts as changed (ANY mouse).
        _call_row("MAG_A", "m1", "40", "5mo_22mo", True, False, replicate="c1"),
        _call_row("MAG_A", "m2", "40", "5mo_22mo", False, False, replicate="c1"),
        # cage c2: m3 alone, not changed.
        _call_row("MAG_A", "m3", "40", "5mo_22mo", False, False, replicate="c2"),
        # cage c3: only an undetermined mouse -> the cage has no call.
        _call_row("MAG_A", "m4", "40", "5mo_22mo", None, None, replicate="c3"),
    ]

    def test_replicate_counts_sit_beside_mouse_counts(self):
        both = classify_mags(_frame(self.ROWS))
        row = both[both.metric == "strain_replacement"].iloc[0]
        # mice: 3 called (m1,m2,m3), 1 changed -> 1 of 3 is no majority
        self.assertEqual((row.n_mice_with_call, row.n_mice_changed), (3, 1))
        self.assertFalse(row.majority_mice_changed)
        # replicates: c1 and c2 called, c3 not; c1 changed -> 1 of 2, still no majority
        self.assertEqual((row.n_replicates_with_call, row.n_replicates_changed), (2, 1))
        self.assertFalse(row.majority_replicates_changed)
        self.assertFalse(row.all_replicates_changed)

    def test_replicate_majority_can_differ_from_mouse_majority(self):
        # c1 = {m1 changed, m2 not, m3 not}; c2 = {m4 changed}.  Mice: 2 of 4 (no
        # majority).  Cages: both changed -> majority AND all.
        rows = [
            _call_row("MAG_A", "m1", "40", "5mo_22mo", True, False, replicate="c1"),
            _call_row("MAG_A", "m2", "40", "5mo_22mo", False, False, replicate="c1"),
            _call_row("MAG_A", "m3", "40", "5mo_22mo", False, False, replicate="c1"),
            _call_row("MAG_A", "m4", "40", "5mo_22mo", True, False, replicate="c2"),
        ]
        both = classify_mags(_frame(rows))
        row = both[both.metric == "strain_replacement"].iloc[0]
        self.assertFalse(row.majority_mice_changed)
        self.assertTrue(row.majority_replicates_changed)
        self.assertTrue(row.all_replicates_changed)

    def test_replicate_equal_to_mouse_makes_the_blocks_agree(self):
        # DRiDO: no replicate column -> replicate == subjectID.
        rows = [_call_row("MAG_A", m, "40", "5mo_22mo", flag, False)
                for m, flag in (("m1", True), ("m2", False), ("m3", None))]
        both = classify_mags(_frame(rows))
        row = both[both.metric == "strain_replacement"].iloc[0]
        self.assertEqual(row.n_replicates_with_call, row.n_mice_with_call)
        self.assertEqual(row.n_replicates_changed, row.n_mice_changed)
        self.assertEqual(row.majority_replicates_changed, row.majority_mice_changed)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# End-to-end: the ``alleleflux-replacement-classification`` command
# ---------------------------------------------------------------------------
import os
import shutil
import subprocess
import tempfile

from alleleflux.scripts.analysis.ani.strain_turnover import TURNOVER_COLUMNS


def _turnover_row(mag, mouse, group, transition, background, replacement, dominant):
    """One row in the exact {mag}_strain_turnover.tsv schema (Task 6 output)."""
    row = {c: "" for c in TURNOVER_COLUMNS}
    row.update({
        "MAG_ID": mag, "subjectID": mouse, "replicate": mouse, "group": group,
        "transition": transition, "sample_t1": f"{mouse}_a", "sample_t2": f"{mouse}_b",
        "compared_bases_count": 1000, "percent_genome_compared": 0.5,
        "conANI": 0.9999, "popANI": 0.99999, "frequency_shift": 0.0,
        "strain_replacement": replacement, "dominant_strain_change": dominant,
        "background": background, "min_compared": 0.1, "pop_threshold": 0.99999,
        "con_threshold": 0.999, "min_cov": 5,
    })
    return row


class TestReplacementClassificationCLI(unittest.TestCase):
    """Two MAGs' turnover files plus one header-only file in a directory.

    MAG_A, group 40, 5mo_22mo: 3 mice, replacement T/T/F -> majority; dominant F/F/F.
    MAG_B, group 40, 5mo_22mo: 1 mouse undetermined -> no verdict of any kind.
    MAG_C: header only (a MAG with no matching transitions) -> contributes nothing.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.turnover = os.path.join(self.tmp, "strain_turnover")
        os.makedirs(self.turnover)
        rows_a = [
            _turnover_row("MAG_A", "m1", "40", "5mo_22mo", "strain_replacement", True, False),
            _turnover_row("MAG_A", "m2", "40", "5mo_22mo", "strain_replacement", True, False),
            _turnover_row("MAG_A", "m3", "40", "5mo_22mo", "stable", False, False),
        ]
        rows_b = [_turnover_row("MAG_B", "m4", "40", "5mo_22mo", "undetermined", "", "")]
        for mag, rows in (("MAG_A", rows_a), ("MAG_B", rows_b), ("MAG_C", [])):
            pd.DataFrame(rows, columns=TURNOVER_COLUMNS).to_csv(
                os.path.join(self.turnover, f"{mag}_strain_turnover.tsv"), sep="\t", index=False)
        self.out = os.path.join(self.tmp, "replacement_classification.tsv")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, turnover_dir=None):
        return subprocess.run([
            "alleleflux-replacement-classification",
            "--turnover_dir", turnover_dir or self.turnover, "--output_path", self.out,
        ], capture_output=True, text=True)

    def test_classifies_every_mag_for_both_metrics(self):
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        got = pd.read_csv(self.out, sep="\t", dtype={"group": str})
        # 2 MAGs with rows x 2 metrics; the header-only MAG_C contributes nothing
        self.assertEqual(sorted(got.MAG_ID.unique()), ["MAG_A", "MAG_B"])
        self.assertEqual(got.metric.value_counts().to_dict(), {"strain_replacement": 2, "dominant_strain_change": 2})
        a = got[(got.MAG_ID == "MAG_A") & (got.metric == "strain_replacement")].iloc[0]
        self.assertEqual((int(a.n_mice_with_call), int(a.n_mice_changed)), (3, 2))
        self.assertTrue(bool(a.majority_mice_changed))
        b = got[(got.MAG_ID == "MAG_B") & (got.metric == "strain_replacement")].iloc[0]
        self.assertEqual((int(b.n_mice_with_call), int(b.n_mice_undetermined)), (0, 1))
        self.assertFalse(bool(b.majority_mice_changed) or bool(b.no_mouse_changed))
        self.assertEqual(list(got.columns[:4]), ["MAG_ID", "group", "transition", "metric"])

    def test_directory_without_turnover_files_fails_loud(self):
        empty = os.path.join(self.tmp, "nothing"); os.makedirs(empty)
        done = self._run(empty)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("strain_turnover", done.stderr)
