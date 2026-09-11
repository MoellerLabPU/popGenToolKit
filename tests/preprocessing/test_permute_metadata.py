#!/usr/bin/env python3
"""Unit tests for the unified metadata permutation primitive.

The whole script is one operation: relabel ``--unit`` units to groups (preserving
counts), optionally within each ``--block``.  These tests pin that contract for
the global case, the within-block case, validation, the --require-change guard,
and the per-comparison-pair restriction.
"""

import unittest

import pandas as pd

from alleleflux.scripts.preprocessing.permute_metadata import (
    permute_group_labels,
    permute_metadata,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _global_sheet():
    """4 independent subjects, 2 groups (A,B), 2 timepoints. No block."""
    rows = []
    for subj, grp in [("s1", "A"), ("s2", "A"), ("s3", "B"), ("s4", "B")]:
        for tp in ("pre", "post"):
            rows.append(
                {
                    "sample_id": f"{subj}_{tp}",
                    "bam_path": f"bams/{subj}_{tp}.bam",
                    "subjectID": subj,
                    "group": grp,
                    "time": tp,
                }
            )
    return pd.DataFrame(rows)


def _cage_block_sheet():
    """Diet-style: each replicate (block) has 1 control cage + 1 fat cage, 2 mice
    per cage, 2 timepoints. unit = cage, block = replicate."""
    rows = []
    for block in ("b1", "b2", "b3", "b4"):
        for suffix, grp in [("control", "control"), ("fat", "fat")]:
            cage = f"{block}_{suffix}"
            for mouse in ("m1", "m2"):
                subj = f"{cage}_{mouse}"
                for tp in ("T0", "T7"):
                    rows.append(
                        {
                            "sample_id": f"{subj}_{tp}",
                            "subjectID": subj,
                            "replicate": block,
                            "cage": cage,
                            "group": grp,
                            "time": tp,
                        }
                    )
    return pd.DataFrame(rows)


def _soil_block_sheet():
    """Soil-style: each replicate (plot) has 1 burned + 1 control subject, 2
    timepoints. unit = subjectID, block = replicate."""
    rows = []
    for plot in ("p1", "p2", "p3", "p4"):
        for subj, grp in [(f"{plot}_burn", "burned"), (f"{plot}_unburn", "control")]:
            for tp in ("T0", "T7"):
                rows.append(
                    {
                        "sample_id": f"{subj}_{tp}",
                        "subjectID": subj,
                        "replicate": plot,
                        "group": grp,
                        "time": tp,
                    }
                )
    return pd.DataFrame(rows)


def _multigroup_sheet():
    """6 subjects across 3 groups (A,B,C) — for per-pair subset tests."""
    rows = []
    for subj, grp in [("s1", "A"), ("s2", "A"), ("s3", "B"),
                      ("s4", "B"), ("s5", "C"), ("s6", "C")]:
        for tp in ("pre", "post"):
            rows.append(
                {"sample_id": f"{subj}_{tp}", "subjectID": subj,
                 "group": grp, "time": tp}
            )
    return pd.DataFrame(rows)


def _unit_counts(df, unit):
    return df.groupby(unit)["group"].first().value_counts().sort_index()


# --------------------------------------------------------------------------- #
# Global shuffle (no block)
# --------------------------------------------------------------------------- #
class TestGlobalShuffle(unittest.TestCase):
    def test_preserves_per_unit_group_counts(self):
        df = _global_sheet()
        out = permute_group_labels(df, unit_col="subjectID", seed=1)
        self.assertTrue(_unit_counts(df, "subjectID").equals(_unit_counts(out, "subjectID")))

    def test_one_group_per_unit(self):
        out = permute_group_labels(_global_sheet(), unit_col="subjectID", seed=3)
        self.assertTrue((out.groupby("subjectID")["group"].nunique() == 1).all())

    def test_only_group_column_changes(self):
        df = _global_sheet()
        out = permute_group_labels(df, unit_col="subjectID", seed=2)
        for col in ("sample_id", "bam_path", "subjectID", "time"):
            pd.testing.assert_series_equal(df[col], out[col])

    def test_reproducible_under_seed(self):
        a = permute_group_labels(_global_sheet(), unit_col="subjectID", seed=7)
        b = permute_group_labels(_global_sheet(), unit_col="subjectID", seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_does_not_mutate_input(self):
        df = _global_sheet()
        before = df.copy()
        permute_group_labels(df, unit_col="subjectID", seed=9)
        pd.testing.assert_frame_equal(df, before)

    def test_some_seed_actually_permutes(self):
        df = _global_sheet()
        changed = any(
            (df["group"].to_numpy()
             != permute_group_labels(df, unit_col="subjectID", seed=s)["group"].to_numpy()).any()
            for s in range(20)
        )
        self.assertTrue(changed)

    def test_unit_can_be_any_column(self):
        # e.g. antibiotic: the cage IS the 'replicate' column -> shuffle cages.
        df = _cage_block_sheet()  # has a 'cage' column; ignore the block here
        out = permute_group_labels(df, unit_col="cage", seed=1)  # global cage shuffle
        self.assertTrue((out.groupby("cage")["group"].nunique() == 1).all())
        self.assertTrue(_unit_counts(df, "cage").equals(_unit_counts(out, "cage")))


# --------------------------------------------------------------------------- #
# Within-block shuffle
# --------------------------------------------------------------------------- #
class TestWithinBlockShuffle(unittest.TestCase):
    def test_cage_never_splits(self):
        df = _cage_block_sheet()
        for seed in range(25):
            out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=seed)
            self.assertTrue(
                (out.groupby("cage")["group"].nunique() == 1).all(),
                msg=f"a cage split across groups at seed={seed}",
            )

    def test_each_block_keeps_its_composition(self):
        df = _cage_block_sheet()
        out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=2)
        for _, sub in out.groupby("replicate"):
            comp = sub.drop_duplicates("cage")["group"].value_counts().to_dict()
            self.assertEqual(comp, {"control": 1, "fat": 1})

    def test_subjectID_unchanged(self):
        df = _cage_block_sheet()
        out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=1)
        pd.testing.assert_series_equal(df["subjectID"], out["subjectID"])

    def test_consistent_across_timepoints(self):
        df = _cage_block_sheet()
        out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=3)
        per_cage_time = out.groupby(["cage", "time"])["group"].first().unstack()
        self.assertTrue((per_cage_time.nunique(axis=1) == 1).all())

    def test_soil_within_plot_swap(self):
        df = _soil_block_sheet()
        out = permute_group_labels(df, unit_col="subjectID", block_col="replicate", seed=1)
        # Each plot still has exactly one burned + one control subject.
        for _, sub in out.groupby("replicate"):
            comp = sub.drop_duplicates("subjectID")["group"].value_counts().to_dict()
            self.assertEqual(comp, {"burned": 1, "control": 1})

    def test_blocks_are_independent(self):
        # Across seeds some blocks flip while others don't (not all-or-nothing).
        df = _cage_block_sheet()
        orig = df.groupby(["replicate", "cage"])["group"].first()
        seen_partial = False
        for seed in range(30):
            out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=seed)
            now = out.groupby(["replicate", "cage"])["group"].first()
            flipped = (orig != now).groupby(level="replicate").any()
            if 0 < int(flipped.sum()) < len(flipped):  # some but not all blocks flipped
                seen_partial = True
                break
        self.assertTrue(seen_partial)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation(unittest.TestCase):
    def test_missing_unit_raises(self):
        with self.assertRaises(ValueError):
            permute_group_labels(_global_sheet(), unit_col="cage")

    def test_missing_block_raises(self):
        with self.assertRaises(ValueError):
            permute_group_labels(_global_sheet(), unit_col="subjectID", block_col="plot")

    def test_unit_spanning_multiple_groups_raises(self):
        # 'replicate' is the block in diet data → it spans both groups → bad unit.
        with self.assertRaises(ValueError):
            permute_group_labels(_cage_block_sheet(), unit_col="replicate")

    def test_unit_spanning_multiple_blocks_raises(self):
        # Force a unit value to appear in two blocks → invalid.
        df = _cage_block_sheet().copy()
        df.loc[df["replicate"] == "b2", "cage"] = "b1_control"  # b1_control now in 2 blocks
        with self.assertRaises(ValueError):
            permute_group_labels(df, unit_col="cage", block_col="replicate")


# --------------------------------------------------------------------------- #
# --swaps count
# --------------------------------------------------------------------------- #
def _n_blocks_flipped(df, out, block, unit):
    o = df.groupby([block, unit])["group"].first()
    n = out.groupby([block, unit])["group"].first()
    return int((o != n).groupby(level=block).any().sum())


def _n_units_changed(df, out, unit):
    o = df.groupby(unit)["group"].first()
    n = out.groupby(unit)["group"].first()
    return int((o != n).sum())


class TestSwaps(unittest.TestCase):
    def test_block_default_is_half_the_blocks(self):
        df = _cage_block_sheet()  # 4 blocks → default 2 flipped
        out = permute_group_labels(df, unit_col="cage", block_col="replicate", seed=1)
        self.assertEqual(_n_blocks_flipped(df, out, "replicate", "cage"), 4 // 2)

    def test_block_explicit_swaps_count(self):
        df = _cage_block_sheet()
        out = permute_group_labels(df, unit_col="cage", block_col="replicate",
                                   seed=1, swaps=3)
        self.assertEqual(_n_blocks_flipped(df, out, "replicate", "cage"), 3)

    def test_global_default_is_half_the_smaller_group(self):
        df = _global_sheet()  # 2 A + 2 B subjects → max 2 swaps → default 1
        out = permute_group_labels(df, unit_col="subjectID", seed=1)
        # one A<->B pair exchanged → 2 units change.
        self.assertEqual(_n_units_changed(df, out, "subjectID"), 2)

    def test_global_explicit_swaps_count(self):
        df = _global_sheet()
        out = permute_group_labels(df, unit_col="subjectID", seed=1, swaps=2)
        self.assertEqual(_n_units_changed(df, out, "subjectID"), 4)  # all 4 swap

    def test_swaps_zero_is_identity(self):
        df = _cage_block_sheet()
        out = permute_group_labels(df, unit_col="cage", block_col="replicate",
                                   seed=1, swaps=0)
        self.assertEqual(list(out["group"]), list(df["group"]))

    def test_too_many_block_swaps_raises(self):
        df = _cage_block_sheet()  # 4 blocks
        with self.assertRaises(ValueError):
            permute_group_labels(df, unit_col="cage", block_col="replicate", swaps=5)

    def test_too_many_global_swaps_raises(self):
        df = _global_sheet()  # max 2
        with self.assertRaises(ValueError):
            permute_group_labels(df, unit_col="subjectID", swaps=3)

    def test_global_swap_needs_two_groups(self):
        # 3 groups, no --block and no pair restriction → ambiguous.
        with self.assertRaises(ValueError):
            permute_group_labels(_multigroup_sheet(), unit_col="subjectID")

    def test_block_needs_matched_pairs(self):
        # unit=subjectID on cage data → 4 subjects per block (not a matched pair).
        with self.assertRaises(ValueError):
            permute_group_labels(_cage_block_sheet(), unit_col="subjectID",
                                 block_col="replicate")


# --------------------------------------------------------------------------- #
# Per-comparison-pair restriction
# --------------------------------------------------------------------------- #
class TestPerPairSubset(unittest.TestCase):
    def test_other_groups_untouched(self):
        df = _multigroup_sheet()
        out = permute_metadata(df, unit="subjectID", seed=1, groups=["A", "B"])
        c_in = df[df["group"] == "C"].reset_index(drop=True)
        c_out = out[out["subjectID"].isin(["s5", "s6"])].reset_index(drop=True)
        pd.testing.assert_frame_equal(c_in, c_out)

    def test_pair_counts_preserved(self):
        df = _multigroup_sheet()
        out = permute_metadata(df, unit="subjectID", seed=2, groups=["A", "B"])
        for grp in ("A", "B", "C"):
            self.assertEqual((df["group"] == grp).sum(), (out["group"] == grp).sum())

    def test_row_order_preserved(self):
        df = _multigroup_sheet()
        out = permute_metadata(df, unit="subjectID", seed=3, groups=["A", "B"])
        pd.testing.assert_series_equal(df["sample_id"], out["sample_id"])

    def test_empty_pair_raises(self):
        with self.assertRaises(ValueError):
            permute_metadata(_multigroup_sheet(), unit="subjectID", seed=1,
                             groups=["X", "Y"])

    def test_per_pair_with_block(self):
        # Orchestrator threads unit + block through on a comparison pair.
        df = _cage_block_sheet()
        out = permute_metadata(df, unit="cage", block="replicate", seed=1,
                               groups=["control", "fat"])
        self.assertTrue((out.groupby("cage")["group"].nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
