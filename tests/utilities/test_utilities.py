#!/usr/bin/env python3
"""Unit tests for ``alleleflux.scripts.utilities.utilities``.

Currently covers the two helpers that form the in-memory seam of the
reuse-based null/permutation feature:

* ``relabel_groups_from_metadata`` — re-derives permuted ``group`` labels by
  joining a loaded frame to a permuted metadata sheet on ``sample_id``.  Tests
  pin the contract every cache consumer relies on: correct join, idempotency,
  identity no-op, categorical preservation, graceful handling of a missing key,
  and the numeric-group → str coercion guard.
* ``input_has_column`` — the cheap header/schema probe that guards the *other*
  half of that seam (force the ``sample_id`` read only when the input actually
  carries it).  Tests pin present-vs-absent for TSV, TSV.gz, parquet, and the
  list-of-paths form.

Add new ``TestX`` classes here for other utilities helpers rather than spawning
a separate file per function.
"""

import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from alleleflux.scripts.utilities.utilities import (
    build_contig_length_index,
    input_has_column,
    relabel_groups_from_metadata,
)


def _cache_like():
    """Frame shaped like a loaded cache slice: per-sample rows + a value column."""
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "group": ["A", "A", "A", "A", "B", "B"],
            "subjectID": ["s1", "s1", "s2", "s2", "s3", "s3"],
            "A_frequency": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def _permuted_meta():
    """Permuted sheet: s2 moves A->B, s3 moves B->A (per-sample labels)."""
    return pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "group": ["A", "B", "A"],
            "subjectID": ["s1", "s2", "s3"],
        }
    )


class TestRelabelGroups(unittest.TestCase):
    def test_relabels_group_by_sample_id(self):
        out = relabel_groups_from_metadata(_cache_like(), _permuted_meta())
        # s1->A (unchanged), s2->B, s3->A
        self.assertEqual(list(out["group"]), ["A", "A", "B", "B", "A", "A"])

    def test_frequency_column_untouched(self):
        df = _cache_like()
        before = df["A_frequency"].copy()
        out = relabel_groups_from_metadata(df, _permuted_meta())
        pd.testing.assert_series_equal(before, out["A_frequency"])

    def test_subjectid_never_touched(self):
        # Permutation moves ONLY the group label; subjectID must stay tied to its
        # original rows even if the permuted sheet carries different subjectIDs
        # (rewriting it would risk re-coercing the dtype of the merge key).
        meta = _permuted_meta()
        meta["subjectID"] = ["s1", "X2", "X3"]  # deliberately different
        out = relabel_groups_from_metadata(_cache_like(), meta)
        self.assertEqual(list(out["subjectID"]), ["s1", "s1", "s2", "s2", "s3", "s3"])
        # group still relabeled as expected
        self.assertEqual(list(out["group"]), ["A", "A", "B", "B", "A", "A"])

    def test_idempotent(self):
        once = relabel_groups_from_metadata(_cache_like(), _permuted_meta())
        twice = relabel_groups_from_metadata(once.copy(), _permuted_meta())
        pd.testing.assert_frame_equal(once, twice)

    def test_identity_metadata_is_noop(self):
        df = _cache_like()
        identity = df[["sample_id", "group", "subjectID"]].drop_duplicates()
        out = relabel_groups_from_metadata(df.copy(), identity)
        self.assertEqual(list(out["group"]), ["A", "A", "A", "A", "B", "B"])

    def test_preserves_categorical_dtype(self):
        df = _cache_like()
        df["group"] = df["group"].astype("category")
        out = relabel_groups_from_metadata(df, _permuted_meta())
        self.assertIsInstance(out["group"].dtype, pd.CategoricalDtype)
        self.assertEqual(list(out["group"]), ["A", "A", "B", "B", "A", "A"])

    def test_missing_key_in_df_returns_unchanged(self):
        # An already-aggregated frame that dropped sample_id is returned as-is.
        df = _cache_like().drop(columns=["sample_id"])
        out = relabel_groups_from_metadata(df.copy(), _permuted_meta())
        pd.testing.assert_frame_equal(out, df)

    def test_sample_absent_from_metadata_raises(self):
        meta = _permuted_meta()
        meta = meta[meta["sample_id"] != "s3"]  # drop s3 from the map
        with self.assertRaises(ValueError) as ctx:
            relabel_groups_from_metadata(_cache_like(), meta)
        self.assertIn("s3", str(ctx.exception))

    def test_numeric_groups_coerced_to_str(self):
        # DRIDO-style numeric group names must relabel as strings, not ints.
        df = pd.DataFrame(
            {"sample_id": ["s1", "s2"], "group": ["20", "40"], "v": [1.0, 2.0]}
        )
        meta = pd.DataFrame({"sample_id": ["s1", "s2"], "group": [40, 20]})  # int
        out = relabel_groups_from_metadata(df, meta)
        self.assertEqual(list(out["group"]), ["40", "20"])
        self.assertTrue(all(isinstance(v, str) for v in out["group"]))

    def test_missing_key_in_metadata_raises(self):
        bad_meta = _permuted_meta().rename(columns={"sample_id": "id"})
        with self.assertRaises(ValueError):
            relabel_groups_from_metadata(_cache_like(), bad_meta)

    def test_accepts_tsv_path(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "perm.tsv"
            _permuted_meta().to_csv(p, sep="\t", index=False)
            out = relabel_groups_from_metadata(_cache_like(), str(p))
        self.assertEqual(list(out["group"]), ["A", "A", "B", "B", "A", "A"])


class TestInputHasColumn(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # cache-like frame: carries sample_id (the per-sample join key)
        self.cache = pd.DataFrame(
            {"sample_id": ["s1", "s2"], "group": ["A", "B"], "A_frequency": [0.1, 0.2]}
        )
        # wide across_time-like frame: keyed by replicate, NO sample_id
        self.wide = pd.DataFrame(
            {"replicate": ["A", "B"], "group": ["A", "B"], "A_frequency_pre": [0.1, 0.2]}
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_tsv_present(self):
        p = self.tmp / "cache.tsv"
        self.cache.to_csv(p, sep="\t", index=False)
        self.assertTrue(input_has_column(p, "sample_id"))

    def test_tsv_absent(self):
        p = self.tmp / "wide.tsv"
        self.wide.to_csv(p, sep="\t", index=False)
        self.assertFalse(input_has_column(p, "sample_id"))

    def test_tsv_gz_present(self):
        p = self.tmp / "cache.tsv.gz"
        self.cache.to_csv(p, sep="\t", index=False, compression="gzip")
        self.assertTrue(input_has_column(p, "sample_id"))

    def test_parquet_present(self):
        p = self.tmp / "cache.parquet"
        self.cache.to_parquet(p)
        self.assertTrue(input_has_column(p, "sample_id"))

    def test_parquet_absent(self):
        """The bug scenario: a wide across_time parquet has no sample_id."""
        p = self.tmp / "wide.parquet"
        self.wide.to_parquet(p)
        self.assertFalse(input_has_column(p, "sample_id"))

    def test_list_of_paths_probes_first(self):
        p = self.tmp / "cache.parquet"
        self.cache.to_parquet(p)
        self.assertTrue(input_has_column([p], "sample_id"))
        self.assertFalse(input_has_column([str(p)], "nonexistent_col"))




class TestBuildContigLengthIndex(unittest.TestCase):
    """The shared contig-length helper: .fai fast path versus FASTA fallback."""

    def _write_reference(self, tmp, with_fai, fai_lengths=None):
        fasta = Path(tmp) / "ref.fa"
        fasta.write_text(">c1\nACGT\n>c2\nACGTAC\n")
        if with_fai:
            lengths = fai_lengths or {"c1": 4, "c2": 6}
            lines = [f"{name}\t{length}\t0\t60\t61" for name, length in lengths.items()]
            (Path(tmp) / "ref.fa.fai").write_text("\n".join(lines) + "\n")
        mapping = Path(tmp) / "map.tsv"
        mapping.write_text("mag_id\tcontig_id\nM1\tc1\nM1\tc2\n")
        return str(fasta), str(mapping)

    def test_fai_index_is_preferred_when_present(self):
        """A deliberately wrong length in the .fai proves which source was read."""
        with TemporaryDirectory() as tmp:
            fasta, mapping = self._write_reference(
                tmp, with_fai=True, fai_lengths={"c1": 7, "c2": 6}
            )
            self.assertEqual(build_contig_length_index(fasta, mapping), {"c1": 7, "c2": 6})

    def test_falls_back_to_parsing_the_fasta(self):
        with TemporaryDirectory() as tmp:
            fasta, mapping = self._write_reference(tmp, with_fai=False)
            self.assertEqual(build_contig_length_index(fasta, mapping), {"c1": 4, "c2": 6})

    def test_both_paths_agree_and_filter_to_mapped_contigs(self):
        """Same answer with and without the index, and unmapped contigs stay out."""
        with TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "ref.fa"
            fasta.write_text(">c1\nACGT\n>unmapped\nACGTACGT\n")
            (Path(tmp) / "ref.fa.fai").write_text("c1\t4\t0\t60\t61\nunmapped\t8\t0\t60\t61\n")
            mapping = Path(tmp) / "map.tsv"
            mapping.write_text("mag_id\tcontig_id\nM1\tc1\n")
            via_fai = build_contig_length_index(str(fasta), str(mapping))
            (Path(tmp) / "ref.fa.fai").unlink()
            via_fasta = build_contig_length_index(str(fasta), str(mapping))
            self.assertEqual(via_fai, via_fasta)
            self.assertEqual(via_fai, {"c1": 4})


if __name__ == "__main__":
    unittest.main()
