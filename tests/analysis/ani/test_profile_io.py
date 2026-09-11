"""Tests for profile discovery, the QC sample gate, and dense count arrays.

Fixtures mirror the production schema exactly: profiles are written with the same
header ``profile_mags.py`` emits (both the current one and the older one that still
carries a ``mapq_scores`` column, which older profiles have), and
QC fixtures use the real column names from ``quality_control.py`` -- including a
numeric group label, to guard the "numeric labels silently become int64" trap.
"""
import gzip
import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.profile_io import (
    contig_lengths_for_mag,
    dense_contig_counts,
    load_profile,
    profile_path,
    qc_passing_samples,
)

HEADER = "contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tgene_id\n"
HEADER_WITH_MAPQ = (
    "contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tmapq_scores\tgene_id\n"
)


def _write_profile(path, rows, header=HEADER):
    """rows: (contig, position, ref, A, C, G, T, N).

    ``total_coverage`` is written as A+C+G+T+N, exactly as profile_mags.py does --
    the loader must NOT use it.
    """
    with_mapq = "mapq_scores" in header
    with gzip.open(path, "wt") as handle:
        handle.write(header)
        for contig, pos, ref, a, c, g, t, n in rows:
            fields = [contig, pos, ref, a + c + g + t + n, a, c, g, t, n]
            if with_mapq:
                fields.append("40,40")
            fields.append("")  # gene_id, empty for intergenic
            handle.write("\t".join(str(f) for f in fields) + "\n")


def _write_qc(path, mag_id, rows):
    """rows: (sample_id, group, subjectID, time, replicate, passed).

    Column names and their order match quality_control.py's real output (a subset).
    """
    pd.DataFrame(
        {
            "sample_id": [r[0] for r in rows],
            "MAG_ID": [mag_id] * len(rows),
            "file_path": ["/scratch/profiles/x.tsv.gz"] * len(rows),
            "group": [r[1] for r in rows],
            "subjectID": [r[2] for r in rows],
            "replicate": [r[4] for r in rows],
            "time": [r[3] for r in rows],
            "coverage_threshold_passed": [r[5] for r in rows],
        }
    ).to_csv(path, sep="\t", index=False)


class TestProfilePath(unittest.TestCase):
    def test_follows_the_profiling_rule_layout(self):
        got = profile_path("/profiles", "SLG1007", "MAG_1")
        self.assertEqual(got, "/profiles/SLG1007/SLG1007_MAG_1_profiled.tsv.gz")


class TestLoadProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_keeps_file_base_order_and_computes_coverage_without_n(self):
        """Columns stay in profile order A,C,G,T (the set-based classifier never
        needs a reorder) and coverage excludes the N calls."""
        path = os.path.join(self.tmp, "p.tsv.gz")
        _write_profile(path, [("ctg1", 0, "A", 10, 1, 2, 3, 5)])
        table = load_profile(path)
        self.assertEqual(
            list(table.columns), ["contig", "position", "A", "C", "G", "T", "coverage"]
        )
        row = table.iloc[0]
        self.assertEqual([int(row[b]) for b in "ACGT"], [10, 1, 2, 3])
        # coverage is 10+1+2+3 = 16, NOT the file's total_coverage of 21 (5 Ns).
        self.assertEqual(int(row["coverage"]), 16)

    def test_tolerates_the_older_header_with_mapq_scores(self):
        """Profiles written before the samtools-mpileup optimisation carry an extra
        mapq_scores column (older real files do); usecols must
        shrug it off."""
        path = os.path.join(self.tmp, "old.tsv.gz")
        _write_profile(path, [("ctg1", 7, "G", 0, 0, 9, 1, 0)], header=HEADER_WITH_MAPQ)
        table = load_profile(path)
        self.assertEqual(int(table.loc[0, "G"]), 9)
        self.assertEqual(int(table.loc[0, "coverage"]), 10)

    def test_gene_id_opt_in(self):
        """gene_id rides along only on request: the ANI comparison never needs it,
        candidate annotation downstream does (blank stays "")."""
        path = os.path.join(self.tmp, "g.tsv.gz")
        _write_profile(path, [("ctg1", 0, "A", 5, 0, 0, 0, 0)])
        with_genes = load_profile(path, include_gene_id=True)
        self.assertEqual(list(with_genes.columns)[-1], "gene_id")
        self.assertEqual(with_genes.loc[0, "gene_id"], "")
        self.assertNotIn("gene_id", load_profile(path).columns)

    def test_rejects_duplicate_positions(self):
        """Two rows for one (contig, position) means a corrupt profile -- fail loud."""
        path = os.path.join(self.tmp, "dup.tsv.gz")
        _write_profile(
            path,
            [("ctg1", 5, "A", 4, 0, 0, 0, 0), ("ctg1", 5, "A", 4, 0, 0, 0, 0)],
        )
        with self.assertRaises(ValueError):
            load_profile(path)

    def test_missing_required_column_raises(self):
        """A file without one of the count columns is not a profile."""
        path = os.path.join(self.tmp, "bad.tsv.gz")
        with gzip.open(path, "wt") as handle:
            handle.write("contig\tposition\tref_base\ttotal_coverage\tC\tG\tT\tN\tgene_id\n")
            handle.write("ctg1\t0\tA\t4\t4\t0\t0\t0\t\n")
        with self.assertRaises(ValueError):
            load_profile(path)


class TestQcPassingSamples(unittest.TestCase):
    MAG = "MAG_1"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_keeps_only_passing_samples(self):
        qc = os.path.join(self.tmp, f"{self.MAG}_QC.tsv")
        _write_qc(qc, self.MAG, [
            ("S1", "fat", "m1", "pre", 1, True),
            ("S2", "control", "m2", "pre", 1, False),
            ("S3", "fat", "m3", "end", 1, True),
        ])
        got = qc_passing_samples([qc], self.MAG)
        self.assertEqual(sorted(got["sample_id"]), ["S1", "S3"])

    def test_unions_across_files_without_duplicating(self):
        """A sample judged usable at two timepoint combinations appears once."""
        paths = []
        for name in ("qc_a.tsv", "qc_b.tsv"):
            path = os.path.join(self.tmp, name)
            _write_qc(path, self.MAG, [("S1", "fat", "m1", "pre", 1, True)])
            paths.append(path)
        got = qc_passing_samples(paths, self.MAG)
        self.assertEqual(len(got), 1)

    def test_numeric_labels_stay_strings(self):
        """DRIDO's groups are 20/40 and subject ids are numbers; if pandas sniffs
        them as int64 every downstream string comparison silently matches nothing."""
        qc = os.path.join(self.tmp, "numeric_QC.tsv")
        _write_qc(qc, self.MAG, [
            ("S1", 20, 533, 4, 1, True),
            ("S2", 40, 545, 4, 1, True),
        ])
        got = qc_passing_samples([qc], self.MAG)
        self.assertEqual(sorted(got["group"]), ["20", "40"])
        self.assertEqual(sorted(got["subjectID"]), ["533", "545"])
        self.assertTrue(all(isinstance(v, str) for v in got["time"]))

    def test_zero_passing_samples_is_an_empty_frame_not_an_error(self):
        """A low-abundance MAG where nothing passes QC is a legitimate outcome --
        the CLI writes header-only outputs, the pipeline stays green."""
        qc = os.path.join(self.tmp, "none_QC.tsv")
        _write_qc(qc, self.MAG, [("S1", "fat", "m1", "pre", 1, False)])
        got = qc_passing_samples([qc], self.MAG)
        self.assertTrue(got.empty)
        self.assertIn("sample_id", got.columns)

    def test_raises_when_verdict_column_missing(self):
        qc = os.path.join(self.tmp, "bad_QC.tsv")
        pd.DataFrame({"sample_id": ["S1"]}).to_csv(qc, sep="\t", index=False)
        with self.assertRaises(ValueError):
            qc_passing_samples([qc], self.MAG)


class TestDenseContigCounts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _load(self, rows):
        path = os.path.join(self.tmp, "p.tsv.gz")
        _write_profile(path, rows)
        return load_profile(path)

    def test_places_rows_and_leaves_gaps_at_zero(self):
        """An absent position means zero coverage, not missing data."""
        table = self._load([("ctg1", 0, "A", 4, 0, 0, 0, 0), ("ctg1", 3, "C", 0, 6, 0, 0, 0)])
        dense = dense_contig_counts(table, {"ctg1": 5})
        self.assertEqual(dense["ctg1"].shape, (5, 4))
        self.assertEqual(int(dense["ctg1"][0, 0]), 4)      # A at position 0
        self.assertEqual(int(dense["ctg1"][3, 1]), 6)      # C at position 3
        self.assertEqual(int(dense["ctg1"][1].sum()), 0)   # uncovered position
        self.assertEqual(dense["ctg1"].dtype, np.uint16)

    def test_every_mag_contig_gets_an_array_even_if_uncovered(self):
        """All samples must share one position index per contig, so a contig this
        sample never touched still gets a zero array."""
        table = self._load([("ctg1", 0, "A", 4, 0, 0, 0, 0)])
        dense = dense_contig_counts(table, {"ctg1": 2, "ctg2": 3})
        self.assertEqual(dense["ctg2"].shape, (3, 4))
        self.assertEqual(int(dense["ctg2"].sum()), 0)

    def test_rejects_positions_beyond_contig_length(self):
        table = self._load([("ctg1", 99, "A", 4, 0, 0, 0, 0)])
        with self.assertRaises(ValueError):
            dense_contig_counts(table, {"ctg1": 5})

    def test_rejects_contigs_the_mag_does_not_own(self):
        """A profile row on a contig outside the MAG's mapping means the mapping and
        the profiles are out of sync -- fail loud, don't silently drop data."""
        table = self._load([("ctg1", 0, "A", 4, 0, 0, 0, 0), ("ghost", 0, "A", 4, 0, 0, 0, 0)])
        with self.assertRaises(ValueError):
            dense_contig_counts(table, {"ctg1": 5})


class TestContigLengthsForMag(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mapping = os.path.join(self.tmp, "map.tsv")
        pd.DataFrame(
            {"mag_id": ["MAG_1", "MAG_1", "MAG_2"], "contig_id": ["c1", "c2", "other"]}
        ).to_csv(self.mapping, sep="\t", index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _fasta(self, name, records, with_fai=False):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            for contig, seq in records:
                handle.write(f">{contig}\n{seq}\n")
        if with_fai:
            with open(path + ".fai", "w") as handle:
                offset = 0
                for contig, seq in records:
                    offset += len(contig) + 2  # ">" + name + newline
                    handle.write(f"{contig}\t{len(seq)}\t{offset}\t{len(seq)}\t{len(seq) + 1}\n")
                    offset += len(seq) + 1
        return path

    def test_reads_lengths_from_the_fai_index(self):
        """The .fai beside the FASTA is a 1.5 MB text file versus a 0.5 GB parse."""
        fasta = self._fasta("ref.fa", [("c1", "ACGT"), ("c2", "ACGTAC"), ("other", "AC")], with_fai=True)
        self.assertEqual(contig_lengths_for_mag(fasta, self.mapping, "MAG_1"), {"c1": 4, "c2": 6})

    def test_falls_back_to_parsing_the_fasta(self):
        fasta = self._fasta("noindex.fa", [("c1", "ACGT"), ("c2", "ACGTAC")])
        self.assertEqual(contig_lengths_for_mag(fasta, self.mapping, "MAG_1"), {"c1": 4, "c2": 6})

    def test_raises_when_mag_absent_from_mapping(self):
        fasta = self._fasta("r2.fa", [("c1", "ACGT")])
        with self.assertRaises(ValueError):
            contig_lengths_for_mag(fasta, self.mapping, "MAG_MISSING")

    def test_raises_when_mapping_contig_missing_from_reference(self):
        """Mapping names c2 but the reference lacks it: the two inputs disagree."""
        fasta = self._fasta("r3.fa", [("c1", "ACGT")])
        with self.assertRaises(ValueError):
            contig_lengths_for_mag(fasta, self.mapping, "MAG_1")


if __name__ == "__main__":
    unittest.main()
