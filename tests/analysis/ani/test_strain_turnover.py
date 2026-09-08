"""Tests for per-mouse strain-background calls from the pairwise ANI table.

The rows fed in mimic what the engine emits: ``sample1 < sample2`` by ID, which
is NOT chronological -- orientation by time is one of the things under test.
"""
import unittest

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.ani.strain_turnover import (
    alleles_present_at,
    call_transitions,
    scan_for_new_alleles,
)

# Q30 / FDR 1e-6: threshold is 3 reads for coverages 5..99 (see null_model docs).
MODEL = build_error_model(min_base_quality=30, fdr=1e-6)


def _pair_row(s1, s2, subj, t1, t2, popani, pgc, conani=None, group="40"):
    """One engine-style pair row; conANI defaults to popANI when not given."""
    return {
        "MAG_ID": "MRGM_0841", "sample1": s1, "sample2": s2,
        "subjectID_1": subj, "subjectID_2": subj,
        "time_1": t1, "time_2": t2, "group_1": group, "group_2": group,
        "replicate_1": "1", "replicate_2": "1",
        "popANI": popani, "conANI": popani if conani is None else conani,
        "percent_genome_compared": pgc,
        "compared_bases_count": int(pgc * 4_000_000), "min_cov": 5,
    }


class TestCallTransitions(unittest.TestCase):
    TRANSITIONS = [("5mo", "22mo")]

    def _call(self, rows, **overrides):
        kwargs = dict(min_compared=0.1, pop_threshold=0.99999, con_threshold=0.999)
        kwargs.update(overrides)
        return call_transitions(pd.DataFrame(rows), self.TRANSITIONS, **kwargs)

    def test_orients_by_time_not_by_sample_id(self):
        # "Abe" sorts before "Zed" so the engine put the 22mo sample in sample1.
        got = self._call([_pair_row("Abe", "Zed", "m1", "22mo", "5mo", 0.999999, 0.8)])
        self.assertEqual(got.iloc[0]["sample_t1"], "Zed")      # the 5mo sample
        self.assertEqual(got.iloc[0]["sample_t2"], "Abe")
        self.assertEqual(got.iloc[0]["transition"], "5mo_22mo")

    def test_four_backgrounds(self):
        got = self._call([
            _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.9999990, 0.8, conani=0.9995),  # stable
            _pair_row("B1", "B2", "m2", "5mo", "22mo", 0.9999950, 0.8, conani=0.9985),  # dominant only
            _pair_row("C1", "C2", "m3", "5mo", "22mo", 0.9999800, 0.8, conani=0.9995),  # replacement only
            _pair_row("D1", "D2", "m4", "5mo", "22mo", 0.9900000, 0.8, conani=0.9800),  # both
        ])
        self.assertEqual(list(got["background"]), [
            "stable", "dominant_strain_change", "strain_replacement",
            "strain_replacement+dominant_strain_change"])
        self.assertEqual(list(got["strain_replacement"]), [False, False, True, True])
        self.assertEqual(list(got["dominant_strain_change"]), [False, True, False, True])
        # popANI >= conANI always, so the shift is never negative.
        self.assertTrue(bool((got["frequency_shift"] >= 0).all()))

    def test_threshold_is_strict_less_than(self):
        # Exactly AT the threshold is NOT a change (>= threshold means same strain).
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", 0.99999, 0.8, conani=0.999)])
        self.assertEqual(got.iloc[0]["background"], "stable")

    def test_undetermined_leaves_both_flags_blank(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", 0.5, 0.01)])
        self.assertEqual(got.iloc[0]["background"], "undetermined")
        self.assertTrue(pd.isna(got.iloc[0]["strain_replacement"]))
        self.assertTrue(pd.isna(got.iloc[0]["dominant_strain_change"]))

    def test_min_compared_is_configurable(self):
        row = _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.99, 0.05, conani=0.99)
        self.assertEqual(self._call([row]).iloc[0]["background"], "undetermined")  # 5% < 10%
        self.assertEqual(self._call([row], min_compared=0.05).iloc[0]["background"],
                         "strain_replacement+dominant_strain_change")            # 5% >= 5%

    def test_nan_ani_is_undetermined(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "22mo", np.nan, 0.0)])
        self.assertEqual(got.iloc[0]["background"], "undetermined")

    def test_non_transition_and_cross_subject_rows_are_excluded(self):
        rows = [
            _pair_row("A1", "A2", "m1", "5mo", "28mo", 0.9999990, 0.8),   # not a configured transition
            _pair_row("A1", "B1", "m1", "5mo", "22mo", 0.9999990, 0.8),   # two different mice
        ]
        rows[1]["subjectID_2"] = "m2"
        self.assertEqual(len(self._call(rows)), 0)

    def test_empty_input_keeps_the_schema(self):
        got = self._call([_pair_row("A1", "A2", "m1", "5mo", "28mo", 0.9999990, 0.8)])
        for col in ("transition", "sample_t1", "sample_t2", "strain_replacement",
                    "dominant_strain_change", "frequency_shift", "background"):
            self.assertIn(col, got.columns)

    def test_group_or_replicate_mismatch_within_a_mouse_raises(self):
        # A pair is one mouse at two times; its diet group and replicate cannot differ.
        row = _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.999999, 0.8)
        row["group_2"] = "20"                                   # group_1 is "40"
        with self.assertRaisesRegex(ValueError, "group"):
            self._call([row])
        row = _pair_row("A1", "A2", "m1", "5mo", "22mo", 0.999999, 0.8)
        row["replicate_2"] = "2"
        with self.assertRaisesRegex(ValueError, "replicate"):
            self._call([row])

    def test_rejects_degenerate_transition(self):
        with self.assertRaises(ValueError):
            call_transitions(pd.DataFrame([_pair_row("A", "B", "m", "5mo", "22mo", 1.0, 1.0)]),
                             [("5mo", "5mo")], 0.1, 0.99999, 0.999)


def _dense(rows, length):
    """Dense (length, 4) uint16 counts from {position: (A, C, G, T)}; unlisted rows are zero."""
    arr = np.zeros((length, 4), dtype=np.uint16)
    for pos, counts in rows.items():
        arr[pos] = counts
    return arr


class TestScanForNewAlleles(unittest.TestCase):
    """One mouse, baseline (t1) vs later (t2) dense counts; min_freq 5 %, min_cov 5."""

    def _scan(self, t1, t2, length=1):
        return scan_for_new_alleles(_dense({0: t1}, length), _dense({0: t2}, length), MODEL, 0.05, 5)

    def test_new_minor_allele_without_majority_flip(self):
        """The case no SNP table can show: A stays consensus, C appears at 20 %."""
        got = self._scan((30, 0, 0, 0), (24, 6, 0, 0), length=2)
        self.assertEqual(list(got["position"]), [0])
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["t1_reads"]), [0])
        self.assertEqual(list(got["t1_threshold"]), [int(MODEL[30])])
        self.assertEqual(list(got["t1_evidence"]), ["absent"])
        self.assertEqual(list(got["fully_replaced"]), [False])
        self.assertEqual(list(got["t2_consensus"]), [False])          # A still leads at t2
        self.assertEqual(list(got["t2_reads"]), [6])
        self.assertEqual(list(got["t2_threshold"]), [int(MODEL[30])])
        self.assertAlmostEqual(float(got["freq_t1"][0]), 0.0)
        self.assertAlmostEqual(float(got["freq_t2"][0]), 0.2)

    def test_full_sweep_is_flagged(self):
        got = self._scan((30, 0, 0, 0), (0, 28, 0, 0))
        self.assertEqual(list(got["fully_replaced"]), [True])
        self.assertEqual(list(got["t1_evidence"]), ["absent"])

    def test_majority_flip_with_old_allele_still_present_is_not_fully_replaced(self):
        # C takes over but A (5/30 = 17 %) is still present at t2 -> shared allele.
        got = self._scan((30, 0, 0, 0), (5, 25, 0, 0))
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["fully_replaced"]), [False])
        self.assertEqual(list(got["t2_consensus"]), [True])           # ...but C IS the majority now

    def test_thin_t1_majority_is_not_reported_as_new(self):
        # A has 2 of 5 reads at t1 (bar 3): fails presence but IS the consensus -> credible.
        got = self._scan((2, 1, 1, 1), (30, 0, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_below_min_cov_positions_are_never_scanned(self):
        got = self._scan((3, 0, 0, 0), (10, 10, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_allele_already_present_at_t1_is_not_new(self):
        # C had 5 of 25 at t1 (bar 3, 20 %) -> present -> credible -> not new.
        got = self._scan((20, 5, 0, 0), (18, 9, 0, 0))
        self.assertEqual(len(got["position"]), 0)

    def test_one_read_short_at_t1_is_below_detection(self):
        """t1 has 1 C read of 30 (bar 3): not credible, reported, and labelled so."""
        got = self._scan((29, 1, 0, 0), (24, 6, 0, 0))
        self.assertEqual(list(got["base"]), ["C"])
        self.assertEqual(list(got["t1_reads"]), [1])
        self.assertEqual(list(got["t1_threshold"]), [3])
        self.assertEqual(list(got["t1_evidence"]), ["below_detection"])

    def test_under_five_percent_at_t1_is_below_detection(self):
        # 4 of 100 passes the bar (4 at 100x) but fails the 5 % floor.
        got = self._scan((96, 4, 0, 0), (80, 20, 0, 0))
        self.assertEqual(list(got["t1_evidence"]), ["below_detection"])
        self.assertEqual(list(got["t1_reads"]), [4])

    def test_raw_rows_ride_along(self):
        got = self._scan((30, 0, 0, 0), (24, 6, 0, 0))
        self.assertEqual(got["counts_t1_rows"].tolist(), [[30, 0, 0, 0]])
        self.assertEqual(got["counts_t2_rows"].tolist(), [[24, 6, 0, 0]])
        self.assertEqual((list(got["coverage_t1"]), list(got["coverage_t2"])), ([30], [30]))

    def test_empty_result_has_every_key(self):
        got = self._scan((30, 0, 0, 0), (30, 0, 0, 0))
        for key in ("position", "base", "t1_reads", "t1_threshold", "t1_evidence", "freq_t1",
                    "t2_reads", "t2_threshold", "freq_t2", "t2_consensus", "fully_replaced", "counts_t1_rows", "coverage_t1", "counts_t2_rows", "coverage_t2"):
            self.assertEqual(len(got[key]), 0, key)


class TestAllelesPresentAt(unittest.TestCase):
    def test_presence_lookup_at_named_positions(self):
        dense = _dense({0: (30, 0, 0, 0), 2: (24, 6, 0, 0)}, 3)
        got = alleles_present_at(dense, np.array([0, 1, 2]), MODEL, 0.05)
        self.assertEqual(got.shape, (3, 4))
        self.assertTrue(bool(got[0, 0]))          # A present at pos 0
        self.assertFalse(got[1].any())            # zero coverage -> nothing present
        self.assertTrue(bool(got[2, 1]))          # C (6/30) present at pos 2
        self.assertTrue(bool(got[2, 0]))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# End-to-end: the ``alleleflux-strain-turnover`` command
# ---------------------------------------------------------------------------
import gzip
import os
import shutil
import subprocess
import tempfile

PROFILE_HEADER = "contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tgene_id\n"


def _write_profile(directory, sample, mag, rows):
    """rows: (contig, position, ref, A, C, G, T, gene_id)."""
    sample_dir = os.path.join(directory, sample)
    os.makedirs(sample_dir, exist_ok=True)
    with gzip.open(os.path.join(sample_dir, f"{sample}_{mag}_profiled.tsv.gz"), "wt") as handle:
        handle.write(PROFILE_HEADER)
        for contig, pos, ref, a, c, g, t, gene in rows:
            handle.write(f"{contig}\t{pos}\t{ref}\t{a + c + g + t}\t{a}\t{c}\t{g}\t{t}\t0\t{gene}\n")


class TestStrainTurnoverCLI(unittest.TestCase):
    """Runs the REAL pairwise-ANI command to make the pair table, then the
    turnover command on top of it (both via ``conda run``-installed scripts).

    Two mice over a 6 bp contig, transition pre -> end:
      m1 (fat):     S1 pre = 20 A everywhere; S3 end = same except position 2
                    is 14 A + 6 C (a NEW allele at 30 %) and position 5 has 2
                    reads (below min_cov).  -> stable background, 1 candidate.
      m2 (control): S2 pre = 20 A everywhere; S4 end = 20 G everywhere.
                    -> every compared base a fixed difference: strain_replacement
                    + dominant_strain_change, NOT scanned.
    """
    MAG = "MAG_T"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.profiles = os.path.join(self.tmp, "profiles")
        self.pairwise = os.path.join(self.tmp, "pairwise")
        self.out = os.path.join(self.tmp, "turnover")
        A = lambda p, gene="g1": ("c1", p, "A", 20, 0, 0, 0, gene)
        _write_profile(self.profiles, "S1", self.MAG, [A(p) for p in range(6)])
        _write_profile(self.profiles, "S3", self.MAG,
                       [A(0), A(1), ("c1", 2, "A", 14, 6, 0, 0, "g1"), A(3), A(4), ("c1", 5, "A", 2, 0, 0, 0, "")])
        _write_profile(self.profiles, "S2", self.MAG, [A(p) for p in range(6)])
        _write_profile(self.profiles, "S4", self.MAG, [("c1", p, "A", 0, 0, 20, 0, "g1") for p in range(6)])
        self.qc = os.path.join(self.tmp, f"{self.MAG}_QC.tsv")
        pd.DataFrame({
            "sample_id": ["S1", "S2", "S3", "S4"], "MAG_ID": [self.MAG] * 4, "file_path": ["x"] * 4,
            "group": ["fat", "control", "fat", "control"], "subjectID": ["m1", "m2", "m1", "m2"],
            "replicate": ["r1", "r2", "r1", "r2"], "time": ["pre", "pre", "end", "end"],
            "genome_size": [6] * 4, "breadth": [1.0] * 4, "coverage_threshold_passed": [True] * 4,
        }).to_csv(self.qc, sep="\t", index=False)
        self.fasta = os.path.join(self.tmp, "ref.fa")
        with open(self.fasta, "w") as handle:
            handle.write(">c1\nAAAAAA\n")
        with open(self.fasta + ".fai", "w") as handle:
            handle.write("c1\t6\t4\t6\t7\n")
        self.mag_mapping = os.path.join(self.tmp, "mag_mapping.tsv")
        pd.DataFrame({"mag_id": [self.MAG], "contig_id": ["c1"]}).to_csv(self.mag_mapping, sep="\t", index=False)
        # The pair table this command consumes, made by the real upstream command.
        done = subprocess.run([
            "alleleflux-pairwise-ani", "--mag", self.MAG, "--profiles_dir", self.profiles,
            "--qc_files", self.qc, "--fasta", self.fasta, "--mag_mapping", self.mag_mapping,
            "--output_dir", self.pairwise, "--pairs", "transitions", "--transitions", "pre:end",
        ], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, *extra):
        return subprocess.run([
            "alleleflux-strain-turnover", "--mag", self.MAG,
            "--pair_table", os.path.join(self.pairwise, f"{self.MAG}_pairwise_ani.tsv"),
            "--profiles_dir", self.profiles, "--fasta", self.fasta, "--mag_mapping", self.mag_mapping,
            "--output_dir", self.out, "--transitions", "pre:end", *extra,
        ], capture_output=True, text=True)

    def _read(self, suffix, **kw):
        return pd.read_csv(os.path.join(self.out, f"{self.MAG}{suffix}"), sep="\t", **kw)

    def test_turnover_table_has_one_row_per_mouse_with_both_verdicts(self):
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        table = self._read("_strain_turnover.tsv").set_index("subjectID")
        self.assertEqual(sorted(table.index), ["m1", "m2"])
        m1, m2 = table.loc["m1"], table.loc["m2"]
        self.assertEqual((m1.sample_t1, m1.sample_t2, m1.transition), ("S1", "S3", "pre_end"))
        self.assertEqual(m1.background, "stable")
        self.assertEqual((int(m1.n_absent), int(m1.n_below_detection), int(m1.n_de_novo)), (1, 0, 1))
        self.assertEqual(m2.background, "strain_replacement+dominant_strain_change")
        self.assertTrue(bool(m2.strain_replacement) and bool(m2.dominant_strain_change))
        self.assertTrue(pd.isna(m2.n_de_novo))          # not scanned -> blank, not 0
        # provenance stamped on every row
        self.assertEqual((float(m1.min_compared), float(m1.pop_threshold), float(m1.con_threshold), int(m1.min_cov)),
                         (0.1, 0.99999, 0.999, 5))
        self.assertEqual(m1.replicate, "r1")

    def test_candidates_file_has_the_new_allele_with_gene_and_counts(self):
        self._run()
        cand = self._read("_de_novo_candidates.tsv.gz")
        self.assertEqual(len(cand), 1)
        row = cand.iloc[0]
        self.assertEqual((row.subjectID, row.group, row.transition, row.contig, int(row.position), row.base),
                         ("m1", "fat", "pre_end", "c1", 2, "C"))
        self.assertEqual(row.gene_id, "g1")
        self.assertEqual((int(row.t1_reads), int(row.t1_threshold), row.t1_evidence), (0, 3, "absent"))
        self.assertEqual((int(row.t2_reads), float(row.freq_t2)), (6, 0.3))
        self.assertEqual([int(row[c]) for c in ("A_t1", "C_t1", "G_t1", "T_t1", "coverage_t1")], [20, 0, 0, 0, 20])
        self.assertEqual([int(row[c]) for c in ("A_t2", "C_t2", "G_t2", "T_t2", "coverage_t2")], [14, 6, 0, 0, 20])
        self.assertFalse(bool(row.t2_consensus) or bool(row.fully_replaced))

    def test_rollup_keeps_groups_separate(self):
        self._run()
        roll = self._read("_turnover_rollup.tsv").set_index("group")
        fat, control = roll.loc["fat"], roll.loc["control"]
        self.assertEqual((int(fat.n_pairs), int(fat.n_stable), int(fat.n_both)), (1, 1, 0))
        self.assertEqual((int(fat.n_absent), int(fat.n_below_detection), int(fat.n_de_novo)), (1, 0, 1))
        # 1 de novo over the 5 compared bases of the one scanned pair
        self.assertAlmostEqual(float(fat.de_novo_per_mb), 1 / 5 * 1e6)
        self.assertEqual((int(control.n_pairs), int(control.n_both), int(control.n_stable)), (1, 1, 0))
        self.assertTrue(pd.isna(control.de_novo_per_mb))   # nothing scanned in this group

    def test_min_cov_must_match_the_pair_table(self):
        done = self._run("--min_cov", "4")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("min_cov", done.stderr)

    def test_no_matching_transition_gives_header_only_outputs(self):
        done = self._run("--transitions", "pre:post")
        self.assertEqual(done.returncode, 0, done.stderr)
        for suffix in ("_strain_turnover.tsv", "_de_novo_candidates.tsv.gz", "_turnover_rollup.tsv"):
            table = self._read(suffix)
            self.assertEqual(len(table), 0, suffix)
            self.assertGreater(len(table.columns), 5, suffix)
