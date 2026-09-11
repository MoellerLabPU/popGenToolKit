"""Tests for ``alleleflux-baseline-presence``: was the significant allele already
there at the baseline timepoint?

Unit tests cover each pure step; the end-to-end test builds a miniature run
directory (summary, per-MAG test file, profiles, metadata, reference) and runs
the real console script.
"""
import gzip
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from alleleflux.scripts.analysis.ani.null_model import build_error_model
from alleleflux.scripts.analysis.baseline_presence import (
    EVIDENCE_ABSENT,
    EVIDENCE_BELOW_DETECTION,
    EVIDENCE_NOT_COVERED,
    EVIDENCE_PRESENT,
    ORIGIN_DE_NOVO,
    ORIGIN_DE_NOVO_BELOW_DETECTION,
    ORIGIN_NO_T0_SAMPLE,
    ORIGIN_ABSENT_AT_T1,
    ORIGIN_BELOW_DETECTION_AT_T1,
    ORIGIN_STANDING,
    ORIGIN_T0_NOT_COVERED,
    ORIGIN_T1_NOT_COVERED,
    label_origin_in_own_mouse,
    allele_p_column,
    assess_allele,
    find_summary_file,
    output_label,
    candidate_alleles,
    load_significant_sites,
    parse_comparison,
    summarise_sites,
)

MODEL = build_error_model(min_base_quality=30, fdr=1e-6)   # bar = 3 reads for 5..99x


class TestParseComparison(unittest.TestCase):
    def test_splits_period_and_groups(self):
        self.assertEqual(parse_comparison("pre_end-fat_control"), ("pre", "end", "fat", "control"))
        self.assertEqual(parse_comparison("5mo_22mo-40_AL"), ("5mo", "22mo", "40", "AL"))

    def test_rejects_malformed_label(self):
        for bad in ("pre_end", "pre-end-fat_control", "pre_mid_end-fat_control"):
            with self.assertRaises(ValueError):
                parse_comparison(bad)


class TestFindSummaryFile(unittest.TestCase):
    def test_family_prefix_does_not_swallow_across_time(self):
        tmp = tempfile.mkdtemp()
        for name in ("p_value_summary_lmm_pre_end.tsv", "p_value_summary_lmm_across_time_pre_end.tsv",
                     "p_value_summary_two_sample_paired_5mo_22mo-40_AL.tsv"):
            open(os.path.join(tmp, name), "w").write("x\n")
        self.assertTrue(find_summary_file(tmp, "lmm", "pre_end").endswith("p_value_summary_lmm_pre_end.tsv"))
        self.assertTrue(find_summary_file(tmp, "lmm_across_time", "pre_end").endswith("across_time_pre_end.tsv"))
        self.assertTrue(find_summary_file(tmp, "two_sample_paired", "5mo_22mo").endswith("-40_AL.tsv"))   # DRiDO naming
        with self.assertRaises(FileNotFoundError):
            find_summary_file(tmp, "two_sample_unpaired", "pre_end")
        shutil.rmtree(tmp)


class TestOutputLabel(unittest.TestCase):
    def test_family_is_always_in_the_name_and_never_repeated(self):
        self.assertEqual(output_label("two_sample_paired", "two_sample_paired_tTest"), "two_sample_paired_tTest")
        self.assertEqual(output_label("lmm", "LMM_abs"), "lmm_LMM_abs")
        self.assertEqual(output_label("lmm_across_time", "LMM"), "lmm_across_time_LMM")   # distinct from lmm's


class TestLoadSignificantSites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "p_value_summary_two_sample_paired_pre_end.tsv")
        pd.DataFrame({
            "period": ["pre_end"] * 4, "mag_id": ["MAG_A"] * 4,
            "contig": ["c1"] * 4, "position": [10, 10, 20, 30],
            "gene_id": ["g1 ", "g1 ", "", "g2 "],                       # real files carry a trailing space
            "test_type": ["two_sample_paired_tTest", "two_sample_paired_Wilcoxon",
                          "two_sample_paired_tTest", "two_sample_paired_tTest"],
            "min_p_value": [0.001, 0.002, 0.03, 0.2], "source_file": ["MAG_A_two_sample_paired.tsv.gz"] * 4,
            "q_value": [0.01, 0.02, 0.06, 0.5],
        }).to_csv(self.path, sep="\t", index=False)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_filters_to_the_test_type_and_the_threshold_column(self):
        by_q = load_significant_sites(self.path, "two_sample_paired_tTest", "q_value", 0.05)
        self.assertEqual(by_q.position.tolist(), [10])                 # Wilcoxon row and q=0.06 row dropped
        self.assertEqual(by_q.gene_id.tolist(), ["g1"])                # stripped
        by_p = load_significant_sites(self.path, "two_sample_paired_tTest", "min_p_value", 0.05)
        self.assertEqual(by_p.position.tolist(), [10, 20])             # p=0.03 passes on p
        self.assertEqual(by_p.group_analyzed.tolist(), ["", ""])       # two-sample: no group column

    def test_any_test_type_in_the_file_is_selectable(self):
        wilcoxon = load_significant_sites(self.path, "two_sample_paired_Wilcoxon", "q_value", 0.05)
        self.assertEqual(wilcoxon.position.tolist(), [10])

    def test_wrong_test_type_finds_nothing_and_says_so(self):
        with self.assertRaisesRegex(ValueError, "single_sample_tTest"):
            load_significant_sites(self.path, "single_sample_tTest", "q_value", 0.05)

    def test_test_type_is_required(self):
        with self.assertRaisesRegex(ValueError, "Wilcoxon"):          # None -> error listing what the file holds
            load_significant_sites(self.path, None, "q_value", 0.05)


class TestCandidateAlleles(unittest.TestCase):
    def test_two_sample_ties_keep_both_bases(self):
        sites = pd.DataFrame({"contig": ["c1", "c1"], "position": [10, 20], "gene_id": ["g1", ""],
                              "min_p_value": [0.001, 0.03], "group_analyzed": ["", ""]})
        source = pd.DataFrame({
            "contig": ["c1", "c1"], "position": [10, 20], "gene_id": ["g1 ", np.nan],   # real-file quirks
            "A_frequency_p_value_tTest": [0.001, 0.5], "C_frequency_p_value_tTest": [1.0, 0.03],
            "G_frequency_p_value_tTest": [0.001, 0.9], "T_frequency_p_value_tTest": [1.0, 1.0],
        })
        got = candidate_alleles(sites, source, allele_p_column("two_sample_paired", "two_sample_paired_tTest"))
        self.assertEqual(got[got.position == 10].allele.tolist(), ["A", "G"])   # complementary pair
        self.assertEqual(got[got.position == 20].allele.tolist(), ["C"])
        self.assertEqual(got.n_alleles_tied_at_min_p.tolist(), [2, 2, 1])

    def test_single_sample_reads_the_group_suffixed_column(self):
        sites = pd.DataFrame({"contig": ["c1"], "position": [10], "gene_id": ["g1"], "min_p_value": [0.004], "group_analyzed": ["fat"]})
        source = pd.DataFrame({
            "contig": ["c1"], "position": [10], "gene_id": ["g1"],
            "A_frequency_p_value_tTest_fat": [0.9], "C_frequency_p_value_tTest_fat": [0.004],
            "G_frequency_p_value_tTest_fat": [0.9], "T_frequency_p_value_tTest_fat": [1.0],
        })
        self.assertEqual(candidate_alleles(sites, source, allele_p_column("single_sample", "single_sample_tTest")).allele.tolist(), ["C"])

    def test_site_missing_from_source_raises_and_names_it(self):
        sites = pd.DataFrame({"contig": ["c1", "c1"], "position": [10, 99], "gene_id": ["g1", "g1"],
                              "min_p_value": [1.0, 0.001], "group_analyzed": ["", ""]})
        source = pd.DataFrame({"contig": ["c1"], "position": [10], "gene_id": ["g1"],
                               **{f"{b}_frequency_p_value_tTest": [1.0] for b in "ACGT"}})
        with self.assertRaisesRegex(ValueError, "1 of 2 sites.*99"):
            candidate_alleles(sites, source, allele_p_column("two_sample_paired", "two_sample_paired_tTest"))

    def test_gene_id_disagreement_is_a_mismatch(self):
        sites = pd.DataFrame({"contig": ["c1"], "position": [10], "gene_id": ["g1"], "min_p_value": [1.0], "group_analyzed": [""]})
        source = pd.DataFrame({"contig": ["c1"], "position": [10], "gene_id": ["g2"],
                               **{f"{b}_frequency_p_value_tTest": [1.0] for b in "ACGT"}})
        with self.assertRaises(ValueError):
            candidate_alleles(sites, source, allele_p_column("two_sample_paired", "two_sample_paired_tTest"))

    def test_column_registry_covers_every_family_that_reports_per_base_p(self):
        col = allele_p_column("two_sample_unpaired", "two_sample_unpaired_MannWhitney_abs")
        self.assertEqual(col("G", ""), "G_frequency_p_value_MannWhitney_abs")
        col = allele_p_column("single_sample", "single_sample_Wilcoxon")
        self.assertEqual(col("A", "fat"), "A_frequency_p_value_Wilcoxon_fat")
        col = allele_p_column("lmm", "LMM_abs")
        self.assertEqual(col("T", ""), "T_p_value_LMM_abs")
        col = allele_p_column("lmm_across_time", "LMM")
        self.assertEqual(col("C", "control"), "C_p_value_LMM")            # one FILE per group, plain columns



class TestAssessAllele(unittest.TestCase):
    def test_four_evidence_tiers(self):
        counts = np.zeros((5, 4), dtype=np.uint16)
        counts[0] = (25, 0, 5, 0)      # G 5 of 30: present
        counts[1] = (29, 0, 1, 0)      # G 1 of 30: below the bar of 3
        counts[2] = (30, 0, 0, 0)      # G 0 of 30: absent
        counts[3] = (2, 0, 0, 0)       # 2 reads total: not covered at min_cov 5
        counts[4] = (96, 0, 4, 0)      # G 4 of 100: clears bar 4 but under 5 %
        got = assess_allele(counts, np.array([0, 1, 2, 3, 4]), base_idx=2, model=MODEL, min_freq=0.05, min_cov=5)
        self.assertEqual(got["allele_status"].tolist(), [EVIDENCE_PRESENT, EVIDENCE_BELOW_DETECTION, EVIDENCE_ABSENT,
                                                    EVIDENCE_NOT_COVERED, EVIDENCE_BELOW_DETECTION])
        self.assertEqual(got["allele_reads"].tolist(), [5, 1, 0, 0, 4])
        self.assertEqual(got["total_reads"].tolist(), [30, 30, 30, 2, 100])
        self.assertEqual(got["detection_threshold_reads"].tolist(), [3, 3, 3, 2, 4])     # bar is 2 reads at 2x, 4 at 100x
        self.assertEqual(got["allele_present"].tolist(), [True, False, False, False, False])
        self.assertTrue(np.isnan(got["allele_frequency"][3]))                     # no frequency without coverage

    def test_min_cov_toggle_off(self):
        counts = np.array([[1, 0, 1, 0]], dtype=np.uint16)            # 2 reads, G 1 of 2
        got = assess_allele(counts, np.array([0]), 2, MODEL, 0.05, min_cov=1)
        self.assertEqual(got["allele_status"].tolist(), [EVIDENCE_BELOW_DETECTION])   # covered now, but 1 < bar


class TestLabelOriginInOwnMouse(unittest.TestCase):
    """t1 rows get a verdict from BOTH their own t1 status and their own mouse's t0
    status; t0 rows stay blank.  One site-allele, seven mice covering every case."""

    CASES = {  # mouse: (t0 status or None for no t0 sample, t1 status)
        "m1": ("present", "present"),            # standing_variation
        "m2": ("below_detection", "present"),    # de_novo_candidate_below_detection_at_t0
        "m3": ("absent", "present"),             # de_novo_candidate
        "m4": ("not_covered", "present"),        # t0_not_covered
        "m5": (None, "present"),                 # no_t0_sample
        "m6": ("absent", "below_detection"),     # allele_below_detection_at_t1 (t0 irrelevant)
        "m7": ("present", "not_covered"),        # t1_not_covered
        "m8": ("present", "absent"),             # allele_absent_at_t1 (t0 irrelevant)
    }

    def _long(self):
        site = {"mag_id": "MAG_A", "contig": "c1", "position": 10, "gene_id": "g1", "allele": "G",
                "test_type": "t", "group_analyzed": ""}
        rows = []
        for mouse, (t0, t1) in self.CASES.items():
            if t0 is not None:
                rows.append({**site, "sample_id": f"{mouse}_t0", "subjectID": mouse, "timepoint_role": "t0", "allele_status": t0})
            rows.append({**site, "sample_id": f"{mouse}_t1", "subjectID": mouse, "timepoint_role": "t1", "allele_status": t1})
        return pd.DataFrame(rows)

    def test_every_label(self):
        got = label_origin_in_own_mouse(self._long()).set_index("sample_id").origin_in_own_mouse
        self.assertEqual(got["m1_t1"], ORIGIN_STANDING)
        self.assertEqual(got["m2_t1"], ORIGIN_DE_NOVO_BELOW_DETECTION)
        self.assertEqual(got["m3_t1"], ORIGIN_DE_NOVO)
        self.assertEqual(got["m4_t1"], ORIGIN_T0_NOT_COVERED)
        self.assertEqual(got["m5_t1"], ORIGIN_NO_T0_SAMPLE)
        self.assertEqual(got["m6_t1"], ORIGIN_BELOW_DETECTION_AT_T1)
        self.assertEqual(got["m8_t1"], ORIGIN_ABSENT_AT_T1)
        self.assertEqual(got["m7_t1"], ORIGIN_T1_NOT_COVERED)
        self.assertTrue(got[[s for s in got.index if s.endswith("_t0")]].isna().all())

class TestSummariseSites(unittest.TestCase):
    """One site, allele G.  t0: m1 present, m2 absent, m3 below, m4 not covered.
    t1: m1..m4 all present.  m1+m2 share replicate r1 (fat), m3+m4 replicate r2 (control)."""

    def _long(self):
        rows = []
        pre = {"m1": EVIDENCE_PRESENT, "m2": EVIDENCE_ABSENT, "m3": EVIDENCE_BELOW_DETECTION, "m4": EVIDENCE_NOT_COVERED}
        meta = {"m1": ("r1", "fat"), "m2": ("r1", "fat"), "m3": ("r2", "control"), "m4": ("r2", "control")}
        for mouse, ev in pre.items():
            rep, grp = meta[mouse]
            for role, evidence, freq in (("t0", ev, {"allele_present": 0.2, "absent": 0.0}.get(ev, 0.02)),
                                          ("t1", EVIDENCE_PRESENT, 0.8)):
                rows.append({"mag_id": "MAG_A", "contig": "c1", "position": 10, "gene_id": "g1", "allele": "G",
                             "test_type": "two_sample_paired_tTest", "group_analyzed": "",
                             "sample_id": f"{mouse}_{role}", "subjectID": mouse, "replicate": rep, "group": grp,
                             "timepoint_role": role, "allele_status": evidence,
                             "allele_present": evidence == EVIDENCE_PRESENT,
                             "allele_frequency": np.nan if evidence == EVIDENCE_NOT_COVERED else freq})
        return pd.DataFrame(rows)

    def test_site_verdict_requires_the_allele_to_be_seen_at_t1(self):
        long = label_origin_in_own_mouse(self._long())
        long.loc[long.timepoint_role == "t1", "allele_status"] = "absent"        # nobody shows it at t1
        long.loc[long.timepoint_role == "t1", "allele_present"] = False
        got = summarise_sites(long).iloc[0]
        self.assertEqual(got.origin_any_mouse, "allele_not_present_at_t1")
        self.assertEqual(int(got.n_t0_samples_allele_present), 1)               # the t0 fact itself is unchanged

    def test_lean_summary_row(self):
        got = summarise_sites(label_origin_in_own_mouse(self._long())).iloc[0]
        self.assertEqual(list(got.index), [
            "mag_id", "contig", "position", "gene_id", "group_analyzed", "allele", "n_alleles_tied_at_min_p", "q_value",
            "origin_any_mouse", "n_t0_samples_allele_present", "n_t0_samples_covered",
            "n_replicates_with_allele_at_t0", "t0_mice_allele_present",
            "n_mice_standing_variation", "n_mice_de_novo_candidate", "n_mice_de_novo_candidate_below_detection_at_t0"])
        self.assertEqual(got.origin_any_mouse, ORIGIN_STANDING)                       # m1 had it at t0
        # PRE: m1 present, m2 absent, m3 below, m4 not covered -> 1 present of 3 covered, 1 replicate (r1)
        self.assertEqual((int(got.n_t0_samples_allele_present), int(got.n_t0_samples_covered),
                          int(got.n_replicates_with_allele_at_t0), got.t0_mice_allele_present), (1, 3, 1, "m1"))
        # per-mouse: m1 standing, m2 de novo, m3 de novo (below det.); m4's t0 not covered is not a summary column
        self.assertEqual((int(got.n_mice_standing_variation), int(got.n_mice_de_novo_candidate),
                          int(got.n_mice_de_novo_candidate_below_detection_at_t0)), (1, 1, 1))

# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
PROFILE_HEADER = "contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tgene_id\n"


def _write_profile(directory, sample, mag, rows):
    """rows: (position, A, C, G, T) on contig c1, gene g1."""
    os.makedirs(os.path.join(directory, sample), exist_ok=True)
    with gzip.open(os.path.join(directory, sample, f"{sample}_{mag}_profiled.tsv.gz"), "wt") as h:
        h.write(PROFILE_HEADER)
        for p, a, c, g, t in rows:
            h.write(f"c1\t{p}\tA\t{a + c + g + t}\t{a}\t{c}\t{g}\t{t}\t0\tg1\n")


class TestBaselinePresenceCLI(unittest.TestCase):
    """8 samples: mice m1,m2 (fat, replicate r1) and m3,m4 (control, r2), each at pre and end.
    One significant site, c1:10, where G rose at end in the fat mice.  t0 G reads:
    m1 5/30 present, m2 0/30 absent, m3 1/30 below detection, m4 2 reads total (not covered)."""
    MAG = "MAG_A"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        run = self.run = os.path.join(self.tmp, "run"); prof = os.path.join(run, "profiles")
        base = [(p, 30, 0, 0, 0) for p in range(6)]
        def with_site(a, c, g, t): return [(p, 30, 0, 0, 0) if p != 3 else (3, a, c, g, t) for p in range(6)] if False else \
            [(p, 30, 0, 0, 0) for p in range(6) if p != 1] + [(1, a, c, g, t)]
        pre = {"m1": (25, 0, 5, 0), "m2": (30, 0, 0, 0), "m3": (29, 0, 1, 0), "m4": (2, 0, 0, 0)}
        end = {"m1": (6, 0, 24, 0), "m2": (9, 0, 21, 0), "m3": (28, 0, 2, 0), "m4": (30, 0, 0, 0)}
        for mouse in pre:
            _write_profile(prof, f"{mouse}_pre", self.MAG, with_site(*pre[mouse]))
            _write_profile(prof, f"{mouse}_end", self.MAG, with_site(*end[mouse]))
        meta = pd.DataFrame([
            {"sample_id": f"{m}_{t}", "subjectID": m, "group": g, "time": t, "replicate": r, "bam_path": "x"}
            for m, g, r in (("m1", "fat", "r1"), ("m2", "fat", "r1"), ("m3", "control", "r2"), ("m4", "control", "r2"))
            for t in ("pre", "end")])
        self.metadata = os.path.join(self.tmp, "metadata.tsv"); meta.to_csv(self.metadata, sep="\t", index=False)
        # p_value_summary + the per-MAG test file the summary points at (site c1:1, G and A tie)
        sd = os.path.join(run, "p_value_summary", "pre_end-fat_control"); os.makedirs(sd)
        pd.DataFrame({"period": ["pre_end"], "mag_id": [self.MAG], "contig": ["c1"], "position": [1], "gene_id": ["g1 "],
                      "test_type": ["two_sample_paired_tTest"], "min_p_value": [0.001],
                      "source_file": [f"{self.MAG}_two_sample_paired.tsv.gz"], "q_value": [0.01]}
                     ).to_csv(os.path.join(sd, "p_value_summary_two_sample_paired_pre_end.tsv"), sep="\t", index=False)
        td = os.path.join(run, "significance_tests", "two_sample_paired_pre_end-fat_control"); os.makedirs(td)
        pd.DataFrame({"contig": ["c1"], "gene_id": ["g1"], "position": [1],
                      "A_frequency_p_value_tTest": [0.001], "C_frequency_p_value_tTest": [1.0],
                      "G_frequency_p_value_tTest": [0.001], "T_frequency_p_value_tTest": [1.0], "num_pairs": [4]}
                     ).to_csv(os.path.join(td, f"{self.MAG}_two_sample_paired.tsv.gz"), sep="\t", index=False, compression="gzip")
        self.fasta = os.path.join(self.tmp, "ref.fa"); open(self.fasta, "w").write(">c1\nAAAAAA\n")
        open(self.fasta + ".fai", "w").write("c1\t6\t4\t6\t7\n")
        self.mapping = os.path.join(self.tmp, "map.tsv")
        pd.DataFrame({"mag_id": [self.MAG], "contig_id": ["c1"]}).to_csv(self.mapping, sep="\t", index=False)
        self.out = os.path.join(self.tmp, "out")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, *extra):
        return subprocess.run([
            "alleleflux-baseline-presence", "--run_dir", self.run, "--comparison", "pre_end-fat_control",
            "--summary", "two_sample_paired", "--test_type", "two_sample_paired_tTest", "--profiles_dir", os.path.join(self.run, "profiles"),
            "--metadata", self.metadata, "--fasta", self.fasta, "--mag_mapping", self.mapping,
            "--output_dir", self.out, "--cpus", "2", *extra,
        ], capture_output=True, text=True)

    def test_long_table_one_row_per_site_allele_sample(self):
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        long = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence.tsv.gz"), sep="\t")
        self.assertEqual(len(long), 2 * 8)                          # 2 candidate alleles x 8 samples
        g = long[(long.allele == "G")].set_index("sample_id")
        self.assertEqual(g.loc["m1_pre"].allele_status, "present");  self.assertEqual(int(g.loc["m1_pre"].allele_reads), 5)
        self.assertEqual(g.loc["m2_pre"].allele_status, "absent")
        self.assertEqual(g.loc["m3_pre"].allele_status, "below_detection"); self.assertEqual(int(g.loc["m3_pre"].detection_threshold_reads), 3)
        self.assertEqual(g.loc["m4_pre"].allele_status, "not_covered"); self.assertTrue(pd.isna(g.loc["m4_pre"].allele_frequency))
        row = g.loc["m1_pre"]
        self.assertEqual((row.mag_id, row.contig, int(row.position), row.gene_id, row.subjectID, row.replicate, row.group,
                          row.timepoint_role, int(row.n_alleles_tied_at_min_p), int(row.min_cov)),
                         (self.MAG, "c1", 1, "g1", "m1", "r1", "fat", "t0", 2, 5))
        self.assertEqual(sorted(long.allele.unique()), ["A", "G"])
        self.assertEqual(g.loc["m1_end"].origin_in_own_mouse, "standing_variation")
        self.assertEqual(g.loc["m2_end"].origin_in_own_mouse, "de_novo_candidate")
        self.assertEqual(g.loc["m3_end"].origin_in_own_mouse, "allele_below_detection_at_t1")   # G 2/30 at end
        self.assertEqual(g.loc["m4_end"].origin_in_own_mouse, "allele_absent_at_t1")            # G 0/30 at end
        self.assertTrue(pd.isna(g.loc["m1_pre"].origin_in_own_mouse))
        self.assertTrue(long.strain_background.isna().all())          # no --turnover_dir given

    def test_summary_counts(self):
        self._run()
        summ = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence_summary.tsv"), sep="\t")
        g = summ[summ.allele == "G"].iloc[0]
        self.assertEqual((int(g.n_t0_samples_allele_present), int(g.n_t0_samples_covered), int(g.n_replicates_with_allele_at_t0)), (1, 3, 1))
        self.assertEqual(g.t0_mice_allele_present, "m1")
        self.assertEqual(g.origin_any_mouse, "standing_variation")
        self.assertEqual((int(g.n_mice_standing_variation), int(g.n_mice_de_novo_candidate)), (1, 1))   # m1; m2
        self.assertEqual(len(summ.columns), 16)

    def test_strain_background_joins_when_given(self):
        tdir = os.path.join(self.tmp, "turnover"); os.makedirs(tdir)
        pd.DataFrame({"MAG_ID": [self.MAG] * 2, "subjectID": ["m1", "m2"], "transition": ["pre_end"] * 2,
                      "background": ["stable", "strain_replacement"]}
                     ).to_csv(os.path.join(tdir, f"{self.MAG}_strain_turnover.tsv"), sep="\t", index=False)
        done = self._run("--turnover_dir", tdir)
        self.assertEqual(done.returncode, 0, done.stderr)
        long = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence.tsv.gz"), sep="\t")
        bg = long.drop_duplicates("subjectID").set_index("subjectID").strain_background
        self.assertEqual((bg["m1"], bg["m2"]), ("stable", "strain_replacement"))
        m2_end = long[(long.sample_id == "m2_end") & (long.allele == "G")].iloc[0]
        self.assertEqual(m2_end.origin_in_own_mouse, "de_novo_candidate")    # label does NOT depend on the strain column
        self.assertTrue(pd.isna(bg["m3"]))                             # mouse not in the turnover table

    def test_sample_without_a_profile_is_not_covered_everywhere(self):
        # A MAG with no reads in a sample gets no profile from profile_mags; the
        # sample must still appear, as not_covered, not crash the run.
        shutil.rmtree(os.path.join(self.run, "profiles", "m2_pre"))
        done = self._run()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("no profile", done.stderr)                          # warned, not silent
        long = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence.tsv.gz"), sep="\t")
        self.assertEqual(len(long), 2 * 8)                                # still 8 samples x 2 alleles
        row = long[(long.sample_id == "m2_pre") & (long.allele == "G")].iloc[0]
        self.assertEqual((int(row.total_reads), int(row.allele_reads), row.allele_status), (0, 0, "not_covered"))
        summ = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence_summary.tsv"), sep="\t")
        self.assertEqual(int(summ[summ.allele == "G"].iloc[0].n_t0_samples_covered), 2)   # m1, m3 (m4 was thin, m2 now missing)

    def test_no_significant_sites_gives_header_only_outputs(self):
        done = self._run("--threshold", "0.0001")                       # q=0.01 site no longer passes
        self.assertEqual(done.returncode, 0, done.stderr)
        long = pd.read_csv(os.path.join(self.out, "pre_end-fat_control_two_sample_paired_tTest_baseline_presence.tsv.gz"), sep="\t")
        self.assertEqual(len(long), 0)
        self.assertIn("allele_status", long.columns)

    def test_missing_summary_for_the_chosen_family_fails_loud(self):
        done = self._run("--summary", "two_sample_unpaired", "--test_type", "two_sample_unpaired_tTest")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("two_sample_unpaired", done.stderr)


if __name__ == "__main__":
    unittest.main()
