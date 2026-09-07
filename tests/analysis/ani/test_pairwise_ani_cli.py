"""End-to-end tests for the ``alleleflux-pairwise-ani`` command over synthetic data.

These run the REAL console script in a subprocess (which is why the whole suite
must be invoked via ``conda run -n alleleflux``): profiles, QC and reference are
built on disk exactly in the production formats, the command runs, and the output
files are read back and checked.
"""
import gzip
import os
import shutil
import subprocess
import tempfile
import unittest

import pandas as pd

PROFILE_HEADER = "contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tgene_id\n"


def _write_profile(directory, sample, mag, rows):
    """rows: (contig, position, ref, A, C, G, T) -- N fixed at 0 for simplicity."""
    sample_dir = os.path.join(directory, sample)
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(sample_dir, f"{sample}_{mag}_profiled.tsv.gz")
    with gzip.open(path, "wt") as handle:
        handle.write(PROFILE_HEADER)
        for contig, pos, ref, a, c, g, t in rows:
            handle.write(f"{contig}\t{pos}\t{ref}\t{a + c + g + t}\t{a}\t{c}\t{g}\t{t}\t0\t\n")


class TestPairwiseAniCLI(unittest.TestCase):
    MAG = "MAG_1"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.profiles = os.path.join(self.tmp, "profiles")
        self.out = os.path.join(self.tmp, "out")

        # Three samples over a 6 bp contig.  S2 carries one fixed difference at
        # position 0 (pure G vs S1's pure A); S3 is identical to S1 except its
        # position 5 has only 2 reads -- below the default min_cov of 5.
        base = [("c1", p, "A", 20, 0, 0, 0) for p in range(6)]
        _write_profile(self.profiles, "S1", self.MAG, base)
        s2 = [("c1", 0, "A", 0, 0, 20, 0)] + [("c1", p, "A", 20, 0, 0, 0) for p in range(1, 6)]
        _write_profile(self.profiles, "S2", self.MAG, s2)
        s3 = [("c1", p, "A", 20, 0, 0, 0) for p in range(5)] + [("c1", 5, "A", 2, 0, 0, 0)]
        _write_profile(self.profiles, "S3", self.MAG, s3)

        # QC file in the real quality_control.py column layout (a subset).
        # S1 and S3 share mouse m1 (pre and end) -- the within-subject pair.
        self.qc = os.path.join(self.tmp, f"{self.MAG}_QC.tsv")
        pd.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "MAG_ID": [self.MAG] * 3,
            "file_path": ["x"] * 3,
            "group": ["fat", "control", "fat"],
            "subjectID": ["m1", "m2", "m1"],
            "replicate": [1, 1, 1],
            "time": ["pre", "pre", "end"],
            "genome_size": [6, 6, 6],           # matches the reference: the happy path
            "breadth": [1.0, 1.0, 1.0],         # matches the profiles: the happy path
            "coverage_threshold_passed": [True, True, True],
        }).to_csv(self.qc, sep="\t", index=False)

        # Tiny reference plus its .fai, matching how the pipeline supplies lengths.
        self.fasta = os.path.join(self.tmp, "ref.fa")
        with open(self.fasta, "w") as handle:
            handle.write(">c1\nAAAAAA\n")
        with open(self.fasta + ".fai", "w") as handle:
            handle.write("c1\t6\t4\t6\t7\n")
        self.mag_mapping = os.path.join(self.tmp, "mag_mapping.tsv")
        pd.DataFrame({"mag_id": [self.MAG], "contig_id": ["c1"]}).to_csv(
            self.mag_mapping, sep="\t", index=False
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, *extra):
        command = [
            "alleleflux-pairwise-ani",
            "--mag", self.MAG,
            "--profiles_dir", self.profiles,
            "--qc_files", self.qc,
            "--fasta", self.fasta,
            "--mag_mapping", self.mag_mapping,
            "--output_dir", self.out,
            # Most tests exercise the full pair matrix; the user-chosen DEFAULT is
            # within_subject and has its own dedicated test below.  argparse takes
            # the LAST occurrence, so *extra can still override this.
            "--pairs", "all",
            *extra,
        ]
        return subprocess.run(command, capture_output=True, text=True)

    def _read(self, suffix):
        return pd.read_csv(os.path.join(self.out, f"{self.MAG}{suffix}"), sep="\t")

    def test_writes_expected_pair_table(self):
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        table = self._read("_pairwise_ani.tsv")

        self.assertEqual(len(table), 3)  # three pairs from three samples
        row = table[(table.sample1 == "S1") & (table.sample2 == "S2")].iloc[0]
        self.assertEqual(int(row["compared_bases_count"]), 6)
        self.assertEqual(int(row["consensus_SNPs"]), 1)
        self.assertEqual(int(row["population_SNPs"]), 1)
        self.assertAlmostEqual(row["conANI"], 5 / 6)
        self.assertAlmostEqual(row["percent_genome_compared"], 1.0)
        self.assertEqual(int(row["min_cov"]), 5)          # the gate travels with the row
        # Metadata attached for BOTH sides, never blank (the 2026-06-18 lesson).
        self.assertEqual(row["subjectID_1"], "m1")
        self.assertEqual(row["group_2"], "control")
        self.assertEqual(row["time_1"], "pre")

        # S3's shallow position 5 drops out of the comparison entirely.
        shallow = table[(table.sample1 == "S1") & (table.sample2 == "S3")].iloc[0]
        self.assertEqual(int(shallow["compared_bases_count"]), 5)
        self.assertAlmostEqual(shallow["popANI"], 1.0)

    def test_min_cov_toggle_reaches_the_engine(self):
        """--min_cov 1 disables the depth gate: S3's 2-read position is compared."""
        self.assertEqual(self._run("--min_cov", "1").returncode, 0)
        table = self._read("_pairwise_ani.tsv")
        shallow = table[(table.sample1 == "S1") & (table.sample2 == "S3")].iloc[0]
        self.assertEqual(int(shallow["compared_bases_count"]), 6)   # was 5 at min_cov 5
        self.assertEqual(int(shallow["min_cov"]), 1)                # stamped accordingly

    def test_writes_sample_table_with_min_cov_breadth(self):
        self.assertEqual(self._run().returncode, 0)
        samples = self._read("_pairwise_ani_samples.tsv")
        self.assertEqual(sorted(samples["sample_id"]), ["S1", "S2", "S3"])
        s3 = samples[samples.sample_id == "S3"].iloc[0]
        self.assertEqual(int(s3["n_positions"]), 6)             # covered everywhere...
        self.assertAlmostEqual(s3["breadth"], 1.0)              # kept, QC-verified
        self.assertAlmostEqual(s3["breadth_minCov"], 5 / 6)     # ...but deep at 5 of 6
        # mean_coverage deliberately absent: QC's average_coverage is authoritative.
        self.assertNotIn("mean_coverage", samples.columns)

    def test_within_subject_mode_restricts_pairs(self):
        self.assertEqual(self._run("--pairs", "within_subject").returncode, 0)
        table = self._read("_pairwise_ani.tsv")
        self.assertEqual(len(table), 1)                          # only S1-S3 share m1
        self.assertEqual({table.iloc[0]["sample1"], table.iloc[0]["sample2"]}, {"S1", "S3"})

    def test_snp_locations_written_for_within_subject_by_default(self):
        self.assertEqual(self._run().returncode, 0)
        path = os.path.join(self.out, f"{self.MAG}_pairwise_ani_snp_locations.tsv.gz")
        self.assertTrue(os.path.exists(path))
        locations = pd.read_csv(path, sep="\t")
        # S1-S3 (the only same-mouse pair) has no SNPs: header present, zero rows.
        self.assertEqual(len(locations), 0)
        self.assertIn("population_SNP", locations.columns)

    def test_snp_locations_all_records_the_fixed_difference(self):
        self.assertEqual(self._run("--store_snp_locations", "all").returncode, 0)
        locations = pd.read_csv(
            os.path.join(self.out, f"{self.MAG}_pairwise_ani_snp_locations.tsv.gz"), sep="\t"
        )
        flagged = locations[(locations.sample1 == "S1") & (locations.sample2 == "S2")]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(int(flagged.iloc[0]["position"]), 0)
        self.assertEqual(int(flagged.iloc[0]["A_1"]), 20)
        self.assertEqual(int(flagged.iloc[0]["G_2"]), 20)

    def test_single_usable_sample_writes_headers_and_exits_cleanly(self):
        solo = pd.read_csv(self.qc, sep="\t")
        solo["coverage_threshold_passed"] = [True, False, False]
        solo.to_csv(self.qc, sep="\t", index=False)
        completed = self._run()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        table = self._read("_pairwise_ani.tsv")
        self.assertEqual(len(table), 0)
        self.assertIn("popANI", table.columns)                   # schema survives emptiness

    def test_default_pairs_mode_is_within_subject(self):
        """A bare invocation (no --pairs flag) compares only same-mouse pairs --
        the default chosen during review 2026-09-01."""
        command = [
            "alleleflux-pairwise-ani", "--mag", self.MAG,
            "--profiles_dir", self.profiles, "--qc_files", self.qc,
            "--fasta", self.fasta, "--mag_mapping", self.mag_mapping,
            "--output_dir", self.out,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        table = self._read("_pairwise_ani.tsv")
        self.assertEqual(len(table), 1)          # only S1-S3 share mouse m1
        self.assertEqual({table.iloc[0]["sample1"], table.iloc[0]["sample2"]}, {"S1", "S3"})

    def test_replicate_travels_to_both_outputs(self):
        """Andy's analyses read out per replicate; both tables must carry it."""
        self.assertEqual(self._run().returncode, 0)
        samples = self._read("_pairwise_ani_samples.tsv")
        self.assertEqual(sorted(samples["replicate"].astype(str)), ["1", "1", "1"])
        table = self._read("_pairwise_ani.tsv")
        self.assertIn("replicate_1", table.columns)
        self.assertIn("replicate_2", table.columns)
        self.assertFalse(table["replicate_1"].isna().any())

    def test_parallel_run_matches_serial_exactly(self):
        """--cpus 2 exercises the process pool AND the chunked thread path; the
        tables must be identical to the serial run, row order included."""
        # store_snp_locations all -> the locations file has rows (2 fixed
        # differences), so its row ORDER is under test too.
        self.assertEqual(self._run("--store_snp_locations", "all").returncode, 0)
        serial = self._read("_pairwise_ani.tsv")
        serial_samples = self._read("_pairwise_ani_samples.tsv")
        serial_locations = self._read("_pairwise_ani_snp_locations.tsv.gz")
        self.assertEqual(self._run("--store_snp_locations", "all", "--cpus", "2").returncode, 0)
        pd.testing.assert_frame_equal(serial, self._read("_pairwise_ani.tsv"))
        pd.testing.assert_frame_equal(serial_samples, self._read("_pairwise_ani_samples.tsv"))
        pd.testing.assert_frame_equal(serial_locations, self._read("_pairwise_ani_snp_locations.tsv.gz"))
        self.assertEqual(len(serial_locations), 2)

    def test_only_paired_samples_are_loaded_and_summarised(self):
        """within_subject: S2 (mouse m2, no partner) takes part in no pair, so it
        is never loaded -- at DRiDO scale that is ~38% of the roster."""
        self.assertEqual(self._run("--pairs", "within_subject").returncode, 0)
        samples = self._read("_pairwise_ani_samples.tsv")
        self.assertEqual(sorted(samples["sample_id"]), ["S1", "S3"])

    def test_transitions_mode_builds_only_the_configured_comparisons(self):
        """pre:end on the fixture -> the single same-mouse pair S1(pre)-S3(end);
        S2 (mouse m2, pre only) has no 'end' partner and is not loaded."""
        self.assertEqual(self._run("--pairs", "transitions", "--transitions", "pre:end").returncode, 0)
        table = self._read("_pairwise_ani.tsv")
        self.assertEqual(len(table), 1)
        self.assertEqual((table.iloc[0]["sample1"], table.iloc[0]["sample2"]), ("S1", "S3"))
        samples = self._read("_pairwise_ani_samples.tsv")
        self.assertEqual(sorted(samples["sample_id"]), ["S1", "S3"])

    def test_transitions_mode_with_no_matching_samples_is_header_only(self):
        """A configured transition nobody has (pre:post) is a legitimate empty
        result, not an error -- the pipeline must stay green."""
        completed = self._run("--pairs", "transitions", "--transitions", "pre:post")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        table = self._read("_pairwise_ani.tsv")
        self.assertEqual(len(table), 0)
        self.assertIn("popANI", table.columns)

    def test_transitions_mode_requires_the_transitions_flag(self):
        completed = self._run("--pairs", "transitions")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("transitions", completed.stderr)

    def test_degenerate_transition_is_rejected(self):
        completed = self._run("--pairs", "transitions", "--transitions", "pre:pre")
        self.assertNotEqual(completed.returncode, 0)

    def test_qc_breadth_mismatch_is_fatal(self):
        """QC recorded a breadth the profile no longer produces -> stale tree."""
        qc = pd.read_csv(self.qc, sep="\t")
        qc["breadth"] = 0.5
        qc.to_csv(self.qc, sep="\t", index=False)
        completed = self._run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("breadth", completed.stderr)

    def test_qc_genome_size_mismatch_is_fatal(self):
        """QC recorded a different genome size than the reference provides -> the
        run must refuse to mix the two references, loudly."""
        qc = pd.read_csv(self.qc, sep="\t")
        qc["genome_size"] = 999
        qc.to_csv(self.qc, sep="\t", index=False)
        completed = self._run()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("genome_size", completed.stderr)

    def test_missing_profile_for_passing_sample_is_fatal(self):
        os.remove(os.path.join(self.profiles, "S2", f"S2_{self.MAG}_profiled.tsv.gz"))
        completed = self._run()
        self.assertNotEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()


class TestEnumeratePairs(unittest.TestCase):
    """Direct unit tests for ``enumerate_pairs`` (no subprocess): a two-mouse,
    three-timepoint roster where mouse m2 was never sampled at 10mo."""

    ROSTER = pd.DataFrame({
        "sample_id": ["S1", "S2", "S3", "S4", "S5"],
        "subjectID": ["m1", "m2", "m1", "m2", "m1"],
        "time":      ["5mo", "5mo", "10mo", "16mo", "16mo"],
    })

    def test_transitions_pair_each_mouse_across_the_configured_timepoints(self):
        from alleleflux.scripts.analysis.ani.pairwise_ani import enumerate_pairs
        pairs = enumerate_pairs(self.ROSTER, "transitions", [("5mo", "10mo"), ("5mo", "16mo")])
        # m1: 5mo->10mo (S1,S3) and 5mo->16mo (S1,S5); m2 only has 5mo->16mo (S2,S4).
        # No later-vs-later pair (S3,S5), which within_subject WOULD emit.
        self.assertEqual(pairs, [("S1", "S3"), ("S1", "S5"), ("S2", "S4")])

    def test_two_samples_for_one_mouse_at_one_timepoint_is_an_error(self):
        from alleleflux.scripts.analysis.ani.pairwise_ani import enumerate_pairs
        roster = pd.concat([self.ROSTER, pd.DataFrame(
            {"sample_id": ["S6"], "subjectID": ["m1"], "time": ["5mo"]})])
        with self.assertRaisesRegex(ValueError, "m1.*5mo"):
            enumerate_pairs(roster, "transitions", [("5mo", "10mo")])
