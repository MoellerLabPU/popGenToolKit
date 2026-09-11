#!/usr/bin/env python3
"""
Comprehensive unit tests for profile_mags.py.

Tests use mocking for pysam objects to avoid dependency on actual BAM/FASTA files.
Updated for the optimized profile_mags.py (removed mapq_scores, tuple-based output,
samtools mpileup parser, pre-computed gene trees).
"""

import os
import tempfile
import unittest
from collections import defaultdict

import pandas as pd
from intervaltree import IntervalTree

from alleleflux.scripts.analysis.profile_mags import (
    _parse_mpileup_bases,
    build_mag_gene_trees,
    extract_contigID,
    lookup_gene,
    map_genes,
    process_contig_mpileup,
)


class TestExtractContigID(unittest.TestCase):
    """Tests for extract_contigID function."""

    def test_standard_gene_id(self):
        gene_id = "SLG1007_DASTool_bins_35.fa_k141_82760_1"
        expected = "SLG1007_DASTool_bins_35.fa_k141_82760"
        self.assertEqual(extract_contigID(gene_id), expected)

    def test_simple_gene_id(self):
        gene_id = "contig_1_gene_5"
        expected = "contig_1_gene"
        self.assertEqual(extract_contigID(gene_id), expected)

    def test_single_underscore(self):
        gene_id = "contig_1"
        expected = "contig"
        self.assertEqual(extract_contigID(gene_id), expected)

    def test_no_underscore(self):
        gene_id = "gene1"
        expected = ""
        self.assertEqual(extract_contigID(gene_id), expected)

    def test_multiple_underscores(self):
        gene_id = "mag_01_contig_123_scaffold_456_gene_789"
        expected = "mag_01_contig_123_scaffold_456_gene"
        self.assertEqual(extract_contigID(gene_id), expected)

    def test_empty_string(self):
        self.assertEqual(extract_contigID(""), "")


class TestMapGenes(unittest.TestCase):
    """Tests for map_genes function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_valid_prodigal_fasta(self):
        fasta_content = """>gene_1 # 100 # 300 # 1 # ID=1_1;partial=00;start_type=ATG
ATGCGTACGTAG
>gene_2 # 400 # 600 # -1 # ID=1_2;partial=00;start_type=ATG
ATGAAACCC
"""
        fasta_path = os.path.join(self.temp_dir, "genes.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)

        genes_df = map_genes(fasta_path)
        self.assertIsInstance(genes_df, pd.DataFrame)
        self.assertEqual(len(genes_df), 2)
        for col in ["gene_id", "start", "end", "contig"]:
            self.assertIn(col, genes_df.columns)

        gene1 = genes_df[genes_df["gene_id"].str.strip() == "gene_1"].iloc[0]
        self.assertEqual(gene1["start"], 99)
        self.assertEqual(gene1["end"], 299)

    def test_malformed_header(self):
        fasta_content = """>gene_1 # 100 # 300
ATGCGT
"""
        fasta_path = os.path.join(self.temp_dir, "malformed.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        with self.assertRaises(ValueError):
            map_genes(fasta_path)

    def test_coordinates_conversion(self):
        fasta_content = """>test_gene # 1 # 3 # 1 # ID=1_1
ATG
"""
        fasta_path = os.path.join(self.temp_dir, "coords.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)
        genes_df = map_genes(fasta_path)
        gene = genes_df.iloc[0]
        self.assertEqual(gene["start"], 0)
        self.assertEqual(gene["end"], 2)

    def test_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            map_genes("/nonexistent/file.fasta")


class TestLookupGene(unittest.TestCase):
    """Tests for the lookup_gene helper."""

    def setUp(self):
        self.genes_df = pd.DataFrame({
            "gene_id": ["gene_1", "gene_2"],
            "contig": ["contig_1", "contig_1"],
            "start": [100, 300], "end": [200, 400],
        })
        mag_mapping = {"contig_1": "MAG_A"}
        self.trees = build_mag_gene_trees(self.genes_df, mag_mapping, mags_to_include=None)["MAG_A"]

    def test_hit(self):
        result = lookup_gene(self.trees, "contig_1", 150)
        self.assertEqual(result, "gene_1")

    def test_miss(self):
        result = lookup_gene(self.trees, "contig_1", 250)
        self.assertIsNone(result)

    def test_unknown_contig(self):
        result = lookup_gene(self.trees, "unknown", 150)
        self.assertIsNone(result)


class TestBuildMagGeneTrees(unittest.TestCase):
    """Tests for build_mag_gene_trees (Change 3)."""

    def test_basic(self):
        genes_df = pd.DataFrame({
            "gene_id": ["g1", "g2", "g3"],
            "contig": ["c1", "c2", "c3"],
            "start": [100, 200, 300], "end": [200, 300, 400],
        })
        mapping = {"c1": "MAG_A", "c2": "MAG_A", "c3": "MAG_B"}
        trees = build_mag_gene_trees(genes_df, mapping, mags_to_include=None)
        self.assertIn("MAG_A", trees)
        self.assertIn("MAG_B", trees)
        self.assertEqual(len(trees["MAG_A"]), 2)  # c1 and c2
        self.assertEqual(len(trees["MAG_B"]), 1)  # c3

    def test_unmapped_contigs_skipped(self):
        genes_df = pd.DataFrame({
            "gene_id": ["g1"], "contig": ["c_unknown"],
            "start": [100], "end": [200],
        })
        trees = build_mag_gene_trees(genes_df, {}, mags_to_include=None)
        self.assertEqual(len(trees), 0)

    def test_mags_to_include_filter(self):
        """Only MAGs in mags_to_include get trees; others are skipped."""
        genes_df = pd.DataFrame({
            "gene_id": ["g1", "g2", "g3"],
            "contig": ["c1", "c2", "c3"],
            "start": [100, 200, 300], "end": [200, 300, 400],
        })
        mapping = {"c1": "MAG_A", "c2": "MAG_A", "c3": "MAG_B"}
        trees = build_mag_gene_trees(genes_df, mapping, mags_to_include={"MAG_A"})
        self.assertIn("MAG_A", trees)
        self.assertNotIn("MAG_B", trees)
        self.assertEqual(len(trees["MAG_A"]), 2)  # c1 and c2 only


class TestParseMpileupBases(unittest.TestCase):
    """Tests for _parse_mpileup_bases (Change 5)."""

    def test_simple_matches(self):
        # . and , mean reference base
        a, c, g, t, n = _parse_mpileup_bases("A", "..,,")
        self.assertEqual((a, c, g, t, n), (4, 0, 0, 0, 0))

    def test_explicit_bases(self):
        a, c, g, t, n = _parse_mpileup_bases("A", "ACGTacgt")
        self.assertEqual((a, c, g, t, n), (2, 2, 2, 2, 0))

    def test_ambiguous_bases(self):
        a, c, g, t, n = _parse_mpileup_bases("A", "NnRr")
        self.assertEqual((a, c, g, t, n), (0, 0, 0, 0, 4))

    def test_read_start(self):
        # ^X means start of read, X is mapping quality char — skip both
        a, c, g, t, n = _parse_mpileup_bases("A", "^~..")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))

    def test_read_end(self):
        a, c, g, t, n = _parse_mpileup_bases("A", "..$")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))

    def test_insertion(self):
        # +2AG means 2-base insertion of AG — skip it
        a, c, g, t, n = _parse_mpileup_bases("A", ".+2AG.")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))

    def test_deletion(self):
        # -3ACG means 3-base deletion — skip it
        a, c, g, t, n = _parse_mpileup_bases("A", ".-3ACG.")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))

    def test_deletion_placeholder(self):
        # * is a deletion placeholder — skip
        a, c, g, t, n = _parse_mpileup_bases("A", ".*.")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))

    def test_empty_string(self):
        a, c, g, t, n = _parse_mpileup_bases("A", "")
        self.assertEqual((a, c, g, t, n), (0, 0, 0, 0, 0))

    def test_mixed_complex(self):
        # ^~. = start(skip) + ref_match(A)
        # C = explicit C
        # +1T = insertion(skip)
        # , = ref_match(G, since ref is G)
        # $ = end(skip)
        a, c, g, t, n = _parse_mpileup_bases("G", "^~.C+1T,$")
        self.assertEqual((a, c, g, t, n), (0, 1, 2, 0, 0))

    def test_ref_skip(self):
        # < and > are reference skips — skip them
        a, c, g, t, n = _parse_mpileup_bases("A", ".<>.")
        self.assertEqual((a, c, g, t, n), (2, 0, 0, 0, 0))







class TestIntegrationScenarios(unittest.TestCase):
    """Integration-style tests combining multiple functions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_full_pipeline_simple(self):
        fasta_content = """>contig_1_gene_1 # 100 # 200 # 1 # ID=1_1
ATGCGT
>contig_1_gene_2 # 300 # 400 # 1 # ID=1_2
ATGAAA
"""
        fasta_path = os.path.join(self.temp_dir, "genes.fasta")
        with open(fasta_path, "w") as f:
            f.write(fasta_content)

        genes_df = map_genes(fasta_path)
        self.assertEqual(len(genes_df), 2)

        mag_mapping = {"contig_1_gene": "MAG_A", "contig_2": "MAG_A"}
        mag_gene_trees = build_mag_gene_trees(genes_df, mag_mapping, mags_to_include=None)
        contig_trees = mag_gene_trees["MAG_A"]

        self.assertIn("contig_1_gene", contig_trees)

        # Test lookup_gene
        gene1 = lookup_gene(contig_trees, "contig_1_gene", 150)
        self.assertEqual(gene1.strip(), "contig_1_gene_1")

        gene_none = lookup_gene(contig_trees, "contig_1_gene", 250)
        self.assertIsNone(gene_none)

        gene2 = lookup_gene(contig_trees, "contig_1_gene", 350)
        self.assertEqual(gene2.strip(), "contig_1_gene_2")


class TestDataFrameConsistency(unittest.TestCase):
    """Test DataFrame column consistency and types."""

    def test_map_genes_output_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">gene_1 # 100 # 200 # 1 # ID=1_1\nATG\n")
            fasta_path = f.name
        try:
            genes_df = map_genes(fasta_path)
            for col in ["gene_id", "start", "end", "contig"]:
                self.assertIn(col, genes_df.columns)
            self.assertTrue(pd.api.types.is_integer_dtype(genes_df["start"]))
            self.assertTrue(pd.api.types.is_integer_dtype(genes_df["end"]))
        finally:
            os.unlink(fasta_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
