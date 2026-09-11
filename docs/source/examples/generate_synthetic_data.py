#!/usr/bin/env python3
"""
Generate Synthetic AlleleFlux Test Data

Produces all input files needed to run the AlleleFlux pipeline WITHOUT real
sequencing data. Instead of starting from BAM files and running the profiling
step, this script writes the profile TSVs directly, simulating what the
profiler would produce.

What gets generated
-------------------
  reference/combined_mags.fasta    — reference genome (random sequences)
  reference/prodigal_genes.fna     — gene coordinates in Prodigal header format
  reference/mag_mapping.tsv        — maps every contig to its MAG
  reference/gtdbtk_taxonomy.tsv    — mock GTDB taxonomy rows
  metadata/sample_metadata.tsv     — 8-row sample sheet
  profiles/{sample}/               — one gzipped TSV per sample × MAG
  significant_sites/               — dummy significant-site table for visualisation
  config_generated.yml             — ready-to-run AlleleFlux config

Key design choice: treatment effect is injected directly into read counts.
This is impossible to do at the BAM level with a generic read simulator (wgsim
treats every sample identically). By writing profiles directly we can control
exactly which positions are "significant" in which samples.

Usage:
    # Regenerate the bundled longitudinal dataset (paired pre/post samples)
    python generate_synthetic_data.py \
        --output_dir example_data_longitudinal \
        --data_type longitudinal --seed 42

    # Regenerate the bundled single-timepoint dataset
    python generate_synthetic_data.py \
        --output_dir example_data_single \
        --data_type single --seed 42

    # Ad-hoc larger dataset for one-off experiments
    python generate_synthetic_data.py \
        --output_dir my_test_data \
        --num_mags 10 --num_samples 20
"""

import argparse
import gzip
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from Bio import SeqIO
from Bio.Data import CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# The four DNA bases in a fixed order.
# The ORDER matters: the treatment effect shifts each position's dominant
# allele one step forward in this cycle (A→T→G→C→A).
BASES = ["A", "T", "G", "C"]

def generate_random_sequence(length: int) -> str:
    """Return a random DNA string of the given length."""
    return "".join(random.choices(BASES, k=length))


def generate_gene_sequence(length: int) -> str:
    """Return a plausible coding sequence: ATG … (non-stop codons) … stop.

    The sequence is padded/trimmed so its length is divisible by 3, starts
    with ATG (met), and ends with a stop codon. Stop codons are explicitly
    excluded from the internal codons so Prodigal-style parsers don't
    truncate the gene early.
    """
    length = (length // 3) * 3   # round down to codon boundary
    if length < 6:
        length = 6

    seq = "ATG"   # mandatory start codon

    # Standard genetic code stop codons (TAA/TAG/TGA), from Biopython's
    # translation table rather than a hardcoded literal.
    stop_codons = CodonTable.standard_dna_table.stop_codons
    stop_set = set(stop_codons)
    while len(seq) < length - 3:          # -3 leaves room for the stop codon
        codon = "".join(random.choices(BASES, k=3))
        if codon not in stop_set:
            seq += codon

    seq += random.choice(stop_codons)   # append stop codon
    return seq


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of an A/T/G/C string.

    Prodigal stores every gene in its own 5'→3' direction; for a strand=-1 gene
    that direction is the reverse strand, so the forward-strand bases occupying
    those coordinates on the contig are the reverse complement of the stored
    gene sequence.
    """
    return str(Seq(seq).reverse_complement())


def create_mag_structure(
    mag_id: str,
    num_contigs: int = 2,
    contig_length_range: Tuple[int, int] = (800, 2000),
    genes_per_contig: int = 3,
) -> Dict:
    """Build the in-memory representation of one MAG.

    Returns a dict:
      {
        "mag_id": "MAG_001",
        "contigs": [
          {
            "contig_id": "MAG_001.fa_contig1",
            "length": 1500,
            "sequence": "ATGC...",
            "genes": [
              {"gene_id": ..., "start": 100, "end": 280, "strand": 1,
               "sequence": "ATG...TAA"},
              ...
            ]
          },
          ...
        ]
      }

    Gene placement uses a rejection-sampling loop: draw a random start, check
    it doesn't overlap any previously placed gene, retry up to 50 times. Genes
    are placed at least 50 bp from either contig end to leave non-genic flanks.

    Contig/gene sequence consistency
    --------------------------------
    The contig sequence is built in TWO stages:
      1. Start with a random A/T/G/C background of the full contig length.
      2. After gene coordinates are chosen, overwrite contig[start:end] with
         the gene's coding sequence — reverse-complemented when strand == -1
         so the forward-strand bases on the contig really ARE the reverse
         complement of the gene sequence stored in prodigal_genes.fna.

    With this in place, contig.sequence, prodigal_genes.fna, and the profile
    ref_base column (derived from contig.sequence in generate_profile_data)
    all agree at gene coordinates — so dN/dS reconstruction's sanity check
    against the Prodigal reference doesn't fire warnings on every variable site.
    """
    mag = {"mag_id": mag_id, "contigs": []}

    for i in range(num_contigs):
        contig_id = f"{mag_id}.fa_contig{i+1}"
        length = random.randint(*contig_length_range)

        # Stage 1: random A/T/G/C background. Intergenic regions stay random;
        # gene regions get overwritten in stage 2 below.
        contig_chars = list(generate_random_sequence(length))

        genes = []
        used_regions = []   # list of (start, end) for overlap checking

        for j in range(genes_per_contig):
            gene_id = f"{contig_id}_gene{j+1}"
            gene_length = random.randint(150, min(400, length // 2))

            for _ in range(50):   # 50 placement attempts per gene
                start = random.randint(50, length - gene_length - 50)
                end = start + gene_length

                # Reject if this region overlaps any already-placed gene.
                # Condition: two intervals [s1,e1] and [s2,e2] do NOT overlap
                # iff e1 < s2 or s1 > e2 — so they DO overlap if neither holds.
                overlap = any(
                    not (end < us or start > ue)
                    for us, ue in used_regions
                )
                if not overlap:
                    strand = random.choice([1, -1])
                    gene_seq = generate_gene_sequence(gene_length)
                    # gene_seq may be 1–2 bp shorter than gene_length because
                    # generate_gene_sequence rounds down to a codon boundary.
                    # Trust the actual embedded length so end stays consistent
                    # with what's physically present in contig.sequence.
                    embed_seq = (
                        reverse_complement(gene_seq) if strand == -1 else gene_seq
                    )
                    embed_len = len(embed_seq)
                    end = start + embed_len

                    # Stage 2: weave the gene's bases into the contig at its
                    # coordinates, so contig.sequence[start:end] reads back as
                    # the (forward-strand) gene sequence.
                    contig_chars[start:end] = list(embed_seq)

                    used_regions.append((start, end))
                    genes.append({
                        "gene_id": gene_id,
                        "start": start,
                        "end": end,
                        "strand": strand,
                        "sequence": gene_seq,
                    })
                    break

        mag["contigs"].append({
            "contig_id": contig_id,
            "length": length,
            "sequence": "".join(contig_chars),
            "genes": sorted(genes, key=lambda x: x["start"]),
        })

    return mag


def write_reference_fasta(mags: List[Dict], output_path: Path) -> None:
    """Write a multi-FASTA with one entry per contig via Biopython's SeqIO.

    Sequences are wrapped at Biopython's default 60 bp/line; wrap width is
    cosmetic — every downstream consumer reads this file with SeqIO.parse.
    """
    records = [
        SeqRecord(Seq(contig["sequence"]), id=contig["contig_id"], description="")
        for mag in mags
        for contig in mag["contigs"]
    ]
    SeqIO.write(records, str(output_path), "fasta")
    logger.info(f"Wrote reference FASTA: {output_path}")


def write_prodigal_genes(mags: List[Dict], output_path: Path) -> None:
    """Write gene predictions in Prodigal FASTA format.

    Prodigal headers look like:
      >gene_id # start # end # strand # ID=...;partial=00;start_type=ATG;...

    AlleleFlux's profile_mags.py parses this format to build a position →
    gene_id lookup table. The gc_cont and other metadata fields are mocked
    but structurally correct.
    """
    records = []
    for mag in mags:
        for contig in mag["contigs"]:
            for gene in contig["genes"]:
                # Prodigal coordinates are 1-based INCLUSIVE; our gene dict
                # stores 0-based half-open [start, end). Convert before writing:
                #   1-based start          = start + 1
                #   1-based inclusive end  = end  (last base is 0-based end-1)
                # Both map_genes (start-1, end-1) and dnds_from_timepoints
                # (forward: pos-(start-1)) decode these as 1-based, so emitting
                # raw 0-based here misaligns forward-strand genes by one base.
                prodigal_start = gene["start"] + 1
                prodigal_end = gene["end"]
                # Biopython writes the header as ">{id} {description}", so the
                # coordinate block goes in description. On read-back,
                # record.description is the full line and splits on '#' into the
                # 5 fields map_genes expects (gene_id, start, end, strand, attrs).
                description = (
                    f"# {prodigal_start} # {prodigal_end} # {gene['strand']} # "
                    f"ID={gene['gene_id']};partial=00;start_type=ATG;"
                    f"rbs_motif=GGAG;gc_cont=0.55"
                )
                records.append(
                    SeqRecord(
                        Seq(gene["sequence"]),
                        id=gene["gene_id"],
                        description=description,
                    )
                )
    SeqIO.write(records, str(output_path), "fasta")
    logger.info(f"Wrote Prodigal genes: {output_path}")


def write_mag_mapping(mags: List[Dict], output_path: Path) -> None:
    """Write the contig → MAG mapping TSV (contig_id, mag_id).

    AlleleFlux uses this to group per-contig pileup data back into per-MAG
    profiles and to route each position to the correct output file.
    """
    with open(output_path, "w") as f:
        f.write("contig_id\tmag_id\n")
        for mag in mags:
            for contig in mag["contigs"]:
                f.write(f"{contig['contig_id']}\t{mag['mag_id']}\n")
    logger.info(f"Wrote MAG mapping: {output_path}")


def write_gtdb_taxonomy(mags: List[Dict], output_path: Path) -> None:
    """Write a GTDB taxonomy table with only the columns AlleleFlux consumes.

    Real GTDB-Tk classify summary.tsv has 19 columns, but the pipeline reads it
    via ``pd.read_csv(..., usecols=["user_genome", "classification"])`` (see
    utilities.read_gtdb) — it selects those two BY NAME and ignores everything
    else. So we emit just those two columns; the other 17 are dead weight here.

    The classification value must be a ';'-separated lineage with the standard
    d__/p__/c__/o__/f__/g__/s__ rank prefixes; parse_classification splits on
    those to build the per-rank columns used by taxon-level score aggregation.

    The lineages are deliberately arranged so they OVERLAP at lower ranks —
    otherwise every taxon-level group would have size 1 and the aggregation step
    (taxa_score_levels: phylum, genus) would have nothing to pool. Cycling order
    (used via taxonomies[i % len]):
      tax0, tax1  → same phylum (Bacteroidota) AND genus (Bacteroides),
                    differing only at species. Default num_mags=2 hits exactly
                    these two, giving an immediate 2-member phylum+genus group.
      tax2, tax3  → same phylum (Bacillota) but different genus, so at
                    num_mags=4 the phylum level has two 2-member groups while
                    genus has one 2-member group plus singletons.
    """
    taxonomies = [
        # --- Bacteroidota / Bacteroides : overlap down to genus --------------
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;"
        "f__Bacteroidaceae;g__Bacteroides;s__Bacteroides vulgatus",
        "d__Bacteria;p__Bacteroidota;c__Bacteroidia;o__Bacteroidales;"
        "f__Bacteroidaceae;g__Bacteroides;s__Bacteroides fragilis",
        # --- Bacillota : shared phylum, divergent genus ----------------------
        "d__Bacteria;p__Bacillota;c__Clostridia;o__Lachnospirales;"
        "f__Lachnospiraceae;g__Lachnoclostridium;s__Lachnoclostridium sp001",
        "d__Bacteria;p__Bacillota;c__Clostridia;o__Oscillospirales;"
        "f__Ruminococcaceae;g__Faecalibacterium;s__Faecalibacterium prausnitzii",
    ]
    with open(output_path, "w") as f:
        f.write("user_genome\tclassification\n")
        for i, mag in enumerate(mags):
            tax = taxonomies[i % len(taxonomies)]   # cycle through the 4 lineages
            f.write(f"{mag['mag_id']}\t{tax}\n")
    logger.info(f"Wrote GTDB taxonomy: {output_path}")


def generate_sample_structure(
    num_samples: int,
    groups: List[str] = ["control", "treatment"],
    timepoints: List[str] = ["pre", "post"],
    data_type: str = "longitudinal",
) -> List[Dict]:
    """Build the list of sample metadata dicts.

    For a longitudinal design with num_samples=8, groups=["control","treatment"],
    timepoints=["pre","post"], the function produces exactly 8 rows:

      control_subj1_pre   / subj1 / control / A / pre
      control_subj1_post  / subj1 / control / A / post
      control_subj2_pre   / subj2 / control / B / pre
      control_subj2_post  / subj2 / control / B / post
      treatment_subj3_pre / subj3 / treatment / A / pre
      treatment_subj3_post/ subj3 / treatment / A / post
      treatment_subj4_pre / subj4 / treatment / B / pre
      treatment_subj4_post/ subj4 / treatment / B / post

    subject_id increments globally (not per-group) so each subject has a unique
    ID across groups. The replicate letter is assigned per subject within a
    group (A for the first subject in each group, B for the second, etc.) and
    is used by the CMH and paired tests to match subjects across timepoints.
    """
    samples = []

    if data_type == "longitudinal":
        # How many subjects per group?
        # e.g. 8 samples / (2 groups × 2 timepoints) = 2 subjects/group
        samples_per_group = num_samples // (len(groups) * len(timepoints))
        if samples_per_group < 1:
            samples_per_group = 1

        subject_id = 1
        replicate_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for group in groups:
            for rep_idx in range(samples_per_group):
                replicate = replicate_letters[rep_idx % len(replicate_letters)]
                # Both timepoints for this subject are emitted together so they
                # share the same subjectID, which is what the paired tests require.
                for timepoint in timepoints:
                    samples.append({
                        "sample_id": f"{group}_subj{subject_id}_{timepoint}",
                        "subjectID": f"subj{subject_id}",
                        "group": group,
                        "replicate": replicate,
                        "time": timepoint,
                    })
                subject_id += 1

    else:
        # Single-timepoint design: one row per sample, no pairing needed.
        samples_per_group = num_samples // len(groups)
        subject_id = 1
        replicate_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        for group in groups:
            for rep_idx in range(samples_per_group):
                replicate = replicate_letters[rep_idx % len(replicate_letters)]
                samples.append({
                    "sample_id": f"{group}_sample{subject_id}",
                    "subjectID": f"subj{subject_id}",
                    "group": group,
                    "replicate": replicate,
                    "time": timepoints[0] if timepoints else "t1",
                })
                subject_id += 1

    return samples


def write_metadata(
    samples: List[Dict],
    output_path: Path,
    *,
    path_resolver: Callable[[Dict], Path],
) -> None:
    """Write the sample metadata TSV that AlleleFlux reads at startup.

    The bam_path column holds the path AlleleFlux uses to locate each sample's
    data. Its meaning depends on the run mode:
      - "profile" mode: bam_path points at the sample's profile *directory*.
        When profiles_path is set in config.yml, AlleleFlux uses these as
        pointers to pre-generated profiles and skips profiling.
      - "BAM" mode: bam_path points at a sample's .bam *file* and AlleleFlux
        runs the profiling step.

    Rather than baking either convention into this function, the caller passes
    a `path_resolver(sample) -> Path` callable that yields the right path. Two
    typical call sites — both passing **relative** paths anchored at the
    dataset's example_dir (so they resolve against the pipeline's cwd, which
    by convention is set to example_dir):

        write_metadata(..., path_resolver=lambda s: Path("profiles") / s["sample_id"])
        write_metadata(..., path_resolver=lambda s: Path("bams") / f"{s['sample_id']}.bam")

    Paths are written AS-GIVEN (no ``.resolve()``).  The README + the
    regression test both ``cd`` into example_dir before running the
    pipeline, so relative-to-example_dir paths resolve correctly.  This makes
    the bundled dataset portable across machines (CI runners, contributor
    laptops, HPC clusters) without re-running the generator — which the
    earlier absolute-path design forbade.
    """
    with open(output_path, "w") as f:
        f.write("sample_id\tbam_path\tsubjectID\tgroup\treplicate\ttime\n")
        for sample in samples:
            bam_path = path_resolver(sample)
            f.write(
                f"{sample['sample_id']}\t{bam_path}\t{sample['subjectID']}\t"
                f"{sample['group']}\t{sample['replicate']}\t{sample['time']}\n"
            )
    logger.info(f"Wrote metadata: {output_path}")


def generate_profile_data(
    sample: Dict,
    mags: List[Dict],
    coverage_range: Tuple[int, int] = (20, 80),
    treatment_effect_rate: float = 0.08,
) -> Dict[str, List[str]]:
    """Simulate the per-position allele count table for one sample.

    This is the core function. It mimics what alleleflux-profile produces from
    a BAM file, but without any reads or alignment. For each genomic position
    it directly constructs the counts (A, C, G, T) that a pileup would return.

    Reference base assignment
    -------------------------
    The reference base at each position is read directly from the contig's
    forward-strand sequence (see create_mag_structure): intergenic positions
    carry the random background; positions inside a gene carry the embedded
    gene-derived bases (reverse-complemented where strand == -1).

    Because contig.sequence is built once and reused across every sample, the
    same position always gets the same ref_base across all samples — which is
    what AlleleFlux expects (the reference genome doesn't change between
    samples). This also keeps ref_base consistent with prodigal_genes.fna, so
    dN/dS reconstruction does not see a mismatch warning at every variable site.

    Allele count model
    ------------------
    For each position, one allele is designated "dominant". It receives
    85–98% of the coverage; the remaining 2–15% reads are distributed randomly
    among the other three bases (capped at 3 reads each to simulate low-level
    sequencing noise rather than real polymorphism).

    Treatment effect — the allele shift
    ------------------------------------
    ONLY in samples where group == "treatment" AND time == "post" does the
    dominant allele have a chance of shifting.

    At each position, a Bernoulli draw with p = treatment_effect_rate (default
    0.08, i.e. 8%) decides whether the shift happens:

      if is_treatment and is_post and random.random() < 0.08:
          ref_idx = BASES.index(ref_base)
          dominant_base = BASES[(ref_idx + 1) % 4]   # rotate one step forward

    The rotation mapping (in the BASES = [A, T, G, C] cycle):
      ref A → dominant becomes T
      ref T → dominant becomes G
      ref G → dominant becomes C
      ref C → dominant becomes A  (wraps around)

    Concrete example — position with ref_base = A, coverage = 50:
      All other samples:               A=44  T=2  G=1  C=0   (A frequency ≈ 0.88)
      treatment_subj3_post (shifted):  T=46  A=2  G=1  C=0   (A frequency ≈ 0.04)

    AlleleFlux computes Δ = post_freq − pre_freq per subject, then tests
    whether those deltas differ between groups. At this position, the treatment
    group has Δ_A ≈ 0.04 − 0.88 = −0.84, while the control group has
    Δ_A ≈ 0.88 − 0.88 = 0 → a large, detectable difference.

    The shift is applied independently and randomly at each position, so ~8%
    of all positions in treatment post-timepoint samples carry this signal.

    Returns
    -------
    dict mapping mag_id → list of tab-separated row strings (one per position)
    """
    is_treatment = sample["group"] == "treatment"
    is_post = sample["time"] == "post"

    profiles = {}

    for mag in mags:
        rows = []

        for contig in mag["contigs"]:
            contig_seq = contig["sequence"]
            # Pre-build a flat list of (start, end, gene_id) for O(n) gene lookup.
            # end is the half-open upper bound: gene occupies [start, end).
            gene_ranges = [
                (gene["start"], gene["end"], gene["gene_id"])
                for gene in contig["genes"]
            ]

            for pos in range(contig["length"]):
                coverage = random.randint(*coverage_range)

                # Reference base is whatever was baked into the contig sequence
                # at this position by create_mag_structure (random background
                # for intergenic, embedded gene bases for genic positions).
                ref_base = contig_seq[pos]
                dominant_base = ref_base

                # ---- Treatment effect ----------------------------------------
                # Independently for each position, flip a biased coin. If it
                # lands heads (prob = treatment_effect_rate), rotate the dominant
                # allele one step forward in the BASES cycle relative to ref_base.
                # This only fires for treatment post-timepoint samples.
                #
                # Why `< treatment_effect_rate`:
                #   random.random() returns a uniform float in [0.0, 1.0). The
                #   fraction of that range below `p` is exactly `p`, so the
                #   probability of `random.random() < p` being True is `p`.
                #   With p = 0.08, ~8% of positions in treatment-post samples
                #   get the shift; ~92% pass through unchanged. The threshold
                #   IS the probability — never use `>` here, that's 1 - p.
                #
                # Why `(ref_idx + 1) % 4`:
                #   BASES has 4 elements (indices 0..3). The `+1` walks one step
                #   forward in the cycle; `% 4` wraps C (index 3) → A (index 0)
                #   instead of crashing with IndexError on `BASES[4]`. This
                #   is the standard "circular index" idiom.
                if is_treatment and is_post and random.random() < treatment_effect_rate:
                    ref_idx = BASES.index(ref_base)
                    dominant_base = BASES[(ref_idx + 1) % 4]
                # --------------------------------------------------------------

                counts = {"A": 0, "T": 0, "G": 0, "C": 0}

                # The dominant allele captures the vast majority of reads.
                # random.uniform(0.85, 0.98) is a uniform float draw on [0.85, 0.98]
                # — every value in that band is equally likely. The band is chosen
                # to land below typical variant-caller minor-allele thresholds
                # (so positions look monomorphic-with-noise, not bi-allelic) but
                # above ~99% so a sliver of reads remains for realistic background.
                dominant_count = int(coverage * random.uniform(0.85, 0.98))
                counts[dominant_base] = dominant_count
                remaining = coverage - dominant_count

                # Distribute leftover reads among the three non-dominant bases.
                # min(remaining, 3) is a TWO-JOB ceiling:
                #   - the `3` is a biological cap — keeps each minor allele count
                #     below the typical 3-read polymorphism threshold so they
                #     read as sequencer noise, not a real variant.
                #   - the `remaining` is arithmetic safety — never roll more
                #     than the unspent budget. Whichever is smaller wins.
                other_bases = [b for b in BASES if b != dominant_base]
                for b in other_bases:
                    if remaining > 0:
                        c = random.randint(0, min(remaining, 3))
                        counts[b] = c
                        remaining -= c

                # NOTE: total_coverage may be < `coverage`. Each minor roll can
                # land on 0, so unspent budget can leak; in the worst case all
                # three minor bases roll 0 and `coverage - dominant_count` reads
                # are "lost". The profile remains internally consistent because
                # we compute total_coverage from the counts (not from the input
                # `coverage`), but the realized depth can be a few reads below
                # what was drawn. Tolerated as realistic noise; fix by adding
                # `counts[dominant_base] += remaining` before the sum if exact
                # depth preservation is needed.
                total_coverage = counts["A"] + counts["C"] + counts["G"] + counts["T"]
                n_count = 0   # synthetic reads have no ambiguous bases

                # Find which gene (if any) this position falls inside.
                # Half-open: a position equal to end is just past the gene's
                # last base, which lives at end - 1.
                gene_id = ""
                for start, end, gid in gene_ranges:
                    if start <= pos < end:
                        gene_id = gid
                        break

                rows.append(
                    f"{contig['contig_id']}\t{pos}\t{ref_base}\t{total_coverage}"
                    f"\t{counts['A']}\t{counts['C']}\t{counts['G']}\t{counts['T']}"
                    f"\t{n_count}\t{gene_id}"
                )

        profiles[mag["mag_id"]] = rows

    return profiles


def write_profiles(
    sample: Dict, profiles: Dict[str, List[str]], output_dir: Path
) -> None:
    """Write one gzipped TSV per MAG for this sample.

    Output path: {output_dir}/{sample_id}/{sample_id}_{mag_id}_profiled.tsv.gz

    The column order matches what alleleflux-profile produces from a real BAM
    so downstream rules can read either source without modification.
    """
    sample_dir = output_dir / sample["sample_id"]
    sample_dir.mkdir(parents=True, exist_ok=True)

    for mag_id, rows in profiles.items():
        filepath = sample_dir / f"{sample['sample_id']}_{mag_id}_profiled.tsv.gz"
        with gzip.open(filepath, "wt") as f:
            f.write("contig\tposition\tref_base\ttotal_coverage\tA\tC\tG\tT\tN\tgene_id\n")
            for row in rows:
                f.write(row + "\n")


def generate_significant_sites(
    mags: List[Dict], output_path: Path, num_sites_per_gene: int = 5
) -> None:
    """Write a mock significant-sites table for visualisation testing.

    These sites are randomly sampled from gene bodies with fabricated p-values.
    They are NOT derived from running statistical tests — they exist solely so
    that visualisation tools (e.g. alleleflux-terminal-nuc-analysis) have
    something to display without requiring a full pipeline run first.
    """
    with open(output_path, "w") as f:
        f.write("mag_id\tcontig\tposition\tgene_id\ttest_type\tmin_p_value\tq_value\n")
        for mag in mags:
            for contig in mag["contigs"]:
                for gene in contig["genes"]:
                    n = min(num_sites_per_gene, gene["end"] - gene["start"])
                    positions = sorted(random.sample(range(gene["start"], gene["end"]), n))
                    for pos in positions:
                        p_val = random.uniform(1e-7, 1e-4)
                        q_val = p_val * random.uniform(5, 20)
                        f.write(
                            f"{mag['mag_id']}\t{contig['contig_id']}\t{pos}\t"
                            f"{gene['gene_id']}\ttwo_sample_paired_tTest\t"
                            f"{p_val:.2e}\t{q_val:.2e}\n"
                        )
    logger.info(f"Wrote significant sites: {output_path}")


def write_config(
    output_dir: Path,
    data_type: str,
    groups: List[str],
    timepoints: List[str],
    *,
    mode: str = "profiles",
) -> Path:
    """Write a ready-to-run AlleleFlux config pointing at the generated data.

    The two run modes differ in just five values; the rest of the YAML is
    identical, so they share one template here instead of duplicating the
    full ~70 lines twice.

    Parameters
    ----------
    mode : "profiles" or "bams"
        - "profiles" (default): config references the pre-generated profiles in
          <output_dir>/profiles/ via profiles_path. Profiling is skipped.
          Filename: config_generated_{data_type}.yml.
        - "bams": config omits profiles_path so the pipeline runs profiling
          from BAMs at <output_dir>/bams/<sample>.bam, with min_base_quality
          lowered to 0 (wgsim writes Phred ~23). metadata_path points at
          sample_metadata_bams.tsv. Output lands in output_bams/.
          Filename: config_with_bams_{data_type}.yml.

    dn_ds_test_type is chosen to match the experimental design:
      - longitudinal → "two_sample_paired_tTest" (pre/post pairing on the same
        subject is the natural unit of comparison).
      - single       → "two_sample_unpaired_tTest" (no pre/post pairing exists
        at a single timepoint). The pipeline silently skips dN/dS in single
        mode regardless (see dynamic_targets.smk), so this is informational,
        but the wrong value would confuse a reader inspecting the config.

    Filename reflects the data_type so single-mode and longitudinal-mode runs
    in adjacent dirs don't get confused, and a user inspecting a folder can
    tell at a glance which design the config targets.
    """
    if mode not in {"profiles", "bams"}:
        raise ValueError(f"mode must be 'profiles' or 'bams', got {mode!r}")

    if data_type == "single":
        dnds_test_type = "two_sample_unpaired_tTest"
    else:
        dnds_test_type = "two_sample_paired_tTest"

    # ---- mode-specific overrides --------------------------------------------
    # Everything else in the YAML is shared. Tweak just these five values plus
    # the filename and header comment between the two modes.
    if mode == "profiles":
        header = (
            "# AlleleFlux Configuration (Generated)\n"
            "# =====================================\n"
            "# Paths are RELATIVE to this config's directory (example_data_<dt>/).\n"
            "# Run the pipeline from inside example_data_<dt>/ so they resolve correctly:\n"
            "#   cd docs/source/examples/example_data_" + data_type + "\n"
            "#   alleleflux run --config " + f"config_generated_{data_type}.yml\n"
        )
        run_name = "generated_test"
        metadata_filename = "sample_metadata.tsv"
        # profiles_path line is present in profile mode.
        profiles_line = (
            f'  # Use the pre-generated profiles directly — skips the profiling step.\n'
            f'  profiles_path: "profiles"'
        )
        output_subdir = "output"
        # 30 is the default upstream threshold; works for real sequencing data.
        min_base_quality = 30
        bq_comment = ""
        config_filename = f"config_generated_{data_type}.yml"
    else:  # mode == "bams"
        header = (
            "# AlleleFlux Configuration (Generated, BAM mode)\n"
            "# =================================================\n"
            "# Paths are RELATIVE to this config's directory (example_data_<dt>/).\n"
            "# This config runs the FULL pipeline starting from BAM files — the\n"
            "# profiling step is NOT skipped. Generate the BAMs first with:\n"
            "#\n"
            f"#   bash generate_bams.sh --data-dir example_data_{data_type}\n"
            "#\n"
            "# Then run the pipeline from inside example_data_<dt>/:\n"
            f"#   cd docs/source/examples/example_data_{data_type}\n"
            f"#   alleleflux run --config config_with_bams_{data_type}.yml\n"
        )
        run_name = "generated_test_bams"
        metadata_filename = "sample_metadata_bams.tsv"
        # profiles_path is intentionally absent — pipeline profiles from BAMs.
        profiles_line = (
            "  # profiles_path intentionally omitted — pipeline profiles directly from BAMs."
        )
        output_subdir = "output_bams"
        # wgsim writes Phred ~23 regardless of error rate; raise to ≥20 for real data.
        min_base_quality = 0
        bq_comment = "  # wgsim writes Phred ~23 to every base; raise to ≥20 for real data.\n"
        config_filename = f"config_with_bams_{data_type}.yml"

    config = f"""{header}
run_name: "{run_name}"

input:
  fasta_path: "reference/combined_mags.fasta"
  prodigal_path: "reference/prodigal_genes.fna"
  metadata_path: "metadata/{metadata_filename}"
  gtdb_path: "reference/gtdbtk_taxonomy.tsv"
  mag_mapping_path: "reference/mag_mapping.tsv"
{profiles_line}

output:
  root_dir: "./{output_subdir}"

log_level: "INFO"

analysis:
  data_type: "{data_type}"
  allele_analysis_only: False
  # LMM and CMH require more replicates than the small example provides.
  # Enable them when you generate larger datasets.
  use_lmm: False
  use_significance_tests: True
  use_cmh: False
  # Regional contrast requires min_replicates per group (default 4). The small
  # synthetic dataset has only 2 subjects per group by default, so this is
  # off out of the box — enable it after generating a larger dataset
  # (e.g. --num_samples 20 in longitudinal mode → 5 subjects per group).
  use_regional_contrast: False
  use_gene_scores: False
  use_outlier_detection: False

  taxa_score_levels:
    - phylum
    - genus

  timepoints_combinations:
    - timepoint: {timepoints}
      focus: "{timepoints[-1] if len(timepoints) > 1 else timepoints[0]}"

  groups_combinations:
    - treatment: "{groups[1]}"
      control: "{groups[0]}"

quality_control:
  min_sample_num: 2
  breadth_threshold: 0.1
  coverage_threshold: 1
  disable_zero_diff_filtering: False

profiling:
  ignore_orphans: True
{bq_comment}  min_base_quality: {min_base_quality}
  min_mapping_quality: 2
  ignore_overlaps: True

statistics:
  filter_type: "t-test"
  preprocess_between_groups: True
  preprocess_within_groups: True
  max_zero_count: 2
  p_value_threshold: 0.05
  fdr_group_by_mag_id: False
  min_positions_after_preprocess: 1

dnds:
  p_value_column: "q_value"
  # dN/dS is only run when data_type == "longitudinal"; ignored for single mode.
  dn_ds_test_type: "{dnds_test_type}"

regional_contrast:
  mode: "both"
  window_size: 500
  agg_method: "median"
  min_informative_sites: 2
  min_informative_fraction: 0.0
  use_fisher: True

resources:
  threads_per_job: 4
  mem_per_job: "4G"
  time: "01:00:00"
  retries: 1
  mem_step: "2G"
  time_step: "0:30:00"
"""
    config_path = output_dir / config_filename
    with open(config_path, "w") as f:
        f.write(config)
    logger.info(f"Wrote config ({mode} mode): {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic AlleleFlux test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output directory for generated data")
    parser.add_argument("--num_mags", "-m", type=int, default=2,
                        help="Number of MAGs to generate")
    parser.add_argument("--num_samples", "-s", type=int, default=8,
                        help="Number of samples (must be divisible by groups × timepoints)")
    parser.add_argument("--num_contigs_per_mag", type=int, default=2,
                        help="Number of contigs per MAG")
    parser.add_argument("--genes_per_contig", type=int, default=3,
                        help="Number of genes to place per contig")
    parser.add_argument("--contig_length_min", type=int, default=800,
                        help="Minimum contig length (bp)")
    parser.add_argument("--contig_length_max", type=int, default=2000,
                        help="Maximum contig length (bp)")
    parser.add_argument("--data_type", type=str, choices=["single", "longitudinal"],
                        default="longitudinal",
                        help="Experimental design: single timepoint or longitudinal")
    parser.add_argument("--groups", type=str, nargs="+",
                        default=["control", "treatment"],
                        help="Experimental group names")
    parser.add_argument("--timepoints", type=str, nargs="+",
                        default=None,
                        help="Timepoint labels. Auto-defaults based on --data_type if omitted: "
                             "'pre post' for longitudinal, 't1' for single.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed — use the same seed to reproduce identical data")
    parser.add_argument("--coverage_min", type=int, default=20,
                        help="Minimum per-position coverage")
    parser.add_argument("--coverage_max", type=int, default=80,
                        help="Maximum per-position coverage")

    args = parser.parse_args()

    # ---- Resolve & validate timepoints against data_type --------------------
    # Single-timepoint designs have no pre/post pairing, so they need exactly
    # one timepoint label. Longitudinal designs need ≥2. If --timepoints was
    # omitted we pick a sensible default based on --data_type; if it was given
    # explicitly we still validate the count matches the design.
    if args.timepoints is None:
        args.timepoints = ["t1"] if args.data_type == "single" else ["pre", "post"]
    if args.data_type == "single" and len(args.timepoints) != 1:
        parser.error(
            f"--data_type single requires exactly 1 timepoint, got "
            f"{len(args.timepoints)}: {args.timepoints}"
        )
    if args.data_type == "longitudinal" and len(args.timepoints) < 2:
        parser.error(
            f"--data_type longitudinal requires ≥2 timepoints, got "
            f"{len(args.timepoints)}: {args.timepoints}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve to absolute so every path embedded in the generated configs and
    # metadata is anchored to a real filesystem location, not the user's cwd at
    # invocation time. Without this, configs like {output_dir}/reference/...
    # would be relative ("my_bam_test/reference/...") and only resolve correctly
    # if the pipeline is later invoked from the same cwd as the generator.
    output_dir = Path(args.output_dir).resolve()
    for subdir in ["reference", "metadata", "profiles", "significant_sites"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating synthetic data in: {output_dir}")
    logger.info(f"{args.num_mags} MAGs, {args.num_samples} samples, {args.data_type} design")

    # --- MAG structures -------------------------------------------------
    mags = []
    for i in range(args.num_mags):
        mag_id = f"MAG_{i+1:03d}"   # zero-padded: MAG_001, MAG_002, ...
        mags.append(create_mag_structure(
            mag_id,
            num_contigs=args.num_contigs_per_mag,
            contig_length_range=(args.contig_length_min, args.contig_length_max),
            genes_per_contig=args.genes_per_contig,
        ))

    # --- Reference files ------------------------------------------------
    write_reference_fasta(mags, output_dir / "reference" / "combined_mags.fasta")
    write_prodigal_genes(mags, output_dir / "reference" / "prodigal_genes.fna")
    write_mag_mapping(mags, output_dir / "reference" / "mag_mapping.tsv")
    write_gtdb_taxonomy(mags, output_dir / "reference" / "gtdbtk_taxonomy.tsv")

    # --- Sample structure -----------------------------------------------
    samples = generate_sample_structure(
        args.num_samples,
        groups=args.groups,
        timepoints=args.timepoints,
        data_type=args.data_type,
    )

    profiles_dir = output_dir / "profiles"
    # Profile-mode metadata: bam_path points at the per-sample profile directory,
    # written RELATIVE to example_dir so the dataset is portable.  Pipeline
    # convention is to cd into example_dir before invoking, so "profiles/<id>"
    # resolves correctly against cwd.
    write_metadata(
        samples,
        output_dir / "metadata" / "sample_metadata.tsv",
        path_resolver=lambda s: Path("profiles") / s["sample_id"],
    )
    # BAM-mode metadata: bam_path points at bams/<sample>.bam, relative to
    # example_dir (same convention as the profile-mode metadata above).  BAMs
    # don't exist yet (user runs generate_bams.sh after this); the file is
    # written eagerly so the BAM-mode config can reference it without a second
    # invocation.  The pipeline validates BAM existence at runtime.
    write_metadata(
        samples,
        output_dir / "metadata" / "sample_metadata_bams.tsv",
        path_resolver=lambda s: Path("bams") / f"{s['sample_id']}.bam",
    )

    # --- Profiles (the slow step for large datasets) --------------------
    logger.info(f"Generating profiles for {len(samples)} samples...")
    for i, sample in enumerate(samples):
        # Throttle progress logs: print every 10th sample plus the final one.
        # `(i + 1) % 10 == 0` fires at 1-indexed milestones 10, 20, 30, ...
        # `i == len(samples) - 1` guarantees a final "N/N" line even when the
        # total isn't a multiple of 10. Without the second clause a 17-sample
        # run would log "10/17" and then go silent.
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            logger.info(f"  {i+1}/{len(samples)}: {sample['sample_id']}")
        profiles = generate_profile_data(
            sample, mags, coverage_range=(args.coverage_min, args.coverage_max)
        )
        write_profiles(sample, profiles, profiles_dir)

    # --- Ancillary files ------------------------------------------------
    generate_significant_sites(
        mags, output_dir / "significant_sites" / "significant_sites.tsv"
    )
    config_path = write_config(
        output_dir, args.data_type, args.groups, args.timepoints, mode="profiles"
    )
    config_bams_path = write_config(
        output_dir, args.data_type, args.groups, args.timepoints, mode="bams"
    )

    # Write a minimal README next to the generated data showing BOTH run paths.
    # Configs use relative paths anchored at example_dir, so the pipeline must
    # be invoked from inside this directory (or with cwd set to it).
    config_basename = config_path.name
    config_bams_basename = config_bams_path.name
    readme = (
        f"# Generated AlleleFlux Test Data\n\n"
        f"Seed: {args.seed} | MAGs: {args.num_mags} | "
        f"Samples: {len(samples)} ({args.data_type})\n\n"
        f"All paths in the configs and metadata are RELATIVE to this directory.\n"
        f"`cd` into this directory before running the pipeline.\n\n"
        f"## Option A — pre-generated profiles (fastest)\n\n"
        f"```bash\n"
        f"cd {output_dir}\n"
        f"alleleflux run --config {config_basename}\n"
        f"```\n\n"
        f"## Option B — full pipeline from BAMs (requires wgsim + bwa)\n\n"
        f"```bash\n"
        f"# 1. Generate BAMs from the reference (one-time)\n"
        f"bash generate_bams.sh --data-dir {output_dir}\n\n"
        f"# 2. Run the pipeline (includes profiling step)\n"
        f"cd {output_dir}\n"
        f"alleleflux run --config {config_bams_basename}\n"
        f"```\n"
    )
    (output_dir / "README.md").write_text(readme)

    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    logger.info(f"Done. Total size: {total_size / 1024:.1f} KB")
    logger.info(
        f"Run with pre-generated profiles: "
        f"cd {output_dir} && alleleflux run --config {config_basename}"
    )
    logger.info(
        f"Or generate BAMs + run full pipeline: "
        f"bash generate_bams.sh --data-dir {output_dir} && "
        f"cd {output_dir} && alleleflux run --config {config_bams_basename}"
    )


if __name__ == "__main__":
    main()
