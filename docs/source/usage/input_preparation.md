# Input Preparation

AlleleFlux requires several input files. This guide covers preparation and formatting.

## Required Files

| File | Format | Description |
|------|--------|-------------|
| **BAM files** | `.bam` + `.bai` | Sorted and indexed alignments of metagenomic reads to MAGs |
| **Reference FASTA** | `.fa` / `.fasta` | Combined MAG contigs. Header format: `<MAG_ID>.fa_<contig_ID>` |
| **Prodigal genes** | `.fna` | Nucleotide ORF predictions matching reference contig IDs |
| **Metadata TSV** | `.tsv` | Sample information with `sample_id`, `bam_path`, `subjectID`, `group`, `replicate`. For longitudinal: add `time` |
| **MAG mapping** | `.tsv` | Contig → MAG assignments (`contig_name`, `mag_id`) |
| **GTDB taxonomy** | `.tsv` (optional) | `gtdbtk.bac120.summary.tsv` for taxonomic aggregation |

## Metadata File Format

**Longitudinal Study:**

```text
sample_id  bam_path                  subjectID  group      replicate  time
S1         /data/S1.sorted.bam       mouse1     control    A          pre
S2         /data/S2.sorted.bam       mouse2     control    B          pre
S3         /data/S3.sorted.bam       mouse3     treatment  A          pre
S4         /data/S4.sorted.bam       mouse4     treatment  B          pre
S5         /data/S5.sorted.bam       mouse1     control    A          post
S6         /data/S6.sorted.bam       mouse2     control    B          post
S7         /data/S7.sorted.bam       mouse3     treatment  A          post
S8         /data/S8.sorted.bam       mouse4     treatment  B          post
```

**Single Timepoint:**

```text
sample_id  bam_path                  subjectID  group    replicate
S1         /data/S1.sorted.bam       subject1   disease  A
S2         /data/S2.sorted.bam       subject2   disease  B
S3         /data/S3.sorted.bam       subject3   healthy  A
S4         /data/S4.sorted.bam       subject4   healthy  B
```

## Minimal Configuration

Create `config.yml` with paths to your files:

```yaml
data_type: "longitudinal"  # or "single"

input:
  fasta_path: "reference.fa"
  prodigal_path: "genes.fna"
  metadata_path: "metadata.tsv"
  mag_mapping_path: "mag_mapping.tsv"
  gtdb_path: "gtdbtk.tsv"  # optional

output:
  root_dir: "output/"

analysis:
  timepoints_combinations:
    - timepoint: ["pre", "post"]
      focus: "post"
  groups_combinations:
    - ["treatment", "control"]
  use_lmm: true
  use_significance_tests: true
  use_cmh: true
```

See [Configuration Reference](../reference/configuration.md) for all options.

## Preparation Utilities

**Create MAG mapping** (combines individual MAG FASTAs):

```bash
alleleflux-create-mag-mapping --dir mag_fastas/ --extension fa \
    --output-fasta combined.fasta --output-mapping mapping.tsv
```

**Add BAM paths** to existing metadata:

```bash
alleleflux-add-bam-path --metadata metadata.tsv \
    --bam-dir bamfiles/ --output metadata_with_bam.tsv
```

**Generate Prodigal predictions**:

```bash
prodigal -i combined.fasta -d genes.fna -a genes.faa -p meta
```

For detailed options: `alleleflux-create-mag-mapping --help`

## Next Steps

Once inputs are prepared:

1. Create configuration file: [Configuration Reference](../reference/configuration.md)
2. Run the pipeline: [Running the Workflow](running_workflow.md)
3. Examine example data: [Example Data](../examples/example_data_longitudinal/README.md)
