# Filters and rules in the pairwise ANI comparison

Every gate, threshold and decision rule applied by `alleleflux-pairwise-ani`,
and which of the two metrics each one touches. (Verified against the
implementation 2026-09-01; the worked example is the real pair SLG1007 vs
SLG1102 on MAG `SLG1007_DASTool_bins_35`.)

## The two metrics in one line each

```
conANI = (compared − consensus SNPs)  / compared     "do the majorities match?"
popANI = (compared − population SNPs) / compared     "is ANY credible allele shared?"
```

A **consensus SNP** = the two samples' consensus sets are disjoint.
A **population SNP** = the two samples' *credible-allele* sets are disjoint
(a fixed difference). Population ⊆ consensus by construction, so
**popANI ≥ conANI always**.

## Filter-by-filter

| # | Filter / rule | conANI | popANI | What it does |
|---|--------------|:------:|:------:|--------------|
| 1 | **Sample QC gate** (`coverage_threshold_passed`) | ✅ | ✅ | Only samples that passed AlleleFlux QC (breadth ≥ threshold at ≥1×, coverage ≥ threshold — from the per-MAG `*_QC.tsv`) enter any comparison. Unioned across all timepoint-combination QC files, de-duplicated per sample. |
| 2 | **`min_cov` position gate** (default **5**, in **both** samples) | ✅ | ✅ | A position is *compared* only with ≥ min_cov reads in both samples; it defines the denominator of both ANIs. Part of the metric, not an option — but exposed as a toggle (`--min_cov`, 1 = off, exploration only) and **stamped into every output row**. This is the **only** per-position filter conANI has. |
| 3 | **Coverage = A+C+G+T (N excluded)** | ✅ | ✅ | N calls are reads that cast no vote; they are structurally absent from every denominator (the profile's `total_coverage`, which counts N, is never used). |
| 4 | **Tie rule: consensus is a *set*** | ✅ | ✅ | Bases tied at the max count are *all* consensus (e.g. 3A+3G → {A,G}). No arbitrary tie-break: a tie vs a sample carrying either tied base is **not** a SNP of any kind. (inStrain breaks ties by column order; on the real pair this rule removes 30 of its 242 consensus SNPs — all certified ties.) |
| 5 | **Null model** (sequencing-error threshold) | ❌ | ✅ | `model[coverage]` = minimum reads before a *minor* allele beats sequencing error: smallest k with `3·P[Binom(C, p/3) ≥ k] < fdr`, `p = 10^(−Q/10)`. Computed from the run's `profiling.min_base_quality` (default Q30) and `--fdr` (default 1e-6); at Q30: 3 reads up to ~30×, 4 at 100×, 7 at 1000×. Never consulted by conANI. |
| 6 | **`min_freq` floor** (default **0.05** = 5 %) | ❌ | ✅ | An allele must also be ≥ 5 % of the position's reads. Complements #5: the error model binds below ~70×, the fraction floor binds above (contamination/index-hopping scale with depth; error does not). Never consulted by conANI. |
| 7 | **Majority allele is always credible** — even when it fails the null model | – | ✅ | The credible set = consensus set ∪ {bases passing #5 AND #6}. A thin majority (`[2,1,1,1]` at 5×, bar = 3) still counts as that sample's allele: it is the best available statement of what is there, and excluding it would let empty credible sets manufacture phantom fixed differences. (Same stance inStrain takes implicitly via its consensus shortcut; the majority always has ≥ 25 % of reads, so #6 can never exclude it either.) conANI: not applicable — its verdict uses the consensus set directly. |
| 8 | **Zero compared → NaN, never 1.0** | ✅ | ✅ | An ANI over no data is undefined, not perfect. Read every ANI alongside `percent_genome_compared` (= compared / genome length). |

## Consistency guards (fail-loud, not filters)

* QC's recorded `genome_size` must equal the reference-derived genome length
  (else QC and this run used different references).
* Each sample's profile-derived breadth must match QC's recorded breadth
  (else the profile changed after QC ran).
* Corrupt inputs raise: duplicate `(contig, position)` rows, positions beyond a
  contig's length, contigs missing from mapping/reference, profiles missing for
  QC-passing samples.

## Defaults at a glance

| knob | default | source |
|---|---|---|
| `min_cov` | 5 | `analysis.pairwise_ani.min_cov` / `--min_cov` |
| `min_freq` | 0.05 | `analysis.pairwise_ani.min_freq` / `--min_freq` |
| `fdr` | 1e-6 | `analysis.pairwise_ani.fdr` / `--fdr` |
| base-quality floor (sets error rate) | 30 | `profiling.min_base_quality` (same key the profiler used) |
| `pairs` | `within_subject` | `analysis.pairwise_ani.pairs` / `--pairs`; `transitions` restricts to the configured `--transitions EARLIER:LATER` comparisons (DRiDO: 970 pairs instead of 2,570) |
| `store_snp_locations` | `within_subject` | (`all` produced 7.7 GB in inStrain — use deliberately) |
| `--cpus` | 1 | worker processes for profile loading (the bottleneck) + threads for comparison; output identical for any value |

Only samples that take part in at least one pair are loaded and appear in the
samples table (with `pairs: within_subject` on DRiDO that skips ~38 % of the
QC-passing roster). Memory = genome length × 4 × 2 bytes per loaded sample, all
held at once — size `mem_per_job` from that.

## Worked position (real data)

`k141_17312:13581` — SLG1007 `[A=2, C=0, G=28, T=0]` at 30×; SLG1102
`[A=5, C=0, G=1, T=0]` at 6×. Gate #2: 30 ≥ 5 and 6 ≥ 5 → compared. Consensus
sets {G} vs {A} disjoint → **consensus SNP** (conANI). Credibility: A in SLG1007
is 2 reads < 3 (#5 kills it despite 6.7 % passing #6); G in SLG1102 is 1 read
< 3. Credible sets {G} vs {A} disjoint → **population SNP** (popANI) — one of
only 29 fixed differences across 1,649,007 compared positions of this pair.
