"""Regional contrast analysis rule.

This module runs per-MAG regional contrast analysis, which detects genes or
sliding windows with differential allele-frequency evolution between treatment
and control groups across paired hosts.

The analysis operates directly on the ``*_allele_frequency_changes_mean.tsv.gz``
file produced by allele analysis (no statistical preprocessing required), and is
therefore only applicable to longitudinal data.

Rules are independent per MAG and run in parallel across the DAG.
"""

import os
import numpy as np
import pandas as pd
from snakemake.logging import logger

localrules: combine_regional_contrast_scores


rule regional_contrast:
    """Detect genomic regions with differential allele-frequency evolution.

    For each MAG, computes Total-Variation distance site scores, aggregates
    them into gene and/or sliding-window regions per host × group, ranks
    scores into genome-wide percentiles, contrasts treatment vs control, and
    tests the contrast across hosts with a Wilcoxon signed-rank test and a
    one-sample t-test (both directional), followed by Benjamini–Hochberg FDR
    correction.

    Runs independently per MAG; Snakemake will parallelise across MAGs.
    """
    input:
        get_allele_analysis_input_path(gr_wildcard="{treatment}_{control}"),
    output:
        per_host=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
            "{mag}_regional_contrast_per_host_region.tsv.gz",
        ),
        summary=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
            "{mag}_regional_contrast_region_summary.tsv",
        ),
        # Per-region-type summaries written by regional_contrast.py so that
        # the scoring rule can take them as direct inputs (no filtering needed).
        per_type_summaries=[
            os.path.join(
                OUTDIR,
                "regional_contrast",
                "regional_contrast_{timepoints}-{treatment}_{control}",
                "{mag}_regional_contrast_" + rt + "_region_summary.tsv",
            )
            for rt in _get_rc_region_types()
        ],
    params:
        output_dir=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
        ),
        prefix="{mag}_regional_contrast",
        mode=config.get("regional_contrast", {}).get("mode", "both"),
        window_size=config.get("regional_contrast", {}).get("window_size", 1000),
        agg_method=config.get("regional_contrast", {}).get("agg_method", "median"),
        trim_fraction=(
            "--trim_fraction "
            + str(config.get("regional_contrast", {}).get("trim_fraction", 0.1))
            if config.get("regional_contrast", {}).get("agg_method", "median")
            == "trimmed_mean"
            else ""
        ),
        min_informative_sites=config.get("regional_contrast", {}).get(
            "min_informative_sites", 5
        ),
        min_informative_fraction=config.get("regional_contrast", {}).get(
            "min_informative_fraction", 0.0
        ),
        min_replicates=config.get("regional_contrast", {}).get("min_replicates", 4),
        use_fisher=(
            "--use_fisher"
            if config.get("regional_contrast", {}).get("use_fisher", True)
            else ""
        ),
    retries: get_retries("regional_contrast")
    resources:
        mem_mb=get_mem_mb("regional_contrast"),
        time=get_time("regional_contrast"),
        runtime=get_runtime("regional_contrast"),
    shell:
        """
        alleleflux-regional-contrast \
            --input {input} \
            --output_dir {params.output_dir} \
            --prefix {params.prefix} \
            --treatment_group {wildcards.treatment} \
            --control_group {wildcards.control} \
            --mode {params.mode} \
            --window_size {params.window_size} \
            --agg_method {params.agg_method} \
            {params.trim_fraction} \
            --min_informative_sites {params.min_informative_sites} \
            --min_informative_fraction {params.min_informative_fraction} \
            --min_replicates {params.min_replicates} \
            {params.use_fisher}
        """


rule regional_contrast_scores:
    """Calculate significance scores for regional contrast results, split by region type.

    Uses the per-region-type summary written directly by ``alleleflux-regional-contrast``
    as input, so no filtering or temporary files are needed.  Gene-level and
    window-level scores use their own independent denominators.
    Output is written under ``regional_contrast/scores/``.
    """
    input:
        region_summary=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
            "{mag}_regional_contrast_{region_type}_region_summary.tsv",
        ),
        gtdb_taxonomy=config["input"]["gtdb_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
    output:
        scores=os.path.join(
            OUTDIR,
            "regional_contrast",
            "scores",
            "regional_contrast_scores_{timepoints}-{treatment}_{control}",
            "{mag}_regional_contrast_{region_type}_scores.tsv",
        ),
    wildcard_constraints:
        region_type="gene|window",
    params:
        p_value_threshold=config["statistics"].get("p_value_threshold", 0.05),
    retries: get_retries("regional_contrast_scores")
    resources:
        mem_mb=get_mem_mb("regional_contrast_scores"),
        time=get_time("regional_contrast_scores"),
        runtime=get_runtime("regional_contrast_scores"),
    shell:
        """
        mkdir -p $(dirname {output.scores})
        alleleflux-scores \
            --gtdb_taxonomy {input.gtdb_taxonomy} \
            --pValue_table {input.region_summary} \
            --group_by_column MAG_ID \
            --pValue_threshold {params.p_value_threshold} \
            --out_fPath {output.scores} \
            --mag_mapping_file {input.mag_mapping}
        """


rule combine_regional_contrast_scores:
    """Combine per-MAG regional contrast scores into a single summary table.

    Concatenates all ``{mag}_regional_contrast_{region_type}_scores.tsv`` files
    for a given timepoint-group combination and region type into one file for
    easy viewing and downstream analysis.  Gene and window scores are kept in
    separate combined files so their denominators remain independent.
    """
    input:
        scores=lambda wc: expand(
            os.path.join(
                OUTDIR,
                "regional_contrast",
                "scores",
                "regional_contrast_scores_{timepoints}-{treatment}_{control}",
                "{mag}_regional_contrast_{region_type}_scores.tsv",
            ),
            timepoints=wc.timepoints,
            treatment=wc.treatment,
            control=wc.control,
            region_type=wc.region_type,
            mag=_get_mags_by_eligibility(
                wc.timepoints, f"{wc.treatment}_{wc.control}", eligibility_type="all"
            ),
        ),
    output:
        combined=os.path.join(
            OUTDIR,
            "regional_contrast",
            "scores",
            "regional_contrast_scores_{timepoints}-{treatment}_{control}",
            "combined_regional_contrast_{region_type}_scores.tsv",
        ),
    wildcard_constraints:
        region_type="gene|window",
    retries: get_retries("combine_regional_contrast_scores")
    resources:
        mem_mb=get_mem_mb("combine_regional_contrast_scores"),
        time=get_time("combine_regional_contrast_scores"),
        runtime=get_runtime("combine_regional_contrast_scores"),
    run:
        dfs = []
        for file in input.scores:
            df = pd.read_csv(file, sep="\t")
            if not df.empty:
                dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        combined_df.to_csv(output.combined, sep="\t", index=False)
        logger.info(
            f"Combined {len(dfs)} MAG {wildcards.region_type} score files "
            f"into {output.combined}"
        )


rule regional_contrast_summary:
    """Aggregate per-MAG regional contrast summaries and apply cross-MAG BH-FDR.

    Collects all per-MAG ``{mag}_regional_contrast_{region_type}_region_summary.tsv``
    files produced by the ``regional_contrast`` rule for a given timepoint-group
    combination and region type, concatenates them into a single cross-MAG table,
    and applies a second round of Benjamini–Hochberg FDR correction across all MAGs.
    The per-MAG BH correction already written by ``alleleflux-regional-contrast``
    is preserved alongside the new cross-MAG q-values (``q_value_*_global`` columns).

    Gene-region and window-region summaries are run as separate Snakemake jobs
    so their denominators remain independent (``region_type`` wildcard).

    Output is written to the same directory as the per-MAG input files so that
    all results for a timepoint-group combination stay co-located.
    """
    input:
        summaries=lambda wc: expand(
            os.path.join(
                OUTDIR,
                "regional_contrast",
                "regional_contrast_{timepoints}-{treatment}_{control}",
                "{mag}_regional_contrast_{region_type}_region_summary.tsv",
            ),
            timepoints=wc.timepoints,
            treatment=wc.treatment,
            control=wc.control,
            region_type=wc.region_type,
            mag=_get_mags_by_eligibility(
                wc.timepoints, f"{wc.treatment}_{wc.control}", eligibility_type="all"
            ),
        ),
    output:
        os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
            "region_contrast_summary_{region_type}_region_summary.tsv",
        ),
    wildcard_constraints:
        region_type="gene|window",
    params:
        input_dir=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
        ),
        outdir=os.path.join(
            OUTDIR,
            "regional_contrast",
            "regional_contrast_{timepoints}-{treatment}_{control}",
        ),
        prefix="region_contrast_summary",
    retries: get_retries("regional_contrast_summary")
    resources:
        mem_mb=get_mem_mb("regional_contrast_summary"),
        time=get_time("regional_contrast_summary"),
        runtime=get_runtime("regional_contrast_summary"),
    shell:
        """
        alleleflux-regional-contrast-summary \
            --input-dir {params.input_dir} \
            --outdir {params.outdir} \
            --prefix {params.prefix} \
            --region-types {wildcards.region_type}
        """

