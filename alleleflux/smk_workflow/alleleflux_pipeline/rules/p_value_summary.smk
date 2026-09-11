"""P-value summary aggregation rules.

This module aggregates p-values from significance tests across all MAGs and
positions, applying FDR correction (if configured) and generating summary
tables for downstream analysis including dN/dS calculations.
"""


def get_all_significance_test_results_for_summary(wildcards):
    """
    Get significance results for a specific test type, used as input for p_value_summary.
    
    Respects preprocessing eligibility by triggering the appropriate checkpoint
    and using get_eligible_mags() instead of direct eligibility lookup.
    """
    targets = []
    test_type = wildcards.test_type
    timepoints = wildcards.timepoints
    groups = wildcards.groups

    # No defensive ``checkpoints.preprocessing_eligibility_*.get()`` call here:
    # ``get_eligible_mags`` -> ``get_mags_by_preprocessing_eligibility`` performs
    # that gating itself via ``.output.out_fPath``.  See shared/common.smk.

    if test_type in ['two_sample_unpaired', 'two_sample_paired', 'lmm', 'cmh']:
        mags = get_eligible_mags(timepoints, groups, test_type)
        for mag in mags:
            targets.append(
                os.path.join(
                    OUTDIR,
                    "significance_tests",
                    f"{test_type}_{timepoints}-{groups}",
                    f"{mag}_{test_type}.tsv.gz",
                )
            )

    elif test_type in ["single_sample",'lmm_across_time', 'cmh_across_time'] and DATA_TYPE == "longitudinal":
        sample_entries = get_eligible_mags(timepoints, groups, test_type)
        for mag, group in sample_entries:
            targets.append(
                os.path.join(
                    OUTDIR,
                    "significance_tests",
                    f"{test_type}_{timepoints}-{groups}",
                    f"{mag}_{test_type}_{group}.tsv.gz",
                )
            )
    
    return targets


rule p_value_summary:
    input:
        get_all_significance_test_results_for_summary
    output:
        os.path.join(
            OUTDIR, 
            "p_value_summary", 
            "{timepoints}-{groups}", 
            "p_value_summary_{test_type}_{timepoints}-{groups}.tsv"
        )
    params:
        output_dir=os.path.join(OUTDIR, "p_value_summary", "{timepoints}-{groups}"),
        input_dir=os.path.join(OUTDIR, "significance_tests"),
        prefix="p_value_summary",
        group_by_mag_id=(
            "--fdr-group-by-mag-id"
            if config["statistics"].get("fdr_group_by_mag_id", False)
            else ""
        )
    retries: get_retries("p_value_summary")
    resources:
        mem_mb=get_mem_mb("p_value_summary"),
        time=get_time("p_value_summary"),
        runtime=get_runtime("p_value_summary"),
    shell:
        """
        alleleflux-p-value-summary \
            --input-dir {params.input_dir} \
            --timepoints {wildcards.timepoints} \
            --groups {wildcards.groups} \
            --test-types {wildcards.test_type} \
            --outdir {params.output_dir} \
            --prefix {params.prefix} \
            {params.group_by_mag_id}
        """


def get_all_p_value_summary_outputs(wildcards):
    """Every p_value_summary table across ALL timepoint/group combinations.

    The full input to the run-once ``significant_sites_summary`` rollup: Snakemake will not
    schedule that rule until every one of these files exists.  Reuses the same target
    generator the top-level output loop uses (``generate_p_value_summary_targets``), so the
    rollup input and the pipeline targets stay in lockstep.
    """
    targets = []
    for tp in timepoints_labels:
        for gr in groups_labels:
            targets.extend(generate_p_value_summary_targets(tp, gr))
    return targets


rule significant_sites_summary:
    """Summarize every p_value_summary table into per-(comparison, test, group, MAG)
    significant-site stats -- the input table for the AlleleFlux score/heatmap notebooks.
    Runs once per run, after all p_value_summary outputs are complete."""
    input:
        get_all_p_value_summary_outputs
    output:
        cell_stats=os.path.join(
            OUTDIR, "p_value_summary", "significant_sites_summary",
            "significant_sites_mag_cell_stats_long.tsv",
        ),
        counts_long=os.path.join(
            OUTDIR, "p_value_summary", "significant_sites_summary",
            "significant_sites_mag_counts_long.tsv",
        ),
        counts_pivot=os.path.join(
            OUTDIR, "p_value_summary", "significant_sites_summary",
            "significant_sites_mag_counts_pivot.tsv",
        ),
        sig_sites=os.path.join(
            OUTDIR, "p_value_summary", "significant_sites_summary",
            "significant_sites_sig_sites.tsv",
        ),
    params:
        input_dir=os.path.join(OUTDIR, "p_value_summary"),
        output_dir=os.path.join(OUTDIR, "p_value_summary", "significant_sites_summary"),
        prefix="significant_sites",
        q_threshold=config["statistics"].get("p_value_threshold", 0.05),
        p_threshold=config["statistics"].get("p_value_threshold", 0.05),
    threads: get_threads("significant_sites_summary")
    retries: get_retries("significant_sites_summary")
    resources:
        mem_mb=get_mem_mb("significant_sites_summary"),
        time=get_time("significant_sites_summary"),
        runtime=get_runtime("significant_sites_summary"),
    shell:
        """
        alleleflux-significant-sites-summary \
            --input-dir {params.input_dir} \
            --outdir {params.output_dir} \
            --q-threshold {params.q_threshold} \
            --p-threshold {params.p_threshold} \
            --cpus {threads} \
            --prefix {params.prefix}
        """
