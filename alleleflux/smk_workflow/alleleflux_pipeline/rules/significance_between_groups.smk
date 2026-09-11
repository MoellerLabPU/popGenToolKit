"""Between-groups significance testing rules.

This module contains rules for statistical tests comparing allele frequencies
between experimental groups:
- Two-sample unpaired tests (t-test, Mann-Whitney U)
- Two-sample paired tests (paired t-test, Wilcoxon signed-rank)
- Linear Mixed Models (LMM) for handling repeated measures
- Cochran-Mantel-Haenszel (CMH) tests for stratified analysis
- Preprocessing rules for filtering positions before testing
"""


def _cache_paths_for_combination_between(wildcards):
    """Resolve the per-timepoint (group-independent) Parquet cache paths for one combination."""
    tps = wildcards.timepoints.split("_") if DATA_TYPE == "longitudinal" else [wildcards.timepoints]
    return [
        get_allele_freq_cache_path(
            mag_wildcard=wildcards.mag,
            timepoint_wildcard=tp,
        )
        for tp in tps
    ]


def get_between_group_inputs(test_type=None):
    """Get the appropriate input file path for between-group statistical tests.

    Uses helper functions from common.smk for centralized path construction.

    Parameters:
        test_type: Optional test type. If "cmh" with longitudinal data,
                   returns the per-timepoint cache files (replaces longitudinal).

    Returns:
        str or callable: Path (or input function returning a list of paths) for the rule input.
    """
    # CMH test needs per-sample/per-timepoint data (not mean changes)
    if test_type == "cmh" and DATA_TYPE == "longitudinal":
        return _cache_paths_for_combination_between

    # Check if preprocessing is enabled
    preprocess_enabled = config["statistics"].get("preprocess_between_groups", False)

    if preprocess_enabled:
        return get_preprocessed_between_groups_path()
    else:
        return get_allele_analysis_input_path()

rule preprocess_between_groups:
    input:
        get_allele_analysis_input_path(),
    output:
        outPath=get_preprocessed_between_groups_path(),
        statusPath=os.path.join(
            OUTDIR,
            "significance_tests",
            "preprocessed_between_groups_{timepoints}-{groups}",
            "{mag}_preprocessing_status.json",
        ),
    params:
        p_value_threshold=config["statistics"].get("p_value_threshold", 0.05),
        filter_type=config["statistics"].get("filter_type", "t-test"),
        data_type=DATA_TYPE,
        min_positions=config["statistics"].get("min_positions_after_preprocess", 1),
        min_sample_num=config["quality_control"]["min_sample_num"],
    threads: get_threads("preprocess_between_groups")
    retries: get_retries("preprocess_between_groups")
    resources:
        mem_mb=get_mem_mb("preprocess_between_groups"),
        time=get_time("preprocess_between_groups"),
        runtime=get_runtime("preprocess_between_groups"),
    shell:
        """
        alleleflux-preprocess-between-groups \
            --mean_changes_fPath {input} \
            --cpus {threads} --p_value_threshold {params.p_value_threshold} \
            --output_fPath {output.outPath} \
            --filter_type {params.filter_type} \
            --data_type {params.data_type} \
            --mag_id {wildcards.mag} \
            --min_positions {params.min_positions} \
            --min_sample_num {params.min_sample_num}
            """
            
rule two_sample_unpaired:
    input:
        input_df=get_between_group_inputs(),
    output:
        os.path.join(
            OUTDIR,
            "significance_tests",
            "two_sample_unpaired_{timepoints}-{groups}",
            "{mag}_two_sample_unpaired.tsv.gz",
        ),
    params:
        min_sample_num=config["quality_control"]["min_sample_num"],
        outDir=os.path.join(
            OUTDIR, "significance_tests", "two_sample_unpaired_{timepoints}-{groups}"
        ),
        data_type=DATA_TYPE,
    threads: get_threads("statistical_tests")
    retries: get_retries("statistical_tests")
    resources:
        mem_mb=get_mem_mb("statistical_tests"),
        time=get_time("statistical_tests"),
        runtime=get_runtime("statistical_tests"),
    shell:
        """
        alleleflux-two-sample-unpaired \
            --input_df {input.input_df} \
            --min_sample_num {params.min_sample_num} \
            --mag_id {wildcards.mag} \
            --cpus {threads} \
            --output_dir {params.outDir} \
            --data_type {params.data_type}
            """

rule two_sample_paired:
    input:
        input_df=get_between_group_inputs(),
    output:
        os.path.join(
            OUTDIR,
            "significance_tests",
            "two_sample_paired_{timepoints}-{groups}",
            "{mag}_two_sample_paired.tsv.gz",
        ),
    params:
        min_sample_num=config["quality_control"]["min_sample_num"],
        outDir=os.path.join(
            OUTDIR, "significance_tests", "two_sample_paired_{timepoints}-{groups}"
        ),
        data_type=DATA_TYPE,
    threads: get_threads("statistical_tests")
    retries: get_retries("statistical_tests")
    resources:
        mem_mb=get_mem_mb("statistical_tests"),
        time=get_time("statistical_tests"),
        runtime=get_runtime("statistical_tests"),
    shell:
        """
        alleleflux-two-sample-paired \
            --input_df {input.input_df} \
            --min_sample_num {params.min_sample_num} \
            --mag_id {wildcards.mag} \
            --cpus {threads} \
            --output_dir {params.outDir} \
            --data_type {params.data_type}
            """

rule lmm_analysis:
    input:
        input_df=get_between_group_inputs(),
    output:
        os.path.join(
            OUTDIR,
            "significance_tests",
            "lmm_{timepoints}-{groups}",
            "{mag}_lmm.tsv.gz",
        ),
    params:
        min_sample_num=config["quality_control"]["min_sample_num"],
        outDir=os.path.join(
            OUTDIR, "significance_tests", "lmm_{timepoints}-{groups}"
        ),
        data_type=DATA_TYPE,
    threads: get_threads("statistical_tests")
    retries: get_retries("statistical_tests")
    resources:
        mem_mb=get_mem_mb("statistical_tests"),
        time=get_time("statistical_tests"),
        runtime=get_runtime("statistical_tests"),
    shell:
        """
        alleleflux-lmm \
            --input_df {input.input_df} \
            --min_sample_num {params.min_sample_num} \
            --cpus {threads} \
            --output_dir {params.outDir} \
            --data_type {params.data_type} \
            --mag_id {wildcards.mag}
            """

rule cmh_test:
    input:
        input_df=get_between_group_inputs(test_type="cmh"),
        preprocessed_df=os.path.join(
                OUTDIR,
                "significance_tests",
                "preprocessed_between_groups_{timepoints}-{groups}",
                "{mag}_allele_frequency_changes_mean_preprocessed.tsv.gz"
            )
            if config["statistics"].get("preprocess_between_groups", True) and DATA_TYPE == "longitudinal"
            else [],
        # Permuted (null) run: depend on the per-pair sheet (generate mode only).
        permuted_md=lambda wildcards: permuted_metadata_input(wildcards.groups),
    output:
        os.path.join(
            OUTDIR,
            "significance_tests",
            "cmh_{timepoints}-{groups}",
            "{mag}_cmh.tsv.gz"
        )
    params:
        min_sample_num=config["quality_control"]["min_sample_num"],
        outDir=os.path.join(
            OUTDIR, "significance_tests", "cmh_{timepoints}-{groups}"
        ),
        data_type=DATA_TYPE,
        groups_arg=lambda wildcards: "--groups " + " ".join(wildcards.groups.split("_")),
        # Permuted (null) run: relabel the reused cache from the per-pair sheet.
        permuted_metadata=lambda wildcards: permuted_metadata_flag(wildcards.groups),
        # Conditionally include the preprocessed file argument.
        preprocessed_flag=(
            ("--preprocessed_df")
            if config["statistics"].get("preprocess_between_groups", True) and DATA_TYPE == "longitudinal"
            else ""
        ),
    threads: get_threads("statistical_tests")
    retries: get_retries("statistical_tests")
    resources:
        time=get_time("statistical_tests"),
        runtime=get_runtime("statistical_tests"),
        mem_mb=get_mem_mb("statistical_tests"),
    shell:
        """
        alleleflux-cmh \
            --input_df {input.input_df} \
            --min_sample_num {params.min_sample_num} \
            --mag_id {wildcards.mag} \
            --data_type {params.data_type} \
            {params.groups_arg} \
            {params.permuted_metadata} \
            --cpus {threads} \
            --output_dir {params.outDir} \
            {params.preprocessed_flag} {input.preprocessed_df}

        """

