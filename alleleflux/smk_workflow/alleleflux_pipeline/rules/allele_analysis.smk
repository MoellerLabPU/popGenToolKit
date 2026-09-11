"""Allele frequency analysis rules.

Two-stage flow:
1. ``allele_freq_cache`` — runs once per (MAG, single timepoint) and writes a
   Parquet cache file with per-sample allele frequencies for that timepoint.
   Now group-independent (2B refactor): reused across ALL (timepoint_combination,
   group_combination) pairs that include that timepoint.
2. ``allele_analysis`` — runs once per (MAG, timepoint_combination, group_combination)
   and computes the per-combination diff/aggregate outputs from the two (or one,
   for single data) cache files.

Cache job reduction (DRIDO example):
  Before 2B: 8 unique timepoints × 6 gr_combos = 48 cache-write jobs
  After  2B: 8 unique timepoints × 1            =  8 cache-write jobs
"""


def _allele_analysis_cache_input(wildcards):
    """Resolve the per-timepoint (group-independent) cache paths for one combination."""
    if DATA_TYPE == "longitudinal":
        tps = wildcards.timepoints.split("_")
    else:
        tps = [wildcards.timepoints]
    return [
        get_allele_freq_cache_path(
            mag_wildcard=wildcards.mag,
            timepoint_wildcard=tp,
        )
        for tp in tps
    ]


rule allele_freq_cache:
    input:
        # Single canonical QC file for this timepoint (group-independent after 2A/2B).
        # One file from the first tp_combo in config order that contains this timepoint.
        # Snakemake resolves it as a normal qc → generate_metadata chain — no
        # cross-combination checkpoint dependencies.
        qc_files=lambda wildcards: get_canonical_qc_file(
            wildcards.mag, wildcards.timepoint
            # groups argument removed — QC is now per-timepoint (2A/2B)
        ),
    output:
        cache=os.path.join(
            OUTDIR,
            "allele_freq_cache",
            "{timepoint}",
            "{mag}_{timepoint}_allele_frequency.parquet",
        ),
    params:
        data_type=DATA_TYPE,
        # For longitudinal data the script filters the QC file's rows to this
        # timepoint; for single data the QC file covers one timepoint already
        # so --timepoint is not passed.
        timepoint_arg=lambda wildcards: (
            f"--timepoint {wildcards.timepoint}"
            if DATA_TYPE == "longitudinal"
            else ""
        ),
    threads: get_threads("allele_freq_cache")
    retries: get_retries("allele_freq_cache")
    resources:
        mem_mb=get_mem_mb("allele_freq_cache"),
        time=get_time("allele_freq_cache"),
        runtime=get_runtime("allele_freq_cache"),
    shell:
        """
        alleleflux-cache-allele-freq \
            --magID {wildcards.mag} \
            --qc_files {input.qc_files} \
            {params.timepoint_arg} \
            --data_type {params.data_type} \
            --output_path {output.cache} \
            --cpus {threads}
        """


rule allele_analysis:
    input:
        cache_files=_allele_analysis_cache_input,
        eligibility_table=os.path.join(
            OUTDIR,
            "eligibility_table_{timepoints}-{groups}.tsv",
        ),
        # Permuted (null) run: depend on the per-pair sheet so it is built before
        # this rule relabels the reused cache (generate mode only; [] otherwise).
        permuted_md=lambda wildcards: permuted_metadata_input(wildcards.groups),
    output:
        allele_freq=os.path.join(
            OUTDIR,
            "allele_analysis",
            "allele_analysis_{timepoints}-{groups}",
            (
                "{mag}_allele_frequency_single.parquet"
                if DATA_TYPE == "single"
                else "{mag}_allele_frequency_changes_mean.parquet"
            ),
        ),
        allele_freq_no_zero_diff=os.path.join(
            OUTDIR,
            "allele_analysis",
            "allele_analysis_{timepoints}-{groups}",
            (
                "{mag}_allele_frequency_no_constant.parquet"
                if DATA_TYPE == "single"
                else "{mag}_allele_frequency_changes_no_zero-diff.parquet"
            ),
        ) if not config["quality_control"].get("disable_zero_diff_filtering", False) else [],
    params:
        outDir=os.path.join(
            OUTDIR,
            "allele_analysis",
            "allele_analysis_{timepoints}-{groups}",
        ),
        disable_zero_diff_filtering=(
            "--disable_zero_diff_filtering"
            if config["quality_control"].get("disable_zero_diff_filtering", False)
            else ""
        ),
        # Off by default — the per-subject {mag}_allele_frequency_changes.parquet
        # is not consumed by any downstream rule and the .copy() that builds
        # it adds ~150 GB of transient memory on large MAGs.  Set
        # ``analysis.save_archival_changes: true`` in the config to enable.
        save_archival_changes=(
            "--save_archival_changes"
            if config.get("analysis", {}).get("save_archival_changes", False)
            else ""
        ),
        data_type=DATA_TYPE,
        groups_arg=lambda wildcards: "--groups " + " ".join(wildcards.groups.split("_")),
        # Permuted (null) run: relabel the reused cache from the per-pair sheet.
        # Empty string for a normal run (no behaviour change).
        permuted_metadata=lambda wildcards: permuted_metadata_flag(wildcards.groups),
    threads: 1
    retries: get_retries("allele_analysis")
    resources:
        mem_mb=get_mem_mb("allele_analysis"),
        time=get_time("allele_analysis"),
        runtime=get_runtime("allele_analysis"),
    shell:
        """
        alleleflux-allele-freq \
            --magID {wildcards.mag} \
            --cache_files {input.cache_files} \
            --data_type {params.data_type} \
            --output_dir {params.outDir} \
            {params.groups_arg} \
            {params.permuted_metadata} \
            {params.disable_zero_diff_filtering} \
            {params.save_archival_changes}
        """
