"""
QC-based eligibility table checkpoint.

This checkpoint generates eligibility tables based on QC results, determining
which MAGs have sufficient sample coverage to proceed to downstream analysis.
The checkpoint mechanism allows Snakemake to dynamically update the DAG based
on which MAGs pass QC thresholds.

**2A Refactor — Per-timepoint QC, per-group-pair eligibility:**
The QC directory is now per-timepoints (not per timepoints-groups).  The
eligibility_table checkpoint still runs once per (timepoints, groups) pair —
it reads from the shared per-timepoint QC directory and filters to the
specific group pair via --groups.  This means:
  - QC runs: 15 (was 90)
  - Eligibility runs: 90 (unchanged) — still one per combination
"""

checkpoint eligibility_table:
    input:
        # Depend on the per-timepoints QC sentinel.  The QC directory itself is
        # consumed by the script, but the sentinel is what proves QC finished
        # — see common.smk for the rationale.
        qc_done=qc_sentinel("{timepoints}"),
        # Permuted (null) run: depend on the per-pair sheet so it is built before
        # this checkpoint (generate mode only; [] otherwise — no behaviour change).
        permuted_md=lambda wildcards: permuted_metadata_input(wildcards.groups),
    output:
        out_fPath=os.path.join(OUTDIR, "eligibility_table_{timepoints}-{groups}.tsv"),
    params:
        min_sample_num=config["quality_control"]["min_sample_num"],
        data_type=DATA_TYPE,
        # Filter QC results to the specific group pair for this combination.
        # REUSE_DIR == OUTDIR normally; for a reuse_from (permuted) run it points
        # at the real run's QC dir so this checkpoint reads the existing QC.
        groups_args=lambda wildcards: wildcards.groups.replace("_", " "),
        qc_dir=os.path.join(REUSE_DIR, "QC", "QC_{timepoints}"),
        # Permuted (null) run: relabel the reused QC + recompute counts per pair.
        permuted_metadata=lambda wildcards: permuted_metadata_flag(wildcards.groups),
    retries: get_retries("eligibility_table")
    resources:
        time=get_time("eligibility_table"),
        runtime=get_runtime("eligibility_table"),
    shell:
        """
        alleleflux-eligibility \
            --qc_dir {params.qc_dir} \
            --min_sample_num {params.min_sample_num} \
            --output_file {output.out_fPath} \
            --data_type {params.data_type} \
            --groups {params.groups_args} \
            {params.permuted_metadata}
        """
