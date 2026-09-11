"""
Quality control rules.

This module performs QC on profiled samples, calculating coverage breadth and
depth metrics for each MAG. QC results determine sample eligibility for
downstream statistical analysis based on configurable thresholds.

**2A Refactor — Per-timepoint scope (group-independent):**
QC metrics (breadth, depth) are intrinsic to each sample and independent of
which group-pair comparison will later use them.  The QC rule now runs once
per {timepoints} wildcard (not per groups).  The input metadata directory
contains all groups for this timepoint.

Group-pair eligibility (e.g. how many subjects passed in group A vs B) is
computed downstream in the eligibility_table checkpoint, which reads from the
single per-timepoint QC directory and filters to the specific pair.
"""

rule qc:
    """Run QC once per timepoints combination, covering ALL groups.

    Input is the per-timepoint metadata directory (no {groups} wildcard).
    Output is a single per-timepoint QC directory reused by ALL group-pair
    eligibility_table checkpoints that share this timepoints label.
    """
    input:
        # Depend on the metadata sentinel rather than the directory itself.
        metadata_done=metadata_sentinel("{timepoints}"),
        fasta=config["input"]["fasta_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
    output:
        # Sentinel marker (see common.smk) — replaces directory() output.
        sentinel=qc_sentinel("{timepoints}"),
    params:
        breadth_threshold=config["quality_control"]["breadth_threshold"],
        coverage_threshold=config["quality_control"]["coverage_threshold"],
        data_type=DATA_TYPE,
        # Concrete directory paths the script reads from / writes into.
        metadata_dir=os.path.join(
            OUTDIR, "inputMetadata", "inputMetadata_{timepoints}"
        ),
        outDir=os.path.join(OUTDIR, "QC", "QC_{timepoints}"),
    threads: get_threads("qc")
    retries: get_retries("qc")
    resources:
        time=get_time("qc"),
        runtime=get_runtime("qc"),
        mem_mb=get_mem_mb("qc"),
    shell:
        """
        alleleflux-qc \
            --rootDir {params.metadata_dir} \
            --fasta {input.fasta} \
            --breadth_threshold {params.breadth_threshold} \
            --coverage_threshold {params.coverage_threshold} \
            --cpus {threads} \
            --output_dir {params.outDir} \
            --data_type {params.data_type} \
            --mag_mapping_file {input.mag_mapping}
        touch {output.sentinel}
        """
