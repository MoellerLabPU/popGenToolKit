"""
Sample profiling rules.

This module contains rules for profiling BAM files to extract allele frequency
information at each genomic position. The profiling step is the foundation of
the AlleleFlux pipeline.

Reference files (FASTA, Prodigal, MAG mapping) are wrapped with ancient() to
prevent unnecessary re-runs when these stable files have updated timestamps.

If USE_EXISTING_PROFILES is True (profiles_path specified in config), the
profiling step is skipped entirely - downstream rules read directly from the
existing profiles directory (PROFILES_DIR).

Expected directory structure for profiles_path:
    {profiles_path}/
        {sample1}/
            {sample1}_{MAG1}_profiled.tsv.gz
            {sample1}_{MAG2}_profiled.tsv.gz
            ...
        {sample2}/
            {sample2}_{MAG1}_profiled.tsv.gz
            ...
"""

# Only define the profile rule when NOT using existing profiles.
# When USE_EXISTING_PROFILES is True, downstream rules read directly from PROFILES_DIR
# (which points to the user-specified profiles_path).
if not USE_EXISTING_PROFILES:
    rule profile:
        input:
            bam=lambda wildcards: sample_to_bam_map[wildcards.sample],
            fasta=config["input"]["fasta_path"],
            prodigal=config["input"]["prodigal_path"],
            mag_mapping=config["input"]["mag_mapping_path"],
        output:
            # Sentinel marker (see common.smk).  We deliberately avoid
            # `directory()` so a kill mid-run does not erase already-written
            # profile files and silently invalidate every downstream step.
            sentinel=os.path.join(OUTDIR, "profiles", "{sample}", SENTINEL_PROFILE),
        threads: get_threads("profile")
        retries: get_retries("profile")
        resources:
            mem_mb=get_mem_mb("profile"),
            time=get_time("profile"),
            runtime=get_runtime("profile"),
        params:
            outDir=os.path.join(OUTDIR, "profiles"),
            no_ignore_orphans="--no-ignore-orphans" if not config.get("profiling", {}).get("ignore_orphans", True) else "",
            no_ignore_overlaps="--no-ignore-overlaps" if not config.get("profiling", {}).get("ignore_overlaps", True) else "",
            min_base_quality=config.get("profiling", {}).get("min_base_quality", 30),
            min_mapping_quality=config.get("profiling", {}).get("min_mapping_quality", 2),
            log_level=config.get("log_level", "INFO"),
        shell:
            """
            alleleflux-profile \
                --bam_path {input.bam} \
                --fasta_path {input.fasta} \
                --prodigal_fasta {input.prodigal} \
                --mag_mapping_file {input.mag_mapping} \
                --cpus {threads} \
                --output_dir {params.outDir} \
                --sampleID {wildcards.sample} \
                {params.no_ignore_orphans} \
                {params.no_ignore_overlaps} \
                --min-base-quality {params.min_base_quality} \
                --min-mapping-quality {params.min_mapping_quality} \
                --log-level {params.log_level}
            touch {output.sentinel}
            """
