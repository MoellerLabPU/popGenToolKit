"""Pairwise conANI / popANI between the QC-passing sample pairs of each MAG.

Scope: ONE job per {mag}, independent of group and timepoint combinations.  The
sample universe is the union of everything that passed QC at any timepoint
combination, so a single job covers within-mouse (pre vs end) and between-mouse
comparisons at once; which pairs are actually compared is the CLI's --pairs
setting (config analysis.pairwise_ani.pairs).

The MAG universe is get_tested_mags(): every MAG eligible for at least one
enabled test in at least one comparison, read from the eligibility checkpoints
(so the targets are generated in get_final_pipeline_outputs, after the loop that
forces those checkpoints).  Untested MAGs never get an ANI job.  The rule's own
inputs are just the per-timepoint QC sentinels -- QC already depends on metadata,
which depends on profiles.
"""


rule pairwise_ani:
    input:
        # Depend on the QC sentinels rather than the QC TSVs so an interrupted QC
        # job cannot leave a half-written directory that looks complete.
        qc_sentinels=[qc_sentinel(tp) for tp in timepoints_labels],
        fasta=config["input"]["fasta_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
    output:
        # All three files the CLI writes, so Snakemake tracks (and cleans up on
        # failure) the complete set, not just the primary table.  The primary
        # path comes from the shared helper (called bare, it returns the "{mag}"
        # wildcard pattern) so the rule, the target generator, and any future
        # consumer rule can never spell it differently.
        pair_table=get_pairwise_ani_output_path(),
        sample_table=os.path.join(
            OUTDIR, "pairwise_ani", "{mag}_pairwise_ani_samples.tsv"
        ),
    retries: get_retries("pairwise_ani")
    threads: get_threads("pairwise_ani")
    resources:
        mem_mb=get_mem_mb("pairwise_ani"),
        time=get_time("pairwise_ani"),  # HH:MM:SS -- cluster-generic profile
        runtime=get_runtime("pairwise_ani"),  # minutes  -- native slurm plugin
    params:
        # Wildcard-dependent params need a lambda (bare {wildcards.X} does not
        # interpolate inside params values -- house rule).
        qc_files=lambda wildcards: " ".join(get_all_qc_files_for_mag(wildcards.mag)),
        profiles_dir=PROFILES_DIR,
        output_dir=os.path.join(OUTDIR, "pairwise_ani"),
        min_cov=config["analysis"].get("pairwise_ani", {}).get("min_cov", 5),
        min_freq=config["analysis"].get("pairwise_ani", {}).get("min_freq", 0.05),
        fdr=config["analysis"].get("pairwise_ani", {}).get("fdr", 1e-6),
        # The error model must assume the SAME base-quality floor the profiles
        # were built with -- read it from the profiling section, not a new knob.
        min_base_quality=config.get("profiling", {}).get("min_base_quality", 30),
        pairs=config["analysis"].get("pairwise_ani", {}).get("pairs", "within_subject"),
        # --transitions is only meaningful for pairs == "transitions"; built from the
        # configured timepoint combinations (EARLIER:LATER each), and left empty for
        # single-timepoint data, where the CLI simply never sees the flag.
        transitions_arg=(
            "--transitions " + " ".join(
                f"{tc['timepoint'][0]}:{tc['timepoint'][1]}"
                for tc in config["analysis"]["timepoints_combinations"]
                if len(tc["timepoint"]) == 2
            )
            if DATA_TYPE == "longitudinal"
            and config["analysis"].get("pairwise_ani", {}).get("pairs") == "transitions"
            else ""
        ),
        store_snp_locations=config["analysis"]
        .get("pairwise_ani", {})
        .get("store_snp_locations", "within_subject"),
    shell:
        """
        alleleflux-pairwise-ani \
            --mag {wildcards.mag} \
            --profiles_dir {params.profiles_dir} \
            --qc_files {params.qc_files} \
            --fasta {input.fasta} \
            --mag_mapping {input.mag_mapping} \
            --output_dir {params.output_dir} \
            --min_cov {params.min_cov} \
            --min_freq {params.min_freq} \
            --fdr {params.fdr} \
            --min_base_quality {params.min_base_quality} \
            --pairs {params.pairs} \
            {params.transitions_arg} \
            --store_snp_locations {params.store_snp_locations} \
            --cpus {threads}
        """
