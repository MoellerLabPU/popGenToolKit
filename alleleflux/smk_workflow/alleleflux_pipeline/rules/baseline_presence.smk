"""Was each significant allele already present at the baseline timepoint?

``baseline_presence`` runs ONCE per {timepoints}-{groups} comparison, on the
p_value_summary file of the configured test family (analysis.baseline_presence.summary)
and the per-MAG significance-test files behind it, and reads every sample's
profile once (both groups, both timepoints).  It therefore sits AFTER the
statistics stage: its targets are generated inside get_final_pipeline_outputs
(checkpoint-aware), only where the family's summary target itself exists.

When use_strain_turnover is on, the turnover tables are joined in for the
strain_background column and become an input dependency.
"""


def _baseline_presence_turnover_inputs(wildcards):
    """Turnover tables as inputs only when the strain column is requested."""
    if config["analysis"].get("use_strain_turnover", False):
        return generate_strain_turnover_targets()
    return []


rule baseline_presence:
    input:
        summary=os.path.join(
            OUTDIR,
            "p_value_summary",
            "{timepoints}-{groups}",
            "p_value_summary_" + BASELINE_PRESENCE_FAMILY + "_{timepoints}-{groups}.tsv",
        ),
        turnover=_baseline_presence_turnover_inputs,
        fasta=config["input"]["fasta_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
        metadata=config["input"]["metadata_path"],
    output:
        long=get_baseline_presence_stem() + ".tsv.gz",
        summary=get_baseline_presence_stem() + "_summary.tsv",
    retries: get_retries("baseline_presence")
    threads: get_threads("baseline_presence")
    resources:
        mem_mb=get_mem_mb("baseline_presence"),
        time=get_time("baseline_presence"),
        runtime=get_runtime("baseline_presence"),
    params:
        run_dir=OUTDIR,
        profiles_dir=PROFILES_DIR,
        output_dir=os.path.join(OUTDIR, "baseline_presence"),
        summary_family=BASELINE_PRESENCE_FAMILY,
        test_type=config["analysis"]["baseline_presence"]["test_type"],
        threshold_column=config["analysis"].get("baseline_presence", {}).get("threshold_column", "q_value"),
        threshold=config["analysis"].get("baseline_presence", {}).get("threshold", 0.05),
        # Presence rule: same knobs as pairwise ANI so the two agree by construction.
        min_cov=config["analysis"].get("pairwise_ani", {}).get("min_cov", 5),
        min_freq=config["analysis"].get("pairwise_ani", {}).get("min_freq", 0.05),
        fdr=config["analysis"].get("pairwise_ani", {}).get("fdr", 1e-6),
        min_base_quality=config.get("profiling", {}).get("min_base_quality", 30),
        turnover_arg=(
            f"--turnover_dir {os.path.join(OUTDIR, 'strain_turnover')}"
            if config["analysis"].get("use_strain_turnover", False)
            else ""
        ),
    shell:
        """
        alleleflux-baseline-presence \
            --run_dir {params.run_dir} \
            --comparison {wildcards.timepoints}-{wildcards.groups} \
            --summary {params.summary_family} \
            --test_type {params.test_type} \
            --threshold_column {params.threshold_column} \
            --threshold {params.threshold} \
            --profiles_dir {params.profiles_dir} \
            --metadata {input.metadata} \
            --fasta {input.fasta} \
            --mag_mapping {input.mag_mapping} \
            --output_dir {params.output_dir} \
            --min_cov {params.min_cov} \
            --min_freq {params.min_freq} \
            --fdr {params.fdr} \
            --min_base_quality {params.min_base_quality} \
            {params.turnover_arg} \
            --cpus {threads}
        """
