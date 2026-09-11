"""Per-mouse strain-background calls and the per-MAG classification.

``strain_turnover`` runs ONCE per {mag} on that MAG's pairwise-ANI table (no
profiles are read) and writes the per-mouse verdict table plus a per-group
rollup.  Measured on the 903-pair all-vs-all table: 12 ms of pandas inside a
1.3 s interpreter start-up, so both rules are ``localrules`` -- a SLURM
submission per MAG would cost more in scheduling than the job itself.  ``replacement_classification`` then runs ONCE over every
MAG's turnover table and writes the single table the enrichment filter reads
(both metrics, mouse and replicate blocks).

The MAG universe is the same one pairwise_ani uses -- get_tested_mags(), every
MAG eligible for at least one enabled test in at least one comparison -- which
reads the eligibility checkpoints, so the targets live in
get_final_pipeline_outputs and the classification rule's input is an input
FUNCTION (a bare call at parse time would touch the checkpoints too early).
Requires use_pairwise_ani (the pair table is the input) and longitudinal data
(the transitions come from timepoints_combinations).
"""


# Both rules run where snakemake runs, not as SLURM jobs: seconds each, I/O-light.
localrules:
    strain_turnover,
    replacement_classification,


rule strain_turnover:
    input:
        pair_table=get_pairwise_ani_output_path(),
    output:
        turnover=get_strain_turnover_output_path(),
        rollup=os.path.join(OUTDIR, "strain_turnover", "{mag}_turnover_rollup.tsv"),
    params:
        output_dir=os.path.join(OUTDIR, "strain_turnover"),
        # EARLIER:LATER for every two-timepoint combination -- the same spelling
        # the pairwise-ANI rule uses for pairs == "transitions".
        transitions=" ".join(
            f"{tc['timepoint'][0]}:{tc['timepoint'][1]}"
            for tc in config["analysis"]["timepoints_combinations"]
            if len(tc["timepoint"]) == 2
        ),
        min_compared=config["analysis"].get("strain_turnover", {}).get("min_compared", 0.1),
        pop_threshold=config["analysis"].get("strain_turnover", {}).get("pop_threshold", 0.99999),
        con_threshold=config["analysis"].get("strain_turnover", {}).get("con_threshold", 0.999),
    shell:
        """
        alleleflux-strain-turnover \
            --mag {wildcards.mag} \
            --pair_table {input.pair_table} \
            --output_dir {params.output_dir} \
            --transitions {params.transitions} \
            --min_compared {params.min_compared} \
            --pop_threshold {params.pop_threshold} \
            --con_threshold {params.con_threshold}
        """


rule replacement_classification:
    input:
        # Every tested MAG's turnover table; listing them (not the directory) makes
        # the rule wait for all of them and rerun when any changes.  A lambda so the
        # checkpoint-reading generator runs during DAG evaluation, not at parse time.
        turnover=lambda wildcards: generate_strain_turnover_targets(),
    output:
        get_replacement_classification_path(),
    params:
        turnover_dir=os.path.join(OUTDIR, "strain_turnover"),
    shell:
        """
        alleleflux-replacement-classification \
            --turnover_dir {params.turnover_dir} \
            --output_path {output}
        """
