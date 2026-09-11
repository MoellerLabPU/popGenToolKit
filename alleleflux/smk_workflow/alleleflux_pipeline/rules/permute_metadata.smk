"""
Permuted (null) metadata generation.

For a generate-mode permuted run (``permutation.seed`` set — see the mode table
in common.smk), produce one permuted metadata sheet per comparison pair by
shuffling group labels with a fixed seed.  Each sheet is consumed (relabel the
reused cache/QC in memory) by the group-dependent rules: the ``eligibility_table``
checkpoint, ``allele_analysis``, and the LMM / CMH tests.

This rule is wired into the DAG only when ``GENERATE_PERMUTED`` is true: the
group-dependent rules declare the sheet as an input via ``permuted_metadata_input``,
which returns ``[]`` in BYO / non-permuted mode so this rule stays out of the DAG.

Replaces the eager per-pair ``subprocess.run`` loop that ``run_permutations``
previously executed before Snakemake — so the sheets are now tracked artifacts
(provenance + skip-if-exists) and a dry run writes nothing.
"""

localrules: 
    permute_metadata,

rule permute_metadata:
    input:
        # The real (unpermuted) metadata.  Declared as an input so the sheet is
        # rebuilt if the source metadata changes.
        metadata=config["input"]["metadata_path"],
    output:
        sheet=os.path.join(
            OUTDIR, "permuted_metadata", "permuted_metadata_{groups}.tsv"
        ),
    params:
        unit=PERMUTATION.get("unit", "subjectID"),
        seed=PERMUTATION_SEED,
        # "{treatment}_{control}" -> "--groups treatment control" (mirrors the
        # groups_arg convention used by the group-dependent rules).
        groups_args=lambda wildcards: "--groups " + " ".join(wildcards.groups.split("_")),
        # Optional knobs: emitted only when configured, else an empty string.
        block_flag=(f"--block {PERMUTATION['block']}" if PERMUTATION.get("block") else ""),
        swaps_flag=(
            f"--swaps {PERMUTATION['swaps']}"
            if PERMUTATION.get("swaps") is not None
            else ""
        ),
    retries: get_retries("permute_metadata")
    resources:
        mem_mb=get_mem_mb("permute_metadata"),
        time=get_time("permute_metadata"),
        runtime=get_runtime("permute_metadata"),
    shell:
        """
        alleleflux-permute-metadata \
            --input {input.metadata} \
            --output {output.sheet} \
            --unit {params.unit} \
            {params.groups_args} \
            --seed {params.seed} \
            {params.block_flag} {params.swaps_flag}
        """
