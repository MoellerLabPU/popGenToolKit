"""Preprocessing eligibility checkpoint rules.

These checkpoints aggregate status files from preprocessing steps and generate
eligibility tables that determine which MAGs proceed to downstream statistical
tests based on whether they have sufficient positions remaining after filtering.

Two checkpoints are provided:
- preprocessing_eligibility_between_groups: For tests comparing groups
- preprocessing_eligibility_within_groups: For tests within individual groups
"""


def get_between_groups_status_files(wildcards):
    """
    Get all status files from between-groups preprocessing for eligible MAGs.
    This function is called after the QC eligibility checkpoint.
    """
    # First, get the QC-eligible MAGs
    checkpoint_output = checkpoints.eligibility_table.get(
        timepoints=wildcards.timepoints,
        groups=wildcards.groups
    ).output.out_fPath
    
    # Get MAGs eligible for between-group tests (unpaired OR paired)
    # We exclude single_sample-only MAGs since this is for between-groups preprocessing
    # NOTE: Uses _get_mags_by_eligibility (QC-only) since this determines which MAGs to preprocess
    eligible_mags = set(
        _get_mags_by_eligibility(wildcards.timepoints, wildcards.groups, eligibility_type="two_sample_unpaired")
    ) | set(
        _get_mags_by_eligibility(wildcards.timepoints, wildcards.groups, eligibility_type="two_sample_paired")
    )
    
    status_files = []
    for mag in eligible_mags:
        status_files.append(
            os.path.join(
                OUTDIR,
                "significance_tests",
                f"preprocessed_between_groups_{wildcards.timepoints}-{wildcards.groups}",
                f"{mag}_preprocessing_status.json"
            )
        )
    return status_files


def get_within_groups_status_files(wildcards):
    """
    Get all status files from within-groups preprocessing for eligible MAG-group combinations.
    """
    # First, get the QC-eligible MAG-group combinations
    checkpoint_output = checkpoints.eligibility_table.get(
        timepoints=wildcards.timepoints,
        groups=wildcards.groups
    ).output.out_fPath
    
    # NOTE: Uses _get_single_sample_entries (QC-only) since this determines which MAGs to preprocess
    sample_entries = _get_single_sample_entries(wildcards.timepoints, wildcards.groups)
    
    status_files = []
    for mag, group in sample_entries:
        status_files.append(
            os.path.join(
                OUTDIR,
                "significance_tests",
                f"preprocessed_within_groups_{wildcards.timepoints}-{wildcards.groups}",
                f"{mag}_{group}_preprocessing_status.json"
            )
        )
    return status_files


checkpoint preprocessing_eligibility_between_groups:
    """
    Aggregates status files from preprocess_between_groups and creates
    an eligibility table for between-group tests.
    
    Output columns: two_sample_unpaired_eligible (used by unpaired tests and LMM),
    two_sample_paired_eligible (used by paired tests and CMH).
    """
    input:
        status_files=get_between_groups_status_files
    output:
        out_fPath=os.path.join(
            OUTDIR,
            "preprocessing_eligibility",
            "preprocessing_eligibility_between_groups_{timepoints}-{groups}.tsv"
        )
    params:
        status_dir=os.path.join(
            OUTDIR,
            "significance_tests",
            "preprocessed_between_groups_{timepoints}-{groups}"
        ),
        min_positions=config["statistics"].get("min_positions_after_preprocess", 1),
    retries: get_retries("preprocessing_eligibility_between_groups")
    resources:
        mem_mb=get_mem_mb("preprocessing_eligibility_between_groups"),
        time=get_time("preprocessing_eligibility_between_groups"),
        runtime=get_runtime("preprocessing_eligibility_between_groups"),
    shell:
        """
        alleleflux-preprocessing-eligibility \
            --status_dir {params.status_dir} \
            --output_file {output.out_fPath} \
            --preprocess_type between_groups \
            --min_positions {params.min_positions}
        """


checkpoint preprocessing_eligibility_within_groups:
    """
    Aggregates status files from preprocess_within_groups and creates
    an eligibility table for within-group tests.
    
    Output columns: single_sample_eligible_{group} (used by single_sample,
    lmm_across_time, and cmh_across_time tests).
    """
    input:
        status_files=get_within_groups_status_files
    output:
        out_fPath=os.path.join(
            OUTDIR,
            "preprocessing_eligibility",
            "preprocessing_eligibility_within_groups_{timepoints}-{groups}.tsv"
        )
    params:
        status_dir=os.path.join(
            OUTDIR,
            "significance_tests",
            "preprocessed_within_groups_{timepoints}-{groups}"
        ),
        min_positions=config["statistics"].get("min_positions_after_preprocess", 1),
        # Extract group names from the groups wildcard (e.g., "fat_control" -> ["fat", "control"])
        groups=lambda wildcards: wildcards.groups.split("_"),
    retries: get_retries("preprocessing_eligibility_within_groups")
    resources:
        mem_mb=get_mem_mb("preprocessing_eligibility_within_groups"),
        time=get_time("preprocessing_eligibility_within_groups"),
        runtime=get_runtime("preprocessing_eligibility_within_groups"),
    shell:
        """
        alleleflux-preprocessing-eligibility \
            --status_dir {params.status_dir} \
            --output_file {output.out_fPath} \
            --preprocess_type within_groups \
            --groups {params.groups} \
            --min_positions {params.min_positions}
        """
