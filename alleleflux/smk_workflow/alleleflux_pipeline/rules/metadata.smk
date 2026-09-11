"""
Sample metadata generation rules.

This module generates per-MAG metadata files that organize sample profile paths
by experimental group and timepoint. These metadata files are used as input for
QC analysis and downstream statistical tests.

**2A Refactor — Per-timepoint scope (group-independent):**
The metadata and QC rules previously ran once per (timepoints, groups) combination.
They now run once per timepoints combination only, passing ALL unique group values
from the config.  This means:
  - Before: 15 tp_combos × 6 gr_combos = 90 metadata + 90 QC jobs
  - After:  15 tp_combos × 1            = 15 metadata + 15 QC jobs (6× reduction)

Group-pair-specific eligibility (which MAGs have enough samples in group A vs B)
is still computed per (timepoints, groups) in the eligibility_table checkpoint,
which now reads from the single per-timepoint QC directory.

When USE_EXISTING_PROFILES is True, profiles are read directly from PROFILES_DIR
(the existing profiles path specified in config). No profiling step runs.
The metadata files will contain the actual paths to the existing profile files.
"""

# ALL unique group values across all group combinations — passed to metadata
# so a single job covers every group in this timepoint, not just one pair.
_all_group_values = sorted(
    {gr["treatment"] for gr in config["analysis"]["groups_combinations"]}
    | {gr["control"] for gr in config["analysis"]["groups_combinations"]}
)


# Define input function to conditionally depend on profile rule outputs
def get_metadata_inputs(wildcards):
    """Return inputs for generate_metadata rule.
    
    When using existing profiles, we don't depend on the profile rule outputs.
    When generating new profiles, we depend on all sample directories.
    """
    inputs = {"metadata": config["input"]["metadata_path"]}
    
    if not USE_EXISTING_PROFILES:
        # Only depend on profile outputs when generating new profiles.
        # Depend on the per-sample profile sentinel — see common.smk for the
        # rationale.  Using the sentinel (rather than directory()) means an
        # interrupted profile job cannot silently invalidate downstream rules.
        inputs["sampleDirs"] = [profile_sentinel(s) for s in samples]

    return inputs


rule generate_metadata:
    """Generate per-MAG metadata files for a single timepoints combination.

    Runs once per {timepoints} wildcard (not per groups).  All unique group
    values from the config are passed via --groups so downstream QC sees every
    group in one directory.
    """
    input:
        unpack(get_metadata_inputs),
    output:
        # Sentinel marker (see common.smk) — replaces a directory() output to
        # avoid pre-deletion of the inputMetadata dir on every re-run.
        sentinel=metadata_sentinel("{timepoints}"),
    params:
        # Read profiles from PROFILES_DIR - either existing profiles or newly generated
        rootDir=PROFILES_DIR,
        data_type=DATA_TYPE,
        # The actual output directory the script writes per-MAG metadata into.
        outDir=os.path.join(
            OUTDIR, "inputMetadata", "inputMetadata_{timepoints}"
        ),
        # Pass every unique group so the QC directory covers all groups
        group_args=" ".join(_all_group_values),
        timepoint_args=lambda wildcards: f"--timepoints {wildcards.timepoints.replace('_', ' ')}",
    retries: get_retries("generate_metadata")
    resources:
        mem_mb=get_mem_mb("generate_metadata"),
        time=get_time("generate_metadata"),
        runtime=get_runtime("generate_metadata"),
    shell:
        """
        alleleflux-metadata \
            --rootDir {params.rootDir} \
            --metadata {input.metadata} \
            --outDir {params.outDir} \
            --groups {params.group_args} \
            --data_type {params.data_type} \
            {params.timepoint_args}
        touch {output.sentinel}
        """
