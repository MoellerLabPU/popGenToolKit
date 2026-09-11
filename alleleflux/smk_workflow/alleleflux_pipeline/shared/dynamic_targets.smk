
"""Dynamic target generation functions for Snakemake.

This module provides functions that dynamically generate output targets based on
MAG eligibility and configuration options. These functions are used in the main
Snakefile to determine which files should be produced for each timepoint-group
combination.

Key functions:
- get_eligible_mags(): Unified helper for getting eligible MAGs for any test type
- generate_*_targets(): Functions that return lists of output file paths

These functions are checkpoint-aware and are called after the QC eligibility
checkpoint has been evaluated to ensure the DAG is properly updated.
"""


# Within-group (across-time) test types — gated by analysis.run_within_group_tests.
WITHIN_GROUP_TYPES = ("single_sample", "lmm_across_time", "cmh_across_time")


def _run_within_group_tests():
    """Whether within-group / across-time tests should run.

    Default True.  Permuted (null) runs set this False — the divergence null
    only needs the between-group comparison, so single-sample and across-time
    tests (which measure within-group change over time) are wasted compute.
    Also useful as a standalone knob for non-permuted runs.
    """
    return config["analysis"].get("run_within_group_tests", True)


def get_enabled_test_types():
    within = _run_within_group_tests()
    enabled = []
    if config["analysis"].get("use_significance_tests", True):
        enabled.extend(["two_sample_unpaired", "two_sample_paired"])
        if within:
            enabled.append("single_sample")
    if config["analysis"].get("use_lmm", True):
        enabled.append("lmm")
        if within:
            enabled.append("lmm_across_time")
    if config["analysis"].get("use_cmh", True):
        enabled.append("cmh")
        if within:
            enabled.append("cmh_across_time")
    return enabled


def get_eligible_mags(tp, gr, test_type):
    """
    Unified helper function to get eligible MAGs for any test type.
    
    Uses preprocessing eligibility when the appropriate preprocessing config is enabled,
    otherwise falls back to standard QC eligibility.
    
    When preprocessing is enabled, this function triggers the appropriate preprocessing
    eligibility checkpoint to ensure the DAG is properly updated.
    
    Parameters:
        tp: Timepoint label
        gr: Groups label
        test_type: Type of statistical test. Options:
            - "two_sample_unpaired", "two_sample_paired", "lmm", "cmh": Between-group tests
            - "single_sample", "lmm_across_time", "cmh_across_time": Within-group tests
    
    Returns:
        - For between-group tests: List of MAG IDs
        - For within-group tests: List of (MAG_ID, group) tuples
    """
    # Determine which preprocessing config to check
    if test_type in ["two_sample_unpaired", "two_sample_paired", "lmm", "cmh"]:
        preprocess_enabled = config["statistics"].get("preprocess_between_groups", False)
        
        if preprocess_enabled:
            # The checkpoint is triggered in get_final_pipeline_outputs before this function is called
            return get_mags_by_preprocessing_eligibility(tp, gr, test_type)
        else:
            return _get_mags_by_eligibility(tp, gr, eligibility_type=test_type)
    
    elif test_type in ["single_sample", "lmm_across_time", "cmh_across_time"]:
        # Within-group tests disabled (e.g. permuted/null run): suppress every
        # target for these types from every generator via this single chokepoint.
        if not _run_within_group_tests():
            return []

        preprocess_enabled = config["statistics"].get("preprocess_within_groups", False)

        # First get all QC-eligible entries
        sample_entries = _get_single_sample_entries(tp, gr)
        
        if not preprocess_enabled:
            return sample_entries
        
        # The checkpoint is triggered in get_final_pipeline_outputs before this function is called
        
        # Filter by preprocessing eligibility - cache results per group
        eligible_entries = []
        eligible_mags_by_group = {}
        for mag, grp in sample_entries:
            if grp not in eligible_mags_by_group:
                eligible_mags_by_group[grp] = get_mags_by_preprocessing_eligibility(tp, gr, test_type, group=grp)
            if mag in eligible_mags_by_group[grp]:
                eligible_entries.append((mag, grp))
        
        return eligible_entries
    
    else:
        raise ValueError(
            f"Unknown test_type: {test_type}. Use 'two_sample_unpaired', 'two_sample_paired', "
            "'lmm', 'cmh', 'single_sample', 'lmm_across_time', or 'cmh_across_time'."
        )


def generate_p_value_summary_targets(tp, gr):
    """
    Get the expected summary files for a given timepoint combination and group combination.
    """
    if config["analysis"].get("allele_analysis_only", False):
        return []
    test_types = get_enabled_test_types()
    expected = []
    output_dir = os.path.join(OUTDIR, "p_value_summary", f"{tp}-{gr}")

    for test_type in test_types:
        # Based on eligibility, determine if a file should be created for this test_type
        if test_type in ['two_sample_unpaired', 'two_sample_paired', 'lmm', 'cmh']:
            mags = get_eligible_mags(tp, gr, test_type)
            if mags:
                filename = f"p_value_summary_{test_type}_{tp}-{gr}.tsv"
                expected.append(os.path.join(output_dir, filename))
        
        elif test_type in ["single_sample",'lmm_across_time', 'cmh_across_time'] and DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, test_type)
            if sample_entries:
                filename = f"p_value_summary_{test_type}_{tp}-{gr}.tsv"
                expected.append(os.path.join(output_dir, filename))
                
    return expected

def generate_significant_sites_summary_targets():
    """Run-once rollup of EVERY p_value_summary table into one significant-sites summary.

    Always produced (no config flag) -- the AlleleFlux score/heatmap notebooks consume it --
    but skipped when ``allele_analysis_only`` is set, since then no p_value_summary tables
    exist to summarize.  Not scoped by (tp, gr): it is a single terminal aggregation over the
    whole run, so it is added once, after the per-combination loop in the Snakefile.
    """
    if config["analysis"].get("allele_analysis_only", False):
        return []
    return [
        os.path.join(
            OUTDIR,
            "p_value_summary",
            "significant_sites_summary",
            "significant_sites_mag_cell_stats_long.tsv",
        )
    ]

def generate_allele_analysis_targets(tp, gr):
    """
    Dynamically generate targets for allele analysis based on eligible MAGs.
    
    Uses get_allele_analysis_input_path() helper for centralized path construction.
    
    NOTE: Uses _get_mags_by_eligibility (QC-only) because allele analysis runs
    BEFORE preprocessing, so preprocessing eligibility is not yet available.

    When within-group tests are disabled (e.g. permuted/null runs), MAGs that
    are eligible *only* for single-sample (within-group) tests have no
    downstream consumer, so we restrict to between-group eligibility
    ("between_only") to avoid scheduling allele-analysis / allele-freq-cache
    jobs whose output is never used.
    """
    targets = []
    # Drop within-group-only MAGs when within-group tests are off — otherwise
    # use the full QC-eligible set (allele analysis also feeds regional contrast
    # and stands alone as a deliverable in allele_analysis_only mode).
    eligibility_type = "all" if _run_within_group_tests() else "between_only"
    eligible_mags = _get_mags_by_eligibility(tp, gr, eligibility_type=eligibility_type)
    
    # Add targets for each eligible MAG using the centralized path helper
    for mag in eligible_mags:
        targets.append(
            get_allele_analysis_input_path(
                mag_wildcard=mag, tp_wildcard=tp, gr_wildcard=gr
            )
        )
    return targets


def generate_mag_scores_targets(tp, gr):
    """
    Generate MAG-level combined score targets.

    Not produced when ``allele_analysis_only`` is True, because significance
    tests are not run and therefore no per-MAG scores are available.
    """
    if config["analysis"].get("allele_analysis_only", False):
        return []
    targets = []

    if config["analysis"].get("use_significance_tests", True):
        for test_type in ["two_sample_unpaired", "two_sample_paired"]:
            mags = get_eligible_mags(tp, gr, test_type)
            if mags:
                targets.append(
                    os.path.join(
                        OUTDIR, "scores", "processed", "combined", "MAG",
                        f"scores_{test_type}-{tp}-{gr}-MAGs.tsv",
                    )
                )

        # Single-sample (longitudinal only)
        if DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, "single_sample")
            if sample_entries:
                unique_groups = sorted(set([grp for mag, grp in sample_entries]))
                for grp in unique_groups:
                    targets.append(
                        os.path.join(
                            OUTDIR, "scores", "processed", "combined", "MAG",
                            f"scores_single_sample-{tp}-{gr}_{grp}-MAGs.tsv",
                        )
                    )

    if config["analysis"].get("use_lmm", True):
        mags = get_eligible_mags(tp, gr, "lmm")
        if mags:
            targets.append(
                os.path.join(
                    OUTDIR, "scores", "processed", "combined", "MAG",
                    f"scores_lmm-{tp}-{gr}-MAGs.tsv",
                )
            )
        # LMM across time (longitudinal only)
        if DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, "lmm_across_time")
            if sample_entries:
                unique_groups = sorted(set([grp for mag, grp in sample_entries]))
                for grp in unique_groups:
                    targets.append(
                        os.path.join(
                            OUTDIR, "scores", "processed", "combined", "MAG",
                            f"scores_lmm_across_time-{tp}-{gr}_{grp}-MAGs.tsv",
                        )
                    )

    if config["analysis"].get("use_cmh", True):
        # Regular CMH — uses combine_MAG_scores_cmh (focus_tp in filename)
        mags = get_eligible_mags(tp, gr, "cmh")
        if mags:
            focus_tp = focus_timepoints.get(tp)
            if not focus_tp:
                raise ValueError(f"No focus timepoint defined for {tp}.")
            targets.append(
                os.path.join(
                    OUTDIR, "scores", "processed", "combined", "MAG",
                    f"scores_cmh-{tp}-{gr}-MAGs-{focus_tp}.tsv",
                )
            )
        # CMH across time (longitudinal only) — uses combine_MAG_scores (standard path)
        if DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, "cmh_across_time")
            if sample_entries:
                unique_groups = sorted(set([grp for mag, grp in sample_entries]))
                for grp in unique_groups:
                    targets.append(
                        os.path.join(
                            OUTDIR, "scores", "processed", "combined", "MAG",
                            f"scores_cmh_across_time-{tp}-{gr}_{grp}-MAGs.tsv",
                        )
                    )

    return targets


def generate_taxa_scores_targets(tp, gr):
    if config["analysis"].get("allele_analysis_only", False):
        return []
    targets = []
    # Read configured levels from config; default to empty list (MAG level only)
    tax_levels = config["analysis"].get("taxa_score_levels", [])
    
    # For two-sample tests, group_str is empty.
    if config["analysis"].get("use_significance_tests", True):
        for test_type in ["two_sample_unpaired", "two_sample_paired"]:
            
            # Only generate targets if there are eligible MAGs for this test type
            mags = get_eligible_mags(tp, gr, test_type)
            if mags:  # Only proceed if there are eligible MAGs
                group_str = ""  # no group marker for two-sample tests
                for taxon in tax_levels:
                    targets.append(
                        os.path.join(
                            OUTDIR,
                            "scores",
                            "processed",
                            "combined",
                            taxon,
                            f"scores_{test_type}-{tp}-{gr}{group_str}-{taxon}.tsv",
                        )
                    )
        
        # For single-sample tests, group_str is "_" plus the sample group.
        # Only include if data_type is longitudinal
        if DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, "single_sample")
            if sample_entries:  # Only proceed if there are eligible MAGs
                unique_groups = sorted(set([grp for mag, grp in sample_entries]))
                for grp in unique_groups:
                    group_str = f"_{grp}"
                    for taxon in tax_levels:
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "scores",
                                "processed",
                                "combined",
                                taxon,
                                f"scores_single_sample-{tp}-{gr}{group_str}-{taxon}.tsv",
                            )
                        )
    
    # Add LMM taxa targets if LMM is enabled
    if config["analysis"].get("use_lmm", True):
        # Only generate targets if there are eligible MAGs
        mags = get_eligible_mags(tp, gr, "lmm")
        if mags:  # Only proceed if there are eligible MAGs
            group_str = ""  # no group marker for LMM
            for taxon in tax_levels:
                targets.append(
                    os.path.join(
                        OUTDIR,
                        "scores",
                        "processed",
                        "combined",
                        taxon,
                        f"scores_lmm-{tp}-{gr}{group_str}-{taxon}.tsv",
                    )
                )
    # CMH test targets
    if config["analysis"].get("use_cmh", True):
        # CMH uses paired eligibility
        mags = get_eligible_mags(tp, gr, "cmh")
        if mags:
            # Get the focus timepoint from our global mapping
            focus_tp = focus_timepoints.get(tp)
            if not focus_tp:
                raise ValueError(f"No focus timepoint defined for {tp}.")
            for taxon in tax_levels:
                targets.append(
                    os.path.join(
                        OUTDIR,
                        "scores",
                        "processed",
                        "combined",
                        taxon,
                        f"scores_cmh-{tp}-{gr}-{taxon}-{focus_tp}.tsv",
                    )
                )
                        
    # Add CMH across time taxa targets
    if config["analysis"].get("use_cmh", True) and DATA_TYPE == "longitudinal":
        sample_entries = get_eligible_mags(tp, gr, "cmh_across_time")
        if sample_entries:  # Only proceed if there are eligible entries
            unique_groups = sorted(set([grp for mag, grp in sample_entries]))
            for grp in unique_groups:
                group_str = f"_{grp}"
                for taxon in tax_levels:
                    targets.append(
                        os.path.join(
                            OUTDIR,
                            "scores",
                            "processed",
                            "combined",
                            taxon,
                            f"scores_cmh_across_time-{tp}-{gr}{group_str}-{taxon}.tsv",
                        )
                    )
    
    # Add LMM across time taxa targets
    if config["analysis"].get("use_lmm", True) and DATA_TYPE == "longitudinal":
        sample_entries = get_eligible_mags(tp, gr, "lmm_across_time")
        if sample_entries:  # Only proceed if there are eligible entries
            unique_groups = sorted(set([grp for mag, grp in sample_entries]))
            for grp in unique_groups:
                group_str = f"_{grp}"
                for taxon in tax_levels:
                    targets.append(
                        os.path.join(
                            OUTDIR,
                            "scores",
                            "processed",
                            "combined",
                            taxon,
                            f"scores_lmm_across_time-{tp}-{gr}{group_str}-{taxon}.tsv",
                        )
                    )
                                
    return targets


def generate_gene_scores_targets(tp, gr):
    """
    Generate gene-level score targets.

    Only produces targets when ``use_gene_scores`` is True and
    ``allele_analysis_only`` is False in the config.
    Covers all enabled test types (standard, LMM, CMH, across-time variants).
    """
    if config["analysis"].get("allele_analysis_only", False):
        return []
    targets = []

    if not config["analysis"].get("use_gene_scores", False):
        return targets

    base_subdir = f"gene_scores_{tp}-{gr}"

    def _gene_score_path(prefix):
        return os.path.join(
            OUTDIR, "scores", "processed", base_subdir,
            f"{prefix}_gene_scores_combined.tsv",
        )
    if config["analysis"].get("use_significance_tests", True):
        for test_type in ["two_sample_unpaired", "two_sample_paired"]:
            mags = get_eligible_mags(tp, gr, test_type)
            for mag in mags:
                targets.append(_gene_score_path(f"{mag}_{test_type}"))

        # Single-sample (longitudinal only)
        if DATA_TYPE == "longitudinal":
            for mag, grp in get_eligible_mags(tp, gr, "single_sample"):
                targets.append(_gene_score_path(f"{mag}_single_sample_{grp}"))

    if config["analysis"].get("use_lmm", True):
        for mag in get_eligible_mags(tp, gr, "lmm"):
            targets.append(_gene_score_path(f"{mag}_lmm"))

        # LMM across time (longitudinal only)
        if DATA_TYPE == "longitudinal":
            for mag, grp in get_eligible_mags(tp, gr, "lmm_across_time"):
                targets.append(_gene_score_path(f"{mag}_lmm_across_time_{grp}"))

    if config["analysis"].get("use_cmh", True):
        # Regular CMH — uses cmh_gene_scores rule (focus_tp in prefix)
        mags = get_eligible_mags(tp, gr, "cmh")
        if mags:
            focus_tp = focus_timepoints.get(tp)
            if not focus_tp:
                raise ValueError(f"No focus timepoint defined for {tp}.")
            for mag in mags:
                targets.append(_gene_score_path(f"{mag}_cmh_{focus_tp}"))

        # CMH across time (longitudinal only)
        if DATA_TYPE == "longitudinal":
            for mag, grp in get_eligible_mags(tp, gr, "cmh_across_time"):
                targets.append(_gene_score_path(f"{mag}_cmh_across_time_{grp}"))

    return targets


def generate_outlier_gene_targets(tp, gr):
    if config["analysis"].get("allele_analysis_only", False):
        return []
    targets = []

    if not config["analysis"].get("use_outlier_detection", False):
        return targets

    # Add significance test outlier targets if enabled
    if config["analysis"].get("use_significance_tests", True):
        for test_type in ["two_sample_unpaired", "two_sample_paired"]:
            group_str = ""  # no group marker for two-sample tests
            mags = get_eligible_mags(tp, gr, test_type)
            for mag in mags:
                prefix = f"{mag}_{test_type}{group_str}"
                base_dir = os.path.join(
                    OUTDIR,
                    "outlier_genes",
                    f"{tp}-{gr}",
                )
                targets.append(
                    os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
                )
        # Only include single sample targets if data_type is longitudinal
        if DATA_TYPE == "longitudinal":
            sample_entries = get_eligible_mags(tp, gr, "single_sample")
            for mag, grp in sample_entries:
                group_str = f"_{grp}"
                prefix = f"{mag}_single_sample{group_str}"
                base_dir = os.path.join(
                    OUTDIR,
                    "outlier_genes",
                    f"{tp}-{gr}",
                )
                targets.append(
                    os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
                )
                        
    # Add LMM outlier targets if enabled
    if config["analysis"].get("use_lmm", True):
        group_str = ""  # no group marker for LMM
        mags = get_eligible_mags(tp, gr, "lmm")
        for mag in mags:
            prefix = f"{mag}_lmm{group_str}"
            base_dir = os.path.join(
                OUTDIR,
                "outlier_genes",
                f"{tp}-{gr}",
            )
            targets.append(
                os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
            )
    
    # Add CMH outlier targets if enabled
    if config["analysis"].get("use_cmh", True):
        # Use CMH eligibility
        mags = get_eligible_mags(tp, gr, "cmh")
        # Get the focus timepoint from our global mapping
        focus_tp = focus_timepoints.get(tp)
        if not focus_tp:
            raise ValueError(f"No focus timepoint defined for {tp}.")
        for mag in mags:
            prefix = f"{mag}_cmh_{focus_tp}"
            base_dir = os.path.join(
                OUTDIR,
                "outlier_genes",
                f"{tp}-{gr}",
            )
            targets.append(
                os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
            )

    # Add LMM across time outlier targets if enabled
    if config["analysis"].get("use_lmm", True) and DATA_TYPE == "longitudinal":
        # Get individual groups for across_time analysis
        sample_entries = get_eligible_mags(tp, gr, "lmm_across_time")
        for mag, grp in sample_entries:
            group_str = f"_{grp}"
            prefix = f"{mag}_lmm_across_time{group_str}"
            base_dir = os.path.join(
                OUTDIR,
                "outlier_genes",
                f"{tp}-{gr}",
            )
            targets.append(
                os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
            )
    # Add CMH across time outlier targets if enabled
    if config["analysis"].get("use_cmh", True) and DATA_TYPE == "longitudinal":
        # Get individual groups for across_time analysis
        sample_entries = get_eligible_mags(tp, gr, "cmh_across_time")
        for mag, grp in sample_entries:
            group_str = f"_{grp}"
            prefix = f"{mag}_cmh_across_time{group_str}"
            base_dir = os.path.join(
                OUTDIR,
                "outlier_genes",
                f"{tp}-{gr}",
            )
            targets.append(
                os.path.join(base_dir, f"{prefix}_outlier_genes.tsv")
            )
                    
    return targets


def _get_rc_region_types():
    """Return list of region types expected from the configured mode.

    Used both in dynamic_targets and in regional_contrast.smk rules.
    Defined here (included before regional_contrast.smk) so it is available
    during DAG construction in generate_regional_contrast_targets.
    """
    mode = config.get("regional_contrast", {}).get("mode", "both")
    if mode == "gene":
        return ["gene"]
    elif mode == "window":
        return ["window"]
    else:  # "both"
        return ["gene", "window"]


def generate_regional_contrast_targets(tp, gr):
    """Generate regional contrast output targets for a timepoint-group combination.

    Returns one per-host (``.tsv.gz``) and one summary (``.tsv``) path per
    eligible MAG.  Only generates targets for longitudinal data (the rule
    requires allele-frequency-change columns that do not exist in single-
    timepoint output).

    Parameters:
        tp: Timepoint label (e.g., "pre_post")
        gr: Groups label (e.g., "treatment_control")

    Returns:
        list: Expected output file paths (empty for single-timepoint data or
              when ``use_regional_contrast`` is disabled in config).
    """
    targets = []

    # Regional contrast requires longitudinal mean-change data
    if DATA_TYPE != "longitudinal":
        return targets

    # Honour the opt-out flag; defaults to True (run by default)
    if not config["analysis"].get("use_regional_contrast", True):
        return targets

    # Use the same QC-eligibility set as allele analysis (one output per MAG)
    eligible_mags = _get_mags_by_eligibility(tp, gr, eligibility_type="all")

    base_dir = os.path.join(
        OUTDIR, "regional_contrast", f"regional_contrast_{tp}-{gr}"
    )
    # Determine which region types will be produced based on the configured mode
    region_types = _get_rc_region_types()
    for mag in eligible_mags:
        targets.append(
            os.path.join(base_dir, f"{mag}_regional_contrast_per_host_region.tsv.gz")
        )
        targets.append(
            os.path.join(base_dir, f"{mag}_regional_contrast_region_summary.tsv")
        )
        # Scoring targets are split by region_type (gene vs window) so that
        # each type uses its own independent denominator.
        score_base_dir = os.path.join(
            OUTDIR,
            "regional_contrast",
            "scores",
            f"regional_contrast_scores_{tp}-{gr}",
        )
        for rt in region_types:
            targets.append(
                os.path.join(
                    score_base_dir, f"{mag}_regional_contrast_{rt}_scores.tsv"
                )
            )

    # Add combined scores targets — one per region_type per timepoint-group combination
    if eligible_mags:
        score_base_dir = os.path.join(
            OUTDIR,
            "regional_contrast",
            "scores",
            f"regional_contrast_scores_{tp}-{gr}",
        )
        for rt in region_types:
            targets.append(
                os.path.join(
                    score_base_dir,
                    f"combined_regional_contrast_{rt}_scores.tsv",
                )
            )

        # Add cross-MAG FDR summary targets — one per region_type (gene / window).
        # Always generate summaries alongside regional contrast analysis.
        for rt in region_types:
            targets.append(
                os.path.join(
                    base_dir,
                    f"region_contrast_summary_{rt}_region_summary.tsv",
                )
            )

    return targets


def generate_dnds_analysis_targets(tp, gr):
    """
    Generate dN/dS analysis targets, which are now directories, one for each subject.
    
    This function checks MAG eligibility before generating targets. If no MAGs
    are eligible for the test type used by dN/dS analysis, no targets are generated.
    This prevents downstream rules (like p_value_summary) from running when
    there's no data to process.
    
    Triggers the preprocessing eligibility checkpoint to ensure the eligibility
    file exists and is up-to-date before checking eligibility.
    """
    targets = []

    # dN/dS is only applicable to longitudinal data.
    if DATA_TYPE != "longitudinal":
        return targets

    # Honour the opt-out flag; defaults to True (run by default)
    if not config["analysis"].get("use_dnds", True):
        return targets

    # Map dN/dS test type to base eligibility test type using shared helper.
    eligibility_test_type = get_base_test_type(DN_DS_TEST_TYPE)

    # No defensive ``checkpoints.preprocessing_eligibility_*.get()`` call here:
    # ``get_eligible_mags`` -> ``get_mags_by_preprocessing_eligibility`` performs
    # that gating itself via ``.output.out_fPath``.  See shared/common.smk.
    eligible_mags = get_eligible_mags(tp, gr, eligibility_test_type)
    if not eligible_mags:
        # No eligible MAGs - don't generate any dN/dS targets
        return targets

    # Get subject pairs to determine how many directories to expect.
    subject_pairs = parse_metadata_for_timepoint_pairs(tp, gr)
    subjects = [str(subject_id) for subject_id, _, _ in subject_pairs]

    # A directory is created for each subject; we target its sentinel so
    # Snakemake's notion of "done" matches a successful subprocess exit
    # rather than the directory simply existing.
    for subject in subjects:
        targets.append(dnds_sentinel(tp, gr, subject))
    return targets

# Test the following functions before uncommenting. They are not all 100% up to date and might need modification. Be careful.
"""
def get_two_sample_targets(test_type):
    # Only generate targets if use_significance_tests is enabled
    if not config["analysis"].get("use_significance_tests", True):
        return []
        
    # Define subdirectory and file suffix based on the test type.
    if test_type == "two_sample_unpaired":
        subdir = "two_sample_unpaired"
        suffix = "_two_sample_unpaired.tsv.gz"
    elif test_type == "two_sample_paired":
        subdir = "two_sample_paired"
        suffix = "_two_sample_paired.tsv.gz"
    else:
        raise ValueError("test_type must be either 'unpaired' or 'paired'")
    targets = []
    for tp in timepoints_labels:
        for gr in groups_labels:
            for mag in get_mags_by_eligibility(tp, gr, eligibility_type=test_type):
                targets.append(
                    os.path.join(
                        OUTDIR,
                        "significance_tests",
                        f"{subdir}_{tp}-{gr}",
                        f"{mag}{suffix}",
                    )
                )
    return targets

def get_single_sample_targets():
    # Only generate targets if use_significance_tests is enabled
    if not config["analysis"].get("use_significance_tests", True):
        return []
        
    targets = []
    for tp in timepoints_labels:
        for gr in groups_labels:
            for mag, group in get_single_sample_entries(tp, gr):
                targets.append(
                    os.path.join(
                        OUTDIR,
                        "significance_tests",
                        f"single_sample_{tp}-{gr}",
                        f"{mag}_single_sample_{group}.tsv.gz",
                    )
                )
    return targets

def get_lmm_targets():
    # Only generate targets if use_lmm is enabled
    if not config["analysis"].get("use_lmm", True):
        return []
        
    targets = []
    for tp in timepoints_labels:
        for gr in groups_labels:
            for mag in get_mags_by_eligibility(tp, gr):
                targets.append(
                    os.path.join(
                        OUTDIR,
                        "significance_tests",
                        f"lmm_{tp}-{gr}",
                        f"{mag}_lmm.tsv.gz",
                    )
                )
    return targets

    


def get_gene_scores_targets():
    targets = []
    if config["analysis"].get("use_significance_tests", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                for test_type in ["two_sample_unpaired", "two_sample_paired"]:

                    # Only generate targets if there are eligible MAGs
                    mags = get_mags_by_eligibility(tp, gr, eligibility_type=test_type)
                    if mags:  # Only proceed if there are eligible MAGs
                        group_str = ""  # no group marker for two-sample tests
                        for mag in mags:
                            prefix = f"{mag}_{test_type}{group_str}"
                            base_dir = os.path.join(
                                OUTDIR,
                                "scores",
                                "processed",
                                f"gene_scores_{tp}-{gr}",
                            )
                            targets.extend(
                                [
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_combined.tsv"
                                    ),
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_individual.tsv"
                                    ),
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_overlapping.tsv"
                                    ),
                                ]
                            )
                # Only include single sample targets if data_type is longitudinal
                if DATA_TYPE == "longitudinal":
                    sample_entries = get_single_sample_entries(tp, gr)
                    if sample_entries:  # Only proceed if there are eligible MAGs
                        for mag, grp in sample_entries:
                            group_str = f"_{grp}"
                            prefix = f"{mag}_single_sample{group_str}"
                            base_dir = os.path.join(
                                OUTDIR,
                                "scores",
                                "processed",
                                f"gene_scores_{tp}-{gr}",
                            )
                            targets.extend(
                                [
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_combined.tsv"
                                    ),
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_individual.tsv"
                                    ),
                                    os.path.join(
                                        base_dir, f"{prefix}_gene_scores_overlapping.tsv"
                                    ),
                                ]
                            )
    return targets

def get_cmh_test_targets():
    targets = []
    if config["analysis"].get("use_cmh", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                # Only process timepoint combinations with two timepoints
                # CMH uses paired eligibility
                mags = get_mags_by_eligibility(tp, gr, eligibility_type="cmh")
                if mags:  # Only proceed if there are eligible MAGs
                    for mag in mags:
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "significance_tests",
                                f"cmh_{tp}-{gr}",
                                f"{mag}_cmh.tsv.gz",
                            )
                        )
    return targets



def get_significance_scores_targets():
    targets = []
    if config["analysis"].get("use_significance_tests", True):
        # For two-sample tests (unpaired and paired)
        for tp in timepoints_labels:
            for gr in groups_labels:
                for test_type in ["two_sample_unpaired", "two_sample_paired"]:
                    mags = get_mags_by_eligibility(tp, gr, eligibility_type=test_type)
                    if mags:  # Only proceed if there are eligible MAGs
                        for mag in mags:
                            # Using group_str for non-CMH test types
                            group_str = ""  # no group marker for two-sample tests
                            targets.append(
                                os.path.join(
                                    OUTDIR,
                                    "scores",
                                    "intermediate",
                                    f"MAG_scores_{tp}-{gr}",
                                    f"{mag}_score_{test_type}{group_str}.tsv",
                                )
                            )
        # For single-sample tests - only include if data_type is longitudinal
        if DATA_TYPE == "longitudinal":
            for tp in timepoints_labels:
                for gr in groups_labels:
                    sample_entries = get_single_sample_entries(tp, gr)
                    for mag, group in sample_entries:
                        # Using group_str for single-sample test type
                        group_str = f"_{group}"
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "scores",
                                "intermediate",
                                f"MAG_scores_{tp}-{gr}",
                                f"{mag}_score_single_sample{group_str}.tsv",
                            )
                        )
    # Add LMM scores
    if config["analysis"].get("use_lmm", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                group_str = ""  # no group marker for LMM
                # Use the unpaired eligibility type for LMM
                mags = get_mags_by_eligibility(tp, gr, eligibility_type="lmm")
                if mags:
                    for mag in mags:
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "scores",
                                "intermediate",
                                f"MAG_scores_{tp}-{gr}",
                                f"{mag}_score_lmm{group_str}.tsv",
                            )
                        )
    # Add CMH scores
    if config["analysis"].get("use_cmh", True):  # Updated to check for CMH scores 
        for tp in timepoints_labels:
            for gr in groups_labels:
                # Use the paired eligibility type for CMH
                mags = get_mags_by_eligibility(tp, gr, eligibility_type="cmh")
                if mags: 
                    # Get the focus timepoint from our global mapping
                    focus_tp = focus_timepoints.get(tp)
                    if not focus_tp:
                        raise ValueError(f"No focus timepoint defined for {tp}, skipping CMH score targets")
                    for mag in mags:
                        # Using focus for CMH test type, no group_str
                        # focus = f"_{focus_tp}"
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "scores",
                                "intermediate",
                                f"MAG_scores_{tp}-{gr}",
                                f"{mag}_score_cmh_{focus_tp}.tsv",
                            )
                        )
    return targets

def get_combined_scores_targets():
    targets = []
    # Two-sample tests: group_str is empty.
    if config["analysis"].get("use_significance_tests", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                for test_type in ["two_sample_unpaired", "two_sample_paired"]:
                    # Only generate targets if there are eligible MAGs for this test type
                    mags = get_mags_by_eligibility(tp, gr, eligibility_type=test_type)
                    if mags:  # Only proceed if there are eligible MAGs
                        group_str = ""  # no group marker for two-sample tests
                        targets.append(
                            os.path.join(
                                OUTDIR,
                                "scores",
                                "processed",
                                "combined",
                                f"scores_{test_type}-{tp}-{gr}{group_str}-MAGs.tsv",
                            )
                        )
        # Single-sample tests: group_str is "_" plus the sample group.
        # Only include if data_type is longitudinal
        if DATA_TYPE == "longitudinal":
            for tp in timepoints_labels:
                for gr in groups_labels:
                    # get_single_sample_entries returns (mag, group) pairs for a given timepoint and group.
                    sample_entries = get_single_sample_entries(tp, gr)
                    if sample_entries:  # Only proceed if there are eligible MAGs
                        unique_groups = sorted(set([grp for mag, grp in sample_entries]))
                        for grp in unique_groups:
                            group_str = f"_{grp}"
                            targets.append(
                                os.path.join(
                                    OUTDIR,
                                    "scores",
                                    "processed",
                                    "combined",
                                    f"scores_single_sample-{tp}-{gr}{group_str}-MAGs.tsv",
                                )
                            )
    # Add LMM targets if enabled
    if config["analysis"].get("use_lmm", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                # Only generate targets if there are eligible MAGs
                # Use the unpaired eligibility type for LMM
                mags = get_mags_by_eligibility(tp, gr, eligibility_type="lmm")
                if mags:  # Only proceed if there are eligible MAGs
                    group_str = ""  # no group marker for LMM
                    targets.append(
                        os.path.join(
                            OUTDIR,
                            "scores",
                            "processed",
                            "combined",
                            f"scores_lmm-{tp}-{gr}{group_str}-MAGs.tsv",
                        )
                    )
                    
    # Add CMH targets if enabled
    if config["analysis"].get("use_cmh", True):
        for tp in timepoints_labels:
            for gr in groups_labels:
                # Only process if we have eligible MAGs
                mags = get_mags_by_eligibility(tp, gr, eligibility_type="cmh")
                if mags:
                    # Get the focus timepoint from our global mapping
                    focus_tp = focus_timepoints.get(tp)
                    if not focus_tp:
                        raise ValueError(f"No focus timepoint defined for {tp}.")
                    targets.append(
                        os.path.join(
                            OUTDIR,
                            "scores",
                            "processed",
                            "combined",
                            f"scores_cmh-{tp}-{gr}-MAGs-{focus_tp}.tsv",
                        )
                    )
    return targets
"""

def get_tested_mags():
    """The MAG universe for pairwise ANI and everything downstream of it: every MAG
    that is eligible for at least one enabled test in at least one configured
    (timepoints, groups) comparison.

    ANI is group- and timepoint-blind (one job per MAG covers every comparison),
    so the union across comparisons and test types is the right set: a MAG tested
    only in fat-vs-control still gets its ANI table.  The membership comes from the
    SAME readers the statistics targets use (get_eligible_mags -> preprocessing
    eligibility when preprocessing is on, QC eligibility otherwise), so "tested"
    here can never drift from "tested" there.  Because those readers access the
    checkpoints, this must only be called from checkpoint-aware places
    (get_final_pipeline_outputs, or a rule's input *function*), never at parse time.

    allele_analysis_only runs skip preprocessing entirely, so there the universe
    is the QC-eligible set for any test ("all").

    Example (a two-comparison run, pre_end and pre_post x fat_control,
    two-sample + single-sample tests, QC layer): 160 MAGs in the mapping, 78 and 79
    eligible for some test in the two comparisons, 83 in the union -> 83 ANI /
    turnover jobs instead of 160.
    """
    tested = set()
    for tp in timepoints_labels:
        for gr in groups_labels:
            if config["analysis"].get("allele_analysis_only", False):
                tested.update(_get_mags_by_eligibility(tp, gr, eligibility_type="all"))
                continue
            for test_type in get_enabled_test_types():
                entries = get_eligible_mags(tp, gr, test_type)
                # Within-group types return (mag, group) tuples; the group is
                # irrelevant for a group-blind universe, keep only the id.
                tested.update(e[0] if isinstance(e, tuple) else e for e in entries)
    return sorted(tested)


def generate_pairwise_ani_targets():
    """One pairwise-ANI target per TESTED MAG, gated by the use_pairwise_ani flag.

    Not scoped per (timepoints, groups): the rule is group- and timepoint-
    independent, so this is called once, AFTER the combination loop in
    get_final_pipeline_outputs.  The MAG universe is get_tested_mags(), which
    reads the eligibility checkpoints -- so this generator is checkpoint-
    dependent and must not be eager-listed.
    """
    if not config["analysis"].get("use_pairwise_ani", False):
        return []
    return [get_pairwise_ani_output_path(mag_wildcard=mag) for mag in get_tested_mags()]


def generate_strain_turnover_targets():
    """One turnover table per tested MAG (same universe as pairwise ANI), gated by use_strain_turnover.

    Checkpoint-dependent through get_tested_mags(), like the ANI targets.
    Refuses configurations that cannot work rather than producing empty files:
    it needs the ANI table and two-timepoint transitions.
    """
    if not config["analysis"].get("use_strain_turnover", False):
        return []
    if not config["analysis"].get("use_pairwise_ani", False):
        raise ValueError("use_strain_turnover requires use_pairwise_ani (it reads the pairwise table)")
    if DATA_TYPE != "longitudinal":
        raise ValueError("use_strain_turnover needs longitudinal data (transitions come from timepoints_combinations)")
    return [get_strain_turnover_output_path(mag_wildcard=mag) for mag in get_tested_mags()]


def generate_replacement_classification_targets():
    """The single classification table, once every MAG's turnover table exists."""
    if not config["analysis"].get("use_strain_turnover", False):
        return []
    return [get_replacement_classification_path()]


def generate_baseline_presence_targets(tp, gr):
    """Baseline-presence outputs for one comparison, only where its summary file will exist.

    Piggybacks on generate_p_value_summary_targets: if that generator does not
    emit the configured family's summary for this (tp, gr) -- test disabled, or
    no eligible MAG -- there is nothing to annotate and no target is made.
    """
    if not config["analysis"].get("use_baseline_presence", False):
        return []
    if config["analysis"].get("allele_analysis_only", False):
        return []
    wanted = os.path.join(
        OUTDIR, "p_value_summary", f"{tp}-{gr}",
        f"p_value_summary_{BASELINE_PRESENCE_FAMILY}_{tp}-{gr}.tsv",
    )
    if wanted not in generate_p_value_summary_targets(tp, gr):
        return []
    stem = get_baseline_presence_stem(timepoints=tp, groups=gr)
    return [f"{stem}.tsv.gz", f"{stem}_summary.tsv"]
