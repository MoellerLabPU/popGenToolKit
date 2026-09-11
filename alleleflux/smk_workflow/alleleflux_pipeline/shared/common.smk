"""Common Snakemake utilities and configuration.

This module provides shared configuration, helper functions, and wildcard constraints
for the AlleleFlux Snakemake pipeline. It handles:
- Global configuration parsing (data type, output directory, timepoints, groups)
- Resource management (memory parsing, per-rule overrides)
- MAG eligibility functions for QC and preprocessing stages
- Sample metadata parsing for longitudinal analysis
- Wildcard constraints for consistent rule matching
"""

import os
import pandas as pd
from glob import glob
from collections import defaultdict
from pathlib import Path
from snakemake.logging import logger

# Load the configuration file
# configfile: os.path.join(workflow.basedir, "config.yml")


# =============================================================================
# Resource Management
# =============================================================================

def parse_mem(mem_value):
    """
    Convert memory string to MB for Snakemake resources.
    
    Supports formats: "8G", "8GB", "8192M", "8192MB", "8192" (assumes MB)
    Case-insensitive. Uses binary units (1 GB = 1024 MB).
    
    Args:
        mem_value: Memory value as string (e.g., "8G") or int (MB)
    
    Returns:
        int: Memory in MB
    
    Examples:
        >>> parse_mem("8G")
        8192
        >>> parse_mem("100GB")
        102400
        >>> parse_mem("8192M")
        8192
        >>> parse_mem(8192)
        8192
    """
    if isinstance(mem_value, (int, float)):
        return int(mem_value)
    
    mem_str = str(mem_value).strip().upper()
    
    # Remove 'B' suffix if present (e.g., "8GB" -> "8G")
    if mem_str.endswith("B"):
        mem_str = mem_str[:-1]
    
    if mem_str.endswith("G"):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith("M"):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith("K"):
        return max(1, int(float(mem_str[:-1]) / 1024))
    else:
        # Assume MB if no unit
        return int(float(mem_str))


def parse_time_to_minutes(time_str):
    """
    Convert a time string (HH:MM:SS or D-HH:MM:SS) to total minutes.
    
    Args:
        time_str: Time string in HH:MM:SS or D-HH:MM:SS format
    
    Returns:
        int: Total minutes
    
    Examples:
        >>> parse_time_to_minutes("24:00:00")
        1440
        >>> parse_time_to_minutes("4:00:00")
        240
        >>> parse_time_to_minutes("1-00:00:00")
        1440
        >>> parse_time_to_minutes("0:30:00")
        30
    """
    time_str = str(time_str).strip()
    
    # Handle D-HH:MM:SS format
    days = 0
    if "-" in time_str:
        day_part, time_str = time_str.split("-", 1)
        days = int(day_part)
    
    parts = time_str.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        hours, minutes = 0, int(parts[0])
        seconds = int(parts[1])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS or D-HH:MM:SS.")
    
    return days * 24 * 60 + hours * 60 + minutes + (1 if seconds > 0 else 0)


def minutes_to_time_str(total_minutes):
    """
    Convert total minutes to HH:MM:SS or D-HH:MM:SS format.
    
    Uses D-HH:MM:SS format only when days >= 1 for cleaner output.
    
    Args:
        total_minutes: Total minutes as int
    
    Returns:
        str: Time string in HH:MM:SS or D-HH:MM:SS format
    
    Examples:
        >>> minutes_to_time_str(240)
        '4:00:00'
        >>> minutes_to_time_str(1440)
        '1-00:00:00'
        >>> minutes_to_time_str(1500)
        '1-01:00:00'
    """
    total_minutes = max(1, int(total_minutes))
    days = total_minutes // (24 * 60)
    remaining = total_minutes % (24 * 60)
    hours = remaining // 60
    minutes = remaining % 60
    
    if days > 0:
        return f"{days}-{hours:02d}:{minutes:02d}:00"
    else:
        return f"{hours}:{minutes:02d}:00"


def get_resource(rule_name, resource_type, default=None):
    """
    Get resource value for a rule, checking for per-rule override first.
    
    This enables the 'escape hatch' pattern where power users can override
    resources for specific rules via the resources_override config section,
    while most users just use the flat defaults.
    
    Args:
        rule_name: Name of the Snakemake rule (e.g., "profile", "qc", "statistical_tests")
        resource_type: Type of resource ("threads_per_job", "mem_per_job", "time")
        default: Default value if not specified anywhere (uses config default if None)
    
    Returns:
        Resource value (parsed if memory)
    
    Example:
        # In a rule:
        resources:
            mem_mb=get_resource("profile", "mem_per_job"),
            time=get_resource("profile", "time")
    """
    # Check for per-rule override first
    overrides = config.get("resources_override", {}).get(rule_name, {})
    if resource_type in overrides:
        value = overrides[resource_type]
    elif default is not None:
        value = default
    else:
        # Get from main resources section
        value = config.get("resources", {}).get(resource_type)
    
    # Parse memory values to MB
    if resource_type == "mem_per_job" and value is not None:
        return parse_mem(value)
    
    return value


# -- Internal helpers for retry / resource stepping --

def _get_retries_for_rule(rule_name=None):
    """Get retry count, checking per-rule override first, then global default."""
    if rule_name:
        overrides = config.get("resources_override", {}).get(rule_name, {})
        if "retries" in overrides:
            return int(overrides["retries"])
    return int(config.get("resources", {}).get("retries", 2))


def _get_step_value(rule_name, key, global_default):
    """Get a step value (mem_step, time_step) with per-rule override support."""
    if rule_name:
        overrides = config.get("resources_override", {}).get(rule_name, {})
        if key in overrides:
            return overrides[key]
    return config.get("resources", {}).get(key, global_default)


# -- Public resource functions used by rules --

def get_threads(rule_name=None):
    """Get threads_per_job, optionally checking for rule-specific override."""
    if rule_name:
        return get_resource(rule_name, "threads_per_job")
    return config.get("resources", {}).get("threads_per_job", 1)


def get_retries(rule_name=None):
    """
    Get retry count for a rule, for use in the rule-level retries: directive.
    
    Checks per-rule override first, then falls back to global resources.retries.
    Default is 2 retries if not specified.
    
    Args:
        rule_name: Name of the Snakemake rule (optional)
    
    Returns:
        int: Number of retries
    """
    return _get_retries_for_rule(rule_name)


def get_mem_mb(rule_name=None):
    """
    Get memory in MB, with automatic scaling on retry when mem_step is configured.
    
    When retries > 0 and mem_step is set, returns a callable
    ``lambda wildcards, attempt: base + (attempt-1) * step`` so that
    Snakemake allocates more memory on each retry.
    When no stepping is configured, returns a static int (backward-compatible).
    
    Args:
        rule_name: Name of the Snakemake rule (optional)
    
    Returns:
        int or callable: Memory in MB (static or attempt-aware)
    """
    if rule_name:
        base_mb = get_resource(rule_name, "mem_per_job")
    else:
        base_mb = parse_mem(config.get("resources", {}).get("mem_per_job", "8G"))
    
    retries = _get_retries_for_rule(rule_name)
    step_raw = _get_step_value(rule_name, "mem_step", None)
    
    if retries > 0 and step_raw:
        step_mb = parse_mem(step_raw)
        if step_mb > 0:
            return lambda wildcards, attempt: base_mb + (attempt - 1) * step_mb
    
    return base_mb


def get_time(rule_name=None):
    """
    Get wall time, with automatic scaling on retry when time_step is configured.
    
    When retries > 0 and time_step is set, returns a callable
    ``lambda wildcards, attempt: base_time + (attempt-1) * step`` so that
    Snakemake allocates more time on each retry.
    When no stepping is configured, returns a static string (backward-compatible).
    
    Args:
        rule_name: Name of the Snakemake rule (optional)
    
    Returns:
        str or callable: Wall time (static or attempt-aware)
    """
    if rule_name:
        base_time = get_resource(rule_name, "time")
    else:
        base_time = config.get("resources", {}).get("time", "24:00:00")
    
    retries = _get_retries_for_rule(rule_name)
    step_raw = _get_step_value(rule_name, "time_step", None)
    
    if retries > 0 and step_raw:
        base_mins = parse_time_to_minutes(base_time)
        step_mins = parse_time_to_minutes(step_raw)
        if step_mins > 0:
            return lambda wildcards, attempt: minutes_to_time_str(
                base_mins + (attempt - 1) * step_mins
            )

    return base_time


def get_runtime(rule_name=None):
    """Wall time in minutes (int), with retry scaling. Mirrors :func:`get_time`.

    Used by the native ``snakemake-executor-plugin-slurm`` profile, which expects
    ``runtime`` (integer minutes) — the canonical SLURM-plugin name for what
    cluster-generic calls ``time`` (HH:MM:SS string).

    Rules set BOTH ``time=get_time(...)`` (consumed by cluster-generic via
    ``--time={resources.time}`` in its sbatch template) AND
    ``runtime=get_runtime(...)`` (consumed by plugin-slurm). The unused
    directive is ignored by the active executor; no per-profile rule files
    needed.
    """
    if rule_name:
        base_time = get_resource(rule_name, "time")
    else:
        base_time = config.get("resources", {}).get("time", "24:00:00")

    base_mins = parse_time_to_minutes(base_time)
    retries = _get_retries_for_rule(rule_name)
    step_raw = _get_step_value(rule_name, "time_step", None)

    if retries > 0 and step_raw:
        step_mins = parse_time_to_minutes(step_raw)
        if step_mins > 0:
            return lambda wildcards, attempt: base_mins + (attempt - 1) * step_mins

    return base_mins


# =============================================================================
# Global Constants
# =============================================================================

# Taxonomy levels for aggregation (order matters - from broad to specific)
TAXONOMY_LEVELS = ["phylum", "class", "order", "family", "genus", "species"]

# Statistical test types - centralized for consistency
BETWEEN_GROUP_TEST_TYPES = ["two_sample_unpaired", "two_sample_paired", "lmm", "cmh"]
WITHIN_GROUP_TEST_TYPES = ["single_sample", "lmm_across_time", "cmh_across_time"]
ALL_TEST_TYPES = BETWEEN_GROUP_TEST_TYPES + WITHIN_GROUP_TEST_TYPES

# =============================================================================
# Global Configuration
# =============================================================================

# Define the global data_type variable to be used across all Snakemake files
DATA_TYPE = config["analysis"]["data_type"]
OUTDIR = config["output"]["root_dir"]
# Read defensively so a config with ``use_dnds: False`` (and no ``dnds:`` section)
# still parses.  This value is only consumed when dN/dS targets are generated,
# i.e. when ``use_dnds`` is True — in which case the ``dnds:`` section is expected.
DN_DS_TEST_TYPE = config.get("dnds", {}).get("dn_ds_test_type", "two_sample_paired_tTest")


def get_base_test_type(test_type):
    """
    Convert a specific test type to its base eligibility test type.
    
    Maps specific test variants (e.g., 'two_sample_paired_tTest') to their
    base types (e.g., 'two_sample_paired') for eligibility checking.
    
    Parameters:
        test_type: The specific test type string from config
    
    Returns:
        The base test type string for eligibility lookup
    
    Raises:
        ValueError: If the test type is not recognized
    """
    if test_type in ["two_sample_unpaired_tTest", "two_sample_unpaired_MannWhitney", 
                     "two_sample_unpaired_tTest_abs", "two_sample_unpaired_MannWhitney_abs"]:
        return "two_sample_unpaired"
    elif test_type in ["two_sample_paired_tTest", "two_sample_paired_Wilcoxon", 
                       "two_sample_paired_tTest_abs", "two_sample_paired_Wilcoxon_abs"]:
        return "two_sample_paired"
    elif test_type in ["single_sample_tTest", "single_sample_Wilcoxon"]:
        return "single_sample"
    elif test_type in ["lmm", "lmm_abs", "lmm_across_time", "cmh", "cmh_across_time"]:
        return test_type
    else:
        raise ValueError(f"Unsupported test type: {test_type}")


if DATA_TYPE == "single":
    OUTDIR = os.path.join(OUTDIR, "single_timepoint")
elif DATA_TYPE == "longitudinal":
    OUTDIR = os.path.join(OUTDIR, "longitudinal")

# =============================================================================
# Artifact reuse for permuted / null runs (reuse_from)
# =============================================================================
# A permuted (null) run reuses the real run's *group-independent* expensive
# artifacts — profiles, QC breadth/coverage, and the allele-frequency cache —
# and only recomputes the group-dependent tail (eligibility -> Stage 2 -> stats
# -> scores) on relabeled data.  ``input.reuse_from`` points at the real run's
# data-type output dir (e.g. ``.../alleleflux_output_1/longitudinal``).  When
# set, the cache/QC/metadata path helpers below resolve to that external dir, so
# the profiling/metadata/QC/cache rules drop out of the DAG exactly the way
# USE_EXISTING_PROFILES already drops the profiling rule.  When unset,
# REUSE_DIR == OUTDIR, so non-reuse behaviour is byte-for-byte unchanged.
REUSE_FROM = config["input"].get("reuse_from", "")
USE_REUSE = bool(REUSE_FROM)
if USE_REUSE and not os.path.isdir(REUSE_FROM):
    raise ValueError(
        f"input.reuse_from is set but is not a directory: {REUSE_FROM}"
    )
REUSE_DIR = REUSE_FROM if USE_REUSE else OUTDIR
if USE_REUSE:
    logger.info(f"Reusing profiles/QC/allele-freq cache from: {REUSE_DIR}")


# Permutation mode detection
# ---------------------------
# A permuted (null) run can be in one of three modes, distinguished by config:
#
#   * Orchestrate : permutation.enabled, no ``seed``, no ``permuted_metadata_dir``
#                   -> run_permutations() (workflow.py) fans out one leaf per
#                      seed; the Snakemake pipeline never sees this top config.
#   * Generate    : permutation.seed (an int) is set -> the ``permute_metadata``
#                   rule builds the per-pair sheets under OUTDIR/permuted_metadata
#                   and the group-dependent rules depend on them.
#   * BYO         : permutation.permuted_metadata_dir is set (no ``seed``) ->
#                   read user-supplied sheets directly; the permute_metadata rule
#                   is NOT wired into the DAG.
PERMUTATION = config.get("permutation", {})
PERMUTATION_SEED = PERMUTATION.get("seed")
PERMUTED_METADATA_DIR = PERMUTATION.get("permuted_metadata_dir", "")
GENERATE_PERMUTED = bool(PERMUTATION.get("enabled") and PERMUTATION_SEED is not None)
# Where the per-pair sheets live: generated under OUTDIR, or the BYO directory.
PERMUTED_DIR = (
    os.path.join(OUTDIR, "permuted_metadata")
    if GENERATE_PERMUTED
    else PERMUTED_METADATA_DIR
)


def permuted_metadata_path(groups_label):
    """Absolute path to a comparison pair's permuted sheet (generate or BYO).

    Named ``permuted_metadata_<groups_label>.tsv`` (e.g.
    ``permuted_metadata_1D_AL.tsv``) under :data:`PERMUTED_DIR`.
    """
    return os.path.join(PERMUTED_DIR, f"permuted_metadata_{groups_label}.tsv")


def permuted_metadata_flag(groups_label):
    """Return the ``--permuted_metadata`` CLI flag for a group pair, or ``""``.

    Permuted (null) runs relabel the reused cache/QC **in memory** from a
    per-comparison-pair permuted metadata sheet (see
    ``alleleflux-permute-metadata`` and ``relabel_groups_from_metadata``).  In
    generate mode the sheet is produced by the ``permute_metadata`` rule; in BYO
    mode it is the user-supplied file.  Group-dependent rules pass this flag to
    their CLI so each consumer re-derives the permuted labels at load time.
    Returns ``""`` (no flag) when permutation is disabled or no sheet directory
    is configured, so non-permuted runs are unaffected.
    """
    if not PERMUTATION.get("enabled", False):
        return ""
    if not PERMUTED_DIR:
        return ""
    return f"--permuted_metadata {permuted_metadata_path(groups_label)}"


def permuted_metadata_input(groups_label):
    """Return the sheet path as a rule *dependency* — ONLY in generate mode.

    In generate mode the ``permute_metadata`` rule produces the sheet, so every
    consuming rule must declare it as an ``input`` (else Snakemake could
    schedule the consumer before the sheet exists).  In BYO mode the sheet
    already exists on disk and in non-permuted mode there is no sheet, so this
    returns ``[]`` and those runs gain no dependency — behaviour is byte-for-byte
    unchanged.
    """
    return [permuted_metadata_path(groups_label)] if GENERATE_PERMUTED else []

# Profile reuse configuration
# If profiles_path is specified and exists, use existing profiles instead of
# generating new ones.  A reuse_from run defaults profiles to <reuse_from>/profiles
# when profiles_path is not explicitly set (explicit profiles_path still wins).
EXISTING_PROFILES_PATH = config["input"].get("profiles_path", "")
if not EXISTING_PROFILES_PATH and USE_REUSE:
    EXISTING_PROFILES_PATH = os.path.join(REUSE_FROM, "profiles")
USE_EXISTING_PROFILES = bool(EXISTING_PROFILES_PATH and os.path.isdir(EXISTING_PROFILES_PATH))
PROFILES_DIR = EXISTING_PROFILES_PATH if USE_EXISTING_PROFILES else os.path.join(OUTDIR, "profiles")

if USE_EXISTING_PROFILES:
    logger.info(f"Using existing profiles from: {EXISTING_PROFILES_PATH}")
else:
    logger.info(f"Profiles will be generated in: {PROFILES_DIR}")


# =============================================================================
# Sentinel file conventions
# =============================================================================
# Rules that historically used `directory()` outputs are switched to sentinel
# marker files. Snakemake's `directory()` semantics pre-delete the entire
# directory on every re-run, so an interrupted job leaves an empty directory
# which silently invalidates everything downstream (cf. bug where missing
# inputMetadata_{tp} subdirs forced a full pipeline rebuild).
#
# A sentinel file is written ONLY at the very end of a successful job, so:
#   * Interrupted/failed runs leave no sentinel  -> Snakemake re-runs the rule
#   * Successful runs leave the sentinel         -> Snakemake skips the rule
#   * Existing partial outputs are NOT pre-deleted on re-run, so a fresh run
#     can resume cheaply if the script itself is idempotent.
#
# Helpers below build the canonical sentinel path for each rule's output dir.
SENTINEL_PROFILE = ".profile_done"
SENTINEL_METADATA = ".metadata_done"
SENTINEL_QC = ".qc_done"
SENTINEL_DNDS = ".dnds_done"


def profile_sentinel(sample):
    """Path to the sentinel marker for a single-sample profile directory."""
    return os.path.join(PROFILES_DIR, sample, SENTINEL_PROFILE)


def metadata_sentinel(timepoints):
    """Path to the sentinel marker for a per-timepoint inputMetadata directory.

    Resolves under REUSE_DIR so a reuse_from run reads the real run's existing
    sentinel (already on disk) and the generate_metadata rule drops from the DAG.
    """
    return os.path.join(
        REUSE_DIR, "inputMetadata", f"inputMetadata_{timepoints}", SENTINEL_METADATA
    )


def qc_sentinel(timepoints):
    """Path to the sentinel marker for a per-timepoint QC directory.

    Resolves under REUSE_DIR so a reuse_from run reads the real run's existing
    QC sentinel and the quality_control rule drops from the DAG.
    """
    return os.path.join(REUSE_DIR, "QC", f"QC_{timepoints}", SENTINEL_QC)


def dnds_sentinel(timepoints, groups, subject_id):
    """Path to the sentinel marker for a per-subject dN/dS output directory."""
    return os.path.join(
        OUTDIR,
        "dnds_analysis",
        f"{timepoints}-{groups}",
        str(subject_id),
        SENTINEL_DNDS,
    )

timepoints_labels = []
focus_timepoints = {}

for time_combo in config["analysis"]["timepoints_combinations"]:
    # Standardized format: all entries are dictionaries with "timepoint" key
    timepoint = time_combo["timepoint"]
    
    if len(timepoint) == 1 and DATA_TYPE == "single":
        # Single timepoint
        tp = timepoint[0]
        timepoints_labels.append(tp)
        # For single data type, the timepoint itself is the focus
        focus_timepoints[tp] = tp
    elif len(timepoint) == 2 and DATA_TYPE == "longitudinal":
        # Multiple timepoints (for longitudinal analysis)
        label = f"{timepoint[0]}_{timepoint[1]}"
        timepoints_labels.append(label)
        if "focus" in time_combo:
            # Ensure focus is one of the two timepoints
            if time_combo["focus"] in timepoint:
                focus_timepoints[label] = time_combo["focus"]
            else:
                raise ValueError(f"Invalid focus timepoint '{time_combo['focus']}' for combination '{label}'. Must be one of {timepoint}")
        else:
            # Default to second timepoint if focus not specified
            logger.warning(f"No focus specified for timepoint combination {label}. Using '{timepoint[1]}' as default.")
            focus_timepoints[label] = timepoint[1]

# Define valid focus timepoint values for each timepoint label
valid_focus_timepoints = {}
for tp in timepoints_labels:
    if "_" in tp and DATA_TYPE == "longitudinal":  # It's a two-timepoint combination with focus
        valid_focus_timepoints[tp] = tp.split("_")
    elif DATA_TYPE == "single":
        # For single timepoint, the timepoint itself is the focus
        valid_focus_timepoints[tp] = [tp]

groups_labels = [
    f"{gr['treatment']}_{gr['control']}" for gr in config["analysis"]["groups_combinations"]
]  # ["G1_G2", "G2_G4"]

# Flatten the list of group values from the groups_combinations config.
group_values = sorted(
    {gr["treatment"] for gr in config["analysis"]["groups_combinations"]}
    | {gr["control"] for gr in config["analysis"]["groups_combinations"]}
)
# Build the regex: optionally an underscore and one of the allowed group values.
group_str_regex = "(_({}))?".format("|".join(group_values))

# =============================================================================
# Per-timepoint cache helpers
# =============================================================================
# The ``compute_allele_freq_per_timepoint`` rule writes one Parquet cache file
# per (MAG, gr_combo, single timepoint).  Scoping the cache to gr_combo is
# required by the checkpoint architecture: each (tp_combo, gr_combo) resolves
# its eligibility_table checkpoint separately, so only the QC files for ONE
# combination are guaranteed to exist when the cache is first needed.  By
# keying the cache on gr_combo we can use a SINGLE canonical QC file (the
# first tp_combo in config order that contains this timepoint) as the input,
# which Snakemake resolves as a normal rule dependency — no cross-combination
# QC gathering needed.
#
# Deduplication is within each gr_combo, across tp_combos: all tp_combos that
# share the same (gr_combo, timepoint) reuse the same Parquet cache.  Cache files
# are NOT shared across gr_combos — each gr_combo builds its own set.
# For drido (15 tp_combos, 6 gr_combos, 8 unique timepoints):
#   Before: 15 tp_combos × 6 gr_combos × 2 timepoints = 180 profile-read operations
#   After:  8 unique timepoints × 6 gr_combos          =  48 cache-write jobs

# unique_timepoints: ordered list of individual timepoints, each appearing exactly once.
#
# timepoints_labels contains combo labels like "5mo_10mo", "5mo_16mo", "8mo_10mo".
# This loop splits each label into its constituent timepoints ("5mo", "10mo") and
# collects them in first-appearance order using _seen_tps as a visited set.
#
# Example (drido config, 15 tp_combos):
#   "5mo_10mo" → adds "5mo", "10mo"
#   "5mo_16mo" → "5mo" already seen; adds "16mo"
#   "8mo_10mo" → "10mo" already seen; adds "8mo"   ← 8mo appears last (Strategy C)
#   Result: ["5mo", "10mo", "16mo", "22mo", "28mo", "34mo", "40mo", "8mo"]
#
# This list becomes the {timepoint} wildcard constraint — the single-timepoint
# wildcard used in cache filenames, distinct from {timepoints} (the combo label).
unique_timepoints = []
_seen_tps = set()
for tp_label in timepoints_labels:
    if DATA_TYPE == "longitudinal":
        parts = tp_label.split("_")  # "5mo_10mo" → ["5mo", "10mo"]
    else:
        parts = [tp_label]           # single data: label is already one timepoint
    for tp in parts:
        if tp not in _seen_tps:
            _seen_tps.add(tp)
            unique_timepoints.append(tp)

# =============================================================================
# Cache Path Resolution Logic (2B Refactor)
# =============================================================================
# After the 2A/2B refactors, QC runs once per timepoints combination (group-independent).
# The cache is also group-independent: one Parquet file per (MAG, timepoint).
#
# Before 2B: 8 unique timepoints × 6 gr_combos = 48 cache-write jobs (DRIDO)
# After  2B: 8 unique timepoints × 1            =  8 cache-write jobs
#
# timepoint_to_canonical_tp: maps individual_timepoint → the FIRST tp_combo
# label in config order that contains that timepoint.
#
# Example (drido, 15 tp_combos):
#   "5mo_10mo" → "5mo"→"5mo_10mo", "10mo"→"5mo_10mo"
#   "5mo_16mo" → "5mo" already set; "16mo"→"5mo_16mo"
#   "10mo_16mo" → both already set → nothing added
#   Result: {"5mo": "5mo_10mo", "10mo": "5mo_10mo", "16mo": "5mo_16mo", ...}
timepoint_to_canonical_tp = {}
for tp_label in timepoints_labels:
    if DATA_TYPE == "longitudinal":
        parts = tp_label.split("_")  # "5mo_10mo" → ["5mo", "10mo"]
    else:
        parts = [tp_label]
    for tp in parts:
        if tp not in timepoint_to_canonical_tp:
            timepoint_to_canonical_tp[tp] = tp_label  # first in config order wins

# =============================================================================
# Wildcard Constraints
# =============================================================================

wildcard_constraints:
    groups=f"({'|'.join(groups_labels)})",
    timepoints=f"({'|'.join(timepoints_labels)})",
    timepoint=f"({'|'.join(unique_timepoints)})",
    taxon="(" + "|".join(TAXONOMY_LEVELS) + "|domain)",
    test_type="(" + "|".join(ALL_TEST_TYPES) + "|)",
    sub_test="(MannWhitney|Wilcoxon|tTest|LMM|CMH|)",
    group_str=group_str_regex,
    # Constraint for focus timepoints - all possible values from the timepoint pairs
    focus_tp="|".join(set([tp for tp_pair in valid_focus_timepoints.values() for tp in tp_pair if tp])),
    # Subject ID constraint - alphanumeric with optional underscores/hyphens
    subject_id="[a-zA-Z0-9_-]+",
    # Explicit treatment/control wildcards (used by regional_contrast rule)
    treatment=f"({'|'.join(sorted({gr['treatment'] for gr in config['analysis']['groups_combinations']}))})",
    control=f"({'|'.join(sorted({gr['control'] for gr in config['analysis']['groups_combinations']}))})",
    
# Function to get sample information from metadata file
def get_sample_info():
    """
    Helper function to retrieve sample information.

    This function is responsible for loading and parsing sample metadata,
    typically from a configuration file or input source. It should return
    information necessary for processing samples in the workflow.

    Returns:
        dict or pandas.DataFrame: Sample information containing metadata
        required for the pipeline execution.
    """
    metadata_path = config["input"]["metadata_path"]
    
    if not metadata_path:
        raise ValueError("metadata_path must be provided in the config file")
    
    # Read metadata file
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    
    # Validate required columns
    required_cols = ["sample_id", "bam_path"]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Metadata file is missing required columns: {', '.join(missing_cols)}")
    
    # Create mapping from sample ID to BAM path
    sample_to_bam_map = dict(zip(metadata_df["sample_id"], metadata_df["bam_path"]))
    sample_ids = list(metadata_df["sample_id"])
    
    # Validate that all BAM files exist
    for sample_id, bam_path in sample_to_bam_map.items():
        if not os.path.exists(bam_path):
            raise ValueError(f"BAM file not found for sample {sample_id}: {bam_path}")
    
    return sample_ids, sample_to_bam_map

def _get_mags_by_eligibility(timepoints, groups, eligibility_type):
    """
    INTERNAL: Read the eligibility file for a given timepoint-group combination and return a list of MAG IDs.
    
    WARNING: This function only checks QC eligibility, NOT preprocessing eligibility.
    For rule inputs that need to respect preprocessing eligibility, use get_eligible_mags()
    from dynamic_targets.smk instead.
    
    This function should only be called:
    - Within get_eligible_mags() as a fallback when preprocessing is disabled
    - In preprocessing_eligibility.smk to get initial QC-eligible MAGs
    - In generate_allele_analysis_targets() which runs before preprocessing

    Parameters:
        timepoints (str): The timepoints label
        groups (str): The groups label
        eligibility_type (str or None):
            - "two_sample_unpaired": only return MAG IDs where unpaired_test_eligible is True.
            - "two_sample_paired": only return MAG IDs where paired_test_eligible is True.
            - "lmm": only return MAG IDs where unpaired_test_eligible is True.
            - "between_only": MAG IDs eligible for unpaired OR paired tests,
              excluding single-sample (within-group) eligibility.
            - "all": return MAG IDs that are eligible for any of the tests.
    
    Returns:
        list: MAG IDs that are eligible for the specified test type
    
    Raises:
        FileNotFoundError: If the eligibility file does not exist
    """
    eligibility_file = os.path.join(
        OUTDIR, f"eligibility_table_{timepoints}-{groups}.tsv"
    )
    
    if not os.path.exists(eligibility_file):
        raise FileNotFoundError(
            f"Eligibility file not found: {eligibility_file}. "
            f"Ensure the eligibility_table checkpoint has run for {timepoints}-{groups}."
        )
    
    df = pd.read_csv(eligibility_file, sep="\t")

    if eligibility_type == "two_sample_unpaired" or eligibility_type == "lmm":
        return df.loc[df["unpaired_test_eligible"] == True, "MAG_ID"].tolist()
    elif eligibility_type == "two_sample_paired" or eligibility_type == "cmh":
        return df.loc[df["paired_test_eligible"] == True, "MAG_ID"].tolist()
    # Between-group eligibility only — unpaired OR paired, deliberately
    # EXCLUDING single-sample (within-group) columns. Used by allele-analysis
    # target generation when within-group tests are disabled, so that MAGs
    # eligible *only* for within-group tests do not trigger unconsumed
    # allele-analysis / allele-freq-cache jobs.
    elif eligibility_type == "between_only":
        return (
            df[
                (df["unpaired_test_eligible"] == True)
                | (df["paired_test_eligible"] == True)
            ]["MAG_ID"]
            .unique()
            .tolist()
        )
    # Return MAGs from all eligible columns
    elif eligibility_type == "all":
        # Combine unpaired, paired, and any single-sample eligibility columns.
        single_cols = [
            col for col in df.columns if col.startswith("single_sample_eligible_")
        ]
        return (
            df[
                (df["unpaired_test_eligible"] == True)
                | (df["paired_test_eligible"] == True)
                | (df[single_cols].any(axis=1))
            ]["MAG_ID"]
            .unique()
            .tolist()
        )
    else:
        raise ValueError(
            f"Unknown eligibility type: {eligibility_type}. "
            "Please use 'two_sample_unpaired', 'two_sample_paired', 'cmh', "
            "'lmm', 'between_only' or 'all'."
        )



def _get_single_sample_entries(timepoints, groups):
    """
    INTERNAL: Reads the eligibility file and returns a list of tuples (MAG_ID, group)
    for each column matching 'single_sample_eligible_*' that evaluates to True.
    This allows a MAG to be eligible for multiple single-sample tests.
    
    WARNING: This function only checks QC eligibility, NOT preprocessing eligibility.
    For rule inputs that need to respect preprocessing eligibility, use get_eligible_mags()
    from dynamic_targets.smk with test_type='single_sample' instead.
    """
    eligibility_file = os.path.join(
        OUTDIR, f"eligibility_table_{timepoints}-{groups}.tsv"
    )
    df = pd.read_csv(eligibility_file, sep="\t")
    sample_entries = []
    # Identify all columns with the generic pattern.
    sample_cols = [
        col for col in df.columns if col.startswith("single_sample_eligible_")
    ]

    # If there are no single_sample_eligible columns or all are NaN
    # (which happens for single data_type), return empty list
    if not sample_cols or df[sample_cols].isna().all().all():
        return []

    for _, row in df.iterrows():
        mag = row["MAG_ID"]
        for col in sample_cols:
            if pd.notna(row[col]) and row[col] == True:
                # Extract the sample group (e.g. "control", "fat") from the column name.
                group = col.replace("single_sample_eligible_", "")
                sample_entries.append((mag, group))
    return sample_entries


def get_mags_by_preprocessing_eligibility(timepoints, groups, test_type, group=None):
    """
    Return MAG IDs eligible for ``test_type`` after preprocessing.

    Uses the canonical Snakemake checkpoint idiom — ``checkpoints.X.get(...).output``
    — to force Snakemake to materialise the eligibility file on disk before
    this function reads it.  This is what makes the function safe to call
    from target-generator functions during DAG planning: if the file does
    not yet exist, the ``.output`` access raises
    ``IncompleteCheckpointException``, which Snakemake catches and re-
    evaluates the DAG after the checkpoint job has run.  A plain
    ``pd.read_csv`` on a manually-constructed path does NOT do this and
    would crash with ``FileNotFoundError`` on cold-start runs whenever the
    output file is eager-listed elsewhere in the DAG.

    Parameters:
        timepoints (str): The timepoints label (e.g., "pre_post")
        groups (str): The groups label (e.g., "fat_control")
        test_type (str): One of
            "two_sample_unpaired" / "two_sample_paired" / "lmm" / "cmh"
              — between-group tests, gated on ``preprocessing_eligibility_between_groups``
            "single_sample" / "lmm_across_time" / "cmh_across_time"
              — within-group tests, gated on ``preprocessing_eligibility_within_groups``
              (require ``group``)
        group (str, optional): Required for the within-group test types.

    Returns:
        list[str]: MAG IDs eligible for the specified test type.
    """
    # Resolve the eligibility file via the canonical Snakemake checkpoint idiom.
    # Accessing ``.output.out_fPath`` on a checkpoint that hasn't completed
    # raises IncompleteCheckpointException, which Snakemake catches and uses
    # to trigger a DAG re-evaluation after the checkpoint job runs.  This is
    # what makes the function safe to call during DAG planning — and why we
    # don't need an explicit os.path.exists guard or a manual path build.
    if test_type in ["two_sample_unpaired", "two_sample_paired", "lmm", "cmh"]:
        # LMM uses the unpaired eligibility column; CMH uses paired.
        eligible_column = (
            "two_sample_unpaired_eligible"
            if test_type in ["two_sample_unpaired", "lmm"]
            else "two_sample_paired_eligible"
        )
        eligibility_file = checkpoints.preprocessing_eligibility_between_groups.get(
            timepoints=timepoints, groups=groups
        ).output.out_fPath
    elif test_type in ["single_sample", "lmm_across_time", "cmh_across_time"]:
        if group is None:
            raise ValueError(f"group parameter is required for test_type '{test_type}'")
        eligible_column = f"single_sample_eligible_{group}"
        eligibility_file = checkpoints.preprocessing_eligibility_within_groups.get(
            timepoints=timepoints, groups=groups
        ).output.out_fPath
    else:
        raise ValueError(
            f"Unknown test type: {test_type}. "
            "Use 'two_sample_unpaired', 'two_sample_paired', 'lmm', 'cmh', "
            "'single_sample', 'lmm_across_time', or 'cmh_across_time'."
        )

    df = pd.read_csv(eligibility_file, sep="\t")
    
    if eligible_column not in df.columns:
        logger.warning(
            f"Column '{eligible_column}' not found in {eligibility_file}. "
            "Returning empty list."
        )
        return []
    
    return df.loc[df[eligible_column] == True, "MAG_ID"].tolist()


def parse_metadata_for_timepoint_pairs(timepoints_label, groups_label):
    """
    Parse metadata to identify ancestral-derived sample pairs for dN/dS analysis.
    
    For longitudinal data, this function:
    1. Identifies the derived timepoint from the 'focus' field in config
    2. Treats the other timepoint as ancestral
    3. Matches samples by subject ID across timepoints
    4. Validates exactly 2 samples per subject-timepoint combination
    
    Parameters:
        timepoints_label (str): Timepoint combination label (e.g., "pre_post")
        groups_label (str): Group combination label (e.g., "fat_control")
    
    Returns:
        list: Tuples of (subject_id, ancestral_sample_id, derived_sample_id)
    """
    metadata_path = config["input"]["metadata_path"]
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    
    # Parse timepoint and group information
    if "_" in timepoints_label and DATA_TYPE == "longitudinal":
        timepoint1, timepoint2 = timepoints_label.split("_")
        
        # Determine ancestral and derived timepoints based on focus
        focus_tp = focus_timepoints.get(timepoints_label)
        if not focus_tp:
            raise ValueError(f"No focus timepoint defined for {timepoints_label}")
        
        # The focus is the derived timepoint
        if focus_tp == timepoint1:
            derived_tp, ancestral_tp = timepoint1, timepoint2
        else:
            derived_tp, ancestral_tp = timepoint2, timepoint1
    else:
        raise ValueError(f"dN/dS analysis requires longitudinal data with two timepoints, got: {timepoints_label}")
    
    # Parse groups
    group1, group2 = groups_label.split("_")
    
    # Filter metadata for relevant samples
    filtered_df = metadata_df[
        metadata_df["group"].isin([group1, group2]) &
        metadata_df["time"].isin([ancestral_tp, derived_tp])
    ]
    
    # Group by subject and timepoint to validate sample counts
    subject_timepoint_counts = filtered_df.groupby(["subjectID", "time"]).size()
    
    # Validate exactly 2 samples per subject-timepoint
    for (subject, tp), count in subject_timepoint_counts.items():
        if count != 1:
            raise ValueError(
                f"Expected exactly 1 sample for subject {subject} at timepoint {tp}, "
                f"but found {count} samples"
            )
    
    # Build sample pairs
    sample_pairs = []
    subjects = filtered_df["subjectID"].unique()
    
    for subject in subjects:
        subject_df = filtered_df[filtered_df["subjectID"] == subject]
        
        # Get ancestral and derived samples
        ancestral_samples = subject_df[subject_df["time"] == ancestral_tp]["sample_id"].tolist()
        derived_samples = subject_df[subject_df["time"] == derived_tp]["sample_id"].tolist()
        
        if len(ancestral_samples) == 1 and len(derived_samples) == 1:
            sample_pairs.append((subject, ancestral_samples[0], derived_samples[0]))
        else:
            # logger.warning(
            #     f"Skipping subject {subject}: found {len(ancestral_samples)} ancestral "
            #     f"and {len(derived_samples)} derived samples"
            # )
            pass
    
    return sample_pairs


# =============================================================================
# Input Path Helper Functions
# =============================================================================
# These helpers centralize the logic for determining input file paths based on
# data type and configuration options, reducing duplication across rule files.

def get_allele_analysis_input_path(mag_wildcard="{mag}", tp_wildcard="{timepoints}", gr_wildcard="{groups}"):
    """
    Get the appropriate allele analysis input file path based on data type and config.

    Returns Parquet paths (1B refactor): all four allele_freq.py outputs now use
    Parquet/Snappy instead of gzip TSV.  Downstream consumers call
    load_allele_freq_inputs() which detects the format by extension.

    Parameters:
        mag_wildcard: MAG ID wildcard string (default: "{mag}")
        tp_wildcard: Timepoints wildcard string (default: "{timepoints}")
        gr_wildcard: Groups wildcard string (default: "{groups}")

    Returns:
        str: Path to the appropriate input file
    """
    base_dir = os.path.join(
        OUTDIR,
        "allele_analysis",
        f"allele_analysis_{tp_wildcard}-{gr_wildcard}"
    )

    if DATA_TYPE == "single":
        if not config["quality_control"].get("disable_zero_diff_filtering", False):
            return os.path.join(base_dir, f"{mag_wildcard}_allele_frequency_no_constant.parquet")
        else:
            return os.path.join(base_dir, f"{mag_wildcard}_allele_frequency_single.parquet")
    else:  # longitudinal
        return os.path.join(base_dir, f"{mag_wildcard}_allele_frequency_changes_mean.parquet")


def get_allele_freq_cache_path(
    mag_wildcard="{mag}",
    timepoint_wildcard="{timepoint}",
):
    """Path to the per-(MAG, timepoint) allele-frequency Parquet cache file.

    The cache is group-independent (2B refactor): one file per (MAG, timepoint),
    reused across every (timepoint_combination, group_combination) pair that
    includes that timepoint.  The cache rule input is the canonical QC file
    from the single per-timepoint QC directory (also group-independent after
    the 2A refactor).
    """
    return os.path.join(
        REUSE_DIR,
        "allele_freq_cache",
        timepoint_wildcard,
        f"{mag_wildcard}_{timepoint_wildcard}_allele_frequency.parquet",
    )


def get_canonical_qc_file(mag_wildcard, timepoint):
    """Return the single canonical QC TSV path for a (timepoint,) cache.

    Selects the first tp_combo in config order that contains ``timepoint``.
    This canonical QC file is the ONLY input to
    ``compute_allele_freq_per_timepoint`` — Snakemake resolves it as a regular
    rule dependency (``qc`` → ``generate_metadata``), so the DAG is valid
    regardless of which combination's eligibility_table checkpoint fired first.

    After the 2A/2B refactor QC is group-independent, so the path is
    ``QC_{canonical_tp}/`` with no groups component.
    """
    canonical_tp = timepoint_to_canonical_tp.get(timepoint)
    if canonical_tp is None:
        raise ValueError(
            f"No tp_combo found for timepoint={timepoint!r}. "
            "Check that the timepoint appears in the config."
        )
    return os.path.join(
        REUSE_DIR,
        "QC",
        f"QC_{canonical_tp}",
        f"{mag_wildcard}_QC.tsv",
    )




def get_preprocessed_between_groups_path(mag_wildcard="{mag}", tp_wildcard="{timepoints}", gr_wildcard="{groups}"):
    """
    Get the preprocessed between-groups file path based on data type.
    
    Parameters:
        mag_wildcard: MAG ID wildcard string (default: "{mag}")
        tp_wildcard: Timepoints wildcard string (default: "{timepoints}")
        gr_wildcard: Groups wildcard string (default: "{groups}")
    
    Returns:
        str: Path to the preprocessed file
    """
    base_dir = os.path.join(
        OUTDIR,
        "significance_tests",
        f"preprocessed_between_groups_{tp_wildcard}-{gr_wildcard}"
    )
    
    if DATA_TYPE == "single":
        return os.path.join(base_dir, f"{mag_wildcard}_allele_frequency_preprocessed.tsv.gz")
    else:  # longitudinal
        return os.path.join(base_dir, f"{mag_wildcard}_allele_frequency_changes_mean_preprocessed.tsv.gz")


def get_preprocessed_within_groups_path(mag_wildcard="{mag}", group_wildcard="{group}", 
                                        tp_wildcard="{timepoints}", gr_wildcard="{groups}"):
    """
    Get the preprocessed within-groups file path.
    
    Parameters:
        mag_wildcard: MAG ID wildcard string (default: "{mag}")
        group_wildcard: Group wildcard string (default: "{group}")
        tp_wildcard: Timepoints wildcard string (default: "{timepoints}")
        gr_wildcard: Groups wildcard string (default: "{groups}")
    
    Returns:
        str: Path to the preprocessed within-groups file
    """
    return os.path.join(
        OUTDIR,
        "significance_tests",
        f"preprocessed_within_groups_{tp_wildcard}-{gr_wildcard}",
        f"{mag_wildcard}_{group_wildcard}_allele_frequency_changes_mean_zeros_processed.tsv.gz"
    )



def get_pairwise_ani_output_path(mag_wildcard="{mag}"):
    """Path to the per-MAG pairwise conANI/popANI table (the rule's primary output).

    Group- and timepoint-independent: one file per MAG covering every QC-passing
    sample, because within-subject cross-timepoint comparisons are the point.
    Resolves under OUTDIR (NOT REUSE_DIR): unlike profiles/QC/the allele-freq
    cache, this table is a PRODUCT of the current run, not a reusable upstream
    artifact -- a permuted/null run computes its own or, more sensibly, disables
    the feature via use_pairwise_ani.
    """
    return os.path.join(OUTDIR, "pairwise_ani", f"{mag_wildcard}_pairwise_ani.tsv")


def get_strain_turnover_output_path(mag_wildcard="{mag}"):
    """Per-MAG strain-turnover table (the strain_turnover rule's primary output); OUTDIR, like ANI."""
    return os.path.join(OUTDIR, "strain_turnover", f"{mag_wildcard}_strain_turnover.tsv")


def get_replacement_classification_path():
    """The one all-MAG classification table the enrichment filter reads."""
    return os.path.join(OUTDIR, "strain_turnover", "replacement_classification.tsv")


# The baseline-presence summary family, read once: it names the input summary
# file, the source directory and the output stem.
BASELINE_PRESENCE_FAMILY = config["analysis"].get("baseline_presence", {}).get("summary", "two_sample_paired")


def get_baseline_presence_stem(timepoints="{timepoints}", groups="{groups}"):
    """Output stem for one comparison: {comparison}_{family}_{statistic}_baseline_presence.

    Mirrors ``baseline_presence.output_label``: the family prefix is stripped
    from the test_type so "two_sample_paired" + "two_sample_paired_tTest" gives
    "two_sample_paired_tTest", while "lmm" + "LMM_abs" gives "lmm_LMM_abs".
    """
    family = BASELINE_PRESENCE_FAMILY
    test_type = config["analysis"].get("baseline_presence", {}).get("test_type", f"{family}_tTest")
    prefix = f"{family}_"
    stat = test_type[len(prefix):] if test_type.startswith(prefix) else test_type
    return os.path.join(OUTDIR, "baseline_presence", f"{timepoints}-{groups}_{family}_{stat}_baseline_presence")


def get_all_qc_files_for_mag(mag_wildcard="{mag}"):
    """Every per-timepoint-combination QC file for one MAG -- the ANI sample gate.

    Unlike the allele-frequency cache (which wants ONE canonical timepoint's QC),
    pairwise ANI wants every sample judged usable at ANY timepoint combination;
    the CLI unions and de-duplicates them.  Resolves under REUSE_DIR because QC
    IS a reusable upstream artifact (same reasoning as qc_sentinel).
    """
    return [
        os.path.join(REUSE_DIR, "QC", f"QC_{tp}", f"{mag_wildcard}_QC.tsv")
        for tp in timepoints_labels
    ]
