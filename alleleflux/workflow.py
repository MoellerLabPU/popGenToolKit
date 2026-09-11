#!/usr/bin/env python3
"""
AlleleFlux Workflow Orchestration Module.

This module handles the actual execution of Snakemake workflows,
including configuration validation, command building, and real-time
output streaming. It is separate from the CLI to maintain clean
separation of concerns.
"""

import copy
import logging
import multiprocessing
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from importlib import resources as importlib_resources
from pathlib import Path
from typing import List, Optional

import yaml

from alleleflux.scripts.utilities.logging_config import setup_logging

# Note: setup_logging() is NOT called at module level to avoid interfering
# with Snakemake's logging system. It's called inside execute_workflow().
logger = logging.getLogger(__name__)
# =============================================================================
# Memory Utilities
# =============================================================================


def parse_mem(mem_value: str) -> int:
    """
    Parse memory string like "8G", "100GB", "8192M" to MB.
    Uses binary units (1 GB = 1024 MB).

    Args:
        mem_value: Memory specification string or int (MB).

    Returns:
        Memory in MB as integer.

    Raises:
        ValueError: If format is invalid.

    Examples:
        >>> parse_mem("8G")      # 8 gigabytes
        8192
        >>> parse_mem("100GB")   # 100 gigabytes
        102400
        >>> parse_mem("8192M")   # 8192 megabytes
        8192
        >>> parse_mem("8192")    # Assumes MB
        8192
    """
    if isinstance(mem_value, (int, float)):
        return int(mem_value)

    mem_str = str(mem_value).strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(G|GB|M|MB)?$", mem_str)

    if not match:
        raise ValueError(
            f"Invalid memory format: '{mem_value}'. "
            "Use format like '8G', '64GB', or '8192M'."
        )

    value = float(match.group(1))
    unit = match.group(2) or "M"

    if unit in ("G", "GB"):
        return int(value * 1024)
    return int(value)


# =============================================================================
# Path Helpers
# =============================================================================


def get_snakefile() -> Path:
    """
    Get the path to the packaged Snakefile.

    Returns:
        Path to the Snakefile bundled with the package.

    Raises:
        SystemExit: If the Snakefile cannot be found.
    """
    package_dir = Path(__file__).parent
    snakefile = package_dir / "smk_workflow" / "alleleflux_pipeline" / "Snakefile"

    if not snakefile.exists():
        logger.error(f"Snakefile not found at: {snakefile}")
        logger.error("This may indicate a broken installation.")
        sys.exit(1)

    return snakefile


def get_template_path() -> Path:
    """
    Get the path to the configuration template file.

    Returns:
        Path to the config.template.yml bundled with the package.
    """
    package_dir = Path(__file__).parent
    template = package_dir / "smk_workflow" / "config.template.yml"

    # Fallback to importlib.resources if Path-based lookup fails
    if not template.exists():
        try:
            # For Python 3.9+
            with importlib_resources.as_file(
                importlib_resources.files("alleleflux").joinpath(
                    "smk_workflow/config.template.yml"
                )
            ) as resource_path:
                return Path(resource_path)
        except Exception:
            pass

    return template


# =============================================================================
# Configuration Validation
# =============================================================================


def validate_config(config_path: Path) -> dict:
    """
    Load and validate a configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: If configuration is invalid or missing required fields.
    """
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.error("Generate one with 'alleleflux init' or provide a valid path.")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)

    # Validate required sections
    required_sections = [
        "input",
        "output",
        "analysis",
        "quality_control",
        "profiling",
        "statistics",
        "dnds",
    ]
    missing = [s for s in required_sections if s not in config]
    if missing:
        logger.error(f"Configuration missing required sections: {missing}")
        sys.exit(1)

    # Validate required input paths
    input_config = config.get("input", {})
    required_inputs = [
        "fasta_path",
        "prodigal_path",
        "metadata_path",
        "mag_mapping_path",
        "gtdb_path",
    ]
    missing_inputs = [i for i in required_inputs if not input_config.get(i)]
    if missing_inputs:
        logger.error(f"Configuration missing required input paths: {missing_inputs}")
        sys.exit(1)

    # Validate optional profiles_path if specified
    profiles_path = input_config.get("profiles_path", "")
    if profiles_path:
        if not os.path.isdir(profiles_path):
            logger.error(
                f"Specified profiles_path does not exist or is not a directory: {profiles_path}"
            )
            sys.exit(1)
        logger.info(f"Using existing profiles from: {profiles_path}")
        logger.info("Profiling step will be skipped - profiles will be symlinked.")
    else:
        logger.info(
            "No existing profiles specified - profiling will run from BAM files."
        )

    # Validate optional reuse_from (permuted/null run reusing a real run's
    # group-independent artifacts: profiles, QC, allele-freq cache).
    reuse_from = input_config.get("reuse_from", "")
    if reuse_from:
        if not os.path.isdir(reuse_from):
            logger.error(
                f"Specified reuse_from does not exist or is not a directory: {reuse_from}"
            )
            sys.exit(1)
        # Warn (don't hard-fail) if the expected reusable subdirs are missing —
        # the run will simply rebuild whatever isn't found.
        for sub in ("QC", "allele_freq_cache"):
            if not os.path.isdir(os.path.join(reuse_from, sub)):
                logger.warning(
                    f"reuse_from is missing expected subdir '{sub}/': "
                    f"{os.path.join(reuse_from, sub)}. That stage will be rebuilt."
                )
        logger.info(f"Reusing profiles/QC/allele-freq cache from: {reuse_from}")

    # A permuted (null) run MUST reuse an existing real run — otherwise each
    # permutation would rebuild the whole (expensive) cache from scratch, which
    # defeats the entire purpose.  Enforce reuse_from when permutation is on.
    permutation = config.get("permutation", {})
    if permutation.get("enabled"):
        if not reuse_from:
            logger.error(
                "permutation.enabled is true but input.reuse_from is not set. "
                "A permuted run reuses a completed real run's profiles/QC/cache; "
                "point input.reuse_from at that run's data-type output dir "
                "(e.g. .../alleleflux_output_1/longitudinal)."
            )
            sys.exit(1)
        logger.info("Permutation (null) run enabled.")

    return config


# =============================================================================
# Snakemake Execution
# =============================================================================


def run_snakemake(cmd: str, logfile: Path = None) -> int:
    """
    Run snakemake command with real-time output streaming.

    This follows the SnakePipes pattern of using subprocess.Popen with
    stdout streaming for immediate user feedback during long workflow runs.
    Handles keyboard interrupts gracefully by allowing snakemake to perform
    its own cleanup before exiting.

    Args:
        cmd: The snakemake command string to execute.
        logfile: Optional path to write command output to a log file.

    Returns:
        Exit code from snakemake (0 = success, 130 = interrupted).
    """
    logger.info(f"Executing: {cmd}")

    # Open log file if specified
    log_handle = open(logfile, "w") if logfile else None

    try:
        # shell=True spawns `bash -c <cmd>`, so the immediate child is bash and
        # snakemake is its grandchild. preexec_fn=os.setsid puts bash into a new
        # session/process group with PGID == bash's PID, so a single
        # os.killpg(process.pid, sig) reaches bash + snakemake + any further
        # descendants atomically — without this, terminate() only kills bash
        # and snakemake is orphaned to init.
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
        )

        # Stream output line by line
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_handle:
                log_handle.write(line)
                log_handle.flush()

        # Wait for process to complete
        process.wait()
        return process.returncode

    except KeyboardInterrupt:
        # Handle Ctrl+C with escalating signals: SIGINT (graceful) → SIGTERM
        # (urgent) → SIGKILL (uncatchable). Each step has a bounded timeout, so
        # a snakemake stuck in a non-interruptible sbatch RPC can't strand us in
        # an infinite wait (the original bug — `process.wait()` after `terminate()`
        # blocked forever when SIGTERM was ignored, leaving an orphan driver
        # alive on the login node that kept resubmitting jobs after `scancel`).
        logger.warning("\nInterrupted by user. Waiting for snakemake to clean up...")
        if log_handle:
            log_handle.write("\nInterrupted by user (KeyboardInterrupt)\n")
            log_handle.flush()

        # Because we put the child in its own session (preexec_fn=os.setsid),
        # the terminal's Ctrl+C does NOT reach snakemake automatically — we
        # must forward SIGINT explicitly to give it a chance to gracefully
        # cancel its in-flight SLURM jobs.
        for sig, label, grace_seconds in (
            (signal.SIGINT, "SIGINT (graceful)", 30),
            (signal.SIGTERM, "SIGTERM (urgent)", 10),
            (signal.SIGKILL, "SIGKILL (uncatchable)", 5),
        ):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break  # already dead
            try:
                process.wait(timeout=grace_seconds)
                break  # exited cleanly under this signal
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"{label} did not stop snakemake within {grace_seconds}s; escalating..."
                )
        else:
            # Loop fell through without break — even SIGKILL didn't reap it.
            # This should be impossible on Linux (SIGKILL is uncatchable), but
            # if it ever happens the user needs to know to investigate manually.
            logger.error(
                f"Failed to stop snakemake process group (PGID={process.pid}). "
                "Check for zombie processes with `pgrep -u $USER snakemake`."
            )

        logger.info("Workflow interrupted. You can resume with the same command.")
        return 130  # Standard exit code for SIGINT

    finally:
        if log_handle:
            log_handle.close()


def build_snakemake_command(
    snakefile: Path,
    config_path: Path,
    working_dir: str,
    jobs: Optional[int] = None,
    threads: Optional[int] = None,
    memory_mb: Optional[int] = None,
    profile: Optional[str] = None,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
) -> str:
    """
    Build the snakemake command string.

    Args:
        snakefile: Path to the Snakefile.
        config_path: Path to the configuration file.
        working_dir: Working directory for the workflow.
        jobs: Maximum concurrent jobs (local only, ignored with profile).
        threads: Total threads available (local only, ignored with profile).
        memory_mb: Total memory in MB (local only, ignored with profile).
        profile: Path to Snakemake profile for cluster execution.
        dry_run: If True, add --dry-run flag.
        extra_args: Additional arguments to pass to snakemake.

    Returns:
        Complete snakemake command string.
    """
    cmd_parts = [
        "snakemake",
        f"--snakefile {snakefile}",
        f"--configfile {config_path}",
        f"--directory {working_dir}",
        # "--rerun-incomplete",
        # "--show-failed-logs",
    ]

    # Profile takes precedence - let profile handle all resource scheduling
    if profile:
        cmd_parts.append(f"--profile {profile}")
    else:
        # Local execution: apply thread, job, and memory constraints
        if threads is not None:
            cmd_parts.append(f"--cores {threads}")
        else:
            cmd_parts.append(f"--cores {multiprocessing.cpu_count()}")

        if jobs is not None:
            cmd_parts.append(f"--jobs {jobs}")

        if memory_mb is not None and memory_mb > 0:
            cmd_parts.append(f"--resources mem_mb={memory_mb}")

    # Add dry run flag
    if dry_run:
        cmd_parts.append("--dry-run")

    # Add any extra snakemake arguments
    if extra_args:
        cmd_parts.extend(extra_args)

    return " ".join(cmd_parts)


# =============================================================================
# Permutation (null) orchestration
# =============================================================================


def _resolve_permutation_seeds(perm_config: dict) -> List[int]:
    """Resolve the list of permutation seeds from the ``permutation`` config.

    Precedence: an explicit ``seeds`` list wins; else ``n_permutations`` expands
    to ``1..N``; else a single permutation with seed ``1``.
    """
    seeds = perm_config.get("seeds")
    if seeds:
        return [int(s) for s in seeds]
    n = int(perm_config.get("n_permutations", 1))
    return list(range(1, n + 1))


def _unlock_permutation_leaves(
    config: dict, working_dir: str, snakefile: "str | Path"
) -> int:
    """Unlock every per-seed leaf working dir of a permutation-orchestrating run.

    The head ``--unlock`` only clears the orchestrator's own working dir. Each
    leaf (``<root>/<output_subdir>/perm_<seed>/``) runs Snakemake in its own dir
    with its own ``.snakemake/`` lock, so a crashed leaf leaves a stale lock the
    head unlock cannot reach. This cascades the unlock to each leaf that has a
    persisted leaf config, mirroring the layout written by :func:`run_permutations`.

    No-op (returns 0) when this is not a permutation orchestrator. Leaves with no
    leaf config yet (never started) are silently skipped — nothing to unlock.
    """
    perm = config.get("permutation", {})
    is_orchestrator = (
        perm.get("enabled")
        and perm.get("seed") is None
        and not perm.get("permuted_metadata_dir")
    )
    if not is_orchestrator:
        return 0

    subdir = perm.get("output_subdir", "permuted")
    base_output = Path(config["output"]["root_dir"])
    if not base_output.is_absolute():
        base_output = (Path(working_dir) / base_output).resolve()

    worst = 0
    for seed in _resolve_permutation_seeds(perm):
        seed_dir = base_output / subdir / f"perm_{seed}"
        leaf_config = seed_dir / f"config_perm_{seed}.yml"
        # Skip leaves that never ran (no config, or no lock dir to clear).
        if not leaf_config.exists() or not (seed_dir / ".snakemake").is_dir():
            continue
        logger.info(f"Unlocking permutation leaf: {seed_dir}")
        leaf_cmd = (
            f"snakemake --snakefile {snakefile} "
            f"--configfile {leaf_config.resolve()} "
            f"--directory {seed_dir} --unlock"
        )
        worst = max(worst, run_snakemake(leaf_cmd))
    return worst


def run_permutations(
    config: dict,
    working_dir: str,
    jobs: Optional[int] = None,
    threads: Optional[int] = None,
    memory: Optional[str] = None,
    profile: Optional[str] = None,
    dry_run: bool = False,
    extra_args: Optional[List[str]] = None,
    version: str = "unknown",
) -> int:
    """Fan out N per-comparison-pair permuted (null) runs that reuse a real run.

    For each seed, write a per-seed *leaf* config carrying ``permutation.seed``
    (singular) and run the group-dependent tail through
    :func:`execute_workflow`.  The leaf is a *generate-mode* permuted run: the
    in-DAG ``permute_metadata`` rule builds one permuted metadata sheet per
    ``groups_combinations`` entry from that seed (so the sheets are tracked
    artifacts — provenance + skip-if-exists, and nothing is written on a dry
    run).  Carrying ``seed`` also marks the leaf so it does NOT re-orchestrate,
    and it inherits ``input.reuse_from`` so profiling/QC/cache are reused.

    Layout (under the orchestrating config's ``output.root_dir``)::

        <root>/<output_subdir>/perm_<seed>/
            config_perm_<seed>.yml
            <data_type>/permuted_metadata/permuted_metadata_<t>_<c>.tsv
            <data_type>/...                      # scores, stats, eligibility

    Returns the worst (max) leaf exit code; stops early on the first failure.
    """
    perm = config["permutation"]
    seeds = _resolve_permutation_seeds(perm)
    unit = perm.get("unit", "subjectID")
    block = perm.get("block")
    swaps = perm.get("swaps")
    subdir = perm.get("output_subdir", "permuted")

    wd = Path(working_dir)

    # Resolve the base output and the original metadata to absolute paths so the
    # leaf configs (which run in their own working dir) reference them
    # unambiguously regardless of the leaf's cwd.
    base_output = Path(config["output"]["root_dir"])
    if not base_output.is_absolute():
        base_output = (wd / base_output).resolve()

    orig_md = Path(config["input"]["metadata_path"])
    if not orig_md.is_absolute():
        orig_md = (wd / orig_md).resolve()

    pairs = [
        (g["treatment"], g["control"])
        for g in config["analysis"]["groups_combinations"]
    ]

    logger.info("=" * 60)
    logger.info(
        f"PERMUTATION ORCHESTRATION — {len(seeds)} permutation(s) × "
        f"{len(pairs)} comparison pair(s)"
    )
    logger.info(
        f"  unit={unit}  block={block}  "
        f"swaps={'half (default)' if swaps is None else swaps}  seeds={seeds}"
    )
    logger.info(f"  reuse_from={config['input'].get('reuse_from')}")
    logger.info("=" * 60)

    exit_codes = []
    for idx, seed in enumerate(seeds, start=1):
        seed_dir = base_output / subdir / f"perm_{seed}"

        logger.info(f"[permutation {idx}/{len(seeds)}] seed={seed} → {seed_dir}")

        # Per-seed leaf config.  Carrying ``permutation.seed`` (singular) marks
        # this as a generate-mode leaf: execute_workflow does NOT re-orchestrate,
        # and the in-DAG ``permute_metadata`` rule builds the per-pair sheets from
        # this seed.  ``permuted_metadata_dir`` is removed so BYO mode is not
        # triggered.  ``metadata_path`` is pinned absolute for the rule.
        leaf = copy.deepcopy(config)
        leaf["output"]["root_dir"] = str(seed_dir)
        leaf["input"]["metadata_path"] = str(orig_md)
        leaf.setdefault("permutation", {})
        leaf["permutation"]["enabled"] = True
        leaf["permutation"]["seed"] = seed
        leaf["permutation"].pop("permuted_metadata_dir", None)
        # A divergence null only needs the between-group comparison, so skip the
        # within-group / across-time tests by default (saves compute).  The user
        # can force them back on with analysis.run_within_group_tests: true.
        leaf.setdefault("analysis", {}).setdefault("run_within_group_tests", False)

        # Persist the leaf config beside its outputs on a real run; on a dry run
        # write it to a throwaway temp dir so the output tree stays pristine (no
        # config/TSV litter under ``-n``).  Each leaf runs in its OWN working dir
        # (isolated .snakemake/) — sharing the real run's dir made Snakemake's
        # params rerun-trigger fire on the reused metadata/qc rules and re-touch
        # the real run's sentinels.  Isolation also avoids cross-seed lock
        # contention.
        tmp_dir = None
        if dry_run:
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"alleleflux_perm_{seed}_"))
            leaf_path = tmp_dir / f"config_perm_{seed}.yml"
            leaf_wd = tmp_dir
        else:
            seed_dir.mkdir(parents=True, exist_ok=True)
            leaf_path = seed_dir / f"config_perm_{seed}.yml"
            leaf_wd = seed_dir

        with open(leaf_path, "w") as f:
            yaml.safe_dump(leaf, f, sort_keys=False)

        try:
            rc = execute_workflow(
                config_file=str(leaf_path),
                working_dir=str(leaf_wd),
                jobs=jobs,
                threads=threads,
                memory=memory,
                profile=profile,
                dry_run=dry_run,
                extra_args=extra_args,
                version=version,
            )
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        exit_codes.append(rc)
        if rc != 0:
            logger.error(
                f"Permutation seed={seed} failed (exit {rc}); stopping orchestration."
            )
            break

    logger.info("=" * 60)
    ran = len(exit_codes)
    ok = sum(1 for c in exit_codes if c == 0)
    logger.info(f"PERMUTATION ORCHESTRATION done: {ok}/{ran} permutation(s) succeeded.")
    logger.info("=" * 60)
    return max(exit_codes) if exit_codes else 0


def execute_workflow(
    config_file: str,
    working_dir: str = ".",
    jobs: Optional[int] = None,
    threads: Optional[int] = None,
    memory: Optional[str] = None,
    profile: Optional[str] = None,
    dry_run: bool = False,
    unlock: bool = False,
    extra_args: Optional[List[str]] = None,
    version: str = "unknown",
) -> int:
    """
    Execute the AlleleFlux workflow.

    This is the main entry point for workflow execution, handling all
    setup, validation, and execution logic.

    Args:
        config_file: Path to the configuration file.
        working_dir: Working directory for the workflow.
        jobs: Max concurrent jobs. Local only (ignored with --profile).
        threads: Total threads available. Local only (ignored with --profile).
            Defaults to all CPU cores if not specified.
        memory: Total memory available, e.g. "64G". Local only (ignored with --profile).
            Formats: "8G", "64GB", "8192M".
        profile: Path to Snakemake profile for cluster execution.
        dry_run: If True, perform a dry run without executing jobs.
        unlock: If True, unlock the working directory and exit.
        extra_args: Additional arguments to pass to snakemake.
        version: AlleleFlux version string for logging.

    Returns:
        Exit code (0 = success).
    """
    # Configure logging here (not at module level) to avoid interfering
    # with Snakemake's logging system during import
    setup_logging()

    # Get the Snakefile path
    snakefile = get_snakefile()

    # Validate configuration
    config_path = Path(config_file)
    config = validate_config(config_path)

    # Handle unlock
    if unlock:
        logger.info("Unlocking working directory...")
        unlock_cmd = f"snakemake --snakefile {snakefile} --configfile {config_path.resolve()} --directory {working_dir} --unlock"
        rc = run_snakemake(unlock_cmd)
        # A permutation-orchestrating run drives each per-seed leaf in its OWN
        # working dir with its OWN .snakemake/ — the head unlock above does NOT
        # reach them. Cascade so a single `alleleflux run ... --unlock` clears
        # stale locks left by a crashed leaf (e.g. <root>/permuted/perm_<seed>/).
        rc = max(rc, _unlock_permutation_leaves(config, working_dir, snakefile))
        return rc

    # Permutation orchestration.  When permutation is enabled and this is not
    # already a per-seed *leaf* run, fan out into N permuted runs that reuse the
    # real run's artifacts.  A leaf is marked by either ``permutation.seed``
    # (generate mode — the permute_metadata rule builds the sheets) or
    # ``permutation.permuted_metadata_dir`` (BYO mode — user-supplied sheets);
    # in either case it skips this block and runs the group-dependent pipeline.
    permutation = config.get("permutation", {})
    if (
        permutation.get("enabled")
        and permutation.get("seed") is None
        and not permutation.get("permuted_metadata_dir")
    ):
        return run_permutations(
            config=config,
            working_dir=working_dir,
            jobs=jobs,
            threads=threads,
            memory=memory,
            profile=profile,
            dry_run=dry_run,
            extra_args=extra_args,
            version=version,
        )

    # Get output directory from config for logging
    output_dir = Path(config.get("output", {}).get("root_dir", working_dir))

    # Parse memory if provided
    memory_mb = None
    if memory:
        try:
            memory_mb = parse_mem(memory)
        except ValueError as e:
            logger.error(f"Invalid --memory value: {e}")
            logger.error("Use format like '8G', '64GB', or '8192M'.")
            return 1

    # Only create output directory and logfile if not doing a dry run
    logfile = None
    runtime_config_path = None
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logfile = output_dir / f"alleleflux_{timestamp}.log"

        # Generate runtime config with timestamp
        runtime_config_path = output_dir / f"config_runtime_{timestamp}.yml"
        runtime_config = config.copy()
        runtime_config["_runtime"] = {
            "version": version,
            "timestamp": timestamp,
            "working_dir": str(Path(working_dir).resolve()),
            "config_file": str(config_path.resolve()),
            "threads": threads,
            "jobs": jobs,
            "memory": memory,
            "profile": profile,
        }

        # Custom representer to use flow style (inline) only for simple lists
        # This produces ["T2", "T4"] instead of block style for readability
        def represent_list(dumper, data):
            is_simple = all(isinstance(item, (str, int, float, bool)) for item in data)
            return dumper.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=is_simple
            )

        yaml.add_representer(list, represent_list)

        with open(runtime_config_path, "w") as f:
            yaml.dump(runtime_config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Runtime config saved: {runtime_config_path}")

    # Build snakemake command
    cmd = build_snakemake_command(
        snakefile=snakefile,
        config_path=config_path,
        working_dir=working_dir,
        jobs=jobs,
        threads=threads,
        memory_mb=memory_mb,
        profile=profile,
        dry_run=dry_run,
        extra_args=extra_args,
    )

    # Log startup info
    logger.info("=" * 60)
    logger.info("AlleleFlux Workflow Starting")
    logger.info("=" * 60)
    logger.info(f"Version: {version}")
    logger.info(f"Snakefile: {snakefile}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Working directory: {working_dir}")
    if profile:
        logger.info(f"Profile: {profile}")
    if dry_run:
        logger.warning("DRY RUN MODE - No jobs will be executed")
    else:
        logger.info(f"Log file: {logfile}")
    logger.info("=" * 60)

    # Write command to log (skip if dry run)
    if logfile:
        with open(logfile, "w") as f:
            f.write(
                f"AlleleFlux run started at {datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\n"
            )
            f.write(f"Snakemake command: {cmd}\n\n")

    # Run snakemake
    try:
        exit_code = run_snakemake(cmd, logfile=logfile)
    except KeyboardInterrupt:
        # This catches any keyboard interrupt not caught in run_snakemake
        logger.warning("\nWorkflow interrupted by user.")
        return 130

    # Report result
    if exit_code == 0:
        logger.info("=" * 60)
        logger.info("AlleleFlux workflow completed successfully!")
        if not dry_run:
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Runtime config: {runtime_config_path}")
            logger.info(f"Log file: {logfile}")
        logger.info("=" * 60)
    elif exit_code == 130:
        # Already logged in run_snakemake, just return
        pass
    else:
        logger.error("=" * 60)
        logger.error(f"AlleleFlux workflow failed with exit code {exit_code}")
        if logfile:
            logger.error(f"Check log file for details: {logfile}")
        logger.error("=" * 60)

    return exit_code
