#!/usr/bin/env python3
"""
AlleleFlux CLI - Command Line Interface for the AlleleFlux workflow.

This module defines the CLI commands and arguments using Click.
Workflow orchestration is handled by the separate workflow module.
"""

import logging
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import questionary
import yaml

from alleleflux.scripts.utilities.logging_config import setup_logging

from .workflow import execute_workflow, get_snakefile, get_template_path

# Get version from package metadata
__version__ = version("AlleleFlux")

# =============================================================================
# Banner Display
# =============================================================================

# ASCII art banner for AlleleFlux
BANNER = r"""
    ___    ____     __     ____                
   /   |  / / /__  / /__  / __/ / /  ___  __
  / /| | / / / _ \/ / _ \/ /_  / /  / / / / \/ /
 / ___ |/ / /  __/ /  __/ __/ / /__/ /_/ />  < 
/_/  |_/_/_/\___/_/\___/_/   /____/\__,_/_/\_\
"""


def print_banner():
    """Print the AlleleFlux banner with colors."""
    # Print banner in cyan
    click.echo(click.style(BANNER, fg="cyan", bold=True))
    # Print version line
    version_line = f"  Version {__version__}  |  Evolution Detection in Metagenomes"
    click.echo(click.style(version_line, fg="bright_blue"))
    click.echo(click.style("─" * 60, fg="cyan"))
    click.echo()


class BannerGroup(click.Group):
    """Custom Click Group that shows banner before help."""

    def format_help(self, ctx, formatter):
        """Print banner before showing help."""
        print_banner()
        super().format_help(ctx, formatter)


# =============================================================================
# Helper Functions for Interactive Prompts
# =============================================================================


def prompt_text(
    message: str,
    default: str = None,
    validate_file: bool = False,
    required: bool = False,
    show_default_used: bool = True,
) -> str:
    """
    Prompt for text input with optional file validation.

    Args:
        message: The prompt message to display.
        default: Default value (None means no default shown).
        validate_file: If True, validate that the input is an existing file.
        required: If True, input cannot be empty.
        show_default_used: If True, print message when default is used.

    Returns:
        The user's input string.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """
    kwargs = {"message": message}
    if default is not None:
        kwargs["default"] = default

    # Build validation function
    def validate(text):
        if required and not text.strip():
            return "This field is required."
        if validate_file and text and not Path(text).is_file():
            return "File does not exist."
        return True

    if validate_file or required:
        kwargs["validate"] = validate

    result = questionary.text(**kwargs).ask()
    if result is None:
        raise KeyboardInterrupt

    # Show default used message
    if not result and default is not None and show_default_used:
        click.echo(f"  → Using default: {default}")
        return default

    return result


def prompt_boolean(
    message: str, default: bool = True, show_default_used: bool = True
) -> bool:
    """
    Prompt for a boolean value with true/false text input.

    Args:
        message: The prompt message to display.
        default: Default boolean value.
        show_default_used: If True, print message when default is used.

    Returns:
        The parsed boolean value.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """
    default_str = str(default).lower()

    def validate(text):
        if not text.strip():
            return True  # Empty is OK, will use default
        text = text.lower().strip()
        if text in ["true", "t", "yes", "y", "1", "false", "f", "no", "n", "0"]:
            return True
        return "Please enter true/false, yes/no, or y/n."

    result = questionary.text(message, default=default_str, validate=validate).ask()
    if result is None:
        raise KeyboardInterrupt

    if not result.strip():
        if show_default_used:
            click.echo(f"  → Using default: {default}")
        return default

    result = result.lower().strip()
    if result in ["true", "t", "yes", "y", "1"]:
        return True
    else:
        return False


def prompt_integer(
    message: str,
    default: int,
    min_value: int = None,
    max_value: int = None,
    show_default_used: bool = True,
) -> int:
    """
    Prompt for an integer value with validation.

    Args:
        message: The prompt message to display.
        default: Default integer value.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        show_default_used: If True, print message when default is used.

    Returns:
        The validated integer value.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """

    def validate(text):
        if not text.strip():
            return True  # Empty is OK, will use default
        try:
            value = int(text)
            if min_value is not None and value < min_value:
                return f"Value must be >= {min_value}."
            if max_value is not None and value > max_value:
                return f"Value must be <= {max_value}."
            return True
        except ValueError:
            return "Please enter a valid integer."

    result = questionary.text(message, default=str(default), validate=validate).ask()
    if result is None:
        raise KeyboardInterrupt

    if not result.strip():
        if show_default_used:
            click.echo(f"  → Using default: {default}")
        return default

    return int(result)


def prompt_float(
    message: str,
    default: float,
    min_value: float = None,
    max_value: float = None,
    show_default_used: bool = True,
) -> float:
    """
    Prompt for a float value with optional range validation.

    Args:
        message: The prompt message to display.
        default: Default float value.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        show_default_used: If True, print message when default is used.

    Returns:
        The validated float value.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """

    def validate(text):
        if not text.strip():
            return True  # Empty is OK, will use default
        try:
            value = float(text)
            if min_value is not None and value < min_value:
                return f"Value must be >= {min_value}."
            if max_value is not None and value > max_value:
                return f"Value must be <= {max_value}."
            return True
        except ValueError:
            return "Please enter a valid number."

    result = questionary.text(message, default=str(default), validate=validate).ask()
    if result is None:
        raise KeyboardInterrupt

    if not result.strip():
        if show_default_used:
            click.echo(f"  → Using default: {default}")
        return default

    return float(result)


def prompt_choice(message: str, choices: list) -> str:
    """
    Prompt for a selection from a list of choices.

    Args:
        message: The prompt message to display.
        choices: List of choice values.

    Returns:
        The selected choice.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """
    result = questionary.select(
        message, choices=[{"name": c, "value": c} for c in choices]
    ).ask()
    if result is None:
        raise KeyboardInterrupt
    return result


def prompt_memory(
    message: str, default: str = "8G", show_default_used: bool = True
) -> str:
    """
    Prompt for a memory value with unit validation (e.g., "8G", "100GB", "8192M").

    Args:
        message: The prompt message to display.
        default: Default memory string.
        show_default_used: If True, print message when default is used.

    Returns:
        The validated memory string.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """
    import re

    def validate(text):
        if not text.strip():
            return True  # Empty is OK, will use default
        text = text.strip().upper()
        if re.match(r"^\d+(?:\.\d+)?\s*(G|GB|M|MB)?$", text):
            return True
        return "Please use format like '8G', '100GB', or '8192M'."

    result = questionary.text(message, default=default, validate=validate).ask()
    if result is None:
        raise KeyboardInterrupt

    if not result.strip():
        if show_default_used:
            click.echo(f"  → Using default: {default}")
        return default

    return result.strip()


def prompt_time(
    message: str, default: str = "24:00:00", show_default_used: bool = True
) -> str:
    """
    Prompt for a time value in HH:MM:SS format.

    Args:
        message: The prompt message to display.
        default: Default time string.
        show_default_used: If True, print message when default is used.

    Returns:
        The validated time string.

    Raises:
        KeyboardInterrupt: If user cancels the prompt.
    """
    import re

    def validate(text):
        if not text.strip():
            return True  # Empty is OK, will use default
        text = text.strip()
        # Accept HH:MM:SS or D-HH:MM:SS (SLURM format)
        if re.match(r"^(\d+-)?(\d{1,2}:)?\d{1,2}:\d{2}$", text):
            return True
        return "Please use format like '24:00:00' or '1-00:00:00'."

    result = questionary.text(message, default=default, validate=validate).ask()
    if result is None:
        raise KeyboardInterrupt

    if not result.strip():
        if show_default_used:
            click.echo(f"  → Using default: {default}")
        return default

    return result.strip()


@click.group(cls=BannerGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-v", "--version")
@click.pass_context
def cli(ctx):
    """
    AlleleFlux: Identify genomic targets of natural selection in bacterial communities.

    AlleleFlux is a Snakemake-based workflow for analyzing allele frequency changes
    in metagenomic time-series data to detect parallel evolution.

    \b
    Quick Start:
        1. Initialize a config file: alleleflux init
        2. Edit the config file with your paths
        3. Run the workflow: alleleflux run --config config.yml

    For more information, see: https://github.com/MoellerLabPU/AlleleFlux
    """
    # Print banner for actual command invocations (help is handled by BannerGroup)
    if ctx.invoked_subcommand is not None:
        print_banner()


# =============================================================================
# RUN Command
# =============================================================================


@cli.command(
    "run",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option(
    "-c",
    "--config",
    "config_file",
    type=click.Path(exists=True, resolve_path=True),
    required=True,
    help="Path to the AlleleFlux configuration file (YAML).",
)
@click.option(
    "-w",
    "--working-dir",
    type=click.Path(resolve_path=True),
    default=".",
    show_default=True,
    help="Working directory for the workflow.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=None,
    help="Max concurrent jobs (default: no limit, Snakemake schedules based on --threads). Only for local execution (ignored with --profile).",
)
@click.option(
    "-t",
    "--threads",
    type=int,
    default=None,
    help="Total threads available. Defaults to all CPU cores. Only for local execution (ignored with --profile).",
)
@click.option(
    "-m",
    "--memory",
    type=str,
    default=None,
    help="Total memory available (default: no limit). Formats: '64G', '128GB', '65536M'. Only for local execution (ignored with --profile).",
)
@click.option(
    "-p",
    "--profile",
    type=click.Path(resolve_path=True),
    default=None,
    help="Snakemake profile directory for cluster execution (e.g., SLURM).",
)
@click.option(
    "-n",
    "--dry-run",
    is_flag=True,
    default=False,
    help="Perform a dry run without executing jobs.",
)
@click.option(
    "--unlock",
    is_flag=True,
    default=False,
    help="Unlock the working directory (removes stale locks).",
)
@click.option(
    "--snakemake-args",
    "snakemake_args_str",
    type=str,
    default=None,
    help="Additional Snakemake arguments (quoted string, e.g., '--forceall --reason').",
)
@click.argument("extra_snakemake_args", nargs=-1, type=click.UNPROCESSED)
def run_workflow(
    config_file,
    working_dir,
    jobs,
    threads,
    memory,
    profile,
    dry_run,
    unlock,
    snakemake_args_str,
    extra_snakemake_args,
):
    """
    Run the AlleleFlux workflow.

    This command executes the complete AlleleFlux pipeline including:
    sample profiling, quality control, eligibility filtering,
    allele frequency analysis, statistical testing, and scoring.

    \b
    Resource Options (local execution only):
        --jobs/-j:    Max concurrent jobs (default: no limit)
        --threads/-t: Total threads available (default: all CPU cores)
        --memory/-m:  Total memory available (default: no limit)

    When using --profile for cluster execution (SLURM/PBS), these options
    are ignored and the profile handles resource scheduling.

    \b
    Passing Additional Snakemake Arguments:
        Two equivalent methods are available:

        Method 1 - Using '--' (no quotes needed):\n
            alleleflux run --config config.yml -- --forceall --reason

        Method 2 - Using '--snakemake-args' (quotes required):\n
            alleleflux run --config config.yml --snakemake-args '--forceall --reason'

        Common Snakemake options: --forceall, --forcerun, --reason, --notemp, --keep-going

    \b
    Examples:
        # Run with a config file (uses all CPU cores by default)\n
        alleleflux run --config config.yml

        # Run with limited resources\n
        alleleflux run --config config.yml --threads 16 --memory 64G

        # Dry run to see what would be executed\n
        alleleflux run --config config.yml --dry-run

        # Run with SLURM profile for cluster execution\n
        alleleflux run --config config.yml --profile slurm_profile/

        # Force rerun all jobs\n
        alleleflux run --config config.yml -- --forceall --reason
    """
    import shlex

    # Combine snakemake args from both methods
    combined_args = []
    if snakemake_args_str:
        combined_args.extend(shlex.split(snakemake_args_str))
    if extra_snakemake_args:
        combined_args.extend(list(extra_snakemake_args))

    extra_args = combined_args if combined_args else None

    exit_code = execute_workflow(
        config_file=config_file,
        working_dir=working_dir,
        jobs=jobs,
        threads=threads,
        memory=memory,
        profile=profile,
        dry_run=dry_run,
        unlock=unlock,
        extra_args=extra_args,
        version=__version__,
    )
    sys.exit(exit_code)


# =============================================================================
# INFO Command
# =============================================================================


@cli.command("info")
def show_info():
    """
    Display information about the AlleleFlux installation.

    Shows the version, installation path, and location of bundled files.
    """
    package_dir = Path(__file__).parent
    snakefile = get_snakefile()

    click.echo("")
    click.echo("AlleleFlux Installation Information")
    click.echo("=" * 40)
    click.echo(f"Version:        {__version__}")
    click.echo(f"Package dir:    {package_dir}")
    click.echo(f"Snakefile:      {snakefile}")
    click.echo(f"Workflow dir:   {snakefile.parent}")
    click.echo("")

    # Check if snakemake is available
    try:
        result = subprocess.run(
            ["snakemake", "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        click.echo(f"Snakemake:      {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.echo("Snakemake:      NOT FOUND (please install snakemake)")

    click.echo("")


# =============================================================================
# TOOLS Command
# =============================================================================


# Define all available tools organized by category
ALLELEFLUX_TOOLS = {
    "Analysis": [
        ("alleleflux-profile", "Profile MAGs from BAM files"),
        ("alleleflux-allele-freq", "Calculate allele frequencies"),
        ("alleleflux-scores", "Calculate significance scores"),
        ("alleleflux-taxa-scores", "Aggregate scores by taxonomic level"),
        ("alleleflux-gene-scores", "Calculate gene-level scores"),
        ("alleleflux-outliers", "Detect outlier genes"),
        ("alleleflux-cmh-scores", "Calculate CMH test scores"),
    ],
    "Preprocessing": [
        ("alleleflux-metadata", "Generate MAG metadata files"),
        ("alleleflux-qc", "Quality control for MAG profiles"),
        ("alleleflux-eligibility", "Create eligibility tables"),
        ("alleleflux-preprocess-between-groups", "Preprocess for between-group tests"),
        ("alleleflux-preprocess-within-group", "Preprocess for within-group tests"),
        ("alleleflux-preprocessing-eligibility", "Preprocessing eligibility check"),
    ],
    "Summary": [
        ("alleleflux-p-value-summary", "Summarize p-values across tests"),
        ("alleleflux-significant-sites-summary", "Roll up significant sites per MAG for heatmaps"),
    ],
    "Statistics": [
        ("alleleflux-lmm", "Linear Mixed Models analysis"),
        ("alleleflux-cmh", "Cochran-Mantel-Haenszel test"),
        ("alleleflux-single-sample", "Single-sample statistical test"),
        ("alleleflux-two-sample-paired", "Two-sample paired test"),
        ("alleleflux-two-sample-unpaired", "Two-sample unpaired test"),
    ],
    "Evolution": [
        ("alleleflux-dnds-from-timepoints", "Calculate dN/dS from timepoints"),
    ],
    "Accessory": [
        ("alleleflux-create-mag-mapping", "Create MAG-to-contig mapping file"),
        ("alleleflux-add-bam-path", "Add BAM paths to metadata"),
        ("alleleflux-coverage-allele-stats", "Calculate coverage and allele stats"),
        ("alleleflux-list-mags", "List MAGs in a directory"),
        ("alleleflux-positions-qc", "QC for genomic positions"),
    ],
    "Visualization": [
        ("alleleflux-prepare-metadata", "Prepare metadata for visualization"),
        ("alleleflux-terminal-nucleotide", "Terminal nucleotide analysis"),
        ("alleleflux-track-alleles", "Track allele frequencies"),
        ("alleleflux-plot-trajectories", "Plot allele trajectories"),
    ],
}


@cli.command("tools")
@click.option(
    "--category",
    "-c",
    type=click.Choice(list(ALLELEFLUX_TOOLS.keys()), case_sensitive=False),
    default=None,
    help="Show tools from a specific category only.",
)
def list_tools(category):
    """
    List all available AlleleFlux command-line tools.

    These are standalone commands that can be used independently
    or are called by the main workflow. Each tool has its own --help.

    \b
    Examples:
        # List all tools\n
        alleleflux tools

        # List only analysis tools\n
        alleleflux tools --category Analysis

        # Get help for a specific tool\n
        alleleflux-profile --help
    """
    click.echo("")
    click.echo(click.style("AlleleFlux Command-Line Tools", fg="cyan", bold=True))
    click.echo(click.style("=" * 50, fg="cyan"))
    click.echo("")

    categories = [category] if category else ALLELEFLUX_TOOLS.keys()

    for cat in categories:
        tools = ALLELEFLUX_TOOLS.get(cat, [])
        if not tools:
            continue

        click.echo(click.style(f"  {cat}", fg="bright_blue", bold=True))
        click.echo(click.style("  " + "-" * (len(cat) + 2), fg="bright_blue"))

        for tool_name, description in tools:
            click.echo(f"    {click.style(tool_name, fg='green'):40} {description}")

        click.echo("")

    click.echo(
        click.style("Tip:", fg="yellow")
        + " Run any tool with --help for usage details."
    )
    click.echo(f"     Example: {click.style('alleleflux-profile --help', fg='green')}")
    click.echo("")


# =============================================================================
# INIT Command (Interactive Configuration)
# =============================================================================


@cli.command()
@click.option(
    "--template",
    "use_template",
    is_flag=True,
    help="Print a template config to stdout.",
)
@click.option(
    "--output",
    default="alleleflux_config.yml",
    help="Output config file path.",
    show_default=True,
)
def init_config(use_template, output):
    """Creates a new configuration file for an AlleleFlux analysis."""
    template_path = get_template_path()
    with open(template_path, "r") as f:
        config_template = yaml.safe_load(f)

    if use_template:
        with open(template_path, "r") as f:
            click.echo(f.read())
        return

    click.echo("Welcome to AlleleFlux! Let's create your configuration file.\n")

    # --- Interactive Prompts ---
    try:
        # Output config file path (asked first)
        click.echo("📄 Configuration Output:")
        output = prompt_text("Where should the config file be saved?", default=output)

        # Run name
        run_name = prompt_text(
            "Run name (identifier for this analysis):", default="alleleflux_analysis"
        )

        # Required input files
        click.echo("\n📁 Required Input Files:")
        fasta_path = prompt_text(
            "Path to your reference FASTA file:", validate_file=True, required=True
        )
        metadata_path = prompt_text(
            "Path to metadata file (must include 'file_path' column with BAM paths):",
            validate_file=True,
            required=True,
        )
        prodigal_path = prompt_text(
            "Path to Prodigal genes file (.fna):", validate_file=True, required=True
        )
        gtdb_path = prompt_text(
            "Path to GTDB taxonomy file:", validate_file=True, required=True
        )
        mag_mapping_path = prompt_text(
            "Path to MAG mapping file:", validate_file=True, required=True
        )

        # Optional: Existing profiles for reuse
        click.echo("\n📁 Optional: Existing Profiles (to skip profiling step):")
        click.echo("  If you have profiles from a previous run, you can reuse them.")
        click.echo("  This saves compute time when only changing analysis parameters.")
        profiles_path = prompt_text(
            "Path to existing profiles directory (leave blank to generate new):",
            default="",
            show_default_used=False,
        )
        if profiles_path:
            if not Path(profiles_path).is_dir():
                click.echo(f"  ⚠️  Directory not found: {profiles_path}")
                click.echo("  Profiling will run from scratch.")
                profiles_path = ""
            else:
                click.echo(f"  ✓ Will reuse profiles from: {profiles_path}")

        # Output directory
        click.echo("\n📂 Output Configuration:")
        output_dir = prompt_text(
            "Where should the output files be saved?", default="./alleleflux_output"
        )

        # Analysis type
        click.echo("\n🔬 Analysis Configuration (press Enter to use defaults):")
        click.echo("Analysis Type Options:")
        click.echo("  • single: For analyzing allele frequencies at a single timepoint")
        click.echo(
            "  • longitudinal: For analyzing changes in allele frequencies across multiple timepoints"
        )

        data_type = prompt_text("Type of analysis:", default="longitudinal")
        if data_type not in ["single", "longitudinal"]:
            click.echo(f"Invalid data type '{data_type}'. Using default: longitudinal")
            data_type = "longitudinal"

        # Statistical analysis options
        click.echo("\n📊 Statistical Analysis Options:")
        click.echo("  LMM: Best for repeated measures, accounts for subject variation")
        click.echo("  Significance tests: Simple two-sample and single-sample comparisons")
        click.echo("  CMH: Stratified categorical analysis, detects directional changes")
        use_lmm = prompt_boolean(
            "Enable Linear Mixed Models (LMM)? (true/false):", default=True
        )
        use_significance_tests = prompt_boolean(
            "Enable significance tests (t-test, Mann-Whitney)? (true/false):", default=True
        )
        use_cmh = prompt_boolean(
            "Enable Cochran-Mantel-Haenszel (CMH) test? (true/false):", default=True
        )

        # Timepoint configuration
        click.echo("\n⏱️  Timepoint Configuration:")

        if data_type == "single":
            click.echo(
                "For single timepoint analysis, specify timepoint name(s) one at a time."
            )

            timepoints = []
            while True:
                count_msg = f" ({len(timepoints)} added)" if timepoints else ""
                tp = prompt_text(
                    f"Enter a timepoint name{count_msg} (or leave blank to finish):",
                    default="",
                    show_default_used=False,
                )
                if not tp:
                    if not timepoints:
                        click.echo("  ⚠️  At least one timepoint is required.")
                        continue
                    break
                timepoints.append(tp)
                click.echo(f"  ✓ Added timepoint: {tp}")

            click.echo(f"  → {len(timepoints)} timepoint(s) configured.")
            timepoints_combinations = [{"timepoint": [tp]} for tp in timepoints]

        else:  # longitudinal
            click.echo(
                "For longitudinal analysis, specify timepoint pairs and focus timepoint."
            )

            timepoints_combinations = []
            while True:
                count_msg = (
                    f" ({len(timepoints_combinations)} pair(s) added)"
                    if timepoints_combinations
                    else ""
                )
                tp1 = prompt_text(
                    f"Enter first timepoint name{count_msg} (or leave blank to finish):",
                    default="",
                    show_default_used=False,
                )
                if not tp1:
                    if not timepoints_combinations:
                        click.echo("  ⚠️  At least one timepoint pair is required.")
                        continue
                    break

                tp2 = prompt_text(
                    f"Enter second timepoint name (different from '{tp1}'):",
                    required=True,
                )
                if tp2 == tp1:
                    click.echo(
                        "  ⚠️  Second timepoint must differ from first. Try again."
                    )
                    continue

                # Explain focus timepoint
                click.echo("\n  ℹ️  Focus Timepoint:")
                click.echo("     The focus represents the DERIVED or LATER state.")
                click.echo("     - For dN/dS: focus = derived (Time 2), other = ancestral (Time 1)")
                click.echo("     - For CMH: measures sites significant at focus but not at other")
                click.echo(f"     - Guideline: Choose the LATER timepoint (typically '{tp2}')")
                
                focus = prompt_choice("Select the focus timepoint:", [tp1, tp2])
                timepoints_combinations.append(
                    {"timepoint": [tp1, tp2], "focus": focus}
                )
                click.echo(f"  ✓ Added pair: {tp1} → {tp2} (focus: {focus})")

            click.echo(
                f"  → {len(timepoints_combinations)} timepoint pair(s) configured."
            )

        # Group configuration
        click.echo("\n👥 Group Configuration:")
        click.echo("Specify group pairs to compare (e.g., 'treatment' vs 'control').")

        groups_combinations = []
        while True:
            count_msg = (
                f" ({len(groups_combinations)} pair(s) added)"
                if groups_combinations
                else ""
            )
            treatment = prompt_text(
                f"Enter treatment group name{count_msg} (or leave blank to finish):",
                default="",
                show_default_used=False,
            )
            if not treatment:
                if not groups_combinations:
                    click.echo("  ⚠️  At least one group pair is required.")
                    continue
                break

            control = prompt_text(
                f"Enter control group name (different from '{treatment}'):", required=True
            )
            if control == treatment:
                click.echo("  ⚠️  Control group must differ from treatment. Try again.")
                continue

            groups_combinations.append({"treatment": treatment, "control": control})
            click.echo(f"  ✓ Added comparison: {treatment} (treatment) vs {control} (control)")

        click.echo(f"  → {len(groups_combinations)} group comparison(s) configured.")

        # Quality control parameters
        click.echo("\n⚙️  Quality Control Parameters (press Enter to use defaults):")
        min_sample_num = prompt_integer(
            "Minimum number of samples required for analysis:", default=4, min_value=1
        )
        breadth_threshold = prompt_float(
            "Minimum coverage breadth threshold (0-1):",
            default=0.1,
            min_value=0.0,
            max_value=1.0,
        )
        coverage_threshold = prompt_float(
            "Minimum average coverage depth:", default=1.0, min_value=0.0
        )

        # Resource configuration
        click.echo("\n💻 Resource Configuration (press Enter to use defaults):")
        click.echo("These settings control resource usage for local or cluster runs.")
        threads_per_job = prompt_integer(
            "Threads per job (CPUs allocated to each parallel task):",
            default=16,
            min_value=1,
        )
        mem_per_job = prompt_memory(
            "Memory per job (e.g., '8G', '16GB', '32768M'):", default="8G"
        )
        time_limit = prompt_time(
            "Time limit per job (HH:MM:SS format):", default="24:00:00"
        )

        # Retry configuration
        click.echo("\n🔄 Retry Configuration (press Enter to use defaults):")
        click.echo("Jobs can automatically retry with more memory/time on failure.")
        retries = prompt_integer(
            "Number of retries per job (0 = no retries):",
            default=2,
            min_value=0,
            max_value=5,
        )
        if retries > 0:
            mem_step = prompt_memory(
                "Memory added per retry (e.g., '4G'):", default="4G"
            )
            time_step = prompt_time(
                "Time added per retry (e.g., '4:00:00'):", default="4:00:00"
            )
        else:
            mem_step = "0G"
            time_step = "0:00:00"

    except KeyboardInterrupt:
        click.echo("\nConfiguration cancelled.")
        return

    # --- Populate the config ---
    config_template["run_name"] = run_name
    config_template["input"]["fasta_path"] = fasta_path
    config_template["input"]["metadata_path"] = metadata_path
    config_template["input"]["prodigal_path"] = prodigal_path
    config_template["input"]["gtdb_path"] = gtdb_path
    config_template["input"]["mag_mapping_path"] = mag_mapping_path
    config_template["input"]["profiles_path"] = profiles_path

    config_template["output"]["root_dir"] = output_dir

    # Analysis configuration
    config_template["analysis"]["data_type"] = data_type
    config_template["analysis"]["use_lmm"] = use_lmm
    config_template["analysis"]["use_significance_tests"] = use_significance_tests
    config_template["analysis"]["use_cmh"] = use_cmh

    # Quality control
    config_template["quality_control"]["min_sample_num"] = min_sample_num
    config_template["quality_control"]["breadth_threshold"] = breadth_threshold
    config_template["quality_control"]["coverage_threshold"] = coverage_threshold

    # Resources
    config_template["resources"]["threads_per_job"] = threads_per_job
    config_template["resources"]["mem_per_job"] = mem_per_job
    config_template["resources"]["time"] = time_limit
    config_template["resources"]["retries"] = retries
    config_template["resources"]["mem_step"] = mem_step
    config_template["resources"]["time_step"] = time_step

    # Set timepoints and groups
    config_template["analysis"]["timepoints_combinations"] = timepoints_combinations
    config_template["analysis"]["groups_combinations"] = groups_combinations

    # --- Save the new config file ---
    # Custom representer to use flow style (inline) only for simple lists
    # This produces ["pre", "end"] but keeps complex lists in block style
    def represent_list(dumper, data):
        # Use flow style only for lists of simple types (strings, numbers, bools)
        # Not for lists containing dicts or nested lists
        is_simple = all(isinstance(item, (str, int, float, bool)) for item in data)
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=is_simple
        )

    yaml.add_representer(list, represent_list)

    with open(output, "w") as f:
        yaml.dump(config_template, f, sort_keys=False, default_flow_style=False)

    click.echo(f"\n✅ Configuration file '{output}' was created successfully!")
    click.echo("\n🚀 To start your analysis, run the following command:")
    click.echo(f"   alleleflux run --config {output}")


# =============================================================================
# Entry Point
# =============================================================================


def main():
    """Entry point for the AlleleFlux CLI."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        cli(standalone_mode=False)
    except click.exceptions.Abort:
        # User cancelled (e.g., during init prompts)
        click.echo("\nOperation cancelled.")
        sys.exit(130)
    except KeyboardInterrupt:
        # Catch any keyboard interrupt not caught elsewhere
        click.echo("\nInterrupted.")
        sys.exit(130)
    except SystemExit as e:
        # Propagate exit codes from workflow
        sys.exit(e.code)
    except click.ClickException as e:
        # Let Click handle its own exceptions normally
        e.show()
        sys.exit(e.exit_code)
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":

    main()
