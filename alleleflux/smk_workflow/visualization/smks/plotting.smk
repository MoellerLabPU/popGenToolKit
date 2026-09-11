
# Pre-calculate plotting parameters for use in rules and functions
n_sites_line = config["plotting_params"]["n_sites_line"]
n_label = "ALL" if str(n_sites_line).lower() == "all" else str(n_sites_line)
x_col = config["plotting_params"]["x_col"]
output_fmt = config["plotting_params"]["output_format"]

# Determine x_col for filename based on binning.
# bin_width_days accepts a single int (back-compat) or a list of widths — the
# plotting CLI runs one full pass per width and tags each binned output file
# with _bin<W>d, so the expected filenames must be expanded over the widths.
_bin_cfg = config["plotting_params"].get("bin_width_days")
if _bin_cfg:
    bin_widths = _bin_cfg if isinstance(_bin_cfg, list) else [_bin_cfg]
    x_col_filename = "bin_midpoint"
    bin_tags = [f"_bin{w}d" for w in bin_widths]
else:
    bin_widths = []
    x_col_filename = x_col
    bin_tags = [""]  # unbinned outputs carry no tag

# Whether to also emit one combined line plot per replicate. Declared as a directory output
# below when enabled, so that flipping this flag on triggers a rerun of an already-complete
# run instead of silently leaving the new plots unmade.
combined_per_replicate = bool(config["plotting_params"].get("combined_per_replicate", False))

def get_plotting_targets(wildcards):
    """
    Determines the target plot files based on the MAGs identified in the terminal analysis.
    """
    # Get the output file from the checkpoint
    checkpoint_output = checkpoints.terminal_analysis.get(**wildcards).output.summary
    
    mags = []
    
    # Check if specific MAGs are configured
    specific_mags = config.get("specific_mags", [])
    if specific_mags:
        mags = specific_mags
    elif os.path.isfile(checkpoint_output):
        try:
            df = pd.read_csv(checkpoint_output, sep="\t")
            mags = df[df["sites_processed"] > 0]["mag_id"].unique().tolist()
        except Exception as e:
            print(f"Warning: Could not determine plotting targets: {e}")
            return []

    if mags:
        # We target the combined line plot as a sentinel for completion
        # Note: The filename format must match what plot_allele_trajectory.py produces
        # {mag_id}_combined_n{n_label}_{x_col}_{plot_type}.{output_format}

        targets = expand(
            os.path.join(config["output_dir"], "plots", "{mag_id}", f"{{mag_id}}_combined_n{n_label}_{x_col_filename}_line{{tag}}.{output_fmt}"),
            mag_id=mags,
            tag=bin_tags,
        )

        # The sentinel alone is not sufficient here. If a run already produced the pooled
        # plot and combined_per_replicate is switched on afterwards, requesting only the
        # sentinel leaves the job looking complete and the replicate plots are never made
        # (verified: Snakemake reports "Nothing to be done"). Requesting the directory as
        # well makes the missing output a rerun trigger.
        if combined_per_replicate:
            targets += expand(
                os.path.join(config["output_dir"], "plots", "{mag_id}", "by_replicate"),
                mag_id=mags
            )

        return targets

    return []

rule plot_trajectories:
    input:
        long_table = os.path.join(config["output_dir"], "track_freqs" , "{mag_id}_frequency_table.long.tsv")
    output:
        # We use the combined line plots as the main outputs to track — one per
        # bin width (a single unbinned/single-width run yields exactly one).
        sentinels = expand(
            os.path.join(config["output_dir"], "plots", "{{mag_id}}",
                f"{{{{mag_id}}}}_combined_n{n_label}_{x_col_filename}_line{{tag}}.{output_fmt}"
            ),
            tag=bin_tags,
        ),
        # Only declared when per-replicate plots are requested. The replicate IDs come from
        # the metadata and are not known when the DAG is built, so the containing directory
        # is tracked instead of the individual figures.
        **(
            {
                "rep_dir": directory(
                    os.path.join(config["output_dir"], "plots", "{mag_id}", "by_replicate")
                )
            }
            if combined_per_replicate
            else {}
        )
    params:
        out_dir = directory(os.path.join(config["output_dir"], "plots", "{mag_id}")),
        val_col = config["plotting_params"]["value_col"],
        n_line_arg = f"--n_sites_line {config['plotting_params']['n_sites_line']}" if config["plotting_params"].get("n_sites_line") is not None else "",
        n_dist_arg = f"--n_sites_dist {config['plotting_params']['n_sites_dist']}" if config["plotting_params"].get("n_sites_dist") is not None else "",
        n_per_site_arg = f"--n_sites_per_site {config['plotting_params']['n_sites_per_site']}" if config["plotting_params"].get("n_sites_per_site") is not None else "",
        x_col = config["plotting_params"]["x_col"],
        # Handle optional x_order list
        x_order_arg = f"--x_order {' '.join(config['plotting_params']['x_order'])}" if config["plotting_params"]["x_order"] else "",
        plot_types = " ".join(config["plotting_params"]["plot_types"]),
        per_site = "--per_site" if config["plotting_params"]["per_site"] else "",
        fmt = config["plotting_params"]["output_format"],
        # New parameters
        bin_width_arg = ("--bin_width_days " + " ".join(str(w) for w in bin_widths)) if bin_widths else "",
        min_samples_arg = f"--min_samples_per_bin {config['plotting_params']['min_samples_per_bin']}" if config["plotting_params"].get("min_samples_per_bin") else "",
        group_rep_arg = "--group_by_replicate" if config["plotting_params"].get("group_by_replicate") else "",
        combined_rep_arg = "--combined_per_replicate" if combined_per_replicate else "",
        line_alpha_arg = f"--line_alpha {config['plotting_params']['line_alpha']}" if config["plotting_params"].get("line_alpha") else "",
        # Initial frequency filtering
        max_init_freq_arg = f"--max_initial_freq {config['plotting_params']['max_initial_freq']}" if config["plotting_params"].get("max_initial_freq") else "",
        min_init_freq_arg = f"--min_initial_freq {config['plotting_params']['min_initial_freq']}" if config["plotting_params"].get("min_initial_freq") else "",
        init_freq_group_arg = f"--initial_freq_group {config['plotting_params']['initial_freq_group']}" if config["plotting_params"].get("initial_freq_group") else ""
    threads: 1
    shell:
        """
        alleleflux-plot-trajectories \
            --input_file {input.long_table} \
            --output_dir {params.out_dir} \
            --value_col {params.val_col} \
            {params.n_line_arg} \
            {params.n_dist_arg} \
            {params.n_per_site_arg} \
            --x_col {params.x_col} \
            {params.x_order_arg} \
            --plot_types {params.plot_types} \
            {params.per_site} \
            --output_format {params.fmt} \
            {params.bin_width_arg} \
            {params.min_samples_arg} \
            {params.group_rep_arg} \
            {params.combined_rep_arg} \
            {params.line_alpha_arg} \
            {params.max_init_freq_arg} \
            {params.min_init_freq_arg} \
            {params.init_freq_group_arg}
        """
