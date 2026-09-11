#!/usr/bin/env python
import argparse
import json
import logging
import os
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import alleleflux.scripts.utilities.supress_warning as supress_warning
from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import load_allele_freq_inputs

NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]

logger = logging.getLogger(__name__)


def run_tTest(group1, group2):
    """
    Perform a paired t-test on two related samples of scores.

    Parameters:
    group1 (array-like): The first set of scores.
    group2 (array-like): The second set of scores.

    Returns:
    float: The p-value of the t-test.
    """
    t_stat, t_p = stats.ttest_rel(
        group1,
        group2,
        nan_policy="raise",
        alternative="two-sided",
    )
    return t_p


def run_wilcoxon(group1, group2):
    """
    Perform the Wilcoxon signed-rank test on two related samples.

    Parameters:
    group1 (array-like): The first set of observations.
    group2 (array-like): The second set of observations.

    Returns:
    float: The p-value of the test.

    Raises:
    ValueError: If the input arrays contain NaN values.
    """
    w_stat, w_p = stats.wilcoxon(
        group1,
        group2,
        nan_policy="raise",
        alternative="two-sided",
    )
    return w_p


def paired_test_for_nucleotide(values, filter_type):
    """
    Conducts a paired statistical test (t-test or Wilcoxon test) on a list of nucleotide values.

    Parameters:
    values (list or array-like): A list or array of nucleotide values to be tested.
    filter_type (str): The type of test to perform. Options are "t-test", "wilcoxon", "both", or "either".

    Returns:
    tuple: A tuple containing the p-values of the tests. If filter_type is "t-test", returns (t_p, None).
           If filter_type is "wilcoxon", returns (None, w_p). If filter_type is "both" or "either", returns (t_p, w_p).

    Notes:
    - If the number of values is less than or equal to 3, the function returns (1, None) for "t-test" and (None, 1) for "wilcoxon".
    - If the number of values is odd, the value closest to the median is removed before conducting the test.
    - If the values in both groups are identical, the function returns (1, None) for "t-test" and (None, 1) for "wilcoxon".
    - The function uses `run_tTest` and `run_wilcoxon` to perform the actual statistical tests.
    """
    # Assign p-values of 1 if there are less than 3 values. List of 3 will be converted to 2.
    # paired test using 2 value for t-test generally gives NaN and 1 for wilcoxon
    if len(values) <= 3:
        if filter_type == "t-test":
            return (1, None)
        elif filter_type == "wilcoxon":
            return (None, 1)
        else:
            return (1, 1)

    # Handle odd numbers: remove the value closest to the median.
    if len(values) % 2 != 0:
        median = np.median(values)
        distances = np.abs(values - median)
        # Find all indices with min distance
        min_indices = np.where(distances == distances.min())[0]
        # Remove first occurrence of closest to median
        values = np.delete(values, min_indices[0])

    # np.sort sorts in ascending order, so we use [::-1] to sort in descending order
    sorted_vals = np.sort(values)[::-1]

    # Compute the midpoint index of the sorted array.
    half = len(sorted_vals) // 2
    # First half of sorted values
    group1 = sorted_vals[:half]
    # Second half of sorted values
    group2 = sorted_vals[half:]

    # Check for identical values
    d = group1 - group2
    if np.all(d == 0):
        if filter_type == "t-test":
            # Paired t-test outputs NaN if both groups have identical values
            return (1, None)
        elif filter_type == "wilcoxon":
            # Wilcoxon gives an error if both groups have identical values
            return (None, 1)
        else:
            return (1, 1)

    # Conduct paired t-test.
    if filter_type == "t-test":
        t_p = run_tTest(group1, group2)
        return (t_p, None)
    elif filter_type == "wilcoxon":
        w_p = run_wilcoxon(group1, group2)
        return (None, w_p)
    elif filter_type in ["both", "either"]:
        t_p = run_tTest(group1, group2)
        w_p = run_wilcoxon(group1, group2)
        return (t_p, w_p)


def process_group(site_values, filter_type):
    """
    Run the paired statistical test for each nucleotide at a single site.

    Parameters:
        site_values (dict): Mapping of nucleotide -> 1D numpy array of values for
            that site. The caller (``filter_sites_parallel``) resolves which
            column backs each nucleotide (the ``data_type``-dependent
            ``{nuc}_diff_mean`` vs ``{nuc}`` choice) and extracts the arrays, so
            the worker receives only the minimal data it needs.
        filter_type (str): The type of test to perform on the nucleotide values.

    Returns:
        dict: A dictionary where keys are nucleotides and values are the
              (t_p, w_p) tuples resulting from the paired tests.

    Raises:
        ValueError: If any NA values are found, or if any NaN p-values are
                    encountered in the results.
    """
    pvalues = {}
    for nuc in NUCLEOTIDES:
        values = site_values[nuc]
        if pd.isnull(values).any():
            raise ValueError(f"NA values found for nucleotide '{nuc}'")
        nuc_pvals = paired_test_for_nucleotide(values, filter_type=filter_type)
        # Check that none of the returned p-values are NaN (ignoring None values)
        for p in nuc_pvals:
            if p is not None and np.isnan(p):
                raise ValueError(f"NaN p-value encountered for nucleotide {nuc}")
        pvalues[nuc] = nuc_pvals
    return pvalues


def process_site(args):
    """
    Processes a genomic site to determine if it should be removed based on statistical tests.
    Only remove the site only if ALL 4 nuleotides have nuc_remove as True.
    If any one nucleotide has nuc_remove as False, then remove_site is False.

    Parameters:
        args (tuple): A tuple containing:
            - (contig, position) (tuple): The contig and position of the site.
            - site_values (dict): Mapping of nucleotide -> 1D numpy array of values
              for the site (extracted by the caller).
            - alpha (float): The significance level for the statistical tests.
            - filter_type (str): The type of statistical test to perform. Can be "t-test", "wilcoxon", "both", or "either".

    Returns:
        tuple: A tuple containing:
            - (contig, position) (tuple): The contig and position of the site.
            - remove_site (bool): A boolean indicating whether the site should be removed (True) or not (False).
    """
    (contig, position), site_values, alpha, filter_type = args
    pvals = process_group(site_values, filter_type=filter_type)
    remove_site = True  # assume removal unless one nucleotide prevents it
    for nuc, (t_p, w_p) in pvals.items():
        if filter_type == "t-test":
            nuc_remove = t_p >= alpha
        elif filter_type == "wilcoxon":
            nuc_remove = w_p >= alpha
        elif filter_type == "both":
            # Remove the site if both tests are not significant
            nuc_remove = (t_p >= alpha) and (w_p >= alpha)
        elif filter_type == "either":
            # Remove if either test is non-significant.
            nuc_remove = (t_p >= alpha) or (w_p >= alpha)

        if not nuc_remove:
            remove_site = False
            break
    return ((contig, position), remove_site)


def filter_sites_parallel(grouped, alpha, filter_type, cpus, data_type="longitudinal"):
    """
    Filters sites in parallel using multiprocessing.

    Parameters:
        grouped (iterable): An iterable of grouped data, where each element is a tuple
                            ((contig, position), group_df).
        alpha (float): The significance level for the statistical test.
        filter_type (str): The type of statistical test to perform.
        cpus (int): The number of CPU cores to use for parallel processing.
        data_type (str): The type of data ('longitudinal' or 'single').

    Returns:
        list: A list of sites to remove, where each site is represented by its (contig, position).
    """
    # Resolve, once in the parent, which column backs each nucleotide. The
    # data_type-dependent choice happens here so workers receive only the
    # per-nucleotide value arrays they need -- never whole per-site DataFrames.
    if data_type == "longitudinal":
        nuc_cols = {nuc: f"{nuc}_diff_mean" for nuc in NUCLEOTIDES}
    else:  # single
        nuc_cols = {nuc: nuc for nuc in NUCLEOTIDES}

    def _iter_site_tasks():
        # Lazily extract the minimal payload per site. Crucially we do NOT
        # materialise list(grouped): that would hold ~1 DataFrame object per
        # site (millions for large MAGs), and pickling whole DataFrames through
        # imap drove the process past its memory limit -- an OOM-killed worker
        # then left imap_unordered deadlocked. Shipping a handful of short
        # arrays per site keeps peak memory bounded.
        for (contig, position), group_df in grouped:
            site_values = {
                nuc: group_df[col].values for nuc, col in nuc_cols.items()
            }
            yield ((contig, position), site_values, alpha, filter_type)

    n_sites = grouped.ngroups
    # Batch tasks so 1.7M-site MAGs don't pay per-task IPC overhead. Correctness
    # is independent of chunksize (results are reduced order-independently).
    chunksize = max(1, min(1000, (n_sites // (cpus * 8)) or 1))
    with Pool(processes=cpus) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(
                    process_site, _iter_site_tasks(), chunksize=chunksize
                ),
                total=n_sites,
                desc="Processing sites",
            )
        )
    sites_to_remove = [site for site, remove_flag in results if remove_flag]
    return sites_to_remove


def calculate_position_eligibility(
    df: pd.DataFrame, min_sample_num: int
) -> tuple[int, int, int]:
    """
    Calculate position-level eligibility for unpaired and paired statistical tests.

    For each position, determines if there are enough samples to run:
    - Unpaired tests: Both groups need >= min_sample_num samples
    - Paired tests: >= min_sample_num shared replicates between groups

    Parameters:
        df: DataFrame with columns 'contig', 'position', 'group', and optionally 'replicate'
        min_sample_num: Minimum number of samples/replicates required per position

    Returns:
        tuple: (total_positions, positions_eligible_unpaired, positions_eligible_paired)
    """
    if df.empty or "group" not in df.columns:
        return 0, 0, 0

    groups = df["group"].unique()
    if len(groups) != 2:
        logger.warning(f"Expected 2 groups but found {len(groups)}: {groups}")
        return 0, 0, 0

    group_1, group_2 = groups

    # Group by position and count samples per group
    position_grouped = df.groupby(["contig", "position"], dropna=False)
    total_positions = len(position_grouped)

    positions_eligible_unpaired = 0
    positions_eligible_paired = 0

    for (contig, position), pos_df in position_grouped:
        group1_df = pos_df[pos_df["group"] == group_1]
        group2_df = pos_df[pos_df["group"] == group_2]

        # Unpaired: both groups need enough samples
        num_samples_g1 = len(group1_df)
        num_samples_g2 = len(group2_df)
        if num_samples_g1 >= min_sample_num and num_samples_g2 >= min_sample_num:
            positions_eligible_unpaired += 1

        # Paired: need enough shared replicates between groups
        if "replicate" in pos_df.columns:
            reps_g1 = set(group1_df["replicate"].unique())
            reps_g2 = set(group2_df["replicate"].unique())
            shared_reps = reps_g1 & reps_g2
            if len(shared_reps) >= min_sample_num:
                positions_eligible_paired += 1
        else:
            raise ValueError(
                "Column 'replicate' is required for calculating paired test eligibility."
            )

    logger.info(
        f"Position-level eligibility: {total_positions:,} total positions, "
        f"{positions_eligible_unpaired:,} eligible for unpaired tests, "
        f"{positions_eligible_paired:,} eligible for paired tests"
    )

    return total_positions, positions_eligible_unpaired, positions_eligible_paired


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Filter sites using paired tests on sorted values from mean_changes_df.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mean_changes_fPath",
        required=True,
        help="Path to mean changes dataframe",
        type=str,
    )

    parser.add_argument("--output_fPath", required=True, help="Path to output file")
    parser.add_argument(
        "--p_value_threshold", type=float, default=0.05, help="Significance level"
    )

    parser.add_argument(
        "--data_type",
        help="Is the data from a single timepoint or from a time series (longitudinal)?",
        type=str,
        choices=["single", "longitudinal"],
        default="longitudinal",
    )
    parser.add_argument(
        "--filter_type",
        choices=["t-test", "wilcoxon", "both", "either"],
        default="t-test",
        help="""
        - "t-test":   remove site if t_p >= alpha \n
        - "wilcoxon": remove site if w_p >= alpha \n
        - "both":     remove if (t_p >= alpha) AND (w_p >= alpha)
                    (i.e., remove if both tests are non-significant)
        - "either":   remove if (t_p >= alpha) OR (w_p >= alpha)
                    (i.e., remove if at least one test is non-significant)
        """,
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=cpu_count(),
        help="Number of processors to use",
    )
    parser.add_argument(
        "--mag_id",
        type=str,
        required=True,
        help="MAG ID being processed",
    )
    parser.add_argument(
        "--min_positions",
        type=int,
        default=1,
        help="Minimum number of eligible positions required after filtering for a MAG to be considered eligible",
    )
    parser.add_argument(
        "--min_sample_num",
        type=int,
        default=4,
        help="Minimum number of samples per group (unpaired) or shared replicates (paired) required per position",
    )
    parser.add_argument(
        "--status_dir",
        type=str,
        default=None,
        help="Directory to write preprocessing status file. If not provided, status file is written to the same directory as output.",
    )
    args = parser.parse_args()
    start_time = time.time()

    df = load_allele_freq_inputs(args.mean_changes_fPath)

    # Group data by "contig" and "position" (including groups with NA keys).
    grouped = df.groupby(["contig", "position"], dropna=False)

    logger.info(
        f"Determining sites to remove based on paired tests using {args.cpus} cpus. Filter type is set to {args.filter_type}. Data type is {args.data_type}"
    )
    sites_to_remove = filter_sites_parallel(
        grouped, args.p_value_threshold, args.filter_type, args.cpus, args.data_type
    )
    logger.info(f"Removing {len(sites_to_remove):,} sites")

    # Remove these sites from the DataFrame.
    removal_index = pd.MultiIndex.from_tuples(
        sites_to_remove, names=["contig", "position"]
    )
    df_filtered = (
        df.set_index(["contig", "position"])
        .drop(removal_index, errors="raise")
        .reset_index()
    )

    df_filtered.to_csv(args.output_fPath, sep="\t", index=False)
    logger.info(
        f"Retained {len(df_filtered):,} rows. Filtered data written to {args.output_fPath}"
    )

    # Calculate position-level eligibility for each test type
    input_rows = len(df)
    output_rows = len(df_filtered)
    mag_id = args.mag_id
    min_sample_num = args.min_sample_num

    total_positions, positions_eligible_unpaired, positions_eligible_paired = (
        calculate_position_eligibility(df_filtered, min_sample_num)
    )

    # Determine eligibility based on position counts
    eligible_unpaired = positions_eligible_unpaired >= args.min_positions
    eligible_paired = positions_eligible_paired >= args.min_positions

    status = {
        "mag_id": mag_id,
        "preprocess_type": "between_groups",
        "input_rows": input_rows,
        "output_rows": output_rows,
        "min_positions": args.min_positions,
        "min_sample_num": min_sample_num,
        "total_positions": total_positions,
        "positions_eligible_unpaired": positions_eligible_unpaired,
        "positions_eligible_paired": positions_eligible_paired,
        "eligible_unpaired": eligible_unpaired,
        "eligible_paired": eligible_paired,
        "filter_type": args.filter_type,
        "p_value_threshold": args.p_value_threshold,
        "data_type": args.data_type,
    }

    # Determine status file location
    if args.status_dir:
        os.makedirs(args.status_dir, exist_ok=True)
        status_file = os.path.join(
            args.status_dir, f"{mag_id}_preprocessing_status.json"
        )
    else:
        output_dir = os.path.dirname(args.output_fPath)
        status_file = os.path.join(output_dir, f"{mag_id}_preprocessing_status.json")

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    logger.info(f"Preprocessing status written to {status_file}")

    end_time = time.time()
    logger.info(f"Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
