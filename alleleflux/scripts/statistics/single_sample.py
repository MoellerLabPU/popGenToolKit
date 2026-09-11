import argparse
import logging
import os
import time
import warnings
from functools import partial
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


def perform_one_sample_tests(
    func_one_sample, grouped_df, group, cpus, num_tests, output_dir, mag_id
):
    start_time = time.time()

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(func_one_sample, grouped_df)
        records = []
        for result in tqdm(
            results_iter, desc="Performing one-sample tests", total=num_tests
        ):
            name_tuple, p_values, num_samples_dict, notes = result
            contig, gene_id, position = name_tuple
            record = {
                "contig": contig,
                "gene_id": gene_id,
                "position": position,
                **p_values,
                **num_samples_dict,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logger.info(f"Tests performed in {end_time - start_time:.2f} seconds")
    test_results = pd.DataFrame(records)
    # Identify p-value columns
    p_value_columns = [col for col in test_results.columns if "_p_value" in col]

    # Remove rows where all p-value columns are NaN
    test_results.dropna(subset=p_value_columns, how="all", inplace=True)
    logger.info(
        f"Saving single-sample significance results for MAG {mag_id} to {output_dir}"
    )
    test_results.to_csv(
        os.path.join(output_dir, f"{mag_id}_single_sample_{group}.tsv.gz"),
        index=False,
        sep="\t",
        compression="gzip",
    )


def run_one_sample_tests(args, group, min_sample_num):
    name_tuple, df_group = args
    # Initialize p_values with NaN and notes
    p_values = {}
    notes = ""
    num_samples_dict = {}

    num_samples = df_group.shape[0]
    # Set the number of samples
    num_samples_dict[f"num_samples_{group}"] = num_samples

    # Initialize p-values for all nucleotides for this group to NaN
    for nucleotide in NUCLEOTIDES:
        p_values[f"{nucleotide}_p_value_tTest_{group}"] = np.nan
        p_values[f"{nucleotide}_p_value_Wilcoxon_{group}"] = np.nan

    # Only perform the tests if the group has at least min_sample_num data points
    if num_samples >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            nuc_col = f"{nucleotide}_diff_mean"
            data = df_group[nuc_col]
            # Check for zero variance and zero mean ie. 0,0,0,0. T-test is NA, and wilcoxon gives an error is this case. P-value is set at 1.
            var = np.var(data, ddof=1)
            mean = np.mean(data)
            if var == 0 and mean == 0:
                # Store p-value as 1
                p_values[f"{nucleotide}_p_value_tTest_{group}"] = 1.0
                p_values[f"{nucleotide}_p_value_Wilcoxon_{group}"] = 1.0

                notes += f"{nucleotide} in {group}: identical values, p-value for both tests is set to 1; "
            else:
                # Perform one-sample tests
                res_tTest = stats.ttest_1samp(
                    data,
                    0.0,
                    alternative="two-sided",
                    nan_policy="raise",
                )
                p_values[f"{nucleotide}_p_value_tTest_{group}"] = res_tTest.pvalue

                # Perform Wilcoxon signed-rank test
                res_wilcoxon = stats.wilcoxon(
                    data, alternative="two-sided", nan_policy="raise"
                )
                p_values[f"{nucleotide}_p_value_Wilcoxon_{group}"] = res_wilcoxon.pvalue

    return (name_tuple, p_values, num_samples_dict, notes)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run two sample unpaired test for a MAG across different samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--df_fPath",
        required=True,
        help="Path to mean changes dataframe",
        type=str,
    )
    parser.add_argument(
        "--min_sample_num",
        type=int,
        default=4,
        help="Minimum number of replicates per group required",
    )

    parser.add_argument(
        "--mag_id",
        help="MAG ID to process",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--group",
        help="Group to perform tests on,",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cpus",
        help=f"Number of processors to use.",
        default=cpu_count(),
        type=int,
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    # `group` labels can be purely numeric (e.g. "40", "20"); a bare read_csv would
    # sniff them as int64 and the `== args.group` (str) filter below would match zero
    # rows. load_allele_freq_inputs applies the dtype for both the preprocessed TSV
    # (preprocess_within_groups=True) and the raw parquet (=False), so `group` stays a
    # string and the loader transparently handles either input format.
    input_df = load_allele_freq_inputs(
        args.df_fPath, dtype={"gene_id": str, "group": str}
    )

    # Filter the dataframe to only include positions from the specified group.
    input_df = input_df[input_df["group"] == args.group]
    logger.info(
        f"Data filtered for group '{args.group}', resulting in {input_df.shape[0]:,} rows."
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Check if dataframe is empty after preprocessing.
    if input_df.empty:
        raise ValueError(f"{input_df} is empty.")

    # Group the data
    logger.info("Grouping data by contig, gene_id and position")
    grouped_df = input_df.groupby(["contig", "gene_id", "position"], dropna=False)

    num_tests = len(grouped_df)

    logger.info(
        f"Performing tests for {num_tests:,} positions using {args.cpus} cores."
    )
    func_one_sample = partial(
        run_one_sample_tests,
        group=args.group,
        min_sample_num=args.min_sample_num,
    )

    perform_one_sample_tests(
        func_one_sample,
        grouped_df,
        args.group,
        args.cpus,
        num_tests,
        args.output_dir,
        args.mag_id,
    )


if __name__ == "__main__":
    main()
