import argparse
import gc
import glob
import logging
import os
import time
from collections import defaultdict
from functools import partial
from itertools import islice
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]


def calculate_mag_sizes(fasta_file):
    logging.info("Parsing FASTA file to calculate MAG size.")
    mag_sizes = defaultdict(int)
    for record in SeqIO.parse(fasta_file, "fasta"):
        contig_id = record.id
        # Extract MAG ID from contig ID
        # The MAG ID is everything before '.fa' in the contig ID
        mag_id = contig_id.split(".fa")[0]
        # Accumulate the length of the contig to the MAG's total size
        mag_sizes[mag_id] += len(record.seq)
    return mag_sizes


def load_metadata(metadata_file):
    logging.info(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t")
    # Create a dictionary from the metadata where file_prefix is the key and group is the value
    metadata_dict = metadata.set_index("sampleID")[["group", "subjectID"]].to_dict(
        orient="index"
    )
    return metadata_dict


# def load_mag_files(input_dir):
#     mag_files = defaultdict(list)
#     # Traverse the input directory and sub-directories
#     for root, dirs, files in os.walk(input_dir):
#         logging.info(f"Looking for files in: {root}")
#         for file in files:
#             if file.endswith("_profiled.tsv.gz"):
#                 filepath = os.path.join(root, file)
#                 # Extract dirName from the directory name
#                 dirName = os.path.basename(root)
#                 # Remove '_profiled.tsv.gz' from filename
#                 filename_core = file[: -len("_profiled.tsv.gz")]
#                 # Remove dirName and underscore from the filename to get MAG ID
#                 if filename_core.startswith(dirName + "_"):
#                     mag_id = filename_core[len(dirName) + 1 :]
#                 else:
#                     logging.warning(f"Filename does not start with sample ID: {file}")
#                     continue
#                 mag_files[mag_id].append((dirName, filepath))
#     return mag_files


def load_mag_files(metadata_file):
    logging.info(f"Loading MAG files from metadata: {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep="\t")
    mag_files = defaultdict(list)

    for idx, row in metadata.iterrows():
        sample_id = str(row["sampleID"])
        dir_path = str(row["dirPath"])
        dir_path = os.path.abspath(dir_path)  # Get absolute path

        # Construct the expected filename pattern
        # According to your format: {dirPath}_{MAG_ID}_profiled.tsv.gz
        # Since dirPath is already the directory, we can adjust the pattern
        pattern = os.path.join(
            dir_path, f"{os.path.basename(dir_path)}_*_profiled.tsv.gz"
        )

        # Find all matching files
        files = glob.glob(pattern)

        for filepath in files:
            filename = os.path.basename(filepath)
            # Extract mag_id from filename
            # Filename format: {dirPath}_{MAG_ID}_profiled.tsv.gz
            # Remove '{dirPath}_' and '_profiled.tsv.gz' to get mag_id
            prefix = os.path.basename(dir_path) + "_"
            suffix = "_profiled.tsv.gz"
            if filename.startswith(prefix) and filename.endswith(suffix):
                mag_id = filename[len(prefix) : -len(suffix)]
                mag_files[mag_id].append((sample_id, filepath))
            else:
                logging.warning(f"Filename does not match expected pattern: {filename}")
    return mag_files


def calculate_frequencies(df):
    total_coverage = df["total_coverage"]

    # Calculate frequencies directly
    df["A_frequency"] = df["A"] / total_coverage
    df["C_frequency"] = df["C"] / total_coverage
    df["T_frequency"] = df["T"] / total_coverage
    df["G_frequency"] = df["G"] / total_coverage

    return df


def process_sample_file(args):
    sample_id, filepath, mag_id, breath_threshold = args
    df = pd.read_csv(filepath, sep="\t")
    # Add sample_id column
    df["sample_id"] = sample_id
    # Add metadata columns
    metadata_info = metadata_dict.get(
        sample_id, {"group": "Unknown", "subjectID": "Unknown"}
    )
    df["group"] = metadata_info["group"]
    df["subjectID"] = metadata_info["subjectID"]
    # This adds MAG_ID as the first column
    df.insert(0, "MAG_ID", mag_id)
    # df["MAG_ID"] = mag_id

    # Get MAG size (total number of positions in the MAG)
    mag_size = mag_size_dict.get(mag_id)
    if mag_size is None:
        logging.warning(f"Size for MAG {mag_id} not found in sample {sample_id}.")
        return None  # Skip this sample-MAG combination

    # Calculate the number of positions with total_coverage >= 1
    positions_with_coverage = df[df["total_coverage"] >= 1].shape[0]

    # Calculate breadth
    breadth = positions_with_coverage / mag_size

    if breadth < breath_threshold:
        logging.info(
            f"\nMAG {mag_id} in sample {sample_id} has breadth {breadth:.2%}, which is less than {breath_threshold:.2%}. Skipping this sample-MAG combination."
        )
        return None  # Skip this sample-MAG combination
    else:
        df["breadth"] = breadth
        df["genome_size"] = mag_size
        return df


def init_worker(metadata, mag_sizes):
    global metadata_dict
    global mag_size_dict
    metadata_dict = metadata
    mag_size_dict = mag_sizes


def run_wilcoxon_test(args, group_1, group_2, min_sample_num):
    name_tuple, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Initialize notes
    notes = ""

    # subjectID is the main column to merge on here, as that is what's same in the before and after samples
    merge_cols = ["subjectID", "contig", "position", "gene"]
    merged = pd.merge(
        group1, group2, on=merge_cols, suffixes=("_group1", "_group2"), how="inner"
    )

    # Check if we have enough paired samples
    num_samples = merged.shape[0]
    if num_samples >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            x = merged[f"{nucleotide}_group1"]
            y = merged[f"{nucleotide}_group2"]
            # Compute differences
            d = x - y
            # Check if all differences are zero
            # https://stackoverflow.com/a/65227113/12671809 statitics doesn't make sense if all differences are zero
            # https://stackoverflow.com/a/18402696/12671809
            if np.all(d == 0):
                # All differences are zero, set p-value to 1.0
                p_values[f"{nucleotide}_p_value"] = 1.0
                notes += f"{nucleotide}: all differences zero, p-value is set to 1; "
            else:
                res = stats.wilcoxon(
                    x,
                    y,
                    alternative="two-sided",
                    nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
                )
                p_values[f"{nucleotide}_p_value"] = res.pvalue

    return (name_tuple, p_values, num_samples, num_samples, notes)


def run_mannwhitneyu_test(args, group_1, group_2, min_sample_num):
    name_tuple, group = args

    # group is basically the dataframe, grouped by 'scaffold' and 'position'
    # Separate the data into two groups
    group1 = group[group["group"] == group_1]
    group2 = group[group["group"] == group_2]

    # Initialize p_values with NaN
    # Initially for p-values are NA, because for the sites where number of samples are less than 2, None is returned, causing errors
    p_values = {f"{nuc}_p_value": np.nan for nuc in NUCLEOTIDES}

    # Initialize notes
    notes = ""

    # Counts of samples in each group
    num_samples_group1 = group1.shape[0]
    num_samples_group2 = group2.shape[0]

    # Only perform the t-test if both groups have at least 2 data points
    if num_samples_group1 >= min_sample_num and num_samples_group2 >= min_sample_num:
        for nucleotide in NUCLEOTIDES:
            res = stats.mannwhitneyu(
                group1[nucleotide],
                group2[nucleotide],
                alternative="two-sided",
                nan_policy="raise",  # No NaN values should be present. But if present a ValueError will be raised
            )

            p_values[f"{nucleotide}_p_value"] = res.pvalue

    return (name_tuple, p_values, num_samples_group1, num_samples_group2, notes)


def perform_tests_parallel(df, cpus, group_1, group_2, min_sample_num=6, paired=False):
    start_time = time.time()

    # Group the data
    logging.info("\n Grouping data by scaffold and position")
    grouped = df.groupby(["contig", "position", "gene_id"])

    # Mann-Whitney U Test (or Wilcoxon rank-sum test) is used for independent samples
    # Wilcoxon signed-rank test is used for comparing two paired samples
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#mannwhitneyu
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html#wilcoxon

    if paired:
        logging.info(
            f"Performing Wilcoxon signed-rank tests for dependent samples in parallel using {cpus} cores."
        )
        my_func = partial(
            run_wilcoxon_test,
            group_1=group_1,
            group_2=group_2,
            min_sample_num=min_sample_num,
        )
    else:
        logging.info(
            f"Performing Mann-Whitney U tests for independent samples in parallel using {cpus} cores."
        )
        my_func = partial(
            run_mannwhitneyu_test,
            group_1=group_1,
            group_2=group_2,
            min_sample_num=min_sample_num,
        )

    with Pool(processes=cpus) as pool:
        results_iter = pool.imap_unordered(my_func, grouped)
        records = []
        for result in results_iter:
            name_tuple, p_values, num_samples_group1, num_samples_group2, notes = result
            contig, position, gene_id = name_tuple
            record = {
                "contig": contig,
                "position": position,
                "gene_id": gene_id,
                **p_values,
                f"num_samples_{group_1}": num_samples_group1,
                f"num_samples_{group_2}": num_samples_group2,
                "notes": notes,
            }
            records.append(record)

    end_time = time.time()
    logging.info(
        f"Groups added to a list and tests performed in {end_time - start_time:.2f} seconds"
    )
    test_results = pd.DataFrame(records)
    p_value_columns = [nucleotide + "_p_value" for nucleotide in NUCLEOTIDES]
    test_results["min_p_value"] = test_results[p_value_columns].min(axis=1)
    # print(test_results)
    return apply_bh_correction(test_results)


def apply_bh_correction(test_results):
    """
    Apply Benjamini-Hochberg correction to p-values in the given test results DataFrame.

    This function takes a DataFrame containing p-values for allele frequencies (A, T, G, C)
    and applies the Benjamini-Hochberg correction to control the false discovery rate (FDR).
    The corrected p-values are added to the DataFrame as new columns with the suffix '_adj'.

    Parameters:
    test_results (pd.DataFrame): A DataFrame containing p-values for allele frequencies.
                                 The DataFrame must have the following columns:
                                 - "A_frequency_p_value"
                                 - "T_frequency_p_value"
                                 - "G_frequency_p_value"
                                 - "C_frequency_p_value"

    Returns:
    pd.DataFrame: A new DataFrame with the original p-values and the adjusted p-values.
                  The adjusted p-values are added as new columns with the suffix '_adj'.

    Raises:
    ValueError: If NaN values are found in the p-value columns after cleaning.
    """

    # Only keep rows where no p-value column is NaN
    p_value_columns = [nucleotide + "_p_value" for nucleotide in NUCLEOTIDES]
    test_results_cleaned = test_results.dropna(subset=p_value_columns, how="any").copy()
    # mask = ~test_results[p_value_columns].isna().all(axis=1)
    # test_results_cleaned = test_results.loc[mask]

    # Ensure there are no NaN values in p-value columns
    if test_results_cleaned[p_value_columns].isnull().values.any():
        logging.error("NaNs found in p-value columns after cleaning.")
        raise ValueError("NaNs found in p-value columns after cleaning.")

    if test_results_cleaned.empty:
        logging.warning("No p-values to correct.")
        # raise ValueError("No p-values to correct.")
        return test_results_cleaned

    for col in p_value_columns:
        # Apply Benjamini-Hochberg correction
        logging.info(f"Applying Benjamini-Hochberg correction to {col}.")
        p_values = test_results_cleaned[col]
        _, p_adjusted, _, _ = multipletests(p_values, method="fdr_bh", alpha=0.05)

        # Create a Series with adjusted p-values
        adjusted_p_values = pd.Series(p_adjusted, index=p_values.index)

        # Assign adjusted p-values back to the DataFrame
        test_results_cleaned.loc[adjusted_p_values.index, f"{col}_adj"] = (
            adjusted_p_values
        )

    p_value_columns_adj = [column + "_adj" for column in p_value_columns]
    test_results_cleaned["min_p_value_adj"] = test_results_cleaned[
        p_value_columns_adj
    ].min(axis=1)
    return test_results_cleaned


def main():

    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
    )

    parser = argparse.ArgumentParser(
        description="Perform Mann-Whitney U test with BH correction for nucleotide frequencies at each site.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # parser.add_argument(
    #     "--input_dir",
    #     help="Path to directory with input files",
    #     type=str,
    #     required=True,
    #     metavar="filepath",
    # )

    parser.add_argument(
        "--metadata_file",
        help="Path to metadata file",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--fasta",
        help="Path to FASTA file with contigs",
        type=str,
        required=True,
        metavar="filepath",
    )
    parser.add_argument(
        "--breath_threshold",
        help="Breath threshold to use for MAGs.",
        type=float,
        default=0.1,
        metavar="float",
    )

    parser.add_argument(
        "--cpus",
        help=f"Number of processors to use.",
        default=16,
        metavar="int",
        type=int,
    )
    parser.add_argument(
        "--group_1",
        help="First group to compare",
        type=str,
        required=True,
        metavar="str",
    )

    parser.add_argument(
        "--group_2",
        help="Second group to compare",
        type=str,
        required=True,
        metavar="str",
    )

    parser.add_argument(
        "--min_sample_num",
        help="Minimum number of samples per group to perform the significance test",
        type=int,
        default=6,
        metavar="int",
    )

    parser.add_argument(
        "--paired",
        help="Samples are paired or dependent (e.g., before and after treatment)",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--get_taxa_score",
        help="Samples are paired or dependent (e.g., before and after treatment)",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--output_dir",
        help="Path to output directory",
        type=str,
        required=True,
        metavar="filepath",
    )

    args = parser.parse_args()
    start_time = time.time()

    # Calculate MAG sizes
    mag_size_dict = calculate_mag_sizes(args.fasta)

    # Load metadata
    metadata_dict = load_metadata(args.metadata_file)

    groups_in_metadata = set(entry["group"] for entry in metadata_dict.values())

    # Check if args.group_1 and args.group_2 are in the metadata groups
    if args.group_1 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_1} not found in metadata file.")

    if args.group_2 not in groups_in_metadata:
        raise ValueError(f"Group {args.group_2} not found in metadata file.")

    # Load MAG files
    mag_files = load_mag_files(args.metadata_file)

    # mag_files = dict(islice(mag_files.items(), 2))

    # List to collect test results from all MAGs
    test_results_list = []
    # Loop over each MAG
    for mag_id, sample_files in tqdm(
        mag_files.items(), total=len(mag_files), desc="Reading MAG files"
    ):
        sample_files_with_mag_id = [
            (  # dirName.split(".")[0] to get the sample_id. Removes ".sorted" from the sample ID
                dirName.split(".")[0],
                filepath,
                mag_id,
                args.breath_threshold,
            )
            for dirName, filepath in sample_files
        ]
        number_of_processes = min(16, len(sample_files_with_mag_id))

        # Load sample data in parallel
        with Pool(
            processes=number_of_processes,
            initializer=init_worker,
            initargs=(metadata_dict, mag_size_dict),
        ) as pool:
            data_list = list(
                pool.imap_unordered(process_sample_file, sample_files_with_mag_id)
            )

        # Filter out None values (samples with breadth < 50%)
        data_list = [df for df in data_list if df is not None]

        if not data_list:
            logging.info(
                f"No samples for MAG {mag_id} passed the breadth threshold. Skipping this MAG."
            )
            continue  # Skip further processing for this MAG

        # Concatenate data for this MAG
        mag_df = pd.concat(data_list, ignore_index=True)

        # Release memory from data_list
        del data_list
        gc.collect()

        # Calculate frequencies and perform statistical tests
        mag_df = calculate_frequencies(mag_df)

        test_results = perform_tests_parallel(
            mag_df,
            args.cpus,
            args.group_1,
            args.group_2,
            args.min_sample_num,
            args.paired,
        )
        # Add MAG_ID to test_results
        # test_results["MAG_ID"] = mag_id
        # This adds MAG_ID as the first column
        test_results.insert(0, "MAG_ID", mag_id)

        # Collect test_results into the list
        test_results_list.append(test_results)

        # Save test results for this MAG
        # mag_output_dir = os.path.join(args.output_dir, mag_id)
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Saving results for MAG {mag_id} to {args.output_dir}")

        mag_df.to_csv(
            os.path.join(args.output_dir, f"{mag_id}_nucleotide_frequencies.tsv"),
            index=False,
            sep="\t",
        )

        test_results.to_csv(
            os.path.join(args.output_dir, f"{mag_id}_test_results.tsv"),
            index=False,
            sep="\t",
        )

        # Release memory from mag_df and test_results if necessary
        del mag_df, test_results
        gc.collect()

    # After processing all MAGs, concatenate all test results
    combined_test_results = pd.concat(test_results_list, ignore_index=True)

    # Save the combined test_results table
    combined_test_results.to_csv(
        os.path.join(args.output_dir, "combined_test_results.tsv"),
        index=False,
        sep="\t",
    )

    logging.info(
        f"Combined test results saved to {args.output_dir}/combined_test_results.tsv"
    )
    del test_results_list
    gc.collect()
    end_time = time.time()
    logging.info(f"Total time taken: {end_time-start_time:.2f} seconds")


if __name__ == "__main__":
    main()
