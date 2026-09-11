"""Gene-level scoring and outlier detection rules.

This module contains rules for:
- Calculating gene-level significance scores from p-value tables
- Detecting outlier genes based on MAG-level and gene-level score comparisons
- Handling CMH-specific gene score calculations

Empty output files are created when no gene IDs are present in input data
to maintain workflow continuity.
"""

import os
import pandas as pd
from snakemake.logging import logger
import subprocess

def check_for_gene_ids(pvalue_table_path):
    """
    Check if the p-value table contains any gene IDs in the gene_id column.
    Returns True if gene IDs exist, False otherwise.
    """
    # logger = logging.getLogger(__name__)
    
    # Open file based on whether it's gzipped or not
    df = pd.read_csv(pvalue_table_path, sep='\t')

    # If column exists but all values are NaN, then no gene IDs exist
    if df['gene_id'].isna().all():
        logger.warning(f"gene_id column exists but all values are NaN in {pvalue_table_path}")
        return False
    
    # If we got here, gene IDs exist
    return True
    


rule gene_scores:
    input:
        pvalue_table=os.path.join(
            OUTDIR,
            "significance_tests",
            "{test_type}_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}.tsv.gz",
        ),
    output:
        combined=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}_gene_scores_combined.tsv",
        ),
        individual=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}_gene_scores_individual.tsv",
        ),
        overlapping=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}_gene_scores_overlapping.tsv",
        ),
    params:
        prefix="{mag}_{test_type}{group_str}",
        pValue_threshold=config["statistics"].get("p_value_threshold", 0.05),
        outDir=os.path.join(
            OUTDIR, "scores", "processed", "gene_scores_{timepoints}-{groups}"
        ),
    retries: get_retries("gene_scores")
    resources:
        mem_mb=get_mem_mb("gene_scores"),
        time=get_time("gene_scores"),
        runtime=get_runtime("gene_scores"),
    run:
        # Check if input file exists and has gene IDs
        if check_for_gene_ids(input.pvalue_table):
            # Execute the original shell command
            cmd = f"""
            alleleflux-gene-scores \\
                --pValue_table {input.pvalue_table} \\
                --pValue_threshold {params.pValue_threshold} \\
                --output_dir {params.outDir} \\
                --prefix {params.prefix}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e
        else:
            test_type = wildcards.test_type
            if wildcards.group_str:
                group = wildcards.group_str.strip('_')        
            # Define test names based on test_type
            if test_type == "two_sample_paired":
                tests = ["tTest", "Wilcoxon"]
            elif test_type == "two_sample_unpaired":
                tests = ["tTest", "MannWhitney"]
            elif test_type == "lmm" or test_type == "lmm_across_time":
                tests = ["lmm"]
            elif test_type == "cmh_across_time":
                tests = ["cmh"]
            elif test_type == "single_sample":
                tests = ["tTest", "Wilcoxon"]
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Create columns for the empty DataFrame
            columns = ['gene_id']
            
            # Handle the column creation differently based on test_type
            if test_type == "lmm":
                # Regular LMM has only one test
                columns.extend([
                    f'total_sites_per_group_{tests[0]}',
                    f'significant_sites_per_group_{tests[0]}',
                    f'score_{tests[0]} (%)'
                ])
            elif (test_type == "lmm_across_time" or test_type == "cmh_across_time") and group:
                # LMM across time with specific group
                columns.extend([
                    f'total_sites_per_group_{tests[0]}_{group}',
                    f'significant_sites_per_group_{tests[0]}_{group}',
                    f'score_{tests[0]}_{group} (%)'
                ])
            elif test_type == "single_sample" and group:
                # Add group name to column headers for single_sample
                for test in tests:
                    columns.extend([
                        f'total_sites_per_group_{test}_{group}',
                        f'significant_sites_per_group_{test}_{group}',
                        f'score_{test}_{group} (%)'
                    ])
            elif test_type == "two_sample_unpaired" or test_type == "two_sample_paired":
                # For two_sample tests with two statistical tests
                for test in tests:
                    columns.extend([
                        f'total_sites_per_group_{test}',
                        f'significant_sites_per_group_{test}',
                        f'score_{test} (%)'
                    ])
            
            # Create empty output files to satisfy workflow dependencies
            os.makedirs(params.outDir, exist_ok=True)
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output.combined, sep='\t', index=False)
            empty_df.to_csv(output.individual, sep='\t', index=False)
            empty_df.to_csv(output.overlapping, sep='\t', index=False)
            logger.info(f"No gene IDs found in {input.pvalue_table}. Created empty output files.")


rule detect_outlier_genes:
    input:
        mag_score=os.path.join(
            OUTDIR,
            "scores",
            "intermediate",
            "MAG_scores_{timepoints}-{groups}",
            "{mag}_score_{test_type}{group_str}.tsv",
        ),
        gene_scores=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}_gene_scores_individual.tsv",
        ),
    output:
        os.path.join(
            OUTDIR,
            "outlier_genes",
            "{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}_outlier_genes.tsv",
        ),
    retries: get_retries("detect_outlier_genes")
    resources:
        mem_mb=get_mem_mb("detect_outlier_genes"),
        time=get_time("detect_outlier_genes"),
        runtime=get_runtime("detect_outlier_genes"),
    run:
        gene_df = pd.read_csv(input.gene_scores, sep='\t')
        if len(gene_df) == 0 or gene_df['gene_id'].isna().all():
            # Create an empty output file with appropriate columns based on test_type
            test_type = wildcards.test_type
            if wildcards.group_str:
                # Remove leading underscore from group_str
                group = wildcards.group_str.strip('_')
            
            # Define test names based on test_type - consistent with gene_scores rule
            if test_type == "two_sample_paired":
                tests = ["tTest", "Wilcoxon"]
            elif test_type == "two_sample_unpaired":
                tests = ["tTest", "MannWhitney"]
            elif test_type == "lmm" or test_type == "lmm_across_time":
                tests = ["lmm"]
            elif test_type == "cmh_across_time":
                tests = ["cmh"]
            elif test_type == "single_sample":
                tests = ["tTest", "Wilcoxon"]
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Create empty DataFrame with appropriate columns
            columns = ['gene_id']
            
            if test_type == "lmm":
                # Regular LMM has only one test
                test = tests[0]
                columns.extend([
                    f'mag_score_{test} (%)', f'gene_score_{test} (%)',
                    f'total_sites_gene_{test}', f'significant_sites_gene_{test}',
                    f'p_value_binomial_{test}', f'p_value_poisson_{test}'
                ])
            elif (test_type == "lmm_across_time" or test_type == "cmh_across_time") and group:
                # LMM across time with specific group
                test = tests[0]
                columns.extend([
                    f'mag_score_{test}_{group} (%)', f'gene_score_{test}_{group} (%)',
                    f'total_sites_gene_{test}_{group}', f'significant_sites_gene_{test}_{group}',
                    f'p_value_binomial_{test}_{group}', f'p_value_poisson_{test}_{group}'
                ])
            elif test_type == "single_sample" and group:
                # Add group name to column headers for single_sample
                for test in tests:
                    columns.extend([
                        f'mag_score_{test}_{group} (%)', f'gene_score_{test}_{group} (%)',
                        f'total_sites_gene_{test}_{group}', f'significant_sites_gene_{test}_{group}',
                        f'p_value_binomial_{test}_{group}', f'p_value_poisson_{test}_{group}'
                    ])
            elif test_type == "two_sample_unpaired" or test_type == "two_sample_paired":
                # Two sample unpaired tests
                for test in tests:
                    columns.extend([
                        f'mag_score_{test} (%)', f'gene_score_{test} (%)',
                        f'total_sites_gene_{test}', f'significant_sites_gene_{test}',
                        f'p_value_binomial_{test}', f'p_value_poisson_{test}'
                    ])
            
            # Create and save the empty DataFrame
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output[0], sep='\t', index=False)
            logger.info(f"No gene data found in {input.gene_scores}. Created empty output file with appropriate columns.")
        else:
            # Run the outlier detection
            cmd = f"""
            alleleflux-outliers \\
                --mag_file {input.mag_score} \\
                --mag_id {wildcards.mag} \\
                --gene_file {input.gene_scores} \\
                --out_fPath {output}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e

rule cmh_gene_scores:
    input:
        pvalue_table=os.path.join(
            OUTDIR,
            "significance_tests",
            "cmh_{timepoints}-{groups}",
            "{mag}_cmh.tsv.gz",
        ),
    output:
        combined=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_cmh_{focus_tp}_gene_scores_combined.tsv",
        ),
        individual=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_cmh_{focus_tp}_gene_scores_individual.tsv",
        ),
        overlapping=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_cmh_{focus_tp}_gene_scores_overlapping.tsv",
        ),
    params:
        prefix="{mag}_cmh_{focus_tp}",
        pValue_threshold=config["statistics"].get("p_value_threshold", 0.05),
        outDir=os.path.join(
            OUTDIR, "scores", "processed", "gene_scores_{timepoints}-{groups}"
        ),
    retries: get_retries("cmh_gene_scores")
    resources:
        mem_mb=get_mem_mb("cmh_gene_scores"),
        time=get_time("cmh_gene_scores"),
        runtime=get_runtime("cmh_gene_scores"),
    run:
        # Check if input file exists and has gene IDs
        if check_for_gene_ids(input.pvalue_table):
            # Execute the gene scores command
            cmd = f"""
            alleleflux-gene-scores \\
                --pValue_table {input.pvalue_table} \\
                --pValue_threshold {params.pValue_threshold} \\
                --output_dir {params.outDir} \\
                --prefix {params.prefix}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e
        else:
            # Create empty output files with appropriate columns for CMH test
            columns = ['gene_id', 'total_sites_per_group_CMH',
                      'significant_sites_per_group_CMH', 'score_CMH (%)']
            
            # Create empty output files to satisfy workflow dependencies
            os.makedirs(params.outDir, exist_ok=True)
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output.combined, sep='\t', index=False)
            empty_df.to_csv(output.individual, sep='\t', index=False)
            empty_df.to_csv(output.overlapping, sep='\t', index=False)
            logger.info(f"No gene IDs found in {input.pvalue_table}. Created empty output files.")


rule detect_cmh_outlier_genes:
    input:
        mag_score=os.path.join(
            OUTDIR,
            "scores",
            "intermediate",
            "MAG_scores_{timepoints}-{groups}",
            "{mag}_score_cmh_{focus_tp}.tsv",
        ),
        gene_scores=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "gene_scores_{timepoints}-{groups}",
            "{mag}_cmh_{focus_tp}_gene_scores_individual.tsv",
        ),
    output:
        os.path.join(
            OUTDIR,
            "outlier_genes",
            "{timepoints}-{groups}",
            "{mag}_cmh_{focus_tp}_outlier_genes.tsv",
        ),
    retries: get_retries("detect_cmh_outlier_genes")
    resources:
        mem_mb=get_mem_mb("detect_cmh_outlier_genes"),
        time=get_time("detect_cmh_outlier_genes"),
        runtime=get_runtime("detect_cmh_outlier_genes"),
    run:
        gene_df = pd.read_csv(input.gene_scores, sep='\t')
        if len(gene_df) == 0 or gene_df['gene_id'].isna().all():
            # Create an empty output file with appropriate columns for CMH test
            columns = ['gene_id', 'mag_score_CMH (%)', 'gene_score_CMH (%)',
                      'total_sites_gene_CMH', 'significant_sites_gene_CMH',
                      'p_value_binomial_CMH', 'p_value_poisson_CMH']
            
            # Create and save the empty DataFrame
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            empty_df = pd.DataFrame(columns=columns)
            empty_df.to_csv(output[0], sep='\t', index=False)
            logger.info(f"No gene data found in {input.gene_scores}. Created empty output file with appropriate columns.")
        else:
            # Run the outlier detection
            cmd = f"""
            alleleflux-outliers \\
                --mag_file {input.mag_score} \\
                --mag_id {wildcards.mag} \\
                --gene_file {input.gene_scores} \\
                --out_fPath {output}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e