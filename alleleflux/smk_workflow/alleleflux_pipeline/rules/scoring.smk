"""MAG-level and taxa-level scoring rules.

This module contains rules for:
- Calculating per-MAG significance scores from p-value tables
- Combining scores across MAGs for each timepoint-group combination
- Aggregating scores at different taxonomic levels (phylum to species)
- Special handling for CMH test scores with focus timepoint filtering
"""

import pandas as pd
# import logging
# from alleleflux.scripts.utilities.logging_config import setup_logging
from snakemake.logging import logger
import subprocess

rule significance_score_per_MAG_standard:
    input:
        # Standard test types use a single pvalue table
        pvalue_table=os.path.join(
            OUTDIR,
            "significance_tests",
            "{test_type}_{timepoints}-{groups}",
            "{mag}_{test_type}{group_str}.tsv.gz",
        ),
        gtdb_taxonomy=config["input"]["gtdb_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
    output:
        os.path.join(
            OUTDIR,
            "scores",
            "intermediate",
            "MAG_scores_{timepoints}-{groups}",
            "{mag}_score_{test_type}{group_str}.tsv"
        ),
    params:
        group_by_column="MAG_ID",
        pValue_threshold=config["statistics"].get("p_value_threshold", 0.05),
    retries: get_retries("significance_score_per_MAG_standard")
    resources:
        mem_mb=get_mem_mb("significance_score_per_MAG_standard"),
        time=get_time("significance_score_per_MAG_standard"),
        runtime=get_runtime("significance_score_per_MAG_standard"),
    shell:
        """
        alleleflux-scores \
            --gtdb_taxonomy {input.gtdb_taxonomy} \
            --pValue_table {input.pvalue_table} \
            --group_by_column {params.group_by_column} \
            --pValue_threshold {params.pValue_threshold} \
            --out_fPath {output} \
            --mag_mapping_file {input.mag_mapping}
        """


rule significance_score_per_MAG_cmh:
    input:
        pvalue_table=os.path.join(
            OUTDIR,
            "significance_tests",
            "cmh_{timepoints}-{groups}",
            "{mag}_cmh.tsv.gz",
        ),
        gtdb_taxonomy=config["input"]["gtdb_path"],
        mag_mapping=config["input"]["mag_mapping_path"],
    output:
        os.path.join(
            OUTDIR,
            "scores",
            "intermediate",
            "MAG_scores_{timepoints}-{groups}",
            "{mag}_score_cmh_{focus_tp}.tsv"
        ),
    params:
        # CMH-specific parameters
        tp1_name=lambda wildcards: wildcards.timepoints.split("_")[0] if DATA_TYPE == "longitudinal" else "",
        tp2_name=lambda wildcards: wildcards.timepoints.split("_")[1] if DATA_TYPE == "longitudinal" else "",
        pValue_threshold=config["statistics"].get("p_value_threshold", 0.05),
        group_by_column="MAG_ID",
        data_type=DATA_TYPE,
    retries: get_retries("significance_score_per_MAG_cmh")
    resources:
        mem_mb=get_mem_mb("significance_score_per_MAG_cmh"),
        time=get_time("significance_score_per_MAG_cmh"),
        runtime=get_runtime("significance_score_per_MAG_cmh"),
    run:
        if params.data_type == "single":
            cmd = f"""
            alleleflux-scores \\
                --gtdb_taxonomy {input.gtdb_taxonomy} \\
                --pValue_table {input.pvalue_table} \\
                --group_by_column {params.group_by_column} \\
                --pValue_threshold {params.pValue_threshold} \\
                --out_fPath {output} \\
                --mag_mapping_file {input.mag_mapping}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e
        elif params.data_type == "longitudinal":
            cmd = f"""
            alleleflux-cmh-scores \\
                --combined-file {input.pvalue_table} \\
                --tp1-name {params.tp1_name} \\
                --tp2-name {params.tp2_name} \\
                --focus {wildcards.focus_tp} \\
                --gtdb_taxonomy {input.gtdb_taxonomy} \\
                --mag_id {wildcards.mag} \\
                --threshold {params.pValue_threshold} \\
                --out_fPath {output}
            """
            logger.info(f"Executing: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
                raise e



rule combine_MAG_scores:
    input:
        scores=lambda wc: expand(
            os.path.join(
                OUTDIR,
                "scores",
                "intermediate",
                "MAG_scores_{timepoints}-{groups}",
                "{mag}_score_{test_type}{group_str}.tsv",
            ),
            timepoints=wc.timepoints,
            groups=wc.groups,
            test_type=wc.test_type,
            group_str=wc.group_str,
            mag=(
                get_eligible_mags(wc.timepoints, wc.groups, wc.test_type)
                if wc.test_type not in {"single_sample", "lmm_across_time", "cmh_across_time"}
                else [
                    mag
                    for mag, grp in get_eligible_mags(wc.timepoints, wc.groups, wc.test_type)
                    if f"_{grp}" == wc.group_str
                ]
            ),
        ),
    output:
        concatenated=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "MAG",
            "scores_{test_type}-{timepoints}-{groups}{group_str}-MAGs.tsv",
        ),
    retries: get_retries("combine_MAG_scores")
    resources:
        mem_mb=get_mem_mb("combine_MAG_scores"),
        time=get_time("combine_MAG_scores"),
        runtime=get_runtime("combine_MAG_scores"),
    run:
        dfs = []
        for file in input.scores:
            logger.info(f"Reading {file}")
            df = pd.read_csv(file, sep="\t")
            dfs.append(df)

        logger.info(
            f"Combining scores for {wildcards.timepoints}-{wildcards.groups} "
            f"({wildcards.test_type}{wildcards.group_str})"
        )

        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Writing combined scores to {output.concatenated}")
        combined_df.to_csv(output.concatenated, sep="\t", index=False)


rule combine_MAG_scores_cmh:
    input:
        scores=lambda wc: expand(
            os.path.join(
                OUTDIR,
                "scores",
                "intermediate",
                "MAG_scores_{timepoints}-{groups}",
                "{mag}_score_cmh_{focus_tp}.tsv",
            ),
            timepoints=wc.timepoints,
            groups=wc.groups,
            mag=get_eligible_mags(wc.timepoints, wc.groups, "cmh"),
            focus_tp=[wc.focus_tp],
        ),
    output:
        concatenated=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "MAG",
            "scores_cmh-{timepoints}-{groups}-MAGs-{focus_tp}.tsv",
        ),
    retries: get_retries("combine_MAG_scores_cmh")
    resources:
        mem_mb=get_mem_mb("combine_MAG_scores_cmh"),
        time=get_time("combine_MAG_scores_cmh"),
        runtime=get_runtime("combine_MAG_scores_cmh"),
    run:
        dfs = []
        insufficient_data_mags = []
        for file in input.scores:
            logger.info(f"Reading {file}")
            df = pd.read_csv(file, sep="\t")
            # Verify that the focus timepoint in the file matches the expected one
            if "focus_timepoint" in df.columns and df["focus_timepoint"].iloc[0] != wildcards.focus_tp:
                raise ValueError(f"Mismatched focus timepoint in {file}: "
                                 f"expected {wildcards.focus_tp}, found {df['focus_timepoint'].iloc[0]}")
            # Track MAGs with insufficient data (score=0 due to missing timepoint data)
            if "total_sites_per_group_CMH" in df.columns and df["total_sites_per_group_CMH"].iloc[0] == 0:
                mag_id = df["MAG_ID"].iloc[0] if "MAG_ID" in df.columns else file
                insufficient_data_mags.append(mag_id)
            dfs.append(df)

        if insufficient_data_mags:
            logger.warning(
                f"{len(insufficient_data_mags)} MAG(s) had insufficient CMH data "
                f"(score=0) for {wildcards.timepoints}-{wildcards.groups} "
                f"focus_tp={wildcards.focus_tp}: {insufficient_data_mags}"
            )

        logger.info(
            f"Combining CMH scores for {wildcards.timepoints}-{wildcards.groups} "
            f"with focus timepoint {wildcards.focus_tp}"
        )

        if not dfs:
            raise ValueError(f"No valid CMH score files found for focus timepoint {wildcards.focus_tp}")

        combined_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Writing combined scores to {output.concatenated}")
        combined_df.to_csv(output.concatenated, sep="\t", index=False)

rule taxa_scores:
    input:
        concatenated=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "MAG",
            "scores_{test_type}-{timepoints}-{groups}{group_str}-MAGs.tsv",
        ),
    output:
        os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "{taxon}",
            "scores_{test_type}-{timepoints}-{groups}{group_str}-{taxon}.tsv",
        ),
    retries: get_retries("taxa_scores")
    resources:
        mem_mb=get_mem_mb("taxa_scores"),
        time=get_time("taxa_scores"),
        runtime=get_runtime("taxa_scores"),
    shell:
        """
        alleleflux-taxa-scores \
            --input_df {input.concatenated} \
            --group_by_column {wildcards.taxon} \
            --out_fPath {output}
        """


rule taxa_scores_cmh:
    input:
        concatenated=os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "MAG",
            "scores_cmh-{timepoints}-{groups}-MAGs-{focus_tp}.tsv",
        ),
    output:
        os.path.join(
            OUTDIR,
            "scores",
            "processed",
            "combined",
            "{taxon}",
            "scores_cmh-{timepoints}-{groups}-{taxon}-{focus_tp}.tsv",
        ),
    retries: get_retries("taxa_scores_cmh")
    resources:
        mem_mb=get_mem_mb("taxa_scores_cmh"),
        time=get_time("taxa_scores_cmh"),
        runtime=get_runtime("taxa_scores_cmh"),
    shell:
        """
        alleleflux-taxa-scores \
            --input_df {input.concatenated} \
            --group_by_column {wildcards.taxon} \
            --out_fPath {output}
        """
