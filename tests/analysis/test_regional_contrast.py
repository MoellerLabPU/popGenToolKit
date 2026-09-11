#!/usr/bin/env python3
"""Regression tests for regional_contrast.aggregate_region_scores.

Guards against the categorical-groupby explosion: when a grouping column is a
pandas Categorical (as ``group`` is when loaded from parquet), a groupby that
does not pass ``observed=True`` reindexes the result to the full Cartesian
product of every grouping key's levels.  With the collinear region-metadata
columns (region_id/region_start/region_end/region_length) this exploded to a
23.7 PiB allocation in production.
"""

import pandas as pd
import pytest

from alleleflux.scripts.analysis.regional_contrast import (
    FISHER_PVAL_CONTROL_COL,
    FISHER_PVAL_TREATMENT_COL,
    _fisher_merge_keys,
    aggregate_region_scores,
    build_gene_regions,
    fisher_combine_empirical_pvalues,
    load_input_table,
    reshape_treatment_control,
    write_outputs,
)
from alleleflux.scripts.analysis.regional_contrast import (
    test_region_contrasts as run_region_contrasts,
)


def _make_inputs():
    """Two regions on one contig where each group is observed in only one region.

    Observed (host, group, region) combinations are exactly two.  Under
    ``observed=False`` a Categorical ``group`` grouper forces pandas to emit
    phantom rows for the unobserved combinations.
    """
    df = pd.DataFrame(
        {
            "replicate": [1, 1, 1, 1],
            # Categorical, mirroring how parquet loads the group column.
            "group": pd.Categorical(["A", "A", "B", "B"], categories=["A", "B"]),
            "contig": ["c1", "c1", "c1", "c1"],
            "position": [1, 2, 3, 4],
            "site_score": [0.1, 0.2, 0.3, 0.4],
        }
    )
    region_mapping = pd.DataFrame(
        {
            "contig": ["c1", "c1", "c1", "c1"],
            "position": [1, 2, 3, 4],
            "region_id": ["r1", "r1", "r2", "r2"],
            "region_type": ["gene", "gene", "gene", "gene"],
            "region_start": [1, 1, 3, 3],
            "region_end": [2, 2, 4, 4],
            "region_length": [2, 2, 2, 2],
        }
    )
    return df, region_mapping


def test_categorical_group_does_not_explode():
    df, region_mapping = _make_inputs()

    result = aggregate_region_scores(
        df,
        region_mapping,
        host_col="replicate",
        group_col="group",
        contig_col="contig",
        position_col="position",
        score_col="site_score",
        agg_method="median",
        min_sites=0,
        min_fraction=0.0,
    )

    # Only the two observed (host, group, region) combinations should appear.
    assert len(result) == 2
    # No phantom groups: every emitted region must have a real score.
    assert not result["region_score"].isna().any()
    assert set(zip(result["group"], result["region_id"])) == {("A", "r1"), ("B", "r2")}


def _make_paired_df():
    """Two regions, three replicates each, with treatment/control percentiles."""
    return pd.DataFrame(
        {
            "region_id": ["r1", "r1", "r1", "r2", "r2", "r2"],
            "region_type": ["gene"] * 6,
            "contig": ["c1"] * 6,
            "region_start": [1, 1, 1, 100, 100, 100],
            "region_end": [50, 50, 50, 150, 150, 150],
            "replicate": [1, 2, 3, 1, 2, 3],
            "contrast": [0.2, 0.3, 0.25, -0.1, -0.2, -0.15],
            "percentile_treatment": [80.0, 90.0, 85.0, 40.0, 30.0, 35.0],
            "percentile_control": [20.0, 10.0, 15.0, 60.0, 70.0, 65.0],
        }
    )


def test_summary_fisher_merge_joins_on_region_keys_only():
    """summary_df and fisher_df must merge on region identity, not replicate counts.

    fisher_df carries per-group ``n_replicates_treatment`` / ``n_replicates_control``
    columns that summary_df lacks.  Joining on every non-p-value fisher column (the
    old behavior) referenced a column absent from summary_df → KeyError in production.
    """
    paired = _make_paired_df()
    summary = run_region_contrasts(
        paired, host_col="replicate", contig_col="contig", min_replicates=2
    )
    fisher = fisher_combine_empirical_pvalues(
        paired, contig_col="contig", min_replicates=2
    )
    assert not fisher.empty
    assert "n_replicates_treatment" in fisher.columns
    assert "n_replicates_treatment" not in summary.columns

    # Reproduce the original (buggy) key selection: it raises the production KeyError.
    naive_keys = [
        c
        for c in fisher.columns
        if c not in (FISHER_PVAL_TREATMENT_COL, FISHER_PVAL_CONTROL_COL)
    ]
    with pytest.raises(KeyError):
        summary.merge(fisher, on=naive_keys, how="outer")

    # The fixed key selection joins on region-identity columns only and succeeds.
    keys = _fisher_merge_keys(fisher, summary)
    assert set(keys) == {
        "region_id",
        "region_type",
        "contig",
        "region_start",
        "region_end",
    }
    merged = summary.merge(fisher, on=keys, how="outer")
    assert len(merged) == 2  # one row per region, no duplication
    assert FISHER_PVAL_TREATMENT_COL in merged.columns
    assert "n_replicates" in merged.columns  # summary payload preserved
    assert "n_replicates_treatment" in merged.columns  # fisher payload carried along


def test_write_outputs_emits_empty_file_for_absent_region_type(tmp_path):
    """Every --mode-expected per-type summary file must be written.

    A MAG with no surviving window regions must still produce a
    ``*_window_region_summary.tsv`` (empty, with header) so the Snakemake rule's
    declared outputs all exist — otherwise the job fails with MissingOutputException.
    """
    per_host = pd.DataFrame(
        {
            "replicate": [1, 2],
            "region_id": ["g1", "g1"],
            "region_type": ["gene", "gene"],
            "region_score": [0.1, 0.2],
        }
    )
    summary = pd.DataFrame(
        {
            "region_id": ["g1"],
            "region_type": ["gene"],  # note: no "window" rows
            "n_replicates": [2],
            "mean_contrast": [0.1],
            "median_contrast": [0.1],
            "p_value_wilcoxon_greater": [0.5],
            "p_value_wilcoxon_less": [0.5],
        }
    )

    write_outputs(
        per_host,
        summary,
        output_dir=tmp_path,
        prefix="rc",
        region_types=["gene", "window"],
    )

    gene_path = tmp_path / "rc_gene_region_summary.tsv"
    window_path = tmp_path / "rc_window_region_summary.tsv"
    # Both files exist even though summary_df has no window rows.
    assert gene_path.exists()
    assert window_path.exists()

    gene_df = pd.read_csv(gene_path, sep="\t")
    window_df = pd.read_csv(window_path, sep="\t")
    assert len(gene_df) == 1
    assert len(window_df) == 0  # empty subset
    # Empty file still carries the full header (same schema as the summary).
    assert list(window_df.columns) == list(summary.columns)


# ── null gene_id must never become the literal string "<NA>" ─────────────
#
# Production bug: load_input_table coerced the parquet gene_id column with
# .astype(str).  On a nullable/pyarrow string column that renders every null as
# the *string* "<NA>", which passes build_gene_regions' notna() guard.  Every
# unannotated site on a contig then collapsed into one pseudo-gene sharing the
# id "<NA>" across all contigs, and because reshape_treatment_control merged on
# region_id without contig, the treatment×control join fanned out to
# N_contigs**2 rows per host — inflating n_replicates from 3 to 458.


def _write_parquet_with_null_genes(tmp_path):
    """Parquet whose gene_id column is pyarrow-backed with real nulls."""
    df = pd.DataFrame(
        {
            "replicate": [1, 1, 2, 2],
            "group": ["fat", "control", "fat", "control"],
            "contig": ["c1", "c1", "c2", "c2"],
            "position": [10, 10, 20, 20],
            "gene_id": pd.array(["g1", None, None, "g2"], dtype="string[pyarrow]"),
            "A_frequency_diff_mean": [0.1, 0.2, 0.3, 0.4],
            "T_frequency_diff_mean": [0.0, 0.0, 0.0, 0.0],
            "G_frequency_diff_mean": [0.0, 0.0, 0.0, 0.0],
            "C_frequency_diff_mean": [0.0, 0.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "input.parquet"
    df.to_parquet(path)
    return path


def test_load_input_table_preserves_null_gene_ids(tmp_path):
    """Nulls must survive loading as nulls, not as the string "<NA>"."""
    path = _write_parquet_with_null_genes(tmp_path)
    diff_cols = [
        "A_frequency_diff_mean",
        "T_frequency_diff_mean",
        "G_frequency_diff_mean",
        "C_frequency_diff_mean",
    ]
    df = load_input_table(
        path, ["replicate", "group", "contig", "position", "gene_id"], diff_cols
    )

    assert (df["gene_id"] == "<NA>").sum() == 0, "nulls were stringified to '<NA>'"
    assert df["gene_id"].isna().sum() == 2
    # .str accessor must still work — that is why the coercion existed.
    assert df["gene_id"].str.strip().tolist()[0] == "g1"


@pytest.mark.parametrize("sentinel", ["<NA>", "nan", "NaN", "None", "NA", "  "])
def test_build_gene_regions_rejects_null_sentinels(sentinel):
    """String spellings of "missing" must not become a real gene region."""
    df = pd.DataFrame(
        {
            "contig": ["c1", "c1", "c2", "c2"],
            "position": [1, 2, 3, 4],
            "gene_id": ["g1", sentinel, sentinel, "g2"],
        }
    )
    mapping = build_gene_regions(df, "gene_id", "contig", "position")

    assert set(mapping["region_id"]) == {"g1", "g2"}, (
        f"sentinel {sentinel!r} leaked into the gene regions"
    )


def test_reshape_does_not_cross_join_contigs_sharing_a_region_id():
    """A region_id repeated on two contigs must stay two separate regions.

    One host, one region_id present on two contigs, both groups observed.
    Correct output is 2 paired rows (one per contig).  Merging on region_id
    without contig yields 2x2 = 4 rows of cross-contig garbage.
    """
    agg = pd.DataFrame(
        {
            "replicate": [1, 1, 1, 1],
            "group": ["fat", "control", "fat", "control"],
            "region_id": ["shared", "shared", "shared", "shared"],
            "region_type": ["gene"] * 4,
            "contig": ["c1", "c1", "c2", "c2"],
            "region_start": [1, 1, 500, 500],
            "region_end": [100, 100, 600, 600],
            "region_score": [0.5, 0.1, 0.9, 0.2],
            "percentile": [80.0, 20.0, 95.0, 30.0],
            "n_informative_sites": [10, 10, 12, 12],
            "informative_fraction": [0.5, 0.5, 0.6, 0.6],
        }
    )

    paired = reshape_treatment_control(
        agg,
        host_col="replicate",
        group_col="group",
        contig_col="contig",
        treatment_label="fat",
        control_label="control",
    )

    assert len(paired) == 2, f"cross-contig fan-out: {len(paired)} rows, expected 2"
    # Each contig keeps its own treatment/control pair.
    by_contig = paired.set_index("contig")
    assert by_contig.loc["c1", "region_score_treatment"] == 0.5
    assert by_contig.loc["c1", "region_score_control"] == 0.1
    assert by_contig.loc["c2", "region_score_treatment"] == 0.9
    assert by_contig.loc["c2", "region_score_control"] == 0.2
