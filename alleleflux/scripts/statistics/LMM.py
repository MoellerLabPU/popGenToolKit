#!/usr/bin/env python
import argparse
import logging
import os
import re
import warnings
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.formula.api as smf
from tqdm import tqdm

from alleleflux.scripts.utilities.logging_config import setup_logging
from alleleflux.scripts.utilities.utilities import (
    input_has_column,
    load_allele_freq_inputs,
    load_and_filter_data,
    relabel_groups_from_metadata,
)

# --- Constants ---
logger = logging.getLogger(__name__)

# Regex to find frequency columns with a timepoint suffix, but exclude 'diff' columns.
_FREQ_PATTERN = re.compile(r"^[ATGC]_frequency_(?!diff\b).*")
NUCLEOTIDES = ["A_frequency", "T_frequency", "G_frequency", "C_frequency"]


def _ensure_categorical(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure specified columns in a DataFrame are of categorical dtype.

    This helper inspects the provided DataFrame and converts any listed columns
    to pandas 'category' dtype if (a) they exist in the DataFrame and (b) they are
    not already categorical. Columns not present are silently ignored. If no
    columns require conversion, the original DataFrame object is returned
    (un-copied); otherwise, a shallow copy with the converted columns is produced.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame whose columns may be cast to categorical.
    cols : list of str
        Column names to enforce as categorical. Non-existent columns are ignored.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame if no conversion was necessary; otherwise a copy
        with the specified columns converted to categorical dtype.

    Notes
    -----
    - Using pandas.Categorical (via astype('category')) can reduce memory usage and
      speed up groupby / merge operations on low-cardinality string/object columns.
    - Be aware that returning the original DataFrame when no casting is needed
      means in-place modifications after this call may affect the caller's object.
      Code relying on a guaranteed copy should account for this behavior.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": ["x", "y", "x"], "b": [1, 2, 3]})
    >>> df.dtypes
    a    object
    b     int64
    dtype: object
    >>> df2 = _ensure_categorical(df, ["a", "c"])  # 'c' ignored, 'a' converted
    >>> df2.dtypes
    a    category
    b       int64
    dtype: object
    >>> df2 is df
    False
    >>> df3 = _ensure_categorical(df2, ["a"])  # already categorical; returns same object
    >>> df3 is df2
    True
    """
    to_cast = [c for c in cols if c in df.columns and df[c].dtype != "category"]
    if not to_cast:
        return df
    df_out = df.copy()
    for c in to_cast:
        df_out[c] = df_out[c].astype("category")
    return df_out


# --- Data Loading and Preparation Functions ---


def plan_csv_loading(dtype_map: Dict, input_time_format: str, df_path) -> Dict:
    """
    Prepares a data type map for loading the input efficiently.

    Args:
        dtype_map: The base dictionary of data types.
        input_time_format: The format of time information ('column' or 'suffix').
        df_path: Path (or first path in a list) to the input file used to
            inspect its header.

    Returns:
        The updated data type map.
    """
    if input_time_format == "column":
        dtype_map["time"] = "category"
        dtype_map.update({nuc: "float32" for nuc in NUCLEOTIDES})

    if input_time_format == "suffix":
        # Inspect first file's header for suffixed frequency columns.
        sample_path = df_path[0] if isinstance(df_path, (list, tuple)) else df_path
        if str(sample_path).endswith(".parquet"):
            header = pq.read_schema(sample_path).names
        else:
            header = pd.read_csv(sample_path, sep="\t", nrows=0).columns
        for col in header:
            if _FREQ_PATTERN.match(col):
                dtype_map[col] = "float32"
    return dtype_map


def _load_input_for_suffix_conversion(df_source) -> pd.DataFrame:
    """Helper: load a path/list-of-paths into a DataFrame for wide-to-long conversion.

    Delegates to load_allele_freq_inputs() which detects .parquet vs TSV by
    extension — the single source of truth for allele frequency I/O.
    """
    if isinstance(df_source, (list, tuple, str)):
        return load_allele_freq_inputs(df_source)
    # Already a DataFrame — return a copy to avoid mutating the caller's frame.
    return df_source.copy()



def convert_suffix_to_long(df_source) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame of nucleotide frequencies with timepoint suffixes to long format.

    This function detects columns that follow a frequency naming convention and pivots them
    from wide to long using pandas.wide_to_long. Expected column names follow the pattern
    "<stub>_frequency_<time>", where <stub> is one of NUCLEOTIDES and <time> is an arbitrary suffix.

    Args:
        df_source (pd.DataFrame): Input wide-format DataFrame. All columns not matching the
            frequency pattern are treated as identifier variables.

    Returns:
        pd.DataFrame: Long-format DataFrame with:
            - all identifier variables preserved,
            - a 'time' column derived from the suffix part of the frequency columns,
            - one column per stub in NUCLEOTIDES containing the corresponding values.

    Raises:
        ValueError: If no suffixed frequency columns are found for conversion.
        ValueError: If fewer than two distinct timepoint suffixes are detected.

    Notes:
        - The input DataFrame is not modified; the operation is performed on a copy.
        - The 'time' values are parsed from column suffixes and returned as strings.
        - Relies on module-level constants `_FREQ_PATTERN` (regex used to detect frequency columns)
          and `NUCLEOTIDES` (list of stubs used as `stubnames` in wide_to_long).
        - Emits an info-level log message listing the discovered timepoint suffixes.
    """

    if isinstance(df_source, (str, list, tuple)):
        df = _load_input_for_suffix_conversion(df_source)
    else:
        # Input is already a DataFrame
        df = df_source.copy()

    suffixed_cols = [c for c in df.columns if _FREQ_PATTERN.match(c)]
    if not suffixed_cols:
        raise ValueError(
            "No suffixed frequency columns found for wide-to-long conversion."
        )

    suffixes = sorted({c.split("_frequency_", 1)[1] for c in suffixed_cols})
    if len(suffixes) < 2:
        raise ValueError(f"Less than two timepoint suffixes found: {suffixes}")

    # Define id_vars for pivoting
    id_vars = [c for c in df.columns if c not in suffixed_cols]

    # Perform wide-to-long conversion
    df_long = pd.wide_to_long(
        df, stubnames=NUCLEOTIDES, i=id_vars, j="time", sep="_", suffix=r".+"
    ).reset_index()

    # Ensure time is categorical for consistency with column-based loading path
    if df_long["time"].dtype != "category":
        df_long["time"] = df_long["time"].astype("category")

    logger.info(f"Converted wide-format to long with timepoints: {suffixes}")
    return df_long


def detect_frequency_columns(data_type: str) -> List[str]:
    """
    Return the expected allele-frequency column names for a given analysis type.

    Parameters
    ----------
    data_type : str
        The analysis mode. Must be one of:
        - "longitudinal": returns per-nucleotide difference mean columns, e.g., "<NUC>_diff_mean".
        - "single" or "across_time": returns the raw nucleotide column names.

    Returns
    -------
    List[str]
        Column names derived from the module-level NUCLEOTIDES iterable.
        For "longitudinal": ["A_diff_mean", "C_diff_mean", ...] based on NUCLEOTIDES.
        For "single" or "across_time": the NUCLEOTIDES list as-is.

    Raises
    ------
    ValueError
        If `data_type` is not one of the recognized values.

    Notes
    -----
    Relies on a module-level `NUCLEOTIDES` iterable (e.g., ["A", "C", "G", "T"]).

    Examples
    --------
    >>> NUCLEOTIDES = ["A", "C", "G", "T"]
    >>> detect_frequency_columns("longitudinal")
    ['A_frequency_diff_mean', 'C_frequency_diff_mean', 'G_frequency_diff_mean', 'T_frequency_diff_mean']
    >>> detect_frequency_columns("single")
    ['A_frequency', 'C_frequency', 'G_frequency', 'T_frequency']
    """
    if data_type == "longitudinal":
        return [f"{nuc}_diff_mean" for nuc in NUCLEOTIDES]
    elif data_type in ["single", "across_time"]:
        return NUCLEOTIDES
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


# --- Core Modeling Functions ---
def has_perfect_separation(
    df: pd.DataFrame, response_col: str, predictor_col: str
) -> bool:
    """
    Determine whether a predictor perfectly separates a binary or categorical outcome.

    This function tests for "perfect separation" by grouping the response by the
    predictor and checking two conditions:
    - Within each predictor group, the response has zero variance (i.e., is constant).
    - Across groups, there is at least one difference in group means (i.e., not all
        groups have the same constant value).

    Parameters:
            df (pd.DataFrame): Input DataFrame containing the response and predictor columns.
            response_col (str): Name of the numeric response column.
            predictor_col (str): Name of the predictor column used to form groups
                    (typically categorical or discrete).

    Returns:
            bool: True if the predictor perfectly determines the response according to the
            criteria above; False otherwise.

    Notes:
            - Groups with a single observation will have undefined variance (NaN) under the
                default ddof=1 and therefore will not satisfy the zero-variance condition.
            - If there is only one predictor group, or all group means are identical, the
                function returns False.
    df: pd.DataFrame, response_col: str, predictor_col: str
    """
    group_stats = df.groupby(predictor_col, observed=True)[response_col].agg(
        ["mean", "var"]
    )
    return (group_stats["var"] == 0).all() and (group_stats["mean"].nunique() > 1)


def _fit_lmm_and_process_results(
    df: pd.DataFrame, response_col: str, predictor_col: str
) -> Dict[str, Any]:
    """
    Fit a single Linear Mixed-Effects Model (LMM) and summarize key statistics.

    This routine fits a statsmodels MixedLM with the response as the dependent
    variable and the predictor treated as a categorical fixed effect. The loading
    pipeline ensures relevant columns (time/group/replicate) are cast to 'category'
    ahead of model fitting, allowing use of the simpler formula
    ``response_col ~ predictor_col`` without Patsy C(...). A random intercept is
    included for the "replicate" grouping factor. It captures warnings, handles
    common edge cases, and returns a compact summary suited for downstream reporting.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the columns specified by `response_col`,
        `predictor_col`, and "replicate" (grouping factor).
    response_col : str
        Name of the dependent variable column in `df`.
    predictor_col : str
        Name of the independent variable column in `df` (already categorical upstream).

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following entries:
        - p_value (float): P-value for the predictor's fixed effect. May be NaN if
          unavailable or the model fails; set to 1.0 if the response is constant.
        - coef (float): Estimated coefficient corresponding to the predictor effect;
          NaN if unavailable.
        - t_value (float): Test statistic for the coefficient (as provided by
          statsmodels); NaN if unavailable.
        - warnings (str): Concatenation of any warnings emitted during model fitting.
        - n_warnings (int): Number of captured warnings.
        - notes (str): Diagnostic notes describing special conditions (e.g., perfect
          separation, singular matrix, constant response, NaN p-value).
        - converged (bool): Whether the optimizer reported convergence.

    Behavior
    --------
    - Uses a MixedLM with formula: response_col ~ predictor_col (predictor already categorical).
    - Groups are taken from df["replicate"] to fit a random intercept per replicate.
    - Handles cases where:
      * The p-value is NaN due to perfect separation or other numerical issues.
      * The response is constant (p-value forced to 1.0).
      * Linear algebra errors occur (summarized in `notes`).
    - Aggregates all emitted warnings into `warnings` and counts them in `n_warnings`.
    - The `converged` flag reflects the statsmodels result's convergence status.

    Raises
    ------
    ValueError
        If no parameter corresponding to `predictor_col` is found in the fitted model
        (e.g., due to design matrix or coding issues).

    Notes
    -----
    - Requires an external helper: has_perfect_separation(df, response_col, predictor_col).
    - The predictor is categorical upstream (no Patsy C()).
    - The function does not mutate `df`.

    Examples
    --------
    >>> out = _fit_lmm_and_process_results(df, "y", "treatment")
    >>> out["p_value"], out["coef"], out["converged"]
    """
    result_dict = {
        "p_value": np.nan,
        "coef": np.nan,
        "t_value": np.nan,
        "warnings": "",
        "n_warnings": 0,
        "notes": "",
        "converged": False,
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        try:
            # Predictor assumed already categorical upstream via _ensure_categorical.
            formula = f"{response_col} ~ {predictor_col}"
            model = smf.mixedlm(formula, df, groups=df["replicate"])
            # The default is to try BFGS → L-BFGS → CG
            # https://github.com/statsmodels/statsmodels/blob/d0608bb1a8fe1d9019e8e3d71a684c8b49788745/statsmodels/regression/mixed_linear_model.py#L2141-L2142
            result = model.fit()
            # Model fitted successfully, now process results
            coef_name = next(
                (name for name in result.params.index if predictor_col in name), None
            )
            if coef_name is None:
                raise ValueError(
                    f"Coefficient for '{predictor_col}' not found. Available: {result.params.index}"
                )
            p_value = result.pvalues.get(coef_name)
            if pd.isna(p_value):
                # Handle cases where fitting succeeds but p-value is NaN
                if has_perfect_separation(df, response_col, predictor_col):
                    result_dict["notes"] = (
                        "Perfect separation detected post-fit, p-value is NaN; "
                    )
                elif df[response_col].nunique() == 1:
                    result_dict["p_value"] = 1.0
                    result_dict["notes"] = (
                        "Constant values detected post-fit, p-value set to 1; "
                    )
                else:
                    var = df[response_col].var()
                    result_dict["notes"] = (
                        f"model failed. NaN p-value returned (variance: {var:.2e}); "
                    )
            else:
                # Valid result
                result_dict["p_value"] = p_value
                result_dict["coef"] = result.params.get(coef_name)
                result_dict["t_value"] = result.tvalues.get(coef_name)
            # Record convergence flag regardless of p-value status
            result_dict["converged"] = bool(getattr(result, "converged", False))

        except np.linalg.LinAlgError as e:
            if has_perfect_separation(df, response_col, predictor_col):
                result_dict["notes"] = (
                    "Perfect separation detected post-fit, p-value is NaN; "
                )
            elif df[response_col].nunique() == 1:
                result_dict["p_value"] = 1.0
                result_dict["notes"] = (
                    f"Singular matrix error (constant values), p-value set to 1; "
                )
            else:
                result_dict["notes"] = (
                    f"Model failed ({type(e).__name__}), p-value is NaN; "
                )

        # Capture any warnings that occurred during fitting
        warnings_list = [str(warn.message) for warn in w]
        if warnings_list:
            result_dict["warnings"] = "; ".join(warnings_list)
            result_dict["n_warnings"] = len(warnings_list)

    return result_dict


def run_lmm_for_position(args: Tuple) -> Dict[str, Any]:
    """
    Run per-position linear mixed models (LMMs) for nucleotide frequency responses and
    return a per-nucleotide summary for both raw and absolute-transformed values.

    Parameters
    ----------
    args : Tuple
        A 5-tuple (contig, gene_id, position, sub_df, data_type):
        - contig (str): Contig/chromosome identifier.
        - gene_id (str): Gene identifier.
        - position (int | str): Genomic position index.
        - sub_df (pandas.DataFrame): Pre-filtered data for the specific position. Must include:
            * Frequency columns as discovered by `detect_frequency_columns(data_type)`.
            * A predictor column:
                - "time" if `data_type == "across_time"`,
                - otherwise "group".
            The predictor is expected to have exactly two unique levels (enforced upstream).
        - data_type (str): Analysis mode. If "across_time", the predictor is "time";
          otherwise "group". Also determines which frequency columns are analyzed.

    Returns
    -------
    Dict[str, Any]
        If the predictor has exactly two levels, returns a dictionary containing:
        - "contig", "gene_id", "position"
        - "n_rows_position": Number of rows in `sub_df`.
        - "group_analyzed": Included only when `data_type == "across_time"` and there is
          exactly one unique group in `sub_df["group"]`.
        - "notes": Aggregated notes from all model fits (may be an empty string).
        For each detected nucleotide (derived from each frequency column's prefix), the
        following keys are added for the raw response:
        - "{nucleotide}_coef_LMM": Estimated fixed-effect coefficient for the predictor.
        - "{nucleotide}_p_value_LMM": P-value for the fixed effect.
        - "{nucleotide}_t_value_LMM (z-score)": Test statistic from the model fit.
        - "{nucleotide}_warnings": Warning messages captured during fitting.
        - "{nucleotide}_n_warnings": Count of warnings.
        - "{nucleotide}_converged_LMM": Whether the model reported convergence.
        And for the absolute-transformed response (only when data_type is 'longitudinal'):
        - "{nucleotide}_coef_LMM_abs"
        - "{nucleotide}_p_value_LMM_abs"
        - "{nucleotide}_t_value_LMM_abs (z-score)"
        - "{nucleotide}_warnings_abs"
        - "{nucleotide}_n_warnings_abs"
        - "{nucleotide}_converged_LMM_abs"

        If the predictor does not have exactly two levels, returns a dictionary with:
        - "contig", "gene_id", "position"
        - "num_pairs": Number of rows in `sub_df`.
        - "notes": "Unexpected number of predictor levels; skipped."

    Notes
    -----
    - The function operates on a copy of `sub_df` and does not mutate the input DataFrame.
    - Frequency columns are discovered via `detect_frequency_columns(data_type)`.
    - Model fitting and result extraction are delegated to `_fit_lmm_and_process_results`.
    """
    contig, gene_id, position, sub_df, data_type = args

    position_result = {
        "contig": contig,
        "gene_id": gene_id,
        "position": position,
        "n_rows_position": len(sub_df),
        "notes": "",
    }

    predictor_col = "time" if data_type == "across_time" else "group"
    # Determine baseline as the other level (expecting exactly two levels per pre-filter logic)
    levels = list(pd.unique(sub_df[predictor_col]))
    if len(levels) != 2:
        # Defensive guard; upstream enforces two unique levels
        return {
            "contig": contig,
            "gene_id": gene_id,
            "position": position,
            "num_pairs": len(sub_df),
            "notes": "Unexpected number of predictor levels; skipped.",
        }
    if (
        data_type == "across_time"
        and "group" in sub_df.columns
        and sub_df["group"].nunique() == 1
    ):
        position_result["group_analyzed"] = sub_df["group"].iloc[0]

    freq_cols = detect_frequency_columns(data_type)
    sub_df_copy = (
        sub_df.copy()
    )  # Create one copy to use for all models in this position

    for freq_col in freq_cols:
        nucleotide = freq_col.split("_")[0]

        # --- 1. Model on Raw Values ---
        raw_model_results = _fit_lmm_and_process_results(
            sub_df_copy, freq_col, predictor_col
        )

        position_result[f"{nucleotide}_coef_LMM"] = raw_model_results["coef"]
        position_result[f"{nucleotide}_p_value_LMM"] = raw_model_results["p_value"]
        position_result[f"{nucleotide}_t_value_LMM (z-score)"] = raw_model_results[
            "t_value"
        ]
        position_result[f"{nucleotide}_warnings"] = raw_model_results["warnings"]
        position_result[f"{nucleotide}_n_warnings"] = raw_model_results["n_warnings"]
        position_result[f"{nucleotide}_converged_LMM"] = raw_model_results["converged"]
        if raw_model_results["notes"]:
            position_result["notes"] += f"{nucleotide}: " + raw_model_results["notes"]

        if data_type == "longitudinal":
            # --- 2. Model on Absolute Values ---
            abs_freq_col = f"{freq_col}_abs"
            sub_df_copy[abs_freq_col] = np.abs(sub_df_copy[freq_col])

            abs_model_results = _fit_lmm_and_process_results(
                sub_df_copy, abs_freq_col, predictor_col
            )

            position_result[f"{nucleotide}_coef_LMM_abs"] = abs_model_results["coef"]
            position_result[f"{nucleotide}_p_value_LMM_abs"] = abs_model_results[
                "p_value"
            ]
            position_result[f"{nucleotide}_t_value_LMM_abs (z-score)"] = (
                abs_model_results["t_value"]
            )
            position_result[f"{nucleotide}_warnings_abs"] = abs_model_results[
                "warnings"
            ]
            position_result[f"{nucleotide}_n_warnings_abs"] = abs_model_results[
                "n_warnings"
            ]
            position_result[f"{nucleotide}_converged_LMM_abs"] = abs_model_results[
                "converged"
            ]
            if abs_model_results["notes"]:
                position_result["notes"] += (
                    f"{nucleotide} (abs): " + abs_model_results["notes"]
                )

    return position_result


# --- Main Execution ---


def prepare_tasks_for_lmm(df: pd.DataFrame, args: argparse.Namespace) -> List[Tuple]:
    """
    Prepare per-site tasks for linear mixed-model (LMM) fitting by validating and cleaning input data.

    This function groups the input DataFrame by (contig, gene_id, position) and, for each
    group, performs a series of checks to ensure the data are suitable for mixed-model analysis.
    Rows with missing values in essential columns are dropped, groups are required to have
    exactly two levels of the fixed-effect factor, and sufficient replicate coverage is enforced.

    The fixed-effect factor is chosen based on args.data_type:
    - If args.data_type == "across_time": use the "time" column.
    - Otherwise: use the "group" column.

    Filtering and validation steps per group:
    - Drop rows with NA in required columns: [filter_col, "replicate"] + frequency columns.
    - Require exactly two unique levels in the fixed-effect column (time/group).
    - Require at least args.min_sample_num unique replicates within each fixed-effect level.
    - Require at least two unique replicate levels overall (needed for the random effect).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format input with at least the following columns:
        - "contig", "gene_id", "position"
        - "replicate"
        - "time" (if args.data_type == "across_time") or "group" (otherwise)
        - Frequency/intensity columns determined by detect_frequency_columns(args.data_type).
    args : argparse.Namespace
        Configuration with required attributes:
        - data_type (str): Determines fixed-effect column ("across_time" -> "time", else "group").
        - min_sample_num (int): Minimum number of unique replicates per fixed-effect level.

    Returns
    -------
    List[Tuple[Any, Any, Any, pandas.DataFrame, str]]
        A list of per-position tasks. Each tuple contains:
        (contig, gene_id, position, df_clean, data_type), where df_clean is the
        filtered DataFrame for that locus, ready for LMM fitting.

    Raises
    ------
    KeyError
        If any required columns are missing from the input DataFrame.

    Notes
    -----
    - The frequency columns required for filtering are determined by
      detect_frequency_columns(args.data_type).
    - This function only prepares/validates tasks; it does not fit the model.
    """
    grouped_positions = []
    freq_cols_to_check = detect_frequency_columns(args.data_type)

    for group_keys, sub_df in df.groupby(
        ["contig", "gene_id", "position"], dropna=False
    ):
        contig, gene_id, position = group_keys

        # Determine the grouping column for filtering
        filter_col = "time" if args.data_type == "across_time" else "group"

        # --- Start of pre-filtering and cleaning logic ---
        # 1. PRE-FILTER NA: Identify and drop rows with missing data in essential columns.
        required_cols = [filter_col, "replicate"] + freq_cols_to_check
        use_idx = sub_df[required_cols].dropna().index

        df_clean = sub_df.loc[use_idx].copy()

        # 2. VALIDATE DATA: After NA drop, ensure there's still valid data to model.
        if df_clean.empty or df_clean[filter_col].nunique() != 2:
            continue

        # 3. VALIDATE REPLICATES: Check for minimum number of unique replicates per group.
        # Use observed=True to avoid FutureWarning about future default and limit category combinations.
        replicate_counts = df_clean.groupby(filter_col, observed=True)[
            "replicate"
        ].nunique()
        if (
            not (replicate_counts >= args.min_sample_num).all()
            and len(replicate_counts) == 2
        ):
            continue

        # A mixed model also requires at least two total replicate levels for the random effect.
        if df_clean["replicate"].nunique() < 2:
            continue

        # --- End of pre-filtering ---

        # If all checks pass, the task is valid.
        grouped_positions.append((contig, gene_id, position, df_clean, args.data_type))

    return grouped_positions


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Run Linear Mixed-Effects Models on allele frequency data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_df",
        required=True,
        nargs="+",
        help=(
            "Path(s) to input allele frequency dataframe (single, across_time) or "
            "mean changes dataframe (longitudinal). Accepts multiple paths; if multiple "
            "are given they are concatenated. Files ending in .parquet are read with "
            "pd.read_parquet, otherwise as TSV."
        ),
        type=str,
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Optional list of group names to keep. When the input cache contains samples "
            "from groups not in the current analysis, restrict to these groups before testing."
        ),
    )
    parser.add_argument(
        "--preprocessed_df",
        help="Optional path to a preprocessed dataframe to filter positions. Only used when --data_type is 'across_time'.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_type",
        help="Type of data to analyze",
        type=str,
        choices=["longitudinal", "single", "across_time"],
        default="longitudinal",
    )
    parser.add_argument(
        "--group_to_analyze",
        help="For 'across_time' data_type only: name of the group to analyze.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--input_time_format",
        help="Format of time information in the input dataframe.",
        type=str,
        choices=["column", "suffix"],
        default="column",
    )
    parser.add_argument(
        "--permuted_metadata",
        type=str,
        default=None,
        help=(
            "Optional path to a permuted metadata TSV (null/control run).  When "
            "given, group labels are re-derived from this sheet (joined on "
            "sample_id) right after the cache is loaded, before any group "
            "filtering or modeling.  Omit for a normal run."
        ),
    )
    parser.add_argument(
        "--min_sample_num",
        type=int,
        default=4,
        help="Minimum number of samples per group required.",
    )
    parser.add_argument(
        "--cpus", help="Number of processors to use.", type=int, default=cpu_count()
    )
    parser.add_argument(
        "--output_dir", help="Path to output directory.", type=str, required=True
    )
    parser.add_argument("--mag_id", help="MAG ID to process", type=str, required=True)
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.input_time_format == "suffix" and args.data_type != "across_time":
        parser.error(
            "--input_time_format 'suffix' is only valid with --data_type 'across_time'."
        )
    if args.data_type == "across_time" and not args.group_to_analyze:
        parser.error("--group_to_analyze is required for --data_type 'across_time'.")

    # --- Data Loading ---
    logger.info("Loading and preparing input data...")
    dtype_map = {
        "subjectID": str,
        "contig": str,
        "gene_id": str,
        "position": int,
        "group": "category",
        "replicate": "category",
    }
    # Permuted run: ensure the per-sample join key is read so the cache can be
    # relabeled (the usecols-based loaders below would otherwise drop it).  Only
    # force it when the input actually carries sample_id: the across_time wide
    # input (allele_analysis output) has none and is already relabeled upstream,
    # so demanding the column would break the usecols-based parquet read.
    if args.permuted_metadata and input_has_column(args.input_df, "sample_id"):
        dtype_map["sample_id"] = str

    # Conditional data loading strategy.
    if args.preprocessed_df and args.data_type == "across_time":
        # If preprocessed_df is provided, load and filter input_df based on it
        logger.info(
            f"Preprocessed dataframe provided. Loading and filtering input_df based on {args.preprocessed_df}"
        )
        df = load_and_filter_data(
            input_df_path=args.input_df,
            preprocessed_df_path=args.preprocessed_df,
            mag_id=args.mag_id,
            dtype_map=plan_csv_loading(
                dtype_map, args.input_time_format, args.input_df
            ),
            # The specific group to isolate for the 'across_time' analysis.
            group_to_analyze=args.group_to_analyze,
        )
        if df.empty:
            raise ValueError(
                f"DataFrame is empty for MAG {args.mag_id} after filtering. Exiting."
            )
        if args.input_time_format == "suffix":
            df = convert_suffix_to_long(df)
    else:
        logger.info(
            f"No preprocessed dataframe provided. Loading input_df directly from {args.input_df}"
        )
        # Standard loading path: load the entire dataframe from one or more files.
        df_path = args.input_df
        if args.input_time_format == "suffix":
            logger.info(
                "Converting frequency columns with timepoint suffixes to long format."
            )
            df = convert_suffix_to_long(df_path)
        else:
            dtype_map = plan_csv_loading(dtype_map, args.input_time_format, df_path)
            df = load_allele_freq_inputs(df_path, dtype=dtype_map)

    # Ensure expected categorical columns regardless of loading path
    df = _ensure_categorical(df, ["group", "replicate", "time"])

    # Permuted run: re-derive group labels from the permuted sheet before any
    # group filtering/modeling.  No-op (helper warns) when sample_id is absent —
    # e.g. an already-relabeled Stage-2 suffix input.
    if args.permuted_metadata:
        df = relabel_groups_from_metadata(df, args.permuted_metadata)
        if "sample_id" in df.columns:
            df = df.drop(columns=["sample_id"])

    if args.groups:
        df = df[df["group"].isin(args.groups)].copy()

    logger.info(f"Loaded {df.shape[0]:,} rows.")

    # --- Data Filtering ---
    if args.data_type == "across_time":
        if args.group_to_analyze not in df["group"].unique():
            raise ValueError(f"Group '{args.group_to_analyze}' not found.")

        df = df[df["group"] == args.group_to_analyze].copy()
        logger.info(f"Filtered data for group: {args.group_to_analyze}")

        # Handle time format based on input_time_format
        if "time" not in df.columns:
            raise ValueError("'time' column not found in input data. Exiting.")

        if df["time"].nunique() != 2:
            raise ValueError(
                f"Only two unique timepoints are required, found {df['time'].nunique()} for group '{args.group_to_analyze}'."
            )

    # --- Task Preparation ---
    logger.info("Preparing tasks for LMM analysis...")
    tasks = prepare_tasks_for_lmm(df, args)
    if not tasks:
        raise ValueError("No positions met the criteria for LMM analysis. Exiting.")
    logger.info(f"Found {len(tasks):,} positions with sufficient data for modeling.")

    # --- Parallel Processing ---
    results = []
    with Pool(processes=args.cpus) as pool:
        for res in tqdm(
            pool.imap_unordered(run_lmm_for_position, tasks),
            total=len(tasks),
            desc="Running Linear Mixed Models",
        ):
            results.append(res)
    results_df = pd.DataFrame(results)

    # --- Save Results ---
    p_value_cols = [c for c in results_df.columns if "p_value_LMM" in c]
    results_df.dropna(subset=p_value_cols, how="all", inplace=True)
    logger.info(
        f"Removed {len(results) - len(results_df):,} rows with all NaN p-values."
    )

    output_file_name = f"{args.mag_id}_lmm.tsv.gz"
    if args.data_type == "across_time":
        output_file_name = (
            f"{args.mag_id}_lmm_across_time_{args.group_to_analyze}.tsv.gz"
        )
    output_path = os.path.join(args.output_dir, output_file_name)

    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False, sep="\t", compression="gzip")
    logger.info(f"LMM modeling complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
