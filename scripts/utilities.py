#!/usr/bin/env python
import logging
from functools import reduce

import pandas as pd

logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(name)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.DEBUG,
)


def extract_test_columns(df):
    p_value_columns = [col for col in df.columns if "p_value" in col]

    test_columns_dict = {}

    for col in p_value_columns:
        if "p_value_" in col:
            # Everything after 'p_value' is part of the test name
            test_name = col.split("p_value_")[1]
        else:
            raise ValueError(
                f"Column {col} does not contain 'p_value' in expected format."
            )

        test_columns_dict.setdefault(test_name, []).append(col)
    logging.info(f"Detected tests: {list(test_columns_dict.keys())}")
    return test_columns_dict


def calculate_score(df, test_columns_dict, group_by_column, p_value_threshold=0.05):

    results = []

    for test_name, test_cols in test_columns_dict.items():
        # Create a subset DataFrame with only the grouping column and test columns
        subset_cols = [group_by_column] + test_cols
        subdf = df[subset_cols].copy()
        subdf.dropna(subset=test_cols, how="all", inplace=True)

        # Check for NaNs in all p-value columns and drop
        if subdf[test_cols].isnull().values.any():
            raise ValueError("NaNs found in p-value column.")

        # NAs will be included. Eg. if a position has no no gene it will still be included
        grouped = subdf.groupby(group_by_column, dropna=False)

        # Total sites = number of present sites per group
        total_sites_per_group = grouped.size()

        # Identify significant sites
        significant_col = f"is_significant_{test_name}"
        subdf[significant_col] = subdf[test_cols].lt(p_value_threshold).any(axis=1)

        # Number of significant sites per group
        significant_sites_per_group = grouped[significant_col].sum()

        # Percentage of significant sites per group
        percentage_significant = (
            significant_sites_per_group / total_sites_per_group
        ) * 100

        test_result = pd.DataFrame(
            {
                group_by_column: total_sites_per_group.index,
                f"total_sites_per_group_{test_name}": total_sites_per_group.values,
                f"significant_sites_per_group_{test_name}": significant_sites_per_group.values,
                f"score_{test_name} (%)": percentage_significant.values,
            }
        )

        results.append(test_result)

    # Merge all test results into a single DataFrame
    final_results = reduce(
        lambda left, right: pd.merge(left, right, on=group_by_column, how="outer"),
        results,
    )
    return final_results
