"""Check that groups are balanced on pre-treatment covariates.

Imbalance here means randomization likely failed (or groups aren't
comparable), independent of whatever the post-treatment metric shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats as sps


@dataclass
class CovariateBalanceResult:
    covariate: str
    group_means: dict[str, float]
    standardized_mean_diff: float
    p_value: float
    is_balanced: bool


def check_covariate_balance(
    df: pd.DataFrame,
    covariate_col: str,
    group_col: str,
    smd_threshold: float = 0.1,
) -> CovariateBalanceResult:
    """Check pre-treatment covariate balance between exactly two groups.

    Two-group only. SMD (Cohen's d on the covariate) is the primary signal —
    p-value is supplementary and gets noisy at large n.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment dataframe containing `covariate_col` and `group_col`.
    covariate_col : str
        Name of the pre-treatment covariate column to check.
    group_col : str
        Name of the column holding the experiment group/variant label.
        Must have exactly 2 distinct non-null values.
    smd_threshold : float, default 0.1
        Absolute standardized-mean-difference threshold above which the
        covariate is flagged as imbalanced. Must be non-negative.

    Returns
    -------
    CovariateBalanceResult
        Group means, standardized mean difference, Welch t-test p-value, and
        `is_balanced` (True if `abs(smd) < smd_threshold`).

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, `covariate_col`/`group_col` is not
        a str, or `smd_threshold` is not a number.
    KeyError
        If `covariate_col` or `group_col` is not a column of `df`.
    ValueError
        If `group_col` does not have exactly 2 distinct non-null values, or
        `smd_threshold` is negative.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(covariate_col, str):
        raise TypeError(f"covariate_col must be a str, got {type(covariate_col).__name__}")
    if not isinstance(group_col, str):
        raise TypeError(f"group_col must be a str, got {type(group_col).__name__}")
    if not isinstance(smd_threshold, (int, float)) or isinstance(smd_threshold, bool):
        raise TypeError(f"smd_threshold must be a number, got {type(smd_threshold).__name__}")
    if smd_threshold < 0:
        raise ValueError("smd_threshold must be non-negative")
    for col in (covariate_col, group_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in dataframe")

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError(f"covariate_balance check expects exactly 2 groups, got {len(groups)}")

    g1_vals = df.loc[df[group_col] == groups[0], covariate_col].dropna()
    g2_vals = df.loc[df[group_col] == groups[1], covariate_col].dropna()

    mean1, mean2 = g1_vals.mean(), g2_vals.mean()
    pooled_std = (
        ((len(g1_vals) - 1) * g1_vals.var(ddof=1) + (len(g2_vals) - 1) * g2_vals.var(ddof=1))
        / (len(g1_vals) + len(g2_vals) - 2)
    ) ** 0.5

    smd = 0.0 if pooled_std == 0 else float((mean1 - mean2) / pooled_std)
    _, p_value = sps.ttest_ind(g1_vals, g2_vals, equal_var=False)

    return CovariateBalanceResult(
        covariate=covariate_col,
        group_means={str(groups[0]): float(mean1), str(groups[1]): float(mean2)},
        standardized_mean_diff=smd,
        p_value=float(p_value),
        is_balanced=abs(smd) < smd_threshold,
    )
