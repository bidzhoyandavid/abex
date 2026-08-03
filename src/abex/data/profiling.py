"""Metric profile — the input contract for the selector layer.

profile_metric() inspects a single metric column split by group and produces
a declarative summary that selector/rules.py matches against method
metadata in selector/registry.py. Nothing here decides which test to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sps

MetricKind = Literal["binary", "count", "continuous"]


@dataclass
class MetricProfile:
    metric_col: str
    group_col: str
    n_groups: int
    group_sizes: dict[str, int]
    kind: MetricKind
    is_balanced_design: bool
    skewness: float
    excess_kurtosis: float
    outlier_share: float
    zero_share: float
    has_pre_period: bool
    min_group_size: int

    @property
    def is_two_group(self) -> bool:
        return self.n_groups == 2


def _infer_kind(values: pd.Series) -> MetricKind:
    """Infer whether a metric is binary, count, or continuous.

    Parameters
    ----------
    values : pd.Series
        Non-null metric values.

    Returns
    -------
    {"binary", "count", "continuous"}
        "binary" if the only distinct values are a subset of {0, 1}; "count"
        if all values are non-negative integers; "continuous" otherwise.
    """
    non_null = values.dropna()
    unique_vals = set(non_null.unique().tolist())
    if unique_vals <= {0, 1}:
        return "binary"
    is_nonneg_integer_valued = (non_null >= 0).all() and np.allclose(non_null, non_null.round())
    if is_nonneg_integer_valued:
        return "count"
    return "continuous"


def _outlier_share_iqr(values: pd.Series) -> float:
    """Share of values flagged as outliers by the 1.5*IQR rule.

    Parameters
    ----------
    values : pd.Series
        Non-null metric values.

    Returns
    -------
    float
        Fraction of `values` outside `[q1 - 1.5*iqr, q3 + 1.5*iqr]`. Returns
        0.0 if there are fewer than 4 values or the IQR is 0.
    """
    if len(values) < 4:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(((values < lower) | (values > upper)).mean())


def profile_metric(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    pre_period_col: str | None = None,
) -> MetricProfile:
    """Build a declarative profile of a metric split by experiment group.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment dataframe containing `metric_col` and `group_col`.
    metric_col : str
        Name of the column holding the metric values.
    group_col : str
        Name of the column holding the experiment group/variant label.
    pre_period_col : str or None, optional
        Name of a pre-treatment period column, if available (used only to
        set `MetricProfile.has_pre_period`). Default is None.

    Returns
    -------
    MetricProfile
        Declarative summary: group sizes, inferred metric kind, skewness,
        kurtosis, outlier/zero shares, and design balance.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or any of `metric_col`, `group_col`,
        `pre_period_col` is not a str (or None for `pre_period_col`).
    KeyError
        If `metric_col` or `group_col` is not a column of `df`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(metric_col, str):
        raise TypeError(f"metric_col must be a str, got {type(metric_col).__name__}")
    if not isinstance(group_col, str):
        raise TypeError(f"group_col must be a str, got {type(group_col).__name__}")
    if pre_period_col is not None and not isinstance(pre_period_col, str):
        raise TypeError(f"pre_period_col must be a str or None, got {type(pre_period_col).__name__}")
    for col in (metric_col, group_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in dataframe")

    values = df[metric_col].dropna()
    group_sizes = {str(k): int(v) for k, v in df[group_col].value_counts().to_dict().items()}
    n_groups = len(group_sizes)

    kind = _infer_kind(values)
    skewness = float(sps.skew(values)) if len(values) > 2 else 0.0
    excess_kurtosis = float(sps.kurtosis(values)) if len(values) > 2 else 0.0
    outlier_share = _outlier_share_iqr(values) if kind == "continuous" else 0.0
    zero_share = float((values == 0).mean()) if len(values) else 0.0

    return MetricProfile(
        metric_col=metric_col,
        group_col=group_col,
        n_groups=n_groups,
        group_sizes=group_sizes,
        kind=kind,
        is_balanced_design=(max(group_sizes.values()) / max(min(group_sizes.values()), 1)) < 1.5
        if group_sizes
        else False,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
        outlier_share=outlier_share,
        zero_share=zero_share,
        has_pre_period=pre_period_col is not None and pre_period_col in df.columns,
        min_group_size=min(group_sizes.values()) if group_sizes else 0,
    )
