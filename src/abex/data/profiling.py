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
    non_null = values.dropna()
    unique_vals = set(non_null.unique().tolist())
    if unique_vals <= {0, 1}:
        return "binary"
    is_nonneg_integer_valued = (non_null >= 0).all() and np.allclose(non_null, non_null.round())
    if is_nonneg_integer_valued:
        return "count"
    return "continuous"


def _outlier_share_iqr(values: pd.Series) -> float:
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
