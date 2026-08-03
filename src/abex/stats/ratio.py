"""Ratio metrics (revenue/user, CTR with repeated impressions per user, ...).

A naive t-test on per-event values is wrong here: observations within a
user are not independent, so the naive variance understates the true
variance and inflates false positives.

Approach: linearization (Deng et al., "Trustworthy Analysis of Online
Controlled Experiments", 2018). Aggregate numerator/denominator per cluster
(usually per user), then replace the ratio metric with a linearized
per-cluster scalar:

    L_i = X_i - R0 * Y_i

where X_i/Y_i are the cluster's numerator/denominator sums and R0 is a
global ratio computed by pooling both groups (avoids bias from picking one
group's ratio as the reference). L_i is now an ordinary per-cluster
continuous metric — clustering is already absorbed into the aggregation,
so it can be fed directly into stats/frequentist.py, stats/bootstrap.py or
the selector exactly like any other continuous metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def compute_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    total_den = denominator.sum()
    if total_den == 0:
        raise ValueError("cannot compute ratio: denominator sums to 0")
    return float(numerator.sum() / total_den)


def pooled_ratio(*num_den_pairs: tuple[pd.Series, pd.Series]) -> float:
    """Global ratio R0 pooled across all passed groups."""
    total_num = sum(num.sum() for num, _ in num_den_pairs)
    total_den = sum(den.sum() for _, den in num_den_pairs)
    if total_den == 0:
        raise ValueError("cannot compute pooled ratio: denominators sum to 0")
    return float(total_num / total_den)


def linearize(numerator: pd.Series, denominator: pd.Series, global_ratio: float) -> pd.Series:
    """Per-cluster linearized value L_i = X_i - R0 * Y_i.

    numerator/denominator must already be aggregated per cluster (e.g. one
    row per user) — this does not aggregate raw event-level data itself.
    """
    if len(numerator) != len(denominator):
        raise ValueError("numerator and denominator must have the same length (one row per cluster)")
    return numerator - global_ratio * denominator


@dataclass
class RatioLinearization:
    global_ratio: float
    control_linearized: pd.Series
    treatment_linearized: pd.Series


def linearize_groups(
    control_num: pd.Series,
    control_den: pd.Series,
    treatment_num: pd.Series,
    treatment_den: pd.Series,
    global_ratio: float | None = None,
) -> RatioLinearization:
    """Linearize both groups against a shared R0, ready to hand to any
    stats/* function that expects two per-cluster continuous series.
    """
    if global_ratio is None:
        global_ratio = pooled_ratio((control_num, control_den), (treatment_num, treatment_den))

    return RatioLinearization(
        global_ratio=global_ratio,
        control_linearized=linearize(control_num, control_den, global_ratio),
        treatment_linearized=linearize(treatment_num, treatment_den, global_ratio),
    )


@dataclass
class RatioEffect:
    control_ratio: float
    treatment_ratio: float
    absolute_diff: float
    relative_lift: float


def ratio_effect(
    control_num: pd.Series,
    control_den: pd.Series,
    treatment_num: pd.Series,
    treatment_den: pd.Series,
) -> RatioEffect:
    """Point-estimate effect on the ratio metric itself (for reporting) —
    significance testing is done separately on the linearized values.
    """
    control_ratio = compute_ratio(control_num, control_den)
    treatment_ratio = compute_ratio(treatment_num, treatment_den)
    absolute_diff = treatment_ratio - control_ratio
    relative_lift = float("nan") if control_ratio == 0 else absolute_diff / control_ratio

    return RatioEffect(
        control_ratio=control_ratio,
        treatment_ratio=treatment_ratio,
        absolute_diff=absolute_diff,
        relative_lift=relative_lift,
    )


def aggregate_by_cluster(
    df: pd.DataFrame,
    cluster_col: str,
    numerator_col: str,
    denominator_col: str,
) -> pd.DataFrame:
    """Collapse raw event-level rows to one numerator/denominator sum per
    cluster (e.g. per user). Clusters with 0 denominator are kept — they
    are valid linearization inputs (L_i = X_i - R0 * 0 = X_i).
    """
    return df.groupby(cluster_col)[[numerator_col, denominator_col]].sum().reset_index()
