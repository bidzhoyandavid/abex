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


def _check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")


def compute_ratio(numerator: pd.Series, denominator: pd.Series) -> float:
    """Aggregate ratio `sum(numerator) / sum(denominator)`.

    Parameters
    ----------
    numerator : pd.Series
        Per-cluster (or per-event) numerator values.
    denominator : pd.Series
        Per-cluster (or per-event) denominator values.

    Returns
    -------
    float
        `numerator.sum() / denominator.sum()`.

    Raises
    ------
    TypeError
        If `numerator` or `denominator` is not a pandas Series.
    ValueError
        If `denominator` sums to 0.
    """
    _check_series("numerator", numerator)
    _check_series("denominator", denominator)
    total_den = denominator.sum()
    if total_den == 0:
        raise ValueError("cannot compute ratio: denominator sums to 0")
    return float(numerator.sum() / total_den)


def pooled_ratio(*num_den_pairs: tuple[pd.Series, pd.Series]) -> float:
    """Global ratio R0 pooled across all passed groups.

    Parameters
    ----------
    *num_den_pairs : tuple[pd.Series, pd.Series]
        One `(numerator, denominator)` pair per group to pool.

    Returns
    -------
    float
        `sum(all numerators) / sum(all denominators)`.

    Raises
    ------
    TypeError
        If any pair is not a 2-tuple of pandas Series.
    ValueError
        If no pairs are given, or the pooled denominators sum to 0.
    """
    if len(num_den_pairs) == 0:
        raise ValueError("pooled_ratio requires at least one (numerator, denominator) pair")
    for i, pair in enumerate(num_den_pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"num_den_pairs[{i}] must be a (numerator, denominator) tuple")
        _check_series(f"num_den_pairs[{i}][0]", pair[0])
        _check_series(f"num_den_pairs[{i}][1]", pair[1])

    total_num = sum(num.sum() for num, _ in num_den_pairs)
    total_den = sum(den.sum() for _, den in num_den_pairs)
    if total_den == 0:
        raise ValueError("cannot compute pooled ratio: denominators sum to 0")
    return float(total_num / total_den)


def linearize(numerator: pd.Series, denominator: pd.Series, global_ratio: float) -> pd.Series:
    """Per-cluster linearized value `L_i = X_i - R0 * Y_i`.

    `numerator`/`denominator` must already be aggregated per cluster (e.g.
    one row per user) — this does not aggregate raw event-level data itself.

    Parameters
    ----------
    numerator : pd.Series
        Per-cluster numerator sums (X_i).
    denominator : pd.Series
        Per-cluster denominator sums (Y_i), same length and order as `numerator`.
    global_ratio : float
        Pooled reference ratio R0, e.g. from `pooled_ratio`.

    Returns
    -------
    pd.Series
        Linearized per-cluster values `numerator - global_ratio * denominator`.

    Raises
    ------
    TypeError
        If `numerator`/`denominator` is not a pandas Series, or `global_ratio`
        is not a number.
    ValueError
        If `numerator` and `denominator` have different lengths.
    """
    _check_series("numerator", numerator)
    _check_series("denominator", denominator)
    if not isinstance(global_ratio, (int, float)):
        raise TypeError(f"global_ratio must be a number, got {type(global_ratio).__name__}")
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

    Parameters
    ----------
    control_num : pd.Series
        Control per-cluster numerator sums.
    control_den : pd.Series
        Control per-cluster denominator sums, aligned with `control_num`.
    treatment_num : pd.Series
        Treatment per-cluster numerator sums.
    treatment_den : pd.Series
        Treatment per-cluster denominator sums, aligned with `treatment_num`.
    global_ratio : float or None, optional
        Pooled reference ratio R0. If None, computed via `pooled_ratio` over
        both groups. Default is None.

    Returns
    -------
    RatioLinearization
        `global_ratio` used, plus `control_linearized`/`treatment_linearized` series.

    Raises
    ------
    TypeError
        If any series argument is not a pandas Series, or `global_ratio` is
        neither a number nor None.
    ValueError
        If a numerator/denominator pair within a group has mismatched lengths.
    """
    for name, val in (
        ("control_num", control_num),
        ("control_den", control_den),
        ("treatment_num", treatment_num),
        ("treatment_den", treatment_den),
    ):
        _check_series(name, val)
    if global_ratio is not None and not isinstance(global_ratio, (int, float)):
        raise TypeError(f"global_ratio must be a number or None, got {type(global_ratio).__name__}")

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

    Parameters
    ----------
    control_num : pd.Series
        Control numerator values (per-cluster or per-event).
    control_den : pd.Series
        Control denominator values, aligned with `control_num`.
    treatment_num : pd.Series
        Treatment numerator values (per-cluster or per-event).
    treatment_den : pd.Series
        Treatment denominator values, aligned with `treatment_num`.

    Returns
    -------
    RatioEffect
        `control_ratio`, `treatment_ratio`, `absolute_diff`, and
        `relative_lift` (nan if `control_ratio` is 0).

    Raises
    ------
    TypeError
        If any argument is not a pandas Series.
    ValueError
        If a group's denominator sums to 0.
    """
    for name, val in (
        ("control_num", control_num),
        ("control_den", control_den),
        ("treatment_num", treatment_num),
        ("treatment_den", treatment_den),
    ):
        _check_series(name, val)

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
    are valid linearization inputs (`L_i = X_i - R0 * 0 = X_i`).

    Parameters
    ----------
    df : pd.DataFrame
        Event-level dataframe containing `cluster_col`, `numerator_col`, and
        `denominator_col`.
    cluster_col : str
        Name of the column identifying the cluster (e.g. user id).
    numerator_col : str
        Name of the numerator column to sum per cluster.
    denominator_col : str
        Name of the denominator column to sum per cluster.

    Returns
    -------
    pd.DataFrame
        One row per distinct `cluster_col` value, with summed numerator/
        denominator columns.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame, or any column argument is not a str.
    KeyError
        If `cluster_col`, `numerator_col`, or `denominator_col` is not a
        column of `df`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    for name, val in (
        ("cluster_col", cluster_col),
        ("numerator_col", numerator_col),
        ("denominator_col", denominator_col),
    ):
        if not isinstance(val, str):
            raise TypeError(f"{name} must be a str, got {type(val).__name__}")
    for col in (cluster_col, numerator_col, denominator_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not found in dataframe")

    return df.groupby(cluster_col)[[numerator_col, denominator_col]].sum().reset_index()
