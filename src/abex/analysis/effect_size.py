"""Effect size measures — separate from significance (p-value)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EffectSizeResult:
    absolute_diff: float
    relative_lift: float
    cohens_d: float
    ci_low: float | None = None
    ci_high: float | None = None


def _check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")
    if values.dropna().empty:
        raise ValueError(f"{name} has no non-null observations")


def cohens_d(control: pd.Series, treatment: pd.Series) -> float:
    """Standardized mean difference (Cohen's d) using the pooled standard deviation.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before computing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before computing.

    Returns
    -------
    float
        ``(mean(treatment) - mean(control)) / pooled_std``. Returns ``0.0`` if
        the pooled standard deviation is 0 (e.g. constant or singleton groups).

    Raises
    ------
    TypeError
        If `control` or `treatment` is not a pandas Series.
    ValueError
        If `control` or `treatment` has no non-null observations.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    control, treatment = control.dropna(), treatment.dropna()
    n1, n2 = len(control), len(treatment)
    if n1 + n2 <= 2:
        raise ValueError("cohens_d requires at least 3 combined non-null observations")
    pooled_std = (
        ((n1 - 1) * control.var(ddof=1) + (n2 - 1) * treatment.var(ddof=1)) / (n1 + n2 - 2)
    ) ** 0.5
    if pooled_std == 0:
        return 0.0
    return float((treatment.mean() - control.mean()) / pooled_std)


def relative_lift(control: pd.Series, treatment: pd.Series) -> float:
    """Relative change of the treatment mean over the control mean.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before computing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before computing.

    Returns
    -------
    float
        ``(mean(treatment) - mean(control)) / mean(control)``. Returns
        ``nan`` if the control mean is 0 (relative lift is undefined).

    Raises
    ------
    TypeError
        If `control` or `treatment` is not a pandas Series.
    ValueError
        If `control` or `treatment` has no non-null observations.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    control_mean = control.dropna().mean()
    if control_mean == 0:
        return float("nan")
    return float((treatment.dropna().mean() - control_mean) / control_mean)


def effect_size_summary(
    control: pd.Series,
    treatment: pd.Series,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> EffectSizeResult:
    """Bundle absolute diff, relative lift and Cohen's d for a two-group comparison.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before computing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before computing.
    ci_low : float or None, optional
        Lower confidence bound to pass through as-is (computed elsewhere,
        e.g. `abex.stats.bootstrap.bootstrap_ci`). Default is None.
    ci_high : float or None, optional
        Upper confidence bound to pass through as-is. Default is None.

    Returns
    -------
    EffectSizeResult
        Dataclass with `absolute_diff`, `relative_lift`, `cohens_d`, and the
        passed-through `ci_low`/`ci_high`.

    Raises
    ------
    TypeError
        If `control` or `treatment` is not a pandas Series, or `ci_low`/`ci_high`
        is neither a number nor None.
    ValueError
        If `control` or `treatment` has no non-null observations.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    for name, bound in (("ci_low", ci_low), ("ci_high", ci_high)):
        if bound is not None and not isinstance(bound, (int, float, np.floating, np.integer)):
            raise TypeError(f"{name} must be a number or None, got {type(bound).__name__}")

    control, treatment = control.dropna(), treatment.dropna()
    absolute_diff = float(treatment.mean() - control.mean())
    return EffectSizeResult(
        absolute_diff=absolute_diff,
        relative_lift=relative_lift(control, treatment),
        cohens_d=cohens_d(control, treatment),
        ci_low=ci_low,
        ci_high=ci_high,
    )
