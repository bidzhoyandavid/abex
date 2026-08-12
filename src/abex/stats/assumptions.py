"""Formal checks of the assumptions parametric tests rely on.

The selector approximates these from the metric profile (skewness, outlier
share) because it must rank methods before any test runs. These functions are
the actual measurements, for reporting what was really violated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import scipy.stats as sps

# Shapiro-Wilk is unreliable above a few thousand points (it starts rejecting
# on trivial deviations), so past this size we switch to D'Agostino-Pearson.
SHAPIRO_MAX_N = 5_000
MIN_N = 3


@dataclass
class NormalityResult:
    group: str
    method: str
    statistic: float
    p_value: float
    n: int
    is_normal: bool
    alpha: float


@dataclass
class EqualVarianceResult:
    method: str
    statistic: float
    p_value: float
    groups: list[str]
    is_equal: bool
    alpha: float


def _check_alpha(alpha: float) -> None:
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")


def _check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")


def check_normality(values: pd.Series, group: str = "", alpha: float = 0.05) -> NormalityResult:
    """Test whether `values` are plausibly normal.

    Parameters
    ----------
    values : pd.Series
        Observations for one group. Nulls are dropped.
    group : str, default ""
        Group label, carried through to the result for reporting.
    alpha : float, default 0.05
        Significance level. `is_normal` is `p_value >= alpha`.

    Returns
    -------
    NormalityResult
        `method` is `"shapiro"` for small samples and `"dagostino"` above
        `SHAPIRO_MAX_N`.

    Raises
    ------
    TypeError
        If `values` is not a Series, `group` is not a str, or `alpha` is not
        a number.
    ValueError
        If `alpha` is not in `(0, 1)` or fewer than 3 non-null observations
        remain.

    Notes
    -----
    A non-rejection is not proof of normality — with small samples the test
    simply lacks power. Treat `is_normal=True` as "no evidence against".
    """
    _check_series("values", values)
    if not isinstance(group, str):
        raise TypeError(f"group must be a str, got {type(group).__name__}")
    _check_alpha(alpha)

    clean = values.dropna()
    if len(clean) < MIN_N:
        raise ValueError(f"need at least {MIN_N} non-null observations, got {len(clean)}")

    if len(clean) <= SHAPIRO_MAX_N:
        method = "shapiro"
        statistic, p_value = sps.shapiro(clean.to_numpy())
    else:
        method = "dagostino"
        statistic, p_value = sps.normaltest(clean.to_numpy())

    return NormalityResult(
        group=group,
        method=method,
        statistic=float(statistic),
        p_value=float(p_value),
        n=int(len(clean)),
        is_normal=bool(p_value >= alpha),
        alpha=alpha,
    )


def check_equal_variance(
    groups: dict[str, pd.Series], alpha: float = 0.05, center: str = "median"
) -> EqualVarianceResult:
    """Levene's test for equal variance across groups.

    Parameters
    ----------
    groups : dict[str, pd.Series]
        Group label -> observations. Nulls are dropped. At least two groups
        with at least 3 non-null observations each are required.
    alpha : float, default 0.05
        Significance level. `is_equal` is `p_value >= alpha`.
    center : {"median", "mean", "trimmed"}, default "median"
        Centering used by Levene's test. The median (Brown-Forsythe) is the
        robust default and is what skewed metrics need.

    Returns
    -------
    EqualVarianceResult

    Raises
    ------
    TypeError
        If `groups` is not a dict of Series or `alpha` is not a number.
    ValueError
        If `alpha` is not in `(0, 1)`, `center` is unsupported, or fewer than
        two usable groups remain.
    """
    if not isinstance(groups, dict):
        raise TypeError(f"groups must be a dict, got {type(groups).__name__}")
    if center not in ("median", "mean", "trimmed"):
        raise ValueError(f"center must be median/mean/trimmed, got {center!r}")
    _check_alpha(alpha)

    usable: dict[str, pd.Series] = {}
    for name, values in groups.items():
        _check_series(f"groups[{name!r}]", values)
        clean = values.dropna()
        if len(clean) >= MIN_N:
            usable[str(name)] = clean

    if len(usable) < 2:
        raise ValueError("need at least two groups with 3+ non-null observations")

    statistic, p_value = sps.levene(*(s.to_numpy() for s in usable.values()), center=center)
    return EqualVarianceResult(
        method=f"levene({center})",
        statistic=float(statistic),
        p_value=float(p_value),
        groups=list(usable.keys()),
        is_equal=bool(p_value >= alpha),
        alpha=alpha,
    )
