"""Sample size, MDE and power calculations for two-sample comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as sps


@dataclass
class PowerResult:
    sample_size_per_group: int
    mde: float | None
    power: float
    alpha: float


def sample_size_proportion(
    baseline_rate: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Required sample size per group for a two-proportion z-test.

    Parameters
    ----------
    baseline_rate : float
        Baseline conversion rate, must be in the open interval `(0, 1)`.
    mde_abs : float
        Minimum detectable absolute effect (e.g. 0.02 for a 2 percentage
        point lift). Must be positive.
    alpha : float, default 0.05
        Two-sided significance level, must be in `(0, 1)`.
    power : float, default 0.8
        Target statistical power, must be in `(0, 1)`.

    Returns
    -------
    PowerResult
        `sample_size_per_group` (rounded up), plus the `mde`/`power`/`alpha` inputs.

    Raises
    ------
    TypeError
        If any argument is not a number.
    ValueError
        If `baseline_rate` is not in `(0, 1)`, `mde_abs` is not positive, or
        `alpha`/`power` is not in `(0, 1)`.
    """
    for name, val in (("baseline_rate", baseline_rate), ("mde_abs", mde_abs), ("alpha", alpha), ("power", power)):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1)")

    p1 = baseline_rate
    p2 = baseline_rate + mde_abs
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    p_bar = (p1 + p2) / 2

    numerator = (z_alpha * (2 * p_bar * (1 - p_bar)) ** 0.5 + z_beta * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5) ** 2
    n = numerator / (mde_abs**2)

    return PowerResult(sample_size_per_group=int(n) + 1, mde=mde_abs, power=power, alpha=alpha)


def sample_size_mean(
    std: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerResult:
    """Required sample size per group for a two-sample t-test on means.

    Parameters
    ----------
    std : float
        Assumed (pooled) standard deviation of the metric. Must be positive.
    mde_abs : float
        Minimum detectable absolute effect on the mean. Must be positive.
    alpha : float, default 0.05
        Two-sided significance level, must be in `(0, 1)`.
    power : float, default 0.8
        Target statistical power, must be in `(0, 1)`.

    Returns
    -------
    PowerResult
        `sample_size_per_group` (rounded up), plus the `mde`/`power`/`alpha` inputs.

    Raises
    ------
    TypeError
        If any argument is not a number.
    ValueError
        If `std` or `mde_abs` is not positive, or `alpha`/`power` is not in `(0, 1)`.
    """
    for name, val in (("std", std), ("mde_abs", mde_abs), ("alpha", alpha), ("power", power)):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
    if std <= 0:
        raise ValueError("std must be positive")
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not (0 < power < 1):
        raise ValueError("power must be in (0, 1)")

    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    n = 2 * ((z_alpha + z_beta) * std / mde_abs) ** 2

    return PowerResult(sample_size_per_group=int(n) + 1, mde=mde_abs, power=power, alpha=alpha)


def achieved_power_proportion(
    baseline_rate: float,
    observed_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Post-hoc power for a two-proportion comparison given the actual sample size.

    Parameters
    ----------
    baseline_rate : float
        Control-group conversion rate, must be in `[0, 1]`.
    observed_rate : float
        Treatment-group conversion rate, must be in `[0, 1]`.
    n_per_group : int
        Sample size per group actually collected. Must be a positive integer.
    alpha : float, default 0.05
        Two-sided significance level, must be in `(0, 1)`.

    Returns
    -------
    float
        Estimated statistical power, in `[0, 1]`. Returns 1.0 if the standard
        error under the alternative is 0 (degenerate case).

    Raises
    ------
    TypeError
        If `baseline_rate`/`observed_rate`/`alpha` is not a number, or
        `n_per_group` is not an int.
    ValueError
        If `baseline_rate`/`observed_rate` is outside `[0, 1]`, `n_per_group`
        is not positive, or `alpha` is not in `(0, 1)`.
    """
    for name, val in (("baseline_rate", baseline_rate), ("observed_rate", observed_rate), ("alpha", alpha)):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"{name} must be a number, got {type(val).__name__}")
    if not isinstance(n_per_group, int) or isinstance(n_per_group, bool):
        raise TypeError(f"n_per_group must be an int, got {type(n_per_group).__name__}")
    if not (0 <= baseline_rate <= 1):
        raise ValueError("baseline_rate must be in [0, 1]")
    if not (0 <= observed_rate <= 1):
        raise ValueError("observed_rate must be in [0, 1]")
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    p1, p2 = baseline_rate, observed_rate
    p_bar = (p1 + p2) / 2
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    se_null = (2 * p_bar * (1 - p_bar) / n_per_group) ** 0.5
    se_alt = (p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group) ** 0.5
    if se_alt == 0:
        return 1.0
    z = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return float(sps.norm.cdf(z))
