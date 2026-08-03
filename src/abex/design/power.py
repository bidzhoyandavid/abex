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
    """Required sample size per group for a two-proportion z-test."""
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be in (0, 1)")
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive")

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
    """Required sample size per group for a two-sample t-test on means."""
    if std <= 0:
        raise ValueError("std must be positive")
    if mde_abs <= 0:
        raise ValueError("mde_abs must be positive")

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
    """Post-hoc power for a two-proportion comparison given the actual sample size."""
    p1, p2 = baseline_rate, observed_rate
    p_bar = (p1 + p2) / 2
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    se_null = (2 * p_bar * (1 - p_bar) / n_per_group) ** 0.5
    se_alt = (p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group) ** 0.5
    if se_alt == 0:
        return 1.0
    z = (abs(p1 - p2) - z_alpha * se_null) / se_alt
    return float(sps.norm.cdf(z))
