"""Bootstrap CI and permutation tests — distribution-free, robust to skew
and outliers at the cost of compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    alpha: float
    n_resamples: int
    random_state: int


def bootstrap_ci(
    control: pd.Series,
    treatment: pd.Series,
    statistic: Callable[[np.ndarray, np.ndarray], float] = lambda c, t: t.mean() - c.mean(),
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI on an arbitrary statistic (default: mean difference)."""
    rng = np.random.default_rng(random_state)
    control_arr = control.dropna().to_numpy()
    treatment_arr = treatment.dropna().to_numpy()

    point_estimate = float(statistic(control_arr, treatment_arr))

    resampled = np.empty(n_resamples)
    for i in range(n_resamples):
        c_sample = rng.choice(control_arr, size=len(control_arr), replace=True)
        t_sample = rng.choice(treatment_arr, size=len(treatment_arr), replace=True)
        resampled[i] = statistic(c_sample, t_sample)

    ci_low, ci_high = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        alpha=alpha,
        n_resamples=n_resamples,
        random_state=random_state,
    )


@dataclass
class PermutationResult:
    observed_statistic: float
    p_value: float
    n_permutations: int
    random_state: int


def permutation_test(
    control: pd.Series,
    treatment: pd.Series,
    statistic: Callable[[np.ndarray, np.ndarray], float] = lambda c, t: t.mean() - c.mean(),
    n_permutations: int = 10_000,
    random_state: int = 42,
) -> PermutationResult:
    """Two-sided permutation test: shuffles the group label under H0 of no effect."""
    rng = np.random.default_rng(random_state)
    control_arr = control.dropna().to_numpy()
    treatment_arr = treatment.dropna().to_numpy()

    observed = float(statistic(control_arr, treatment_arr))

    pooled = np.concatenate([control_arr, treatment_arr])
    n_control = len(control_arr)

    perm_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_stats[i] = statistic(shuffled[:n_control], shuffled[n_control:])

    p_value = float((np.abs(perm_stats) >= abs(observed)).mean())

    return PermutationResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        random_state=random_state,
    )
