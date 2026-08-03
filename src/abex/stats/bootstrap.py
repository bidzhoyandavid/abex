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


def _check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")


def _check_common_args(alpha: float, n_iterations: int, n_iterations_name: str, random_state: int) -> None:
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not isinstance(n_iterations, int) or isinstance(n_iterations, bool):
        raise TypeError(f"{n_iterations_name} must be an int, got {type(n_iterations).__name__}")
    if n_iterations <= 0:
        raise ValueError(f"{n_iterations_name} must be positive")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError(f"random_state must be an int, got {type(random_state).__name__}")


def bootstrap_ci(
    control: pd.Series,
    treatment: pd.Series,
    statistic: Callable[[np.ndarray, np.ndarray], float] = lambda c, t: t.mean() - c.mean(),
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI on an arbitrary statistic (default: mean difference).

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before resampling.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before resampling.
    statistic : Callable[[np.ndarray, np.ndarray], float], optional
        Function `(control_array, treatment_array) -> float` to bootstrap.
        Defaults to the mean difference `treatment.mean() - control.mean()`.
    n_resamples : int, default 10_000
        Number of bootstrap resamples. Must be positive.
    alpha : float, default 0.05
        Two-sided significance level for the percentile CI, must be in `(0, 1)`.
    random_state : int, default 42
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    BootstrapResult
        `point_estimate` on the original data, `ci_low`/`ci_high` percentile
        bounds, plus the `alpha`/`n_resamples`/`random_state` inputs.

    Raises
    ------
    TypeError
        If `control`/`treatment` is not a pandas Series, `statistic` is not
        callable, or `n_resamples`/`alpha`/`random_state` has the wrong type.
    ValueError
        If `control`/`treatment` has no non-null observations, or
        `n_resamples` is not positive, or `alpha` is not in `(0, 1)`.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    if not callable(statistic):
        raise TypeError(f"statistic must be callable, got {type(statistic).__name__}")
    _check_common_args(alpha, n_resamples, "n_resamples", random_state)

    control_arr = control.dropna().to_numpy()
    treatment_arr = treatment.dropna().to_numpy()
    if len(control_arr) == 0:
        raise ValueError("control has no non-null observations")
    if len(treatment_arr) == 0:
        raise ValueError("treatment has no non-null observations")

    rng = np.random.default_rng(random_state)
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
    """Two-sided permutation test: shuffles the group label under H0 of no effect.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before permuting.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before permuting.
    statistic : Callable[[np.ndarray, np.ndarray], float], optional
        Function `(control_array, treatment_array) -> float` to test.
        Defaults to the mean difference `treatment.mean() - control.mean()`.
    n_permutations : int, default 10_000
        Number of label permutations. Must be positive.
    random_state : int, default 42
        Seed for the random number generator, for reproducibility.

    Returns
    -------
    PermutationResult
        `observed_statistic` on the original labeling, two-sided `p_value`,
        plus the `n_permutations`/`random_state` inputs.

    Raises
    ------
    TypeError
        If `control`/`treatment` is not a pandas Series, `statistic` is not
        callable, or `n_permutations`/`random_state` has the wrong type.
    ValueError
        If `control`/`treatment` has no non-null observations, or
        `n_permutations` is not positive.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    if not callable(statistic):
        raise TypeError(f"statistic must be callable, got {type(statistic).__name__}")
    if not isinstance(n_permutations, int) or isinstance(n_permutations, bool):
        raise TypeError(f"n_permutations must be an int, got {type(n_permutations).__name__}")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    if not isinstance(random_state, int) or isinstance(random_state, bool):
        raise TypeError(f"random_state must be an int, got {type(random_state).__name__}")

    control_arr = control.dropna().to_numpy()
    treatment_arr = treatment.dropna().to_numpy()
    if len(control_arr) == 0:
        raise ValueError("control has no non-null observations")
    if len(treatment_arr) == 0:
        raise ValueError("treatment has no non-null observations")

    rng = np.random.default_rng(random_state)
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
