"""Corrections for testing multiple metrics/segments at once."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectionResult:
    method: str
    p_values: list[float]
    adjusted_p_values: list[float]
    reject: list[bool]
    alpha: float


def _check_p_values(p_values: list[float], alpha: float) -> None:
    if not isinstance(p_values, list):
        raise TypeError(f"p_values must be a list, got {type(p_values).__name__}")
    if len(p_values) == 0:
        raise ValueError("p_values must not be empty")
    for i, p in enumerate(p_values):
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise TypeError(f"p_values[{i}] must be a number, got {type(p).__name__}")
        if not (0 <= p <= 1):
            raise ValueError(f"p_values[{i}] must be in [0, 1], got {p!r}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")


def bonferroni(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni correction: multiply each p-value by the number of tests.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from the individual tests, each in `[0, 1]`.
    alpha : float, default 0.05
        Family-wise significance level, must be in `(0, 1)`.

    Returns
    -------
    CorrectionResult
        `adjusted_p_values` (capped at 1.0) and `reject` flags at `alpha`.

    Raises
    ------
    TypeError
        If `p_values` is not a list (or its elements aren't numbers), or
        `alpha` is not a number.
    ValueError
        If `p_values` is empty, any element is outside `[0, 1]`, or `alpha`
        is not in `(0, 1)`.
    """
    _check_p_values(p_values, alpha)
    n = len(p_values)
    adjusted = [min(p * n, 1.0) for p in p_values]
    reject = [p_adj < alpha for p_adj in adjusted]
    return CorrectionResult(
        method="bonferroni", p_values=p_values, adjusted_p_values=adjusted, reject=reject, alpha=alpha
    )


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """BH step-up procedure controlling the false discovery rate.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from the individual tests, each in `[0, 1]`.
    alpha : float, default 0.05
        Target false discovery rate, must be in `(0, 1)`.

    Returns
    -------
    CorrectionResult
        `adjusted_p_values` (BH-adjusted, monotone, capped at 1.0) and
        `reject` flags at `alpha`.

    Raises
    ------
    TypeError
        If `p_values` is not a list (or its elements aren't numbers), or
        `alpha` is not a number.
    ValueError
        If `p_values` is empty, any element is outside `[0, 1]`, or `alpha`
        is not in `(0, 1)`.
    """
    _check_p_values(p_values, alpha)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = np.array(p_values)[order]

    adjusted_ranked = ranked_p * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank down
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0, 1)

    adjusted = np.empty(n)
    adjusted[order] = adjusted_ranked
    reject = adjusted < alpha

    return CorrectionResult(
        method="benjamini_hochberg",
        p_values=p_values,
        adjusted_p_values=adjusted.tolist(),
        reject=reject.tolist(),
        alpha=alpha,
    )
