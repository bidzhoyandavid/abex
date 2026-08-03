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


def bonferroni(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    n = len(p_values)
    adjusted = [min(p * n, 1.0) for p in p_values]
    reject = [p_adj < alpha for p_adj in adjusted]
    return CorrectionResult(
        method="bonferroni", p_values=p_values, adjusted_p_values=adjusted, reject=reject, alpha=alpha
    )


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """BH step-up procedure controlling the false discovery rate."""
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
