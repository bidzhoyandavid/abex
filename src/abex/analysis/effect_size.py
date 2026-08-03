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


def cohens_d(control: pd.Series, treatment: pd.Series) -> float:
    control, treatment = control.dropna(), treatment.dropna()
    n1, n2 = len(control), len(treatment)
    pooled_std = (
        ((n1 - 1) * control.var(ddof=1) + (n2 - 1) * treatment.var(ddof=1)) / (n1 + n2 - 2)
    ) ** 0.5
    if pooled_std == 0:
        return 0.0
    return float((treatment.mean() - control.mean()) / pooled_std)


def relative_lift(control: pd.Series, treatment: pd.Series) -> float:
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
    control, treatment = control.dropna(), treatment.dropna()
    absolute_diff = float(treatment.mean() - control.mean())
    return EffectSizeResult(
        absolute_diff=absolute_diff,
        relative_lift=relative_lift(control, treatment),
        cohens_d=cohens_d(control, treatment),
        ci_low=ci_low,
        ci_high=ci_high,
    )
