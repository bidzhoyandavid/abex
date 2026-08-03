"""Guardrail metric checks — did the primary lift come at the cost of
something else that matters (latency, errors, unsubscribes, ...)?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class GuardrailResult:
    metric: str
    control_mean: float
    treatment_mean: float
    relative_change: float
    max_allowed_degradation: float
    is_violated: bool


def check_guardrail(
    control: pd.Series,
    treatment: pd.Series,
    metric_name: str,
    max_allowed_degradation: float,
    higher_is_better: bool = True,
) -> GuardrailResult:
    """max_allowed_degradation is a positive fraction, e.g. 0.02 for "no more
    than 2% worse". Direction is controlled by `higher_is_better`.
    """
    control_mean = float(control.dropna().mean())
    treatment_mean = float(treatment.dropna().mean())

    if control_mean == 0:
        relative_change = float("nan")
        is_violated = False
    else:
        relative_change = (treatment_mean - control_mean) / control_mean
        degradation = -relative_change if higher_is_better else relative_change
        is_violated = degradation > max_allowed_degradation

    return GuardrailResult(
        metric=metric_name,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        relative_change=relative_change,
        max_allowed_degradation=max_allowed_degradation,
        is_violated=is_violated,
    )
