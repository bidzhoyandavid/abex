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
    """Check whether a guardrail metric degraded beyond an allowed tolerance.

    `max_allowed_degradation` is a positive fraction, e.g. 0.02 for "no more
    than 2% worse". Direction is controlled by `higher_is_better`.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before computing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before computing.
    metric_name : str
        Name of the guardrail metric, carried through to `GuardrailResult.metric`.
    max_allowed_degradation : float
        Maximum tolerated relative degradation, as a positive fraction (e.g.
        0.02 for 2%).
    higher_is_better : bool, default True
        Whether a higher metric value is the desirable direction. If True,
        degradation is a decrease; if False, degradation is an increase.

    Returns
    -------
    GuardrailResult
        Dataclass with control/treatment means, relative change, and whether
        the degradation exceeded `max_allowed_degradation`.

    Raises
    ------
    TypeError
        If `control`/`treatment` is not a pandas Series, `metric_name` is not
        a str, or `max_allowed_degradation`/`higher_is_better` has the wrong type.
    ValueError
        If `control`/`treatment` has no non-null observations, or
        `max_allowed_degradation` is negative.
    """
    if not isinstance(control, pd.Series):
        raise TypeError(f"control must be a pandas Series, got {type(control).__name__}")
    if not isinstance(treatment, pd.Series):
        raise TypeError(f"treatment must be a pandas Series, got {type(treatment).__name__}")
    if not isinstance(metric_name, str):
        raise TypeError(f"metric_name must be a str, got {type(metric_name).__name__}")
    if not isinstance(max_allowed_degradation, (int, float)) or isinstance(max_allowed_degradation, bool):
        raise TypeError(
            f"max_allowed_degradation must be a number, got {type(max_allowed_degradation).__name__}"
        )
    if max_allowed_degradation < 0:
        raise ValueError("max_allowed_degradation must be non-negative")
    if not isinstance(higher_is_better, bool):
        raise TypeError(f"higher_is_better must be a bool, got {type(higher_is_better).__name__}")
    if control.dropna().empty:
        raise ValueError("control has no non-null observations")
    if treatment.dropna().empty:
        raise ValueError("treatment has no non-null observations")

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
