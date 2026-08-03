import pandas as pd

from abex.analysis.guardrails import check_guardrail


def test_guardrail_violated_when_degradation_exceeds_threshold():
    control = pd.Series([100] * 50)
    treatment = pd.Series([90] * 50)
    result = check_guardrail(control, treatment, "latency_ok", max_allowed_degradation=0.05, higher_is_better=True)
    assert result.is_violated


def test_guardrail_not_violated_within_threshold():
    control = pd.Series([100] * 50)
    treatment = pd.Series([99] * 50)
    result = check_guardrail(control, treatment, "latency_ok", max_allowed_degradation=0.05, higher_is_better=True)
    assert not result.is_violated


def test_guardrail_lower_is_better_direction():
    control = pd.Series([100] * 50)
    treatment = pd.Series([110] * 50)
    result = check_guardrail(control, treatment, "error_rate", max_allowed_degradation=0.05, higher_is_better=False)
    assert result.is_violated
