import pandas as pd
import pytest

from abex.analysis.effect_size import cohens_d, effect_size_summary, relative_lift
from abex.analysis.guardrails import check_guardrail


@pytest.mark.parametrize("fn", [cohens_d, relative_lift])
def test_effect_size_rejects_non_series_control(fn):
    with pytest.raises(TypeError):
        fn([1, 2, 3], pd.Series([1, 2, 3]))


@pytest.mark.parametrize("fn", [cohens_d, relative_lift])
def test_effect_size_rejects_non_series_treatment(fn):
    with pytest.raises(TypeError):
        fn(pd.Series([1, 2, 3]), [1, 2, 3])


@pytest.mark.parametrize("fn", [cohens_d, relative_lift])
def test_effect_size_rejects_all_null_series(fn):
    with pytest.raises(ValueError):
        fn(pd.Series([None, None]), pd.Series([1, 2, 3]))


def test_cohens_d_rejects_too_few_combined_observations():
    with pytest.raises(ValueError):
        cohens_d(pd.Series([1.0]), pd.Series([2.0]))


def test_effect_size_summary_rejects_bad_ci_type():
    with pytest.raises(TypeError):
        effect_size_summary(pd.Series([1, 2, 3]), pd.Series([2, 3, 4]), ci_low="low")


def test_check_guardrail_rejects_non_series():
    with pytest.raises(TypeError):
        check_guardrail([1, 2, 3], pd.Series([1, 2, 3]), "metric", 0.05)


def test_check_guardrail_rejects_non_str_metric_name():
    with pytest.raises(TypeError):
        check_guardrail(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), 123, 0.05)


def test_check_guardrail_rejects_negative_degradation():
    with pytest.raises(ValueError):
        check_guardrail(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), "metric", -0.05)


def test_check_guardrail_rejects_non_bool_higher_is_better():
    with pytest.raises(TypeError):
        check_guardrail(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), "metric", 0.05, higher_is_better="yes")


def test_check_guardrail_rejects_empty_series():
    with pytest.raises(ValueError):
        check_guardrail(pd.Series([], dtype=float), pd.Series([1, 2, 3]), "metric", 0.05)
