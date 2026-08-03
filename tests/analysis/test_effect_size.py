import pandas as pd
import pytest

from abex.analysis.effect_size import cohens_d, effect_size_summary, relative_lift


def test_cohens_d_positive_for_positive_shift():
    control = pd.Series([1, 2, 3, 4, 5])
    treatment = pd.Series([2, 3, 4, 5, 6])
    assert cohens_d(control, treatment) > 0


def test_cohens_d_zero_variance_no_div_by_zero():
    control = pd.Series([1.0] * 10)
    treatment = pd.Series([1.0] * 10)
    assert cohens_d(control, treatment) == 0.0


def test_relative_lift_positive():
    control = pd.Series([10, 10, 10])
    treatment = pd.Series([12, 12, 12])
    assert relative_lift(control, treatment) == pytest.approx(0.2)


def test_effect_size_summary_shape():
    control = pd.Series([1, 2, 3])
    treatment = pd.Series([2, 3, 4])
    summary = effect_size_summary(control, treatment)
    assert summary.absolute_diff == 1.0
