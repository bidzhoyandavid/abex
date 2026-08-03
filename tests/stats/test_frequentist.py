import numpy as np
import pandas as pd

from abex.stats.frequentist import (
    anova_oneway,
    kruskal_wallis,
    mann_whitney,
    t_test,
    wilcoxon_signed_rank,
    z_test_proportions,
)


def test_t_test_catches_known_effect():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 500))
    treatment = pd.Series(rng.normal(0.5, 1, 500))
    result = t_test(control, treatment)
    assert result.p_value < 0.05


def test_t_test_no_false_positive_on_null_effect():
    rng = np.random.default_rng(1)
    control = pd.Series(rng.normal(0, 1, 500))
    treatment = pd.Series(rng.normal(0, 1, 500))
    result = t_test(control, treatment)
    assert result.p_value > 0.05


def test_z_test_proportions_catches_known_effect():
    result = z_test_proportions(control_successes=100, control_n=1000, treatment_successes=160, treatment_n=1000)
    assert result.p_value < 0.05


def test_z_test_proportions_null_effect():
    result = z_test_proportions(control_successes=100, control_n=1000, treatment_successes=102, treatment_n=1000)
    assert result.p_value > 0.05


def test_mann_whitney_catches_known_effect():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.exponential(1.0, 500))
    treatment = pd.Series(rng.exponential(1.0, 500) + 1.0)
    result = mann_whitney(control, treatment)
    assert result.p_value < 0.05


def test_wilcoxon_paired_catches_known_effect():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 200))
    treatment = control + 0.8
    result = wilcoxon_signed_rank(control, treatment)
    assert result.p_value < 0.05


def test_kruskal_wallis_requires_three_groups():
    a = pd.Series([1, 2, 3])
    b = pd.Series([4, 5, 6])
    try:
        kruskal_wallis(a, b)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_anova_catches_known_effect_across_three_groups():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 200))
    b = pd.Series(rng.normal(0, 1, 200))
    c = pd.Series(rng.normal(1, 1, 200))
    result = anova_oneway(a, b, c)
    assert result.p_value < 0.05
