import pandas as pd
import pytest

from abex.stats.bootstrap import bootstrap_ci, permutation_test
from abex.stats.frequentist import (
    anova_oneway,
    kruskal_wallis,
    mann_whitney,
    t_test,
    z_test_proportions,
)
from abex.stats.multiple_testing import benjamini_hochberg, bonferroni
from abex.stats.ratio import compute_ratio, linearize, pooled_ratio


def test_t_test_rejects_non_series():
    with pytest.raises(TypeError):
        t_test([1, 2, 3], pd.Series([1, 2, 3]))


def test_t_test_rejects_non_bool_equal_var():
    with pytest.raises(TypeError):
        t_test(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), equal_var="yes")


def test_z_test_proportions_rejects_non_int():
    with pytest.raises(TypeError):
        z_test_proportions(10.5, 100, 12, 100)


def test_z_test_proportions_rejects_successes_over_n():
    with pytest.raises(ValueError):
        z_test_proportions(150, 100, 12, 100)


def test_kruskal_wallis_rejects_fewer_than_three_groups():
    with pytest.raises(ValueError):
        kruskal_wallis(pd.Series([1, 2]), pd.Series([3, 4]))


def test_anova_oneway_rejects_non_series_group():
    with pytest.raises(TypeError):
        anova_oneway(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), [1, 2, 3])


def test_mann_whitney_rejects_non_series():
    with pytest.raises(TypeError):
        mann_whitney([1, 2, 3], pd.Series([1, 2, 3]))


def test_bootstrap_ci_rejects_non_callable_statistic():
    with pytest.raises(TypeError):
        bootstrap_ci(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), statistic="mean")


def test_bootstrap_ci_rejects_non_positive_n_resamples():
    with pytest.raises(ValueError):
        bootstrap_ci(pd.Series([1, 2, 3]), pd.Series([1, 2, 3]), n_resamples=0)


def test_permutation_test_rejects_empty_series():
    with pytest.raises(ValueError):
        permutation_test(pd.Series([], dtype=float), pd.Series([1, 2, 3]))


def test_bonferroni_rejects_empty_p_values():
    with pytest.raises(ValueError):
        bonferroni([])


def test_bonferroni_rejects_out_of_range_p_value():
    with pytest.raises(ValueError):
        bonferroni([0.1, 1.5])


def test_benjamini_hochberg_rejects_non_list():
    with pytest.raises(TypeError):
        benjamini_hochberg((0.1, 0.2))


def test_compute_ratio_rejects_zero_denominator():
    with pytest.raises(ValueError):
        compute_ratio(pd.Series([1, 2, 3]), pd.Series([0, 0, 0]))


def test_pooled_ratio_rejects_empty_pairs():
    with pytest.raises(ValueError):
        pooled_ratio()


def test_linearize_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        linearize(pd.Series([1, 2, 3]), pd.Series([1, 2]), global_ratio=0.5)
