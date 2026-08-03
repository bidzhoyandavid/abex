import pandas as pd
import pytest

from abex.design.covariate_balance import check_covariate_balance
from abex.design.power import (
    achieved_power_proportion,
    sample_size_mean,
    sample_size_proportion,
)
from abex.design.srm import check_srm


def test_sample_size_proportion_rejects_bad_baseline_rate():
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=1.5, mde_abs=0.05)


def test_sample_size_proportion_rejects_non_number_type():
    with pytest.raises(TypeError):
        sample_size_proportion(baseline_rate="0.1", mde_abs=0.05)


def test_sample_size_proportion_rejects_non_positive_mde():
    with pytest.raises(ValueError):
        sample_size_proportion(baseline_rate=0.1, mde_abs=0)


def test_sample_size_mean_rejects_non_positive_std():
    with pytest.raises(ValueError):
        sample_size_mean(std=0, mde_abs=1.0)


def test_achieved_power_proportion_rejects_non_int_n():
    with pytest.raises(TypeError):
        achieved_power_proportion(0.1, 0.12, n_per_group=100.5)


def test_achieved_power_proportion_rejects_out_of_range_rate():
    with pytest.raises(ValueError):
        achieved_power_proportion(1.5, 0.12, n_per_group=100)


def test_check_srm_rejects_non_dict_group_counts():
    with pytest.raises(TypeError):
        check_srm([100, 100])


def test_check_srm_rejects_single_group():
    with pytest.raises(ValueError):
        check_srm({"control": 100})


def test_check_srm_rejects_ratios_not_summing_to_one():
    with pytest.raises(ValueError):
        check_srm({"control": 100, "treatment": 100}, expected_ratios={"control": 0.6, "treatment": 0.6})


def test_check_covariate_balance_rejects_non_dataframe():
    with pytest.raises(TypeError):
        check_covariate_balance([1, 2, 3], "covariate", "group")


def test_check_covariate_balance_rejects_wrong_group_count():
    df = pd.DataFrame({"covariate": [1, 2, 3], "group": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        check_covariate_balance(df, "covariate", "group")


def test_check_covariate_balance_rejects_missing_column():
    df = pd.DataFrame({"covariate": [1, 2, 3], "group": ["a", "a", "b"]})
    with pytest.raises(KeyError):
        check_covariate_balance(df, "missing_col", "group")
