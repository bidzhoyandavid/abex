import numpy as np
import pandas as pd

from abex.design.covariate_balance import check_covariate_balance


def test_balanced_covariate_passes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "group": ["a"] * 500 + ["b"] * 500,
            "pre_metric": np.concatenate([rng.normal(10, 1, 500), rng.normal(10, 1, 500)]),
        }
    )
    result = check_covariate_balance(df, covariate_col="pre_metric", group_col="group")
    assert result.is_balanced


def test_imbalanced_covariate_flagged():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "group": ["a"] * 500 + ["b"] * 500,
            "pre_metric": np.concatenate([rng.normal(10, 1, 500), rng.normal(11, 1, 500)]),
        }
    )
    result = check_covariate_balance(df, covariate_col="pre_metric", group_col="group")
    assert not result.is_balanced


def test_requires_exactly_two_groups():
    df = pd.DataFrame({"group": ["a", "b", "c"], "pre_metric": [1, 2, 3]})
    try:
        check_covariate_balance(df, covariate_col="pre_metric", group_col="group")
        assert False, "expected ValueError"
    except ValueError:
        pass
