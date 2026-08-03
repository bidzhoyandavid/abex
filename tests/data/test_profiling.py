import numpy as np
import pandas as pd

from abex.data.profiling import profile_metric


def test_binary_metric_detected():
    df = pd.DataFrame({"group": ["a", "b"] * 50, "metric": [0, 1] * 50})
    profile = profile_metric(df, metric_col="metric", group_col="group")
    assert profile.kind == "binary"
    assert profile.is_two_group


def test_count_metric_detected():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"group": ["a", "b"] * 50, "metric": rng.poisson(3, size=100)}
    )
    profile = profile_metric(df, metric_col="metric", group_col="group")
    assert profile.kind == "count"


def test_continuous_metric_and_skew():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"group": ["a", "b"] * 500, "metric": rng.exponential(1.0, size=1000)}
    )
    profile = profile_metric(df, metric_col="metric", group_col="group")
    assert profile.kind == "continuous"
    assert profile.skewness > 1.0


def test_three_groups_counted():
    df = pd.DataFrame({"group": ["a", "b", "c"] * 20, "metric": range(60)})
    profile = profile_metric(df, metric_col="metric", group_col="group")
    assert profile.n_groups == 3
    assert not profile.is_two_group
