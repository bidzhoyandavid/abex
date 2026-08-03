import numpy as np
import pandas as pd
import pytest

from abex.stats.frequentist import t_test
from abex.stats.ratio import (
    aggregate_by_cluster,
    compute_ratio,
    linearize,
    linearize_groups,
    pooled_ratio,
    ratio_effect,
)


def _simulate_per_user(n_users: int, revenue_per_session: float, rng: np.random.Generator) -> pd.DataFrame:
    """Users with a variable number of sessions each; revenue accrues per session.
    This is exactly the case a naive per-event t-test gets wrong: users with
    more sessions contribute more rows and are not independent observations.
    """
    sessions = rng.poisson(3, size=n_users) + 1  # avoid 0-session users here
    revenue = sessions * revenue_per_session + rng.normal(0, 1, size=n_users)
    return pd.DataFrame({"sessions": sessions, "revenue": revenue})


def test_compute_ratio_basic():
    num = pd.Series([10, 20, 30])
    den = pd.Series([1, 2, 3])
    assert compute_ratio(num, den) == pytest.approx(10.0)


def test_compute_ratio_zero_denominator_raises():
    with pytest.raises(ValueError):
        compute_ratio(pd.Series([1, 2]), pd.Series([0, 0]))


def test_pooled_ratio_matches_manual_calc():
    r = pooled_ratio((pd.Series([10, 10]), pd.Series([1, 1])), (pd.Series([20, 20]), pd.Series([2, 2])))
    assert r == pytest.approx(10.0)


def test_linearize_requires_equal_length():
    with pytest.raises(ValueError):
        linearize(pd.Series([1, 2, 3]), pd.Series([1, 2]), global_ratio=1.0)


def test_linearize_zero_denominator_row_is_just_numerator():
    l = linearize(pd.Series([5.0]), pd.Series([0.0]), global_ratio=2.0)
    assert l.iloc[0] == 5.0


def test_ratio_effect_known_values():
    control_num, control_den = pd.Series([100.0, 100.0]), pd.Series([10.0, 10.0])
    treatment_num, treatment_den = pd.Series([120.0, 120.0]), pd.Series([10.0, 10.0])
    effect = ratio_effect(control_num, control_den, treatment_num, treatment_den)
    assert effect.control_ratio == pytest.approx(10.0)
    assert effect.treatment_ratio == pytest.approx(12.0)
    assert effect.relative_lift == pytest.approx(0.2)


def test_linearized_t_test_catches_known_effect_under_clustering():
    rng = np.random.default_rng(0)
    control = _simulate_per_user(n_users=500, revenue_per_session=10.0, rng=rng)
    treatment = _simulate_per_user(n_users=500, revenue_per_session=13.0, rng=rng)

    lin = linearize_groups(
        control["revenue"], control["sessions"], treatment["revenue"], treatment["sessions"]
    )
    result = t_test(lin.control_linearized, lin.treatment_linearized)
    assert result.p_value < 0.05


def test_linearized_t_test_no_false_positive_on_null_effect():
    rng = np.random.default_rng(1)
    control = _simulate_per_user(n_users=500, revenue_per_session=10.0, rng=rng)
    treatment = _simulate_per_user(n_users=500, revenue_per_session=10.0, rng=rng)

    lin = linearize_groups(
        control["revenue"], control["sessions"], treatment["revenue"], treatment["sessions"]
    )
    result = t_test(lin.control_linearized, lin.treatment_linearized)
    assert result.p_value > 0.05


def test_aggregate_by_cluster_sums_per_user():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2],
            "revenue": [10, 5, 1, 1, 1],
            "sessions": [1, 1, 1, 1, 1],
        }
    )
    agg = aggregate_by_cluster(df, cluster_col="user_id", numerator_col="revenue", denominator_col="sessions")
    row_user1 = agg.loc[agg["user_id"] == 1].iloc[0]
    row_user2 = agg.loc[agg["user_id"] == 2].iloc[0]
    assert row_user1["revenue"] == 15
    assert row_user2["revenue"] == 3
    assert row_user2["sessions"] == 3
