import numpy as np
import pandas as pd

from abex.data.outliers import cap, detect_outliers, log_transform, trim, winsorize


def _series_with_outliers():
    values = list(np.random.default_rng(0).normal(0, 1, 100)) + [1000, -1000]
    return pd.Series(values)


def test_detect_outliers_iqr_flags_extremes():
    s = _series_with_outliers()
    mask = detect_outliers(s, method="iqr")
    assert mask.iloc[-1] and mask.iloc[-2]
    assert mask.sum() < len(s) / 2


def test_detect_outliers_too_few_points_returns_all_false():
    s = pd.Series([1, 2, 3])
    mask = detect_outliers(s)
    assert not mask.any()


def test_winsorize_reduces_max():
    s = _series_with_outliers()
    result = winsorize(s, lower_q=0.01, upper_q=0.99)
    assert result.treated.max() < s.max()
    assert result.n_affected > 0


def test_trim_removes_flagged_rows():
    s = _series_with_outliers()
    mask = detect_outliers(s, method="iqr")
    result = trim(s, mask)
    assert len(result.treated) == len(s) - mask.sum()


def test_cap_bounds_values():
    s = pd.Series([1, 5, 100])
    result = cap(s, lower=0, upper=10)
    assert result.treated.max() == 10


def test_log_transform_rejects_nonpositive():
    s = pd.Series([-5, 1, 2])
    try:
        log_transform(s)
        assert False, "expected ValueError"
    except ValueError:
        pass
