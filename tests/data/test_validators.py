import pandas as pd

from abex.data.validators import validate


def test_clean_data_has_no_issues():
    df = pd.DataFrame({"group": ["a", "b"] * 50, "metric": range(100)})
    report = validate(df, group_col="group", metric_col="metric")
    assert report.is_clean
    assert report.n_rows == 100


def test_nulls_flagged():
    df = pd.DataFrame({"group": ["a", "b", None], "metric": [1, 2, None]})
    report = validate(df, group_col="group", metric_col="metric")
    assert not report.is_clean
    assert any("null" in issue for issue in report.issues)


def test_single_group_flagged():
    df = pd.DataFrame({"group": ["a"] * 10, "metric": range(10)})
    report = validate(df, group_col="group", metric_col="metric")
    assert not report.is_clean


def test_duplicates_by_id_flagged():
    df = pd.DataFrame({"id": [1, 1, 2], "group": ["a", "a", "b"], "metric": [1, 2, 3]})
    report = validate(df, group_col="group", metric_col="metric", id_col="id")
    assert report.n_duplicates == 1
    assert not report.is_clean
