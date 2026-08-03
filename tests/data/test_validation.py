import pandas as pd
import pytest

from abex.data.loaders import coerce_schema, load_csv, load_parquet
from abex.data.outliers import cap, detect_outliers, log_transform, trim, winsorize
from abex.data.profiling import profile_metric
from abex.data.validators import validate


def test_validate_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate([1, 2, 3], "group", "metric")


def test_validate_rejects_non_str_group_col():
    df = pd.DataFrame({"group": ["a", "b"], "metric": [1, 2]})
    with pytest.raises(TypeError):
        validate(df, 123, "metric")


def test_validate_rejects_missing_column():
    df = pd.DataFrame({"group": ["a", "b"], "metric": [1, 2]})
    with pytest.raises(KeyError):
        validate(df, "group", "missing_col")


def test_profile_metric_rejects_non_dataframe():
    with pytest.raises(TypeError):
        profile_metric([1, 2, 3], "metric", "group")


def test_profile_metric_rejects_missing_column():
    df = pd.DataFrame({"group": ["a", "b"], "metric": [1, 2]})
    with pytest.raises(KeyError):
        profile_metric(df, "missing_metric", "group")


def test_detect_outliers_rejects_non_series():
    with pytest.raises(TypeError):
        detect_outliers([1, 2, 3, 4, 5])


def test_detect_outliers_rejects_unknown_method():
    with pytest.raises(ValueError):
        detect_outliers(pd.Series([1, 2, 3, 4, 5]), method="bogus")


def test_detect_outliers_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        detect_outliers(pd.Series([1, 2, 3, 4, 5]), threshold=0)


def test_winsorize_rejects_bad_quantiles():
    with pytest.raises(ValueError):
        winsorize(pd.Series([1, 2, 3]), lower_q=0.9, upper_q=0.1)


def test_trim_rejects_misaligned_mask():
    values = pd.Series([1, 2, 3], index=[0, 1, 2])
    mask = pd.Series([True, False], index=[0, 1])
    with pytest.raises(ValueError):
        trim(values, mask)


def test_trim_rejects_non_bool_mask():
    values = pd.Series([1, 2, 3])
    mask = pd.Series([1, 0, 1])
    with pytest.raises(TypeError):
        trim(values, mask)


def test_cap_rejects_lower_greater_than_upper():
    with pytest.raises(ValueError):
        cap(pd.Series([1, 2, 3]), lower=10, upper=1)


def test_log_transform_rejects_non_positive_shifted_values():
    with pytest.raises(ValueError):
        log_transform(pd.Series([-2, 0, 1]), offset=1.0)


def test_load_csv_rejects_bad_path_type():
    with pytest.raises(TypeError):
        load_csv(123)


def test_load_parquet_rejects_bad_path_type():
    with pytest.raises(TypeError):
        load_parquet(123)


def test_coerce_schema_rejects_non_dataframe():
    with pytest.raises(TypeError):
        coerce_schema([1, 2, 3], {"a": "int64"})


def test_coerce_schema_rejects_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        coerce_schema(df, {"missing": "int64"})


def test_coerce_schema_rejects_uncastable_column():
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(ValueError):
        coerce_schema(df, {"a": "int64"})
