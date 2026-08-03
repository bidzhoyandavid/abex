"""Read raw experiment data from common sources into a canonical dataframe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file into a dataframe.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    **read_csv_kwargs
        Extra keyword arguments forwarded verbatim to `pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        Parsed dataframe.

    Raises
    ------
    TypeError
        If `path` is not a str or `pathlib.Path`.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    return pd.read_csv(path, **read_csv_kwargs)


def load_parquet(path: str | Path, **read_parquet_kwargs) -> pd.DataFrame:
    """Load a Parquet file into a dataframe.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the Parquet file.
    **read_parquet_kwargs
        Extra keyword arguments forwarded verbatim to `pandas.read_parquet`.

    Returns
    -------
    pd.DataFrame
        Parsed dataframe.

    Raises
    ------
    TypeError
        If `path` is not a str or `pathlib.Path`.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    return pd.read_parquet(path, **read_parquet_kwargs)


def coerce_schema(
    df: pd.DataFrame,
    dtypes: dict[str, str],
) -> pd.DataFrame:
    """Cast columns to expected dtypes, raising a clear error on failure.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe whose columns will be cast. Not mutated; a copy is returned.
    dtypes : dict[str, str]
        Mapping of column name to target dtype string (e.g. `{"user_id": "int64"}`).

    Returns
    -------
    pd.DataFrame
        Copy of `df` with the requested columns cast.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame or `dtypes` is not a dict.
    KeyError
        If a key of `dtypes` is not a column of `df`.
    ValueError
        If a column cannot be cast to its requested dtype.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(dtypes, dict):
        raise TypeError(f"dtypes must be a dict, got {type(dtypes).__name__}")

    out = df.copy()
    for col, dtype in dtypes.items():
        if col not in out.columns:
            raise KeyError(f"expected column {col!r} not found in dataframe")
        try:
            out[col] = out[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot cast column {col!r} to {dtype!r}: {exc}") from exc
    return out
