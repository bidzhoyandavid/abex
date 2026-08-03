"""Read raw experiment data from common sources into a canonical dataframe."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **read_csv_kwargs)


def load_parquet(path: str | Path, **read_parquet_kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **read_parquet_kwargs)


def coerce_schema(
    df: pd.DataFrame,
    dtypes: dict[str, str],
) -> pd.DataFrame:
    """Cast columns to expected dtypes, raising a clear error on failure."""
    out = df.copy()
    for col, dtype in dtypes.items():
        if col not in out.columns:
            raise KeyError(f"expected column {col!r} not found in dataframe")
        try:
            out[col] = out[col].astype(dtype)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"cannot cast column {col!r} to {dtype!r}: {exc}") from exc
    return out
