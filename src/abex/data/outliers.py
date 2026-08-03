"""Outlier detection and treatment for a single continuous metric.

Detection returns a boolean mask; treatment functions are pure — they
return a new Series plus a small summary of what changed, never mutate
the input in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal["iqr", "mad", "zscore"]


def detect_outliers(values: pd.Series, method: Method = "iqr", threshold: float = 1.5) -> pd.Series:
    """Return a boolean mask, True where the value is flagged as an outlier."""
    non_null = values.dropna()
    if len(non_null) < 4:
        return pd.Series(False, index=values.index)

    if method == "iqr":
        q1, q3 = np.percentile(non_null, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(False, index=values.index)
        lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
        mask = (values < lower) | (values > upper)
    elif method == "mad":
        median = non_null.median()
        mad = (non_null - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=values.index)
        modified_z = 0.6745 * (values - median) / mad
        mask = modified_z.abs() > threshold
    elif method == "zscore":
        mean, std = non_null.mean(), non_null.std(ddof=1)
        if std == 0:
            return pd.Series(False, index=values.index)
        mask = ((values - mean) / std).abs() > threshold
    else:
        raise ValueError(f"unknown method: {method!r}")

    return mask.fillna(False)


@dataclass
class OutlierTreatmentResult:
    treated: pd.Series
    n_affected: int
    share_affected: float
    method: str


def winsorize(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> OutlierTreatmentResult:
    lower, upper = values.quantile([lower_q, upper_q])
    treated = values.clip(lower=lower, upper=upper)
    n_affected = int((treated != values).sum())
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=n_affected,
        share_affected=n_affected / len(values) if len(values) else 0.0,
        method="winsorize",
    )


def trim(values: pd.Series, mask: pd.Series) -> OutlierTreatmentResult:
    """Drop flagged rows entirely. `mask` typically comes from detect_outliers()."""
    treated = values[~mask]
    n_affected = int(mask.sum())
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=n_affected,
        share_affected=n_affected / len(values) if len(values) else 0.0,
        method="trim",
    )


def cap(values: pd.Series, lower: float | None = None, upper: float | None = None) -> OutlierTreatmentResult:
    treated = values.clip(lower=lower, upper=upper)
    n_affected = int((treated != values).sum())
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=n_affected,
        share_affected=n_affected / len(values) if len(values) else 0.0,
        method="cap",
    )


def log_transform(values: pd.Series, offset: float = 1.0) -> OutlierTreatmentResult:
    """log1p-style transform; requires values + offset > 0."""
    if (values + offset <= 0).any():
        raise ValueError("log_transform requires values + offset > 0 for all entries")
    treated = np.log(values + offset)
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=len(values),
        share_affected=1.0,
        method="log_transform",
    )
