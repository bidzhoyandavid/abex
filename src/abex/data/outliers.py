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
_METHODS = ("iqr", "mad", "zscore")


def _check_values(values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"values must be a pandas Series, got {type(values).__name__}")


def detect_outliers(values: pd.Series, method: Method = "iqr", threshold: float = 1.5) -> pd.Series:
    """Flag outliers in a series using the IQR, MAD, or z-score rule.

    Parameters
    ----------
    values : pd.Series
        Metric values to check. Null values are never flagged as outliers.
    method : {"iqr", "mad", "zscore"}, default "iqr"
        Detection rule: "iqr" uses the 1.5*IQR (Tukey) fence, "mad" uses the
        median-absolute-deviation modified z-score, "zscore" uses the
        classical mean/std z-score.
    threshold : float, default 1.5
        Rule-specific cutoff: IQR multiplier for "iqr", or the flag threshold
        on the (modified) z-score for "mad"/"zscore".

    Returns
    -------
    pd.Series
        Boolean mask aligned to `values.index`, True where the value is
        flagged as an outlier. All-False if fewer than 4 non-null values or
        the spread statistic (IQR/MAD/std) is 0.

    Raises
    ------
    TypeError
        If `values` is not a pandas Series.
    ValueError
        If `method` is not one of "iqr", "mad", "zscore", or `threshold` is
        not positive.
    """
    _check_values(values)
    if method not in _METHODS:
        raise ValueError(f"unknown method: {method!r}, expected one of {_METHODS}")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError(f"threshold must be a positive number, got {threshold!r}")

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
    else:  # zscore
        mean, std = non_null.mean(), non_null.std(ddof=1)
        if std == 0:
            return pd.Series(False, index=values.index)
        mask = ((values - mean) / std).abs() > threshold

    return mask.fillna(False)


@dataclass
class OutlierTreatmentResult:
    treated: pd.Series
    n_affected: int
    share_affected: float
    method: str


def winsorize(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> OutlierTreatmentResult:
    """Clip values to the `[lower_q, upper_q]` empirical quantile range.

    Parameters
    ----------
    values : pd.Series
        Metric values to winsorize.
    lower_q : float, default 0.01
        Lower quantile (in `[0, 1)`) used as the clip floor.
    upper_q : float, default 0.99
        Upper quantile (in `(0, 1]`), must be greater than `lower_q`, used
        as the clip ceiling.

    Returns
    -------
    OutlierTreatmentResult
        `treated` series plus count/share of values that were clipped.

    Raises
    ------
    TypeError
        If `values` is not a pandas Series.
    ValueError
        If `lower_q`/`upper_q` is outside `[0, 1]` or `lower_q >= upper_q`.
    """
    _check_values(values)
    if not (0 <= lower_q < 1):
        raise ValueError(f"lower_q must be in [0, 1), got {lower_q!r}")
    if not (0 < upper_q <= 1):
        raise ValueError(f"upper_q must be in (0, 1], got {upper_q!r}")
    if lower_q >= upper_q:
        raise ValueError(f"lower_q ({lower_q!r}) must be less than upper_q ({upper_q!r})")

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
    """Drop flagged rows entirely. `mask` typically comes from `detect_outliers()`.

    Parameters
    ----------
    values : pd.Series
        Metric values to trim.
    mask : pd.Series
        Boolean mask aligned to `values`, True for rows to drop.

    Returns
    -------
    OutlierTreatmentResult
        `treated` series (with flagged rows removed) plus count/share dropped.

    Raises
    ------
    TypeError
        If `values` or `mask` is not a pandas Series, or `mask` is not boolean.
    ValueError
        If `mask` is not aligned (same index) with `values`.
    """
    _check_values(values)
    if not isinstance(mask, pd.Series):
        raise TypeError(f"mask must be a pandas Series, got {type(mask).__name__}")
    if mask.dtype != bool:
        raise TypeError(f"mask must be a boolean Series, got dtype {mask.dtype}")
    if not values.index.equals(mask.index):
        raise ValueError("mask must be aligned to the same index as values")

    treated = values[~mask]
    n_affected = int(mask.sum())
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=n_affected,
        share_affected=n_affected / len(values) if len(values) else 0.0,
        method="trim",
    )


def cap(values: pd.Series, lower: float | None = None, upper: float | None = None) -> OutlierTreatmentResult:
    """Clip values to fixed `[lower, upper]` bounds (as opposed to quantiles).

    Parameters
    ----------
    values : pd.Series
        Metric values to cap.
    lower : float or None, optional
        Lower bound; values below it are clipped up to it. None means no
        lower bound. Default is None.
    upper : float or None, optional
        Upper bound; values above it are clipped down to it. None means no
        upper bound. Default is None.

    Returns
    -------
    OutlierTreatmentResult
        `treated` series plus count/share of values that were clipped.

    Raises
    ------
    TypeError
        If `values` is not a pandas Series, or `lower`/`upper` is neither a
        number nor None.
    ValueError
        If both `lower` and `upper` are given and `lower > upper`.
    """
    _check_values(values)
    for name, bound in (("lower", lower), ("upper", upper)):
        if bound is not None and not isinstance(bound, (int, float)):
            raise TypeError(f"{name} must be a number or None, got {type(bound).__name__}")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError(f"lower ({lower!r}) must not exceed upper ({upper!r})")

    treated = values.clip(lower=lower, upper=upper)
    n_affected = int((treated != values).sum())
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=n_affected,
        share_affected=n_affected / len(values) if len(values) else 0.0,
        method="cap",
    )


def log_transform(values: pd.Series, offset: float = 1.0) -> OutlierTreatmentResult:
    """Apply a `log(values + offset)` transform; requires `values + offset > 0`.

    Parameters
    ----------
    values : pd.Series
        Metric values to transform.
    offset : float, default 1.0
        Constant added before taking the log (log1p-style shift).

    Returns
    -------
    OutlierTreatmentResult
        `treated` series with all entries transformed (`n_affected` equals
        `len(values)`, `share_affected` is 1.0).

    Raises
    ------
    TypeError
        If `values` is not a pandas Series, or `offset` is not a number.
    ValueError
        If any `values + offset` is <= 0.
    """
    _check_values(values)
    if not isinstance(offset, (int, float)):
        raise TypeError(f"offset must be a number, got {type(offset).__name__}")
    if (values + offset <= 0).any():
        raise ValueError("log_transform requires values + offset > 0 for all entries")
    treated = np.log(values + offset)
    return OutlierTreatmentResult(
        treated=treated,
        n_affected=len(values),
        share_affected=1.0,
        method="log_transform",
    )
