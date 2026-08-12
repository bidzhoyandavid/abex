"""Metric dynamics over the life of the experiment.

Useful for exactly the things a single aggregate hides: a novelty spike in the
first days, a broken allocation appearing mid-test, or a metric that has not
stabilised yet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from abex.viz._common import apply_layout, check_columns, check_frame, group_color, require_plotly

VALID_FREQUENCIES = ("H", "D", "W")


def plot_timeseries(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    timestamp_col: str,
    freq: str = "D",
    agg: str = "mean",
) -> Any:
    """Per-group metric aggregated into time buckets.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data, one row per observation.
    metric_col : str
        Numeric column to aggregate. Must be a column of `df`.
    group_col : str
        Column holding the group label. Must be a column of `df`.
    timestamp_col : str
        Column holding the observation time. Must be a column of `df` and must
        parse as datetime.
    freq : {"H", "D", "W"}, default "D"
        Resampling bucket: hourly, daily or weekly.
    agg : str, default "mean"
        Aggregation applied per bucket, e.g. `"mean"`, `"sum"`, `"count"`.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError
        If `df` is not a DataFrame or the column names are not strings.
    ValueError
        If a column is missing, `freq` is unsupported, the metric is not
        numeric, or no timestamp could be parsed.
    ImportError
        If plotly is not installed (`pip install 'abex[viz]'`).
    """
    go = require_plotly()
    check_frame("df", df)
    check_columns(df, metric_col=metric_col, group_col=group_col, timestamp_col=timestamp_col)
    if freq not in VALID_FREQUENCIES:
        raise ValueError(f"freq must be one of {VALID_FREQUENCIES}, got {freq!r}")
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        raise ValueError(f"metric_col={metric_col!r} must be numeric")

    frame = df[[timestamp_col, group_col, metric_col]].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], errors="coerce")
    frame = frame.dropna(subset=[timestamp_col])
    if frame.empty:
        raise ValueError(f"timestamp_col={timestamp_col!r} has no parseable timestamps")

    figure = go.Figure()
    for index, (name, sub) in enumerate(frame.groupby(group_col)):
        series = sub.set_index(timestamp_col)[metric_col].resample(freq).agg(agg).dropna()
        figure.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(),
                mode="lines+markers",
                name=str(name),
                line={"color": group_color(index)},
            )
        )

    return apply_layout(
        figure,
        f"{metric_col} по времени ({agg})",
        x_title="Время",
        y_title=metric_col,
    )
