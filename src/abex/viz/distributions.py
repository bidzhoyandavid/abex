"""Distribution of a metric across experiment groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as sps

from abex.viz._common import apply_layout, check_columns, check_frame, group_color, require_plotly

MAX_GROUPS = 12
CURVE_POINTS = 256
# Хвосты за этим квантилем растягивают ось и сплющивают основную массу.
TAIL_QUANTILE = 0.995
MIN_KDE_N = 5


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _density_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE, or None when the sample cannot support one (too few
    points or zero variance — a constant column has no density to draw)."""
    if len(values) < MIN_KDE_N or float(np.std(values)) == 0.0:
        return None
    try:
        return sps.gaussian_kde(values)(grid)
    except (np.linalg.LinAlgError, ValueError):
        return None


def plot_distribution(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    show_mean: bool = True,
    show_histogram: bool = False,
    n_bins: int = 40,
) -> Any:
    """Smoothed per-group density of `metric_col`, with optional mean lines.

    A KDE curve rather than raw bars: the question this chart answers is
    whether the group distributions differ in shape, and bin edges add visual
    noise that has nothing to do with the data. The histogram stays available
    behind `show_histogram` for a sanity check against binning artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment data, one row per observation.
    metric_col : str
        Numeric column to plot. Must be a column of `df`.
    group_col : str
        Column holding the group label. Must be a column of `df`.
    show_mean : bool, default True
        Draw a dashed vertical line at each group's mean.
    show_histogram : bool, default False
        Also draw the underlying histogram, faintly, behind the curves.
    n_bins : int, default 40
        Bins for that histogram. Must be positive.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError
        If `df` is not a DataFrame, the column names are not strings, or
        `n_bins` is not an int.
    ValueError
        If a column is missing, `n_bins` is not positive, the metric is not
        numeric, or the data has no groups / too many to plot.
    ImportError
        If plotly is not installed (`pip install 'abex[viz]'`).
    """
    go = require_plotly()
    check_frame("df", df)
    check_columns(df, metric_col=metric_col, group_col=group_col)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if not pd.api.types.is_numeric_dtype(df[metric_col]):
        raise ValueError(f"metric_col={metric_col!r} must be numeric")

    groups = list(df.groupby(group_col))
    if not groups:
        raise ValueError("no groups to plot")
    if len(groups) > MAX_GROUPS:
        raise ValueError(f"too many groups to plot: {len(groups)} > {MAX_GROUPS}")

    all_values = df[metric_col].dropna()
    if all_values.empty:
        raise ValueError(f"metric_col={metric_col!r} has no non-null values")

    low = float(all_values.quantile(1 - TAIL_QUANTILE))
    high = float(all_values.quantile(TAIL_QUANTILE))
    if high <= low:
        low, high = float(all_values.min()), float(all_values.max())
    if high <= low:
        high = low + 1.0
    grid = np.linspace(low, high, CURVE_POINTS)

    figure = go.Figure()
    for index, (name, sub) in enumerate(groups):
        values = sub[metric_col].dropna().to_numpy()
        color = group_color(index)

        if show_histogram:
            figure.add_trace(
                go.Histogram(
                    x=values,
                    name=f"{name} (гистограмма)",
                    nbinsx=n_bins,
                    histnorm="probability density",
                    opacity=0.25,
                    marker_color=color,
                    showlegend=False,
                )
            )

        density = _density_curve(values, grid)
        if density is None:
            # Нечего сглаживать — честнее показать бары, чем ничего.
            figure.add_trace(
                go.Histogram(
                    x=values,
                    name=str(name),
                    nbinsx=n_bins,
                    histnorm="probability density",
                    opacity=0.6,
                    marker_color=color,
                )
            )
        else:
            figure.add_trace(
                go.Scatter(
                    x=grid,
                    y=density,
                    name=str(name),
                    mode="lines",
                    line={"color": color, "width": 2.5, "shape": "spline"},
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(color, 0.18),
                    hovertemplate=f"{name}<br>%{{x:.4g}}<extra></extra>",
                )
            )

        if show_mean and len(values):
            # Подписи разносятся по вертикали: у близких средних они иначе
            # печатаются друг поверх друга и не читаются вовсе.
            figure.add_vline(
                x=float(np.mean(values)),
                line_dash="dash",
                line_color=color,
                line_width=1.5,
                annotation_text=f"среднее {name}",
                annotation_position="top right",
                annotation_font={"size": 11, "color": color},
                annotation_yshift=-16 * index,
            )

    figure.update_layout(barmode="overlay", hovermode="x unified")
    return apply_layout(
        figure, f"Распределение {metric_col} по группам", x_title=metric_col, y_title="Плотность"
    )
