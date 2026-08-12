"""Shared helpers for the viz layer.

plotly is an optional dependency (`abex[viz]`), so it is imported lazily —
importing `abex.viz` must never break an install that only needs the stats.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

CONTROL_COLOR = "#94A3B8"
TREATMENT_COLOR = "#6366F1"
PALETTE = ("#6366F1", "#94A3B8", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6")

_LAYOUT = {
    "template": "plotly_white",
    # Легенда снизу: сверху она сталкивается с заголовком, и оба становятся
    # нечитаемыми на узких контейнерах.
    "margin": {"l": 64, "r": 24, "t": 64, "b": 64},
    "font": {"size": 13},
    "legend": {"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
}
_TITLE = {"x": 0, "xanchor": "left", "font": {"size": 14}}


def require_plotly() -> Any:
    """Return `plotly.graph_objects`, with an actionable error if missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "plotly is required for abex.viz — install the extra: pip install 'abex[viz]'"
        ) from exc
    return go


def check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")


def check_frame(name: str, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")


def check_columns(df: pd.DataFrame, **columns: str) -> None:
    for role, column in columns.items():
        if not isinstance(column, str):
            raise TypeError(f"{role} must be a str, got {type(column).__name__}")
        if column not in df.columns:
            raise ValueError(f"{role}={column!r} is not a column of the dataframe")


def group_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def apply_layout(figure: Any, title: str, x_title: str = "", y_title: str = "") -> Any:
    # title собирается отдельно: если положить его в _LAYOUT, распаковка
    # столкнётся с явным аргументом и update_layout упадёт на дубликате.
    figure.update_layout(
        title={**_TITLE, "text": title}, xaxis_title=x_title, yaxis_title=y_title, **_LAYOUT
    )
    return figure
