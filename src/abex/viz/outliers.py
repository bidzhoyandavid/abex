"""Before/after view of an outlier treatment.

Shows what the chosen treatment actually did to the metric, so the decision
made in the HITL step stays auditable instead of being a black box.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from abex.viz._common import (
    CONTROL_COLOR,
    TREATMENT_COLOR,
    apply_layout,
    check_series,
    require_plotly,
)


def plot_outlier_treatment(
    original: pd.Series,
    treated: pd.Series,
    method: str = "",
    n_bins: int = 40,
) -> Any:
    """Overlaid histograms of the metric before and after treatment.

    Parameters
    ----------
    original : pd.Series
        Metric values before treatment. Nulls are dropped.
    treated : pd.Series
        Metric values after treatment. Nulls are dropped. May be shorter than
        `original` when the treatment trimmed rows.
    method : str, default ""
        Treatment name, shown in the title (e.g. `"winsorize"`).
    n_bins : int, default 40
        Number of histogram bins. Must be positive.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError
        If the inputs are not Series, `method` is not a str, or `n_bins` is
        not an int.
    ValueError
        If `n_bins` is not positive or both series are empty.
    ImportError
        If plotly is not installed (`pip install 'abex[viz]'`).
    """
    go = require_plotly()
    check_series("original", original)
    check_series("treated", treated)
    if not isinstance(method, str):
        raise TypeError(f"method must be a str, got {type(method).__name__}")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool):
        raise TypeError(f"n_bins must be an int, got {type(n_bins).__name__}")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    before, after = original.dropna(), treated.dropna()
    if before.empty and after.empty:
        raise ValueError("both series are empty")

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(x=before, name="до обработки", nbinsx=n_bins, opacity=0.6, marker_color=CONTROL_COLOR)
    )
    figure.add_trace(
        go.Histogram(x=after, name="после обработки", nbinsx=n_bins, opacity=0.6, marker_color=TREATMENT_COLOR)
    )
    figure.update_layout(barmode="overlay")

    suffix = f" ({method})" if method else ""
    return apply_layout(
        figure, f"Обработка выбросов{suffix}", x_title="Значение метрики", y_title="Наблюдений"
    )
