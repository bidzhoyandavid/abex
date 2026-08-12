"""Confidence intervals for the treatment effect.

A CI plot is the honest counterpart to a p-value: it shows the size of the
effect and the uncertainty around it, and whether the interval crosses zero.
"""

from __future__ import annotations

from typing import Any

from abex.viz._common import apply_layout, require_plotly

NO_EFFECT_COLOR = "#94A3B8"
SIGNIFICANT_COLOR = "#10B981"
INCONCLUSIVE_COLOR = "#6366F1"


def plot_confidence_interval(
    estimates: list[dict],
    title: str = "Эффект и 95% доверительный интервал",
) -> Any:
    """Horizontal CI plot, one row per metric.

    Parameters
    ----------
    estimates : list[dict]
        One entry per metric with keys `metric` (str), `estimate` (float),
        `ci_low` (float) and `ci_high` (float); optional `significant` (bool)
        colours the row. Entries missing an estimate or either bound are
        skipped — a metric whose CI could not be computed has nothing to draw.
    title : str, default "Эффект и 95% доверительный интервал"
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    TypeError
        If `estimates` is not a list of dicts, or `title` is not a str.
    ValueError
        If no entry has a complete estimate and interval.
    ImportError
        If plotly is not installed (`pip install 'abex[viz]'`).
    """
    go = require_plotly()
    if not isinstance(estimates, list):
        raise TypeError(f"estimates must be a list, got {type(estimates).__name__}")
    if not isinstance(title, str):
        raise TypeError(f"title must be a str, got {type(title).__name__}")

    rows = []
    for index, item in enumerate(estimates):
        if not isinstance(item, dict):
            raise TypeError(f"estimates[{index}] must be a dict, got {type(item).__name__}")
        estimate, low, high = item.get("estimate"), item.get("ci_low"), item.get("ci_high")
        if estimate is None or low is None or high is None:
            continue
        rows.append(item)

    if not rows:
        raise ValueError("no estimate has a complete confidence interval")

    figure = go.Figure()
    for item in rows:
        estimate = float(item["estimate"])
        low, high = float(item["ci_low"]), float(item["ci_high"])
        significant = item.get("significant")
        color = (
            SIGNIFICANT_COLOR
            if significant is True
            else NO_EFFECT_COLOR
            if significant is False
            else INCONCLUSIVE_COLOR
        )
        figure.add_trace(
            go.Scatter(
                x=[estimate],
                y=[str(item.get("metric", ""))],
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [high - estimate],
                    "arrayminus": [estimate - low],
                    "color": color,
                },
                mode="markers",
                marker={"size": 10, "color": color},
                name=str(item.get("metric", "")),
                showlegend=False,
                hovertemplate=(f"{item.get('metric', '')}<br>эффект %{{x:.4g}}<br>CI [{low:.4g}; {high:.4g}]<extra></extra>")
            )
        )

    # Zero is the decision line: an interval crossing it means "no proven effect".
    figure.add_vline(x=0, line_dash="dash", line_color="#EF4444")
    figure.update_xaxes(tickformat=".1%")
    apply_layout(figure, title, x_title="Разница относительно контроля", y_title="")

    # Ширину под подписи метрик считает сам plotly: фиксированное поле либо
    # обрезает длинные имена, либо съедает половину узкого контейнера.
    figure.update_yaxes(automargin=True)
    return figure
