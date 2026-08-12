"""Plotly-based figures for experiment data.

Requires the optional extra: `pip install 'abex[viz]'`. Imports here are lazy —
importing `abex.viz` on an install without plotly is fine; only calling a plot
function raises.
"""

from abex.viz.distributions import plot_distribution
from abex.viz.intervals import plot_confidence_interval
from abex.viz.outliers import plot_outlier_treatment
from abex.viz.timeseries import plot_timeseries

__all__ = [
    "plot_confidence_interval",
    "plot_distribution",
    "plot_outlier_treatment",
    "plot_timeseries",
]
