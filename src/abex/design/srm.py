"""Sample ratio mismatch check — did users actually land in groups at the
expected allocation ratio?
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats as sps


@dataclass
class SRMResult:
    observed_counts: dict[str, int]
    expected_ratios: dict[str, float]
    chi2_stat: float
    p_value: float
    has_srm: bool


def check_srm(
    group_counts: dict[str, int] | pd.Series,
    expected_ratios: dict[str, float] | None = None,
    alpha: float = 0.001,
) -> SRMResult:
    """Chi-square goodness-of-fit test for sample ratio mismatch.

    SRM is checked at a strict alpha (default 0.001) — false positives here
    invalidate the whole experiment, so the bar is higher than for the metric test.

    Parameters
    ----------
    group_counts : dict[str, int] or pd.Series
        Observed unit counts per group.
    expected_ratios : dict[str, float] or None, optional
        Expected allocation ratio per group; keys must match `group_counts`
        and values must sum to 1. None assumes an equal split across all
        groups. Default is None.
    alpha : float, default 0.001
        Significance level for flagging SRM. Must be in `(0, 1)`.

    Returns
    -------
    SRMResult
        Observed/expected counts, chi-square statistic, p-value, and
        `has_srm` (True if `p_value < alpha`).

    Raises
    ------
    TypeError
        If `group_counts` is not a dict or pandas Series, `expected_ratios`
        is not a dict or None, or `alpha` is not a number.
    ValueError
        If fewer than 2 groups are given, `expected_ratios` doesn't sum to 1,
        its keys don't match `group_counts`, or `alpha` is not in `(0, 1)`.
    """
    if not isinstance(group_counts, (dict, pd.Series)):
        raise TypeError(f"group_counts must be a dict or pandas Series, got {type(group_counts).__name__}")
    if expected_ratios is not None and not isinstance(expected_ratios, dict):
        raise TypeError(f"expected_ratios must be a dict or None, got {type(expected_ratios).__name__}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    if isinstance(group_counts, pd.Series):
        group_counts = {str(k): int(v) for k, v in group_counts.to_dict().items()}

    groups = list(group_counts.keys())
    if len(groups) < 2:
        raise ValueError("need at least 2 groups to check SRM")

    if expected_ratios is None:
        expected_ratios = {g: 1 / len(groups) for g in groups}
    else:
        total = sum(expected_ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"expected_ratios must sum to 1, got {total}")
        if set(expected_ratios.keys()) != set(groups):
            raise ValueError("expected_ratios keys must match group_counts keys")

    observed = [group_counts[g] for g in groups]
    total_n = sum(observed)
    expected = [expected_ratios[g] * total_n for g in groups]

    chi2_stat, p_value = sps.chisquare(f_obs=observed, f_exp=expected)

    return SRMResult(
        observed_counts=group_counts,
        expected_ratios=expected_ratios,
        chi2_stat=float(chi2_stat),
        p_value=float(p_value),
        has_srm=p_value < alpha,
    )
