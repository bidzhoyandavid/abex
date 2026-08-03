"""Assemble a single JSON-serializable summary for one metric's test result.

Schema: metric, method, effect, ci, p_value, decision, agreement, warnings.
This is the contract an agent parses — human-readable fields only add on top,
never replace these keys.
"""

from __future__ import annotations

from typing import Any

from abex.analysis.effect_size import EffectSizeResult


def build_report(
    metric: str,
    method: str,
    p_value: float | None,
    effect: EffectSizeResult,
    alpha: float = 0.05,
    ci: tuple[float, float] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble a single JSON-serializable summary for one metric's test result.

    Parameters
    ----------
    metric : str
        Name of the metric being reported.
    method : str
        Name of the statistical method used (e.g. "welch_t_test").
    p_value : float or None
        Two-sided p-value from the test, or None if not applicable (e.g. a
        pure bootstrap CI with no test).
    effect : EffectSizeResult
        Effect size summary from `abex.analysis.effect_size.effect_size_summary`.
    alpha : float, default 0.05
        Significance threshold used to derive `decision`. Must be in `(0, 1)`.
    ci : tuple[float, float] or None, optional
        `(low, high)` confidence interval bounds. Default is None.
    warnings : list[str] or None, optional
        Caveats to surface alongside the result. Default is None (empty list).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dict with keys `metric`, `method`, `effect`, `ci`,
        `p_value`, `alpha`, `decision`, `agreement`, `warnings`.

    Raises
    ------
    TypeError
        If `metric`/`method` is not a str, `effect` is not an
        `EffectSizeResult`, `alpha` is not a number, `ci` is not a 2-tuple
        or None, `p_value` is neither a number nor None, or `warnings` is
        neither a list nor None.
    ValueError
        If `alpha` is not in `(0, 1)`.
    """
    if not isinstance(metric, str):
        raise TypeError(f"metric must be a str, got {type(metric).__name__}")
    if not isinstance(method, str):
        raise TypeError(f"method must be a str, got {type(method).__name__}")
    if not isinstance(effect, EffectSizeResult):
        raise TypeError(f"effect must be an EffectSizeResult, got {type(effect).__name__}")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise TypeError(f"alpha must be a number, got {type(alpha).__name__}")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if p_value is not None and (not isinstance(p_value, (int, float)) or isinstance(p_value, bool)):
        raise TypeError(f"p_value must be a number or None, got {type(p_value).__name__}")
    if ci is not None and (not isinstance(ci, tuple) or len(ci) != 2):
        raise TypeError("ci must be a (low, high) tuple or None")
    if warnings is not None and not isinstance(warnings, list):
        raise TypeError(f"warnings must be a list or None, got {type(warnings).__name__}")

    warnings = list(warnings or [])
    decision = None if p_value is None else ("significant" if p_value < alpha else "not_significant")

    return {
        "metric": metric,
        "method": method,
        "effect": {
            "absolute_diff": effect.absolute_diff,
            "relative_lift": effect.relative_lift,
            "cohens_d": effect.cohens_d,
        },
        "ci": {"low": ci[0], "high": ci[1]} if ci is not None else None,
        "p_value": p_value,
        "alpha": alpha,
        "decision": decision,
        "agreement": True,
        "warnings": warnings,
    }


def merge_disagreeing_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine reports for the same metric from multiple methods. Flags
    low_confidence when decisions diverge, per the selector disagreement policy.

    Parameters
    ----------
    reports : list[dict[str, Any]]
        Reports for the same metric produced by `build_report`, each
        expected to have `method`, `p_value`, and `decision` keys.

    Returns
    -------
    dict[str, Any]
        Copy of the first report with `agreement` recomputed, an
        `all_methods` summary list added, and (if methods disagree)
        `decision` set to "low_confidence" with an extra warning appended.

    Raises
    ------
    TypeError
        If `reports` is not a list.
    ValueError
        If `reports` is empty.
    """
    if not isinstance(reports, list):
        raise TypeError(f"reports must be a list, got {type(reports).__name__}")
    if not reports:
        raise ValueError("need at least one report to merge")

    decisions = {r["decision"] for r in reports}
    agreement = len(decisions) <= 1

    primary = reports[0]
    merged = dict(primary)
    merged["agreement"] = agreement
    merged["all_methods"] = [{"method": r["method"], "p_value": r["p_value"], "decision": r["decision"]} for r in reports]

    if not agreement:
        merged["decision"] = "low_confidence"
        merged["warnings"] = merged.get("warnings", []) + [
            "methods disagree on significance — see all_methods"
        ]

    return merged
