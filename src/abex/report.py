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
    """
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
