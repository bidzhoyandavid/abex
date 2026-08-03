"""Single agent-facing entrypoint: profile in, ranked method recommendations out."""

from __future__ import annotations

from dataclasses import dataclass

from abex.data.profiling import MetricProfile
from abex.selector.rules import Candidate, candidates_for


@dataclass
class Recommendation:
    method_name: str
    rank: int
    warnings: tuple[str, ...]
    violated_assumptions: tuple[str, ...]
    fn_path: str


def recommend_test(profile: MetricProfile, paired: bool = False) -> list[Recommendation]:
    """Returns candidates ranked best-first. Empty list means no registered
    method fits this profile — most likely need one of the stubbed methods
    (bayesian/sequential/cuped/ratio) that aren't wired into the registry yet.
    """
    candidates: list[Candidate] = candidates_for(profile, paired=paired)

    return [
        Recommendation(
            method_name=c.method.name,
            rank=i,
            warnings=c.warnings,
            violated_assumptions=c.violated_assumptions,
            fn_path=f"{c.method.fn.__module__}.{c.method.fn.__name__}",
        )
        for i, c in enumerate(candidates)
    ]
