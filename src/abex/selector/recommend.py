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
    """Rank registered test methods for a given metric profile.

    Returns candidates ranked best-first. Empty list means no registered
    method fits this profile — most likely need one of the stubbed methods
    (bayesian/sequential/cuped/ratio) that aren't wired into the registry yet.

    Parameters
    ----------
    profile : MetricProfile
        Metric profile produced by `abex.data.profiling.profile_metric`.
    paired : bool, default False
        Whether the data is paired/matched (e.g. pre/post on the same units).

    Returns
    -------
    list[Recommendation]
        Best-first ranked recommendations, each with the method name, rank,
        warnings, violated assumptions, and importable `fn_path`.

    Raises
    ------
    TypeError
        If `profile` is not a `MetricProfile`, or `paired` is not a bool.
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
