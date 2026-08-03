"""Deterministic profile -> candidate methods matching.

No ML, no scoring black box. Each rule is a plain, auditable check against
MetricProfile + MethodSpec metadata from registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from abex.data.profiling import MetricProfile
from abex.selector.registry import REGISTRY, MethodSpec

SKEW_THRESHOLD = 1.0
OUTLIER_SHARE_THRESHOLD = 0.02


@dataclass
class Candidate:
    method: MethodSpec
    warnings: tuple[str, ...]
    violated_assumptions: tuple[str, ...]


def _group_count_ok(profile: MetricProfile, spec: MethodSpec) -> bool:
    """Check whether a profile's group count satisfies a method's requirements.

    Parameters
    ----------
    profile : MetricProfile
        Profile whose `n_groups` is checked.
    spec : MethodSpec
        Method spec providing `min_n_groups`/`max_n_groups` bounds.

    Returns
    -------
    bool
        True if `spec.min_n_groups <= profile.n_groups <= spec.max_n_groups`
        (no upper bound if `spec.max_n_groups` is None).
    """
    if profile.n_groups < spec.min_n_groups:
        return False
    if spec.max_n_groups is not None and profile.n_groups > spec.max_n_groups:
        return False
    return True


def candidates_for(profile: MetricProfile, paired: bool = False) -> list[Candidate]:
    """Return every method whose hard requirements (kind, group count, sample
    size, pairing) are satisfied, each annotated with soft warnings.

    Parameters
    ----------
    profile : MetricProfile
        Metric profile produced by `abex.data.profiling.profile_metric`.
    paired : bool, default False
        Whether the data is paired/matched (e.g. pre/post on the same units).
        Only methods with a matching `MethodSpec.requires_paired` are returned.

    Returns
    -------
    list[Candidate]
        Candidates sorted best-first: fewer violated assumptions first, then
        fewer soft warnings. Empty if no registered method fits.

    Raises
    ------
    TypeError
        If `profile` is not a `MetricProfile`, or `paired` is not a bool.
    """
    if not isinstance(profile, MetricProfile):
        raise TypeError(f"profile must be a MetricProfile, got {type(profile).__name__}")
    if not isinstance(paired, bool):
        raise TypeError(f"paired must be a bool, got {type(paired).__name__}")

    out: list[Candidate] = []

    for spec in REGISTRY.values():
        if profile.kind not in spec.applicable_kinds:
            continue
        if not _group_count_ok(profile, spec):
            continue
        if profile.min_group_size < spec.min_group_size:
            continue
        if spec.requires_paired != paired:
            continue

        warnings: list[str] = []
        violated: list[str] = []

        if "approx_normal_or_large_n" in spec.assumptions or "approx_normal" in spec.assumptions:
            if abs(profile.skewness) > SKEW_THRESHOLD and profile.min_group_size < 100:
                violated.append("approx_normal")
                warnings.append(
                    f"skewness={profile.skewness:.2f} exceeds {SKEW_THRESHOLD} at small n "
                    "— prefer mann_whitney/bootstrap over a parametric test"
                )

        if "no_heavy_outliers" in spec.assumptions and profile.outlier_share > OUTLIER_SHARE_THRESHOLD:
            violated.append("no_heavy_outliers")
            warnings.append(
                f"outlier_share={profile.outlier_share:.2%} exceeds {OUTLIER_SHARE_THRESHOLD:.0%} "
                "— treat outliers first or prefer a robust/non-parametric method"
            )

        out.append(Candidate(method=spec, warnings=tuple(warnings), violated_assumptions=tuple(violated)))

    # fewer violated assumptions first, then prefer non-omnibus (higher-power) tests
    out.sort(key=lambda c: (len(c.violated_assumptions), len(c.warnings)))
    return out
