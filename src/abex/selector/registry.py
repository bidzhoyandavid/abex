"""Declarative metadata for implemented stats/design functions.

Only functions that are actually implemented are registered — stubs in
stats/bayesian.py, sequential.py, cuped.py, ratio.py stay out until they
have real bodies, so the selector never recommends a NotImplementedError.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from abex.stats import bootstrap, frequentist

MetricKind = Literal["binary", "count", "continuous"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: Callable
    applicable_kinds: tuple[MetricKind, ...]
    min_group_size: int
    min_n_groups: int = 2
    max_n_groups: int | None = None  # None = unbounded
    requires_paired: bool = False
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    returns: str = ""


REGISTRY: dict[str, MethodSpec] = {
    "t_test": MethodSpec(
        name="t_test",
        fn=frequentist.t_test,
        applicable_kinds=("continuous", "count"),
        min_group_size=30,
        max_n_groups=2,
        assumptions=("approx_normal_or_large_n", "no_heavy_outliers"),
        returns="statistic, p_value",
    ),
    "mann_whitney": MethodSpec(
        name="mann_whitney",
        fn=frequentist.mann_whitney,
        applicable_kinds=("continuous", "count"),
        min_group_size=5,
        max_n_groups=2,
        assumptions=("independent_samples",),
        returns="statistic, p_value",
    ),
    "wilcoxon_signed_rank": MethodSpec(
        name="wilcoxon_signed_rank",
        fn=frequentist.wilcoxon_signed_rank,
        applicable_kinds=("continuous", "count"),
        min_group_size=5,
        max_n_groups=2,
        requires_paired=True,
        assumptions=("paired_samples",),
        returns="statistic, p_value",
    ),
    "z_test_proportions": MethodSpec(
        name="z_test_proportions",
        fn=frequentist.z_test_proportions,
        applicable_kinds=("binary",),
        min_group_size=30,
        max_n_groups=2,
        assumptions=("np_and_n(1-p)_at_least_5",),
        returns="statistic, p_value",
    ),
    "kruskal_wallis": MethodSpec(
        name="kruskal_wallis",
        fn=frequentist.kruskal_wallis,
        applicable_kinds=("continuous", "count"),
        min_group_size=5,
        min_n_groups=3,
        assumptions=("independent_samples",),
        returns="statistic, p_value (omnibus, needs post-hoc)",
    ),
    "anova_oneway": MethodSpec(
        name="anova_oneway",
        fn=frequentist.anova_oneway,
        applicable_kinds=("continuous",),
        min_group_size=30,
        min_n_groups=3,
        assumptions=("approx_normal", "homogeneity_of_variance"),
        returns="statistic, p_value (omnibus, needs post-hoc)",
    ),
    "bootstrap_ci": MethodSpec(
        name="bootstrap_ci",
        fn=bootstrap.bootstrap_ci,
        applicable_kinds=("continuous", "count", "binary"),
        min_group_size=10,
        max_n_groups=2,
        assumptions=("independent_samples",),
        returns="point_estimate, ci_low, ci_high",
    ),
    "permutation_test": MethodSpec(
        name="permutation_test",
        fn=bootstrap.permutation_test,
        applicable_kinds=("continuous", "count", "binary"),
        min_group_size=10,
        max_n_groups=2,
        assumptions=("independent_samples",),
        returns="observed_statistic, p_value",
    ),
}
