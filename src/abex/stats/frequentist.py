"""Classical frequentist tests. Every function is pure: takes arrays/series,
returns a TestResult, no I/O, no mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class TestResult:
    method: str
    statistic: float
    p_value: float
    n_control: int
    n_treatment: int


def t_test(control: pd.Series, treatment: pd.Series, equal_var: bool = False) -> TestResult:
    """Welch's t-test by default (equal_var=False) — safer under unequal variances."""
    control, treatment = control.dropna(), treatment.dropna()
    stat, p = sps.ttest_ind(control, treatment, equal_var=equal_var)
    return TestResult(
        method="welch_t_test" if not equal_var else "student_t_test",
        statistic=float(stat),
        p_value=float(p),
        n_control=len(control),
        n_treatment=len(treatment),
    )


def z_test_proportions(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
) -> TestResult:
    """Two-proportion z-test, for binary metrics (e.g. conversion)."""
    p1 = control_successes / control_n
    p2 = treatment_successes / treatment_n
    p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)
    se = (p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n)) ** 0.5
    if se == 0:
        stat, p_value = 0.0, 1.0
    else:
        stat = (p2 - p1) / se
        p_value = 2 * (1 - sps.norm.cdf(abs(stat)))
    return TestResult(
        method="z_test_proportions",
        statistic=float(stat),
        p_value=float(p_value),
        n_control=control_n,
        n_treatment=treatment_n,
    )


def chi2_test(contingency_table: pd.DataFrame | np.ndarray) -> TestResult:
    """Chi-square test of independence for categorical outcomes across groups."""
    table = contingency_table.values if isinstance(contingency_table, pd.DataFrame) else contingency_table
    stat, p, _, _ = sps.chi2_contingency(table)
    return TestResult(
        method="chi2_test",
        statistic=float(stat),
        p_value=float(p),
        n_control=int(table[0].sum()) if table.ndim == 2 else 0,
        n_treatment=int(table[1].sum()) if table.ndim == 2 and table.shape[0] > 1 else 0,
    )


def mann_whitney(control: pd.Series, treatment: pd.Series) -> TestResult:
    """Non-parametric alternative to the t-test — use for skewed / heavy-tailed metrics."""
    control, treatment = control.dropna(), treatment.dropna()
    stat, p = sps.mannwhitneyu(control, treatment, alternative="two-sided")
    return TestResult(
        method="mann_whitney_u",
        statistic=float(stat),
        p_value=float(p),
        n_control=len(control),
        n_treatment=len(treatment),
    )


def wilcoxon_signed_rank(control: pd.Series, treatment: pd.Series) -> TestResult:
    """Paired non-parametric test — control/treatment must be aligned (same index, e.g. pre/post)."""
    paired = pd.DataFrame({"control": control, "treatment": treatment}).dropna()
    stat, p = sps.wilcoxon(paired["control"], paired["treatment"])
    return TestResult(
        method="wilcoxon_signed_rank",
        statistic=float(stat),
        p_value=float(p),
        n_control=len(paired),
        n_treatment=len(paired),
    )


def kruskal_wallis(*groups: pd.Series) -> TestResult:
    """Non-parametric A/B/n omnibus test. Follow up with a post-hoc + multiple_testing
    correction to identify which pairs differ.
    """
    if len(groups) < 3:
        raise ValueError("kruskal_wallis expects 3+ groups; use mann_whitney for 2 groups")
    clean_groups = [g.dropna() for g in groups]
    stat, p = sps.kruskal(*clean_groups)
    return TestResult(
        method="kruskal_wallis",
        statistic=float(stat),
        p_value=float(p),
        n_control=sum(len(g) for g in clean_groups),
        n_treatment=len(clean_groups),
    )


def anova_oneway(*groups: pd.Series) -> TestResult:
    """Parametric A/B/n omnibus test. Assumes normality + homogeneity of variance."""
    if len(groups) < 3:
        raise ValueError("anova_oneway expects 3+ groups; use t_test for 2 groups")
    clean_groups = [g.dropna() for g in groups]
    stat, p = sps.f_oneway(*clean_groups)
    return TestResult(
        method="anova_oneway",
        statistic=float(stat),
        p_value=float(p),
        n_control=sum(len(g) for g in clean_groups),
        n_treatment=len(clean_groups),
    )
