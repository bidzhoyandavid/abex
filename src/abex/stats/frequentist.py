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


def _check_series(name: str, values: pd.Series) -> None:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(values).__name__}")


def t_test(control: pd.Series, treatment: pd.Series, equal_var: bool = False) -> TestResult:
    """Two-sample t-test on means. Welch's t-test by default (equal_var=False)
    — safer under unequal variances.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before testing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before testing.
    equal_var : bool, default False
        If True, run Student's t-test assuming equal population variances.
        If False, run Welch's t-test.

    Returns
    -------
    TestResult
        `method` ("welch_t_test" or "student_t_test"), `statistic`, `p_value`,
        and per-group sample sizes.

    Raises
    ------
    TypeError
        If `control`/`treatment` is not a pandas Series, or `equal_var` is
        not a bool.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
    if not isinstance(equal_var, bool):
        raise TypeError(f"equal_var must be a bool, got {type(equal_var).__name__}")

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
    """Two-proportion z-test, for binary metrics (e.g. conversion).

    Parameters
    ----------
    control_successes : int
        Number of successes (events) in the control group. Must be in
        `[0, control_n]`.
    control_n : int
        Control-group sample size. Must be positive.
    treatment_successes : int
        Number of successes (events) in the treatment group. Must be in
        `[0, treatment_n]`.
    treatment_n : int
        Treatment-group sample size. Must be positive.

    Returns
    -------
    TestResult
        `statistic` (z-score), `p_value` (two-sided), and per-group sample sizes.

    Raises
    ------
    TypeError
        If any argument is not an int.
    ValueError
        If `control_n`/`treatment_n` is not positive, or `control_successes`/
        `treatment_successes` is outside `[0, n]`.
    """
    for name, val in (
        ("control_successes", control_successes),
        ("control_n", control_n),
        ("treatment_successes", treatment_successes),
        ("treatment_n", treatment_n),
    ):
        if not isinstance(val, (int, np.integer)) or isinstance(val, bool):
            raise TypeError(f"{name} must be an int, got {type(val).__name__}")
    if control_n <= 0:
        raise ValueError("control_n must be positive")
    if treatment_n <= 0:
        raise ValueError("treatment_n must be positive")
    if not (0 <= control_successes <= control_n):
        raise ValueError("control_successes must be in [0, control_n]")
    if not (0 <= treatment_successes <= treatment_n):
        raise ValueError("treatment_successes must be in [0, treatment_n]")

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
    """Chi-square test of independence for categorical outcomes across groups.

    Parameters
    ----------
    contingency_table : pd.DataFrame or np.ndarray
        2D contingency table of counts (groups x categories).

    Returns
    -------
    TestResult
        `statistic`, `p_value`, and row sums of the first two rows as
        `n_control`/`n_treatment` (0 if the table has fewer than 2 rows/dims).

    Raises
    ------
    TypeError
        If `contingency_table` is not a pandas DataFrame or numpy ndarray.
    ValueError
        If `contingency_table` is not 2-dimensional.
    """
    if not isinstance(contingency_table, (pd.DataFrame, np.ndarray)):
        raise TypeError(
            f"contingency_table must be a pandas DataFrame or numpy ndarray, "
            f"got {type(contingency_table).__name__}"
        )
    table = contingency_table.values if isinstance(contingency_table, pd.DataFrame) else contingency_table
    if table.ndim != 2:
        raise ValueError(f"contingency_table must be 2-dimensional, got {table.ndim} dimensions")

    stat, p, _, _ = sps.chi2_contingency(table)
    return TestResult(
        method="chi2_test",
        statistic=float(stat),
        p_value=float(p),
        n_control=int(table[0].sum()) if table.ndim == 2 else 0,
        n_treatment=int(table[1].sum()) if table.ndim == 2 and table.shape[0] > 1 else 0,
    )


def mann_whitney(control: pd.Series, treatment: pd.Series) -> TestResult:
    """Non-parametric alternative to the t-test — use for skewed / heavy-tailed metrics.

    Parameters
    ----------
    control : pd.Series
        Control-group observations. Null values are dropped before testing.
    treatment : pd.Series
        Treatment-group observations. Null values are dropped before testing.

    Returns
    -------
    TestResult
        `method` ("mann_whitney_u"), `statistic`, `p_value` (two-sided), and
        per-group sample sizes.

    Raises
    ------
    TypeError
        If `control` or `treatment` is not a pandas Series.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
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
    """Paired non-parametric test — control/treatment must be aligned (same
    index, e.g. pre/post).

    Parameters
    ----------
    control : pd.Series
        "Before"/control observations, aligned to `treatment` by index.
    treatment : pd.Series
        "After"/treatment observations, aligned to `control` by index.

    Returns
    -------
    TestResult
        `method` ("wilcoxon_signed_rank"), `statistic`, `p_value`, and the
        number of complete pairs used for both `n_control`/`n_treatment`.

    Raises
    ------
    TypeError
        If `control` or `treatment` is not a pandas Series.
    """
    _check_series("control", control)
    _check_series("treatment", treatment)
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
    """Non-parametric A/B/n omnibus test. Follow up with a post-hoc +
    multiple_testing correction to identify which pairs differ.

    Parameters
    ----------
    *groups : pd.Series
        Three or more group series to compare. Null values are dropped from
        each before testing.

    Returns
    -------
    TestResult
        `method` ("kruskal_wallis"), `statistic`, `p_value` (omnibus),
        `n_control` (total non-null observations across groups), and
        `n_treatment` (number of groups).

    Raises
    ------
    TypeError
        If any element of `groups` is not a pandas Series.
    ValueError
        If fewer than 3 groups are given.
    """
    if len(groups) < 3:
        raise ValueError("kruskal_wallis expects 3+ groups; use mann_whitney for 2 groups")
    for i, g in enumerate(groups):
        _check_series(f"groups[{i}]", g)
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
    """Parametric A/B/n omnibus test. Assumes normality + homogeneity of variance.

    Parameters
    ----------
    *groups : pd.Series
        Three or more group series to compare. Null values are dropped from
        each before testing.

    Returns
    -------
    TestResult
        `method` ("anova_oneway"), `statistic` (F-statistic), `p_value`
        (omnibus), `n_control` (total non-null observations across groups),
        and `n_treatment` (number of groups).

    Raises
    ------
    TypeError
        If any element of `groups` is not a pandas Series.
    ValueError
        If fewer than 3 groups are given.
    """
    if len(groups) < 3:
        raise ValueError("anova_oneway expects 3+ groups; use t_test for 2 groups")
    for i, g in enumerate(groups):
        _check_series(f"groups[{i}]", g)
    clean_groups = [g.dropna() for g in groups]
    stat, p = sps.f_oneway(*clean_groups)
    return TestResult(
        method="anova_oneway",
        statistic=float(stat),
        p_value=float(p),
        n_control=sum(len(g) for g in clean_groups),
        n_treatment=len(clean_groups),
    )
