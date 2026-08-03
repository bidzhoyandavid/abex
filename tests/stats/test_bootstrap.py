import numpy as np
import pandas as pd

from abex.stats.bootstrap import bootstrap_ci, permutation_test


def test_bootstrap_ci_excludes_zero_for_known_effect():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 300))
    treatment = pd.Series(rng.normal(0.5, 1, 300))
    result = bootstrap_ci(control, treatment, n_resamples=2000, random_state=42)
    assert result.ci_low > 0


def test_bootstrap_ci_includes_zero_for_null_effect():
    rng = np.random.default_rng(1)
    control = pd.Series(rng.normal(0, 1, 300))
    treatment = pd.Series(rng.normal(0, 1, 300))
    result = bootstrap_ci(control, treatment, n_resamples=2000, random_state=42)
    assert result.ci_low < 0 < result.ci_high


def test_bootstrap_ci_deterministic_with_fixed_seed():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 100))
    treatment = pd.Series(rng.normal(0.3, 1, 100))
    r1 = bootstrap_ci(control, treatment, n_resamples=500, random_state=7)
    r2 = bootstrap_ci(control, treatment, n_resamples=500, random_state=7)
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


def test_permutation_test_catches_known_effect():
    rng = np.random.default_rng(0)
    control = pd.Series(rng.normal(0, 1, 300))
    treatment = pd.Series(rng.normal(0.5, 1, 300))
    result = permutation_test(control, treatment, n_permutations=2000, random_state=42)
    assert result.p_value < 0.05


def test_permutation_test_no_false_positive_on_null_effect():
    rng = np.random.default_rng(1)
    control = pd.Series(rng.normal(0, 1, 300))
    treatment = pd.Series(rng.normal(0, 1, 300))
    result = permutation_test(control, treatment, n_permutations=2000, random_state=42)
    assert result.p_value > 0.05
