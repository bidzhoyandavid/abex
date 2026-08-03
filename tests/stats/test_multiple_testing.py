from abex.stats.multiple_testing import benjamini_hochberg, bonferroni


def test_bonferroni_scales_p_values():
    result = bonferroni([0.01, 0.02, 0.5], alpha=0.05)
    assert result.adjusted_p_values[0] == 0.03
    assert result.reject == [True, False, False]


def test_bonferroni_caps_at_one():
    result = bonferroni([0.5, 0.6], alpha=0.05)
    assert all(p <= 1.0 for p in result.adjusted_p_values)


def test_benjamini_hochberg_less_conservative_than_bonferroni():
    p_values = [0.001, 0.01, 0.02, 0.04, 0.5]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    bonf = bonferroni(p_values, alpha=0.05)
    assert sum(bh.reject) >= sum(bonf.reject)
