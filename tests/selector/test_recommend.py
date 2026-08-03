from abex.data.profiling import MetricProfile
from abex.selector.recommend import recommend_test


def _profile(**overrides) -> MetricProfile:
    defaults = dict(
        metric_col="metric",
        group_col="group",
        n_groups=2,
        group_sizes={"a": 500, "b": 500},
        kind="continuous",
        is_balanced_design=True,
        skewness=0.1,
        excess_kurtosis=0.0,
        outlier_share=0.0,
        zero_share=0.0,
        has_pre_period=False,
        min_group_size=500,
    )
    defaults.update(overrides)
    return MetricProfile(**defaults)


def test_normal_continuous_two_groups_recommends_t_test_first():
    profile = _profile()
    recs = recommend_test(profile)
    assert recs[0].method_name == "t_test"
    assert recs[0].violated_assumptions == ()


def test_skewed_small_sample_deprioritizes_t_test():
    profile = _profile(skewness=3.0, min_group_size=40, group_sizes={"a": 40, "b": 40})
    recs = recommend_test(profile)
    names_in_order = [r.method_name for r in recs]
    assert names_in_order.index("mann_whitney") < names_in_order.index("t_test")


def test_binary_metric_recommends_z_test():
    profile = _profile(kind="binary", min_group_size=500)
    recs = recommend_test(profile)
    assert any(r.method_name == "z_test_proportions" for r in recs)
    assert all(r.method_name != "t_test" for r in recs)


def test_three_groups_only_recommends_omnibus_tests():
    profile = _profile(n_groups=3, group_sizes={"a": 200, "b": 200, "c": 200}, min_group_size=200)
    recs = recommend_test(profile)
    names = {r.method_name for r in recs}
    assert names <= {"kruskal_wallis", "anova_oneway", "bootstrap_ci", "permutation_test"}


def test_paired_flag_selects_wilcoxon_only():
    profile = _profile()
    recs = recommend_test(profile, paired=True)
    names = {r.method_name for r in recs}
    assert names == {"wilcoxon_signed_rank"}


def test_too_small_sample_returns_no_parametric_candidates():
    profile = _profile(min_group_size=3, group_sizes={"a": 3, "b": 3})
    recs = recommend_test(profile)
    assert all(r.method_name != "t_test" for r in recs)
