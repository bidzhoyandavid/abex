import pytest

from abex.analysis.effect_size import EffectSizeResult
from abex.report import build_report, merge_disagreeing_reports


def _effect():
    return EffectSizeResult(absolute_diff=1.0, relative_lift=0.1, cohens_d=0.2)


def test_build_report_rejects_non_str_metric():
    with pytest.raises(TypeError):
        build_report(123, "t_test", 0.03, _effect())


def test_build_report_rejects_non_effect_size_result():
    with pytest.raises(TypeError):
        build_report("metric", "t_test", 0.03, {"absolute_diff": 1.0})


def test_build_report_rejects_bad_alpha():
    with pytest.raises(ValueError):
        build_report("metric", "t_test", 0.03, _effect(), alpha=1.5)


def test_build_report_rejects_bad_ci_shape():
    with pytest.raises(TypeError):
        build_report("metric", "t_test", 0.03, _effect(), ci=(1.0, 2.0, 3.0))


def test_merge_disagreeing_reports_rejects_empty_list():
    with pytest.raises(ValueError):
        merge_disagreeing_reports([])


def test_merge_disagreeing_reports_rejects_non_list():
    with pytest.raises(TypeError):
        merge_disagreeing_reports("not_a_list")


def test_merge_disagreeing_reports_flags_low_confidence_on_disagreement():
    report_a = build_report("metric", "t_test", 0.03, _effect())
    report_b = build_report("metric", "mann_whitney", 0.4, _effect())
    merged = merge_disagreeing_reports([report_a, report_b])
    assert merged["decision"] == "low_confidence"
    assert merged["agreement"] is False
