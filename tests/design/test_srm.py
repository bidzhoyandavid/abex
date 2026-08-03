from abex.design.srm import check_srm


def test_balanced_groups_no_srm():
    result = check_srm({"control": 5000, "treatment": 5000})
    assert not result.has_srm


def test_imbalanced_groups_flags_srm():
    result = check_srm({"control": 5200, "treatment": 4800})
    assert result.has_srm


def test_custom_expected_ratio():
    result = check_srm({"control": 3000, "treatment": 7000}, expected_ratios={"control": 0.3, "treatment": 0.7})
    assert not result.has_srm


def test_mismatched_ratio_keys_raises():
    try:
        check_srm({"control": 100, "treatment": 100}, expected_ratios={"a": 0.5, "b": 0.5})
        assert False, "expected ValueError"
    except ValueError:
        pass
