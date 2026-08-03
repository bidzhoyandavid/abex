import pytest

from abex.selector.recommend import recommend_test
from abex.selector.rules import candidates_for


def test_candidates_for_rejects_non_profile():
    with pytest.raises(TypeError):
        candidates_for("not_a_profile")


def test_candidates_for_rejects_non_bool_paired():
    with pytest.raises(TypeError):
        candidates_for("not_a_profile", paired="yes")


def test_recommend_test_rejects_non_profile():
    with pytest.raises(TypeError):
        recommend_test({"kind": "continuous"})
