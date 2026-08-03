from abex.design.power import sample_size_mean, sample_size_proportion


def test_sample_size_proportion_smaller_mde_needs_more_n():
    small_mde = sample_size_proportion(baseline_rate=0.1, mde_abs=0.01)
    large_mde = sample_size_proportion(baseline_rate=0.1, mde_abs=0.05)
    assert small_mde.sample_size_per_group > large_mde.sample_size_per_group


def test_sample_size_mean_smaller_mde_needs_more_n():
    small_mde = sample_size_mean(std=1.0, mde_abs=0.05)
    large_mde = sample_size_mean(std=1.0, mde_abs=0.2)
    assert small_mde.sample_size_per_group > large_mde.sample_size_per_group


def test_sample_size_proportion_rejects_invalid_baseline():
    try:
        sample_size_proportion(baseline_rate=1.5, mde_abs=0.01)
        assert False, "expected ValueError"
    except ValueError:
        pass
