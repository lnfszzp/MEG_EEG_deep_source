import numpy as np

from analyze_strict_statistics import _bootstrap_mean_ci, _holm, _rank_biserial


def test_holm_preserves_order_and_monotonic_adjustment():
    assert np.allclose(_holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_rank_biserial_excludes_zero_differences():
    assert np.isclose(_rank_biserial(np.array([1.0, 2.0, -0.5, 0.0])), 2.0 / 3.0)


def test_paired_bootstrap_keeps_constant_difference():
    low, high = _bootstrap_mean_ci(np.full(6, 0.25), np.random.default_rng(7), 100)
    assert low == high == 0.25
