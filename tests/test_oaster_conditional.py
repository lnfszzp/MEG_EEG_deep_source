"""Cheap checks for observable conditions; no inverse or full-grid runs."""
import inspect

import numpy as np
import pytest

from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_conditional import observable_quality
from candidates.oaster_predictive import conformal_decision


def test_quality_formula_bins_and_half_swap():
    baseline = np.arange(6) < 5
    basis = np.array([[0., 0., 0., 0., 0., 1.]])
    a = np.array([[-2., -1., 0., 1., 2., 30.],
                  [-2., -1., 0., 1., 2., 27.]])
    b = np.array([[-2., -1., 0., 1., 2., 1.],
                  [-2., -1., 0., 1., 2., 1.]])
    info = observable_quality(a, b, baseline, basis, (1, 1))
    assert info["cross_energy"] == [30., 27.]
    assert info["expected_noise_half_a"] == info["expected_noise_half_b"] == [3., 3.]
    assert info["q_eeg"] == pytest.approx(10.)
    assert info["q_meg"] == pytest.approx(9.)
    assert info["observable_bin"] == "HL"
    assert info == observable_quality(b, a, baseline, basis, (1, 1))
    assert not info["truth_used"]
    assert not {"truth", "snr", "labels", "gain"} & set(inspect.signature(observable_quality).parameters)


def test_modality_scaling_offsets_and_basis_rotation_do_not_change_quality():
    rng = np.random.default_rng(20261010)
    baseline = np.arange(80) < 40
    active = np.arange(80) >= 40
    a, b = rng.normal(size=(2, 4, 80))
    a[:, active] += 5.
    b[:, active] += 5.
    basis = _smooth_temporal_basis(a, baseline, active)[0]
    info = observable_quality(a, b, baseline, basis, (2, 2))
    scale = np.array([.2, .2, 3., 3.])[:, None]
    offset = np.arange(4)[:, None]
    transformed = observable_quality(a * scale + offset, b * scale - offset,
                                     baseline, basis, (2, 2))
    rotation = np.linalg.qr(rng.normal(size=(len(basis), len(basis))))[0]
    rotated = observable_quality(a, b, baseline, rotation @ basis, (2, 2))
    for key in ("q_eeg", "q_meg", "q_eeg_db", "q_meg_db"):
        assert transformed[key] == pytest.approx(info[key])
        assert rotated[key] == pytest.approx(info[key])
    assert transformed["observable_bin"] == rotated["observable_bin"] == info["observable_bin"]


def test_negative_cross_energy_is_low_and_serializable_without_infinity():
    baseline = np.arange(6) < 5
    basis = np.eye(6)[-1:]
    a = np.tile(np.array([-2., -1., 0., 1., 2., 10.]), (2, 1))
    b = a.copy()
    b[:, -1] *= -1
    info = observable_quality(a, b, baseline, basis, (1, 1))
    assert info["q_eeg"] == info["q_meg"] == 0.
    assert info["q_eeg_db"] is info["q_meg_db"] is None
    assert info["observable_bin"] == "LL"


@pytest.mark.parametrize("invalid", ["shape", "sizes", "float_sizes", "zero_noise", "nonfinite", "basis", "baseline"])
def test_invalid_observations_fail_before_binning(invalid):
    baseline = np.arange(6) < 5
    basis = np.eye(6)[-1:]
    a = np.tile(np.array([-2., -1., 0., 1., 2., 10.]), (2, 1))
    b, sizes = a.copy(), (1, 1)
    if invalid == "shape":
        b = b[:, :-1]
    elif invalid == "sizes":
        sizes = (1, 2)
    elif invalid == "float_sizes":
        sizes = (1., 1.)
    elif invalid == "zero_noise":
        b[:, baseline] = 0.
    elif invalid == "nonfinite":
        a[0, -1] = np.nan
    elif invalid == "basis":
        basis *= 2.
    else:
        baseline = np.zeros(6, bool)
    with pytest.raises(ValueError):
        observable_quality(a, b, baseline, basis, sizes)


def test_global_and_conditional_use_same_rank_level_and_conservative_ties():
    bank = np.arange(60, dtype=float)
    global_decision = conformal_decision(np.nextafter(57., np.inf), bank, alpha=.05)
    conditional_decision = conformal_decision(np.nextafter(19., np.inf), bank[:20], alpha=.05)
    assert global_decision["deep_present"] and conditional_decision["deep_present"]
    assert not conformal_decision(57., bank, alpha=.05)["deep_present"]
    assert not conformal_decision(19., bank[:20], alpha=.05)["deep_present"]
    assert not conformal_decision(-1., np.full(20, -2.), alpha=.05)["deep_present"]
    with pytest.raises(ValueError):
        conformal_decision(99., bank[:18], alpha=.05)
