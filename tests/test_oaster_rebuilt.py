from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import sparse

from candidates import oaster_rebuilt as oaster


def _kernels(n_surf: int):
    identity = sparse.eye(n_surf, format="csc")
    return tuple((scale, identity.copy()) for scale in oaster.SURFACE_SCALES_MM)


def test_temporal_basis_is_observation_selected() -> None:
    rng = np.random.default_rng(9)
    data = rng.normal(scale=0.05, size=(6, 400))
    wave = np.sin(2.0 * np.pi * 11.0 * np.arange(200) / 200.0)
    data[:, 200:] += 8.0 * np.arange(1, 7)[:, None] * wave

    basis = oaster._temporal_basis(data)

    assert 1 <= basis.shape[0] <= data.shape[0]
    assert basis.shape[1] == data.shape[1]
    assert np.array_equal(basis[:, :200], np.zeros((basis.shape[0], 200)))
    assert np.allclose(basis @ basis.T, np.eye(basis.shape[0]), atol=1e-12)
    assert oaster._temporal_basis(np.zeros_like(data)).shape == (0, 400)


def test_ebic_deduplicates_multiscale_templates() -> None:
    basis = np.zeros((1, 400))
    basis[0, 200:] = 1.0 / np.sqrt(200.0)
    leadfield = np.eye(4)
    data = 20.0 * leadfield[:, [0]] @ basis

    estimate, count = oaster._ebic_templates(
        data, leadfield, 3, _kernels(3), basis
    )

    assert count == 1
    assert np.linalg.norm(estimate[0]) > 0
    assert np.array_equal(estimate[1:], np.zeros((3, 400)))


def test_reconstruct_is_finite_and_oracle_free() -> None:
    assert list(inspect.signature(oaster.reconstruct).parameters) == [
        "eeg_data",
        "meg_data",
        "gain_eeg",
        "gain_meg",
        "n_surf",
        "kernels",
    ]
    rng = np.random.default_rng(17)
    gain_eeg = rng.normal(size=(6, 4))
    gain_meg = rng.normal(size=(5, 4))
    source = np.zeros((4, 400))
    t = np.arange(200)
    source[0, 200:] = 3.0 * np.sin(2.0 * np.pi * 9.0 * t / 200.0)
    source[3, 200:] = 2.0 * np.cos(2.0 * np.pi * 15.0 * t / 200.0)
    eeg = gain_eeg @ source + rng.normal(scale=0.05, size=(6, 400))
    meg = gain_meg @ source + rng.normal(scale=0.05, size=(5, 400))

    estimate, diagnostics = oaster.reconstruct(
        eeg, meg, gain_eeg, gain_meg, 3, _kernels(3)
    )

    assert estimate.shape == source.shape
    assert np.isfinite(estimate).all()
    assert np.linalg.norm(estimate[:, 200:]) > 0
    assert diagnostics["temporal_rank"] >= 1
    assert diagnostics["selected_templates"] >= 1


@pytest.mark.parametrize(("delta", "accepted"), ((-1.0, True), (1.0, False)))
def test_reconstruct_applies_rescue_conditionally_before_spectral_fusion(
    monkeypatch: pytest.MonkeyPatch, delta: float, accepted: bool
) -> None:
    data = np.zeros((2, 4))
    leadfield = np.eye(2)
    basis = np.ones((1, 4)) / 2.0
    primary = np.zeros((2, 4))
    primary[0] = 1.0
    rescue = np.zeros_like(primary)
    rescue[1] = 2.0
    spectral = np.full_like(primary, 0.25)
    fused_primary = []

    monkeypatch.setattr(
        oaster.protected,
        "whitened_joint_system",
        lambda *_args: (data, leadfield),
    )
    monkeypatch.setattr(oaster, "_temporal_basis", lambda _data: basis)
    monkeypatch.setattr(
        oaster,
        "_ebic_templates",
        lambda *_args: (primary.copy(), 1),
    )
    monkeypatch.setattr(
        oaster,
        "deep_rescue_trial",
        lambda *_args: (
            rescue,
            {
                "accepted_without_tau": delta < 0.0,
                "deep_local": 0,
                "ebic_delta_without_tau": delta,
                "universe": 1,
                "recovered_design_rank": 1,
            },
        ),
    )
    monkeypatch.setattr(
        oaster.protected,
        "multiscale_spectral_evidence_source",
        lambda *_args: spectral,
        raising=False,
    )

    def fuse(actual_primary, evidence, fraction):
        fused_primary.append(actual_primary.copy())
        assert np.array_equal(evidence, spectral)
        assert fraction == oaster.SPECTRAL_FRACTION
        return actual_primary + evidence

    monkeypatch.setattr(oaster.protected, "add_scaled_evidence", fuse, raising=False)
    estimate, diagnostics = oaster.reconstruct(
        data,
        data,
        leadfield,
        leadfield,
        1,
        ((4.0, sparse.eye(1, format="csc")),),
    )

    expected_primary = primary + rescue if accepted else primary
    assert np.array_equal(fused_primary[0], expected_primary)
    assert np.array_equal(estimate, expected_primary + spectral)
    assert diagnostics["deep_rescue_accepted"] is accepted
    assert diagnostics["deep_rescue_ebic_delta"] == delta
