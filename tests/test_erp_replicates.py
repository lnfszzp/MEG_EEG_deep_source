"""Frozen-truth replication, expected SNR, and no confirmation preprocessing leak."""
import numpy as np
import pytest

from benchmark import erp_protocol, erp_replicates


def _toy():
    rng = np.random.default_rng(15)
    shared = dict(times=np.arange(601) / 1000 - .2, n_surf=2, n_deep=1)
    for modality, channels in (("eeg", 3), ("meg", 4)):
        shared["gain_" + modality] = rng.normal(size=(channels, 3))
        shared["noise_factor_" + modality] = np.linalg.cholesky(.3 ** np.abs(np.subtract.outer(np.arange(channels), np.arange(channels))))
    case = dict(case_id="replicate-toy", surface_centers=[], deep_index=2,
                deep_surface_ratio=None, correlation=None, snr_db=0,
                eeg_snr_db=-5, meg_snr_db=10)
    return shared, case


def test_frozen_truth_independent_means_and_determinism():
    shared, case = _toy()
    old = erp_protocol.simulate_case(shared, case, seed_root=9)
    result = erp_replicates.simulate_replicated_case(shared, case, 9)
    repeated = erp_replicates.simulate_replicated_case(shared, case, 9)
    np.testing.assert_array_equal(result["truth"], old[2])
    np.testing.assert_array_equal(result["baseline"], old[4])
    np.testing.assert_array_equal(result["active"], old[6])
    np.testing.assert_array_equal(result["groups"][0], old[3][0])
    for modality in ("eeg", "meg"):
        train, confirm = result[modality + "_train"], result[modality + "_confirmation"]
        np.testing.assert_array_equal(train, repeated[modality + "_train"])
        assert not np.array_equal(train, confirm)
        np.testing.assert_allclose(train[:, result["baseline"]].mean(axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(confirm[:, result["baseline"]].mean(axis=1), 0, atol=1e-15)
    np.testing.assert_array_equal(old[0], erp_protocol.simulate_case(shared, case, seed_root=9)[0])
    assert result["metadata"]["n_trials_per_half"] == 20
    with pytest.raises(ValueError, match="seed_root"):
        erp_replicates.simulate_replicated_case(shared, case, -1)
    with pytest.raises(ValueError, match="seed_root"):
        erp_replicates.simulate_replicated_case(shared, case, 1.5)


def test_expected_combined_snr_without_realization_normalization():
    shared, case = _toy()
    ratios = {modality: [] for modality in ("eeg", "meg")}
    scales = []
    for seed in range(80):
        result = erp_replicates.simulate_replicated_case(shared, case, seed)
        scales.append(result["metadata"]["noise_scale"])
        active = result["active"]
        for modality in ratios:
            clean = shared["gain_" + modality] @ result["truth"]
            train = result[modality + "_train"] - clean
            confirm = result[modality + "_confirmation"] - clean
            target_energy = np.sum(clean[:, active] ** 2) / 10 ** (case[modality + "_snr_db"] / 10)
            ratios[modality].append([np.sum(noise[:, active] ** 2) / target_energy
                                     for noise in (train, confirm, (train + confirm) / 2)])
    assert all(scale == scales[0] for scale in scales)
    for observations in ratios.values():
        np.testing.assert_allclose(np.mean(observations, axis=0), [2, 2, 1], rtol=.08)
        assert np.std(np.asarray(observations)[:, 2]) > .03


def test_explicit_half_roots_override_fallback_and_change_only_the_requested_half():
    shared, case = _toy()
    case["replica_seed_roots"] = dict(fit=2026092206, check=2026092207)
    original = erp_replicates.simulate_replicated_case(shared, case, 13)
    other_fallback = erp_replicates.simulate_replicated_case(shared, case, 999)
    for modality in ("eeg", "meg"):
        for half in ("train", "confirmation"):
            np.testing.assert_array_equal(original[modality + "_" + half], other_fallback[modality + "_" + half])
    case["replica_seed_roots"]["check"] = 2026092209
    changed = erp_replicates.simulate_replicated_case(shared, case, 13)
    for modality in ("eeg", "meg"):
        np.testing.assert_array_equal(original[modality + "_train"], changed[modality + "_train"])
        assert not np.array_equal(original[modality + "_confirmation"], changed[modality + "_confirmation"])
    assert original["metadata"]["replica_seed_roots"] == dict(fit=2026092206, check=2026092207)


def test_confirmation_cannot_change_whitening_or_weights(monkeypatch):
    shared, case = _toy()
    prepared = erp_replicates.prepare_replicated_case(shared, case, 13)
    simulator = erp_replicates.simulate_replicated_case

    def change_confirmation(*args, **kwargs):
        result = simulator(*args, **kwargs)
        for modality in ("eeg", "meg"):
            result[modality + "_confirmation"] *= 1000
        return result

    monkeypatch.setattr(erp_replicates, "simulate_replicated_case", change_confirmation)
    changed = erp_replicates.prepare_replicated_case(shared, case, 13)
    for key in ("gain", "training", "channel_weights"):
        np.testing.assert_array_equal(prepared[key], changed[key])
    assert not np.array_equal(prepared["confirmation"], changed["confirmation"])
    assert prepared["training"].shape == prepared["confirmation"].shape
    assert prepared["gain"].shape[0] == len(prepared["channel_weights"])
