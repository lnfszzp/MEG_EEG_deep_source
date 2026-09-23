"""Same raw means, independent training covariance, and correct mean-noise scale."""
import copy

import numpy as np

from benchmark import erp_replicates, erp_trial_covariance


def _toy():
    factor = np.linalg.cholesky(.4 ** np.abs(np.subtract.outer(np.arange(3), np.arange(3))))
    shared = dict(times=np.arange(601) / 1000 - .2, n_surf=2, n_deep=1,
                  gain_eeg=np.eye(3), gain_meg=np.eye(3),
                  noise_factor_eeg=factor, noise_factor_meg=factor)
    case = dict(case_id="trial-covariance-toy", surface_centers=[], deep_index=2,
                deep_surface_ratio=None, correlation=None, snr_db=0,
                eeg_snr_db=-5, meg_snr_db=10)
    return shared, case


def test_original_means_determinism_and_confirmation_independence():
    shared, case = _toy()
    case["replica_seed_roots"] = dict(fit=123, check=456)
    raw = erp_replicates.simulate_replicated_case(shared, case, 9)
    prepared = erp_trial_covariance.prepare_trial_covariance_case(shared, case, 9)
    repeated = erp_trial_covariance.prepare_trial_covariance_case(shared, case, 9)
    for modality in ("eeg", "meg"):
        for half in ("train", "confirmation"):
            np.testing.assert_array_equal(prepared[modality + "_" + half], raw[modality + "_" + half])
    for name in ("training", "confirmation", "gain", "channel_weights"):
        np.testing.assert_array_equal(prepared[name], repeated[name])
    changed_case = copy.deepcopy(case)
    changed_case["replica_seed_roots"]["check"] += 1
    changed = erp_trial_covariance.prepare_trial_covariance_case(shared, changed_case, 9)
    for name in ("training", "gain", "channel_weights"):
        np.testing.assert_array_equal(prepared[name], changed[name])
    assert not np.array_equal(prepared["confirmation"], changed["confirmation"])
    info = prepared["metadata"]["trial_baseline_covariance"]
    assert info["contrast_degrees_of_freedom"] == 19 * (raw["baseline"].sum() - 1)
    assert not info["generator_covariance_passed_to_inverse"]


def test_empirical_contrast_covariance_has_twenty_trial_mean_expectation():
    shared, case = _toy()
    estimates = []
    for seed in range(30):
        result = erp_trial_covariance.prepare_trial_covariance_case(shared, case, seed)
        # Identity toy gain makes the first modality's returned gain equal W.
        whitener = result["gain"][:3]
        inverse = np.linalg.inv(whitener)
        estimates.append(inverse @ inverse.T)
    expected = (shared["noise_factor_eeg"] @ shared["noise_factor_eeg"].T
                * result["metadata"]["noise_scale"]["eeg"] ** 2 / 20)
    relative_error = np.linalg.norm(np.mean(estimates, axis=0) - expected) / np.linalg.norm(expected)
    assert relative_error < .035
