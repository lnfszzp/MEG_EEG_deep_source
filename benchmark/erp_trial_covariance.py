"""ERP preprocessing from independent Gaussian trial-baseline contrasts.

Keep the original two half-means exactly. Additional within-training-trial
baseline contrasts estimate their mean-noise covariance. Mean and contrasts are
independent Gaussian sufficient statistics, not forty stored trial epochs.
"""
from __future__ import annotations

import numpy as np

from . import erp_protocol, erp_replicates


def prepare_trial_covariance_case(shared: dict, case: dict,
                                  seed_root=erp_protocol.ERP_SEED_ROOT) -> dict:
    """Return the existing replicate API, changing only the training whitener.

    The simulator uses F to generate contrast samples; the inverse only receives
    their empirical whitener, not the known generating covariance. Confirmation
    data and its seed cannot change the training covariance or modality weights.
    """
    result = erp_replicates.simulate_replicated_case(shared, case, seed_root)
    metadata = result["metadata"]
    baseline, active = result["baseline"], result["active_windows"][0]
    n_trials = int(metadata["n_trials_per_half"])
    n_baseline = int(baseline.sum())
    if n_trials < 2 or n_baseline < 2:
        raise ValueError("trial contrasts require at least two training trials and baseline samples")
    entropy = [metadata["replica_seed_roots"]["fit"], metadata["seed_case_key"], 6002]
    modality_seeds = np.random.SeedSequence(entropy).spawn(2)
    degrees_of_freedom = (n_trials - 1) * (n_baseline - 1)
    training, confirmation, gains = [], [], []
    diagnostics = {}
    for modality, seed in zip(("eeg", "meg"), modality_seeds):
        factor = np.asarray(shared["noise_factor_" + modality], dtype=float)
        channels = factor.shape[0]
        contrast = factor @ np.random.default_rng(seed).standard_normal((channels, n_trials * n_baseline))
        contrast *= metadata["noise_scale"][modality]
        contrast = contrast.reshape(channels, n_trials, n_baseline)
        contrast -= contrast.mean(axis=1, keepdims=True)
        contrast -= contrast.mean(axis=2, keepdims=True)
        samples = contrast.reshape(channels, -1)
        covariance = samples @ samples.T / (degrees_of_freedom * n_trials)
        values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        if not np.isfinite(values).all() or values[-1] <= 0:
            raise ValueError("trial-baseline contrasts must provide finite nonzero covariance")
        keep = values > values[-1] * 1e-8
        whitener = (vectors[:, keep] / np.sqrt(values[keep])).T
        training.append(whitener @ result[modality + "_train"])
        confirmation.append(whitener @ result[modality + "_confirmation"])
        gains.append(whitener @ np.asarray(shared["gain_" + modality], dtype=float))
        diagnostics[modality] = dict(seed_entropy=entropy, seed_spawn_key=list(seed.spawn_key),
            retained_channels=int(keep.sum()), covariance_trace=float(np.trace(covariance)),
            eigenvalue_condition_before_cutoff=float(values[-1] / values[0]) if values[0] > 0 else None,
            eigenvalue_condition_retained=float(values[-1] / values[keep][0]),
            eigenvalue_relative_cutoff=1e-8)
    ratios = np.array([np.mean(block[:, active] ** 2) /
                       max(float(np.mean(block[:, baseline] ** 2)), np.finfo(float).tiny)
                       for block in training])
    excess = np.maximum(ratios - 1, 0)
    weights = np.sqrt(excess / excess.max()) if excess.max() > 0 else np.ones(2)
    result.update(training=np.vstack(training), confirmation=np.vstack(confirmation),
                  gain=np.vstack(gains),
                  channel_weights=np.repeat(weights, [len(block) for block in training]))
    metadata.update(preprocessing="raw independent half-means unchanged; empirical training-trial baseline contrast covariance defines both whiteners; training-only modality evidence weights",
                    modality_weights=weights.tolist(), retained_channels=[len(block) for block in training],
                    trial_baseline_covariance=dict(
                        rule="C_mean = Z Z.T / ((N-1)*(B-1)*N), Z centered over trials and baseline time",
                        representation="independent Gaussian half-mean and training-baseline contrasts; sufficient-statistic equivalent, no stored forty epochs",
                        generator_covariance_passed_to_inverse=False,
                        independent_of_training_mean_under_gaussian_model=True,
                        confirmation_used_for_covariance=False,
                        n_training_trials=n_trials, n_baseline=n_baseline,
                        contrast_degrees_of_freedom=degrees_of_freedom,
                        modalities=diagnostics))
    return result
