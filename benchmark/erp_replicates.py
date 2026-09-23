"""Independent ERP means for fitting and confirmation; no realized-SNR scaling.

The original protocol supplies only the frozen source truth and masks. Noise is
regenerated as two independent 20-trial-mean equivalents, whose average has the
requested expected 40-trial-mean SNR. Forty individual epochs are not generated.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import erp_protocol, methods


def simulate_replicated_case(shared: dict, case: dict, seed_root=erp_protocol.ERP_SEED_ROOT) -> dict:
    """Keep old truth; generate independent training/confirmation observations.

    Explicit case replica_seed_roots['fit'/'check'] take precedence over seed_root.
    Each half also uses the case_id hash, a replicate-protocol domain tag, and
    modality/half indices. No inverse or decision receives truth.
    """
    if int(seed_root) != seed_root or seed_root < 0:
        raise ValueError("seed_root must be a nonnegative integer")
    seed_root = int(seed_root)
    roots = case.get("replica_seed_roots", dict(fit=seed_root, check=seed_root))
    if (not isinstance(roots, dict) or not {"fit", "check"}.issubset(roots)
            or any(int(roots[key]) != roots[key] or roots[key] < 0 for key in ("fit", "check"))):
        raise ValueError("replica_seed_roots must provide nonnegative integer fit/check roots")
    roots = {key: int(roots[key]) for key in ("fit", "check")}
    for modality in ("eeg", "meg"):
        gain = np.asarray(shared["gain_" + modality], dtype=float)
        factor = np.asarray(shared["noise_factor_" + modality], dtype=float)
        if (gain.ndim != 2 or factor.shape != (gain.shape[0], gain.shape[0])
                or not np.isfinite(gain).all() or not np.isfinite(factor).all()):
            raise ValueError("finite gain and square channel noise factor required")
    _, _, truth, groups, baseline, windows, active, old_metadata = erp_protocol.simulate_case(
        shared, case, seed_root=seed_root)
    sensor_ratio = case.get("deep_surface_sensor_amplitude_ratio")
    if sensor_ratio is not None:
        sensor_ratio = float(sensor_ratio)
        deep_index = case.get("deep_index")
        if (not np.isfinite(sensor_ratio) or sensor_ratio <= 0 or deep_index is None
                or not case.get("surface_centers")):
            raise ValueError("sensor-balanced scaling requires a positive ratio and mixed surface/deep truth")
        deep = np.zeros_like(truth)
        deep[int(deep_index)] = truth[int(deep_index)]
        surface = truth - deep
        energies = {}
        for layer, source in (("surface", surface), ("deep", deep)):
            energies[layer] = 0.
            for modality in ("eeg", "meg"):
                factor = np.asarray(shared["noise_factor_" + modality], float)
                covariance = factor @ factor.T
                values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
                keep = values > values[-1] * 1e-8
                whitener = (vectors[:, keep] / np.sqrt(values[keep])).T
                gain = np.asarray(shared["gain_" + modality], float)
                energies[layer] += (np.linalg.norm((whitener @ gain @ source)[:, active]) ** 2
                                    / keep.sum())
        if min(energies.values()) <= 0 or not np.isfinite(list(energies.values())).all():
            raise ValueError("mixed components need positive finite noise-normalized sensor energy")
        scale = sensor_ratio * np.sqrt(energies["surface"] / energies["deep"])
        truth = surface + scale * deep
        old_metadata.update(
            source_deep_surface_ratio_before_sensor_balance=old_metadata["deep_surface_ratio_actual"],
            deep_surface_ratio_actual=float(np.linalg.norm(scale * deep) / np.linalg.norm(surface)),
            deep_surface_sensor_amplitude_ratio=sensor_ratio,
            deep_sensor_scale=float(scale),
            component_scaling="joint EEG/MEG noise-normalized sensor amplitude; covariance eigenvalue cutoff 1e-8; modality energies divided by retained rank")
    case_key = int.from_bytes(hashlib.sha256(str(case["case_id"]).encode("utf-8")).digest()[:8], "little")
    entropy = {half: [roots[half], case_key, 6001, index]
               for index, half in enumerate(("fit", "check"))}
    half_seeds = [np.random.SeedSequence(entropy[half]).spawn(2) for half in ("fit", "check")]
    total_trials = erp_protocol.N_TRIALS
    half_trials = total_trials / 2
    result = dict(truth=truth, groups=groups, baseline=baseline,
                  active_windows=windows, active=active)
    actual_snr, expected_snr, noise_scales, seed_spawns = {}, {}, {}, {}
    for modality_index, modality in enumerate(("eeg", "meg")):
        clean = np.asarray(shared["gain_" + modality], dtype=float) @ truth
        factor = np.asarray(shared["noise_factor_" + modality], dtype=float)
        target = float(case.get(modality + "_snr_db", case.get("snr_db")))
        signal_energy = float(np.sum(clean[:, active] ** 2))
        # Baseline centering adds 1/n_baseline to the active noise variance.
        expected_noise_energy = float(np.sum(factor ** 2) * len(active) * (1 + 1 / baseline.sum()))
        if (not np.isfinite(target) or signal_energy <= 0 or expected_noise_energy <= 0):
            raise ValueError("finite SNR and nonzero active signal/noise energy required")
        noise_scale = np.sqrt(signal_energy * total_trials / (expected_noise_energy * 10 ** (target / 10)))
        noises = []
        spawn_keys = []
        for half, seed in zip(("train", "confirmation"),
                              (seeds[modality_index] for seeds in half_seeds)):
            noise = factor @ np.random.default_rng(seed).standard_normal(clean.shape)
            noise *= noise_scale / np.sqrt(half_trials)
            noise -= noise[:, baseline].mean(axis=1, keepdims=True)
            result[modality + "_" + half] = clean + noise
            noises.append(noise)
            spawn_keys.append(list(seed.spawn_key))
        actual_snr[modality] = {
            label: float(10 * np.log10(signal_energy / np.sum(noise[:, active] ** 2)))
            for label, noise in zip(("train", "confirmation", "combined"),
                                   (*noises, (noises[0] + noises[1]) / 2))}
        expected_snr[modality] = dict(combined=target, each_half=target - 10 * np.log10(2))
        noise_scales[modality] = float(noise_scale)
        seed_spawns[modality] = dict(zip(("train", "confirmation"), spawn_keys))
    metadata = dict(old_metadata)
    metadata.pop("baseline_corrected_before_snr_scaling", None)
    metadata.update(protocol="phase-locked-erp-independent-means-v1",
                    noise_model="two independent Gaussian 20-trial-mean equivalents; no stored individual epochs",
                    n_trials_combined=total_trials, n_trials_per_half=half_trials,
                    snr_level="expected combined 40-trial-mean active-window energy ratio",
                    actual_snr_db=actual_snr, expected_snr_db=expected_snr,
                    noise_scale=noise_scales,
                    snr_realization_normalized=False,
                    baseline_centering="each half independently; analytic scale includes active variance from baseline centering",
                    noise_scale_rule="analytic expected baseline-centered noise energy; never normalize a noise realization",
                    replica_seed_roots=roots, seed_case_key=case_key,
                    seed_entropy=entropy, seed_spawn_keys=seed_spawns,
                    seed_rule="explicit replica_seed_roots fit/check (fallback seed_root) + SHA256(case_id)[:8] little endian + domain 6001 + half index; modality spawn")
    result["metadata"] = metadata
    return result


def prepare_replicated_case(shared: dict, case: dict, seed_root=erp_protocol.ERP_SEED_ROOT) -> dict:
    """Apply one training-derived whitener and evidence weights to both halves.

    Returned training/confirmation/gain are whitened but not evidence-weighted;
    pass the separate channel_weights to the inverse for both observations.
    """
    result = simulate_replicated_case(shared, case, seed_root)
    baseline, active = result["baseline"], result["active_windows"][0]
    training_blocks, confirmation_blocks, gains = [], [], []
    for modality in ("eeg", "meg"):
        train, confirm = result[modality + "_train"], result[modality + "_confirmation"]
        width = train.shape[1]
        if width < 200 or train.shape != confirm.shape:
            raise ValueError("matching ERP halves with at least 200 samples required")
        whitened, gain = methods.whiten(np.hstack((train, confirm)), shared["gain_" + modality], noise_samples=200)
        train_white, confirm_white = whitened[:, :width], whitened[:, width:]
        training_blocks.append(train_white)
        confirmation_blocks.append(confirm_white)
        gains.append(gain)
    ratios = np.array([np.mean(block[:, active] ** 2) /
                       max(float(np.mean(block[:, baseline] ** 2)), np.finfo(float).tiny)
                       for block in training_blocks])
    excess = np.maximum(ratios - 1, 0)
    weights = np.sqrt(excess / excess.max()) if excess.max() > 0 else np.ones(2)
    result.update(training=np.vstack(training_blocks), confirmation=np.vstack(confirmation_blocks),
                  gain=np.vstack(gains),
                  channel_weights=np.repeat(weights, [len(block) for block in training_blocks]))
    result["metadata"].update(preprocessing="raw halves independently baseline-centered; training first 200 samples define each whitener; training modality evidence weights applied to both halves",
                              modality_weights=weights.tolist(),
                              retained_channels=[len(block) for block in training_blocks])
    return result
