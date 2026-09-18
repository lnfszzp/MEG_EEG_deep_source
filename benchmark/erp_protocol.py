"""Deterministic phase-locked ERP simulation on the corrected-v2 source grid."""

from __future__ import annotations

import hashlib

import numpy as np

from . import protocol


ERP_SEED_ROOT = 20260918
N_TIMES = 601
EVENT_SAMPLE = 200
N_TRIALS = 40
ACTIVE_LIMITS_S = (0.020, 0.120)
BASELINE_LIMITS_S = (-0.180, -0.020)


def _masks(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float).ravel()
    if times.size != N_TIMES or not np.all(np.diff(times) > 0):
        raise ValueError("ERP protocol requires 601 strictly increasing time samples")
    relative = times - times[EVENT_SAMPLE]
    tolerance = np.min(np.diff(times)) / 4.0
    if abs(relative[np.argmin(abs(relative))]) > tolerance:
        raise ValueError("ERP protocol requires the event at sample 200")
    active = (relative >= ACTIVE_LIMITS_S[0] - tolerance) & (
        relative <= ACTIVE_LIMITS_S[1] + tolerance
    )
    baseline = (relative >= BASELINE_LIMITS_S[0] - tolerance) & (
        relative <= BASELINE_LIMITS_S[1] + tolerance
    )
    if not active.any() or active.sum() > baseline.sum() or np.any(active & baseline):
        raise ValueError("ERP active window must be non-empty and no wider than baseline")
    return baseline, active


def _waveforms(
    times: np.ndarray, active: np.ndarray, count: int, correlation: float
) -> np.ndarray:
    if count not in (1, 2, 3) or not 0.0 <= correlation < 1.0:
        raise ValueError("ERP truth supports 1-3 sources and correlation in [0, 1)")
    relative = np.asarray(times)[active] - np.asarray(times)[EVENT_SAMPLE]
    centers = np.linspace(0.040, 0.100, count)
    taper = np.sin(np.linspace(0.0, np.pi, relative.size)) ** 2
    candidates = taper[:, None] * np.column_stack(
        [np.exp(-0.5 * ((relative - center) / 0.010) ** 2) for center in centers]
    )
    candidates -= taper[:, None] * (
        candidates.mean(axis=0) / taper.mean()
    )
    basis, _ = np.linalg.qr(candidates)
    target = np.full((count, count), correlation)
    np.fill_diagonal(target, 1.0)
    active_waves = np.linalg.cholesky(target) @ basis[:, :count].T
    waves = np.zeros((count, len(times)))
    waves[:, active] = active_waves
    return waves


def _noise(
    clean: np.ndarray,
    factor: np.ndarray,
    baseline: np.ndarray,
    active: np.ndarray,
    snr_db: float,
    seed: np.random.SeedSequence,
) -> tuple[np.ndarray, float]:
    noise = (
        np.asarray(factor, dtype=float)
        @ np.random.default_rng(seed).standard_normal(clean.shape)
        / np.sqrt(N_TRIALS)
    )
    noise -= noise[:, baseline].mean(axis=1, keepdims=True)
    signal_energy = float(np.sum(clean[:, active] ** 2))
    noise_energy = float(np.sum(noise[:, active] ** 2))
    if signal_energy == 0.0 or noise_energy == 0.0:
        raise ValueError("ERP signal and colored noise need non-zero active-window energy")
    noise *= np.sqrt(signal_energy / (noise_energy * 10.0 ** (snr_db / 10.0)))
    actual = 10.0 * np.log10(signal_energy / np.sum(noise[:, active] ** 2))
    return clean + noise, float(actual)


def simulate_case(shared: dict, case: dict) -> tuple:
    """Return EEG, MEG, truth, groups, masks, indices, and audit metadata."""
    times = np.asarray(shared["times"], dtype=float).ravel()
    baseline, active = _masks(times)
    surface_parts = [
        protocol._surface_patch(shared, int(center))
        for center in case["surface_centers"]
    ]
    groups = [indices for indices, _weights in surface_parts]
    if case.get("deep_index") is not None:
        groups.append(np.array([int(case["deep_index"])], dtype=int))

    correlation = float(case.get("correlation") or 0.0)
    waves = _waveforms(times, active, len(groups), correlation)
    fallback_assignment_key = int.from_bytes(
        hashlib.sha256(
            str(
                case.get(
                    "source_case_id",
                    case.get("configuration_id", case["case_id"]),
                )
            ).encode("utf-8")
        ).digest()[:8],
        "little",
    )
    assignment_key = (
        int(case["location"]) // 6
        if len(groups) > 1 and case.get("location") is not None
        else int(case.get("configuration_number", fallback_assignment_key))
    )
    waveform_assignment = np.roll(
        np.arange(len(groups)), assignment_key % len(groups)
    )
    n_sources = int(shared["n_surf"]) + int(shared["n_deep"])
    components: list[tuple[str, np.ndarray]] = []
    for component_number, (indices, weights) in enumerate(surface_parts):
        wave = waves[waveform_assignment[component_number]]
        component = np.zeros((n_sources, N_TIMES))
        component[indices] = weights[:, None] * wave
        components.append(("surface", component))
    if case.get("deep_index") is not None:
        component = np.zeros((n_sources, N_TIMES))
        component[int(case["deep_index"])] = waves[
            waveform_assignment[len(surface_parts)]
        ]
        components.append(("deep", component))

    surface = sum(
        (part for layer, part in components if layer == "surface"),
        start=np.zeros((n_sources, N_TIMES)),
    )
    deep = sum(
        (part for layer, part in components if layer == "deep"),
        start=np.zeros((n_sources, N_TIMES)),
    )
    if np.linalg.norm(surface) and np.linalg.norm(deep):
        deep *= (
            float(case["deep_surface_ratio"])
            * np.linalg.norm(surface)
            / np.linalg.norm(deep)
        )
    truth = surface + deep

    clean_eeg = np.asarray(shared["gain_eeg"], dtype=float) @ truth
    clean_meg = np.asarray(shared["gain_meg"], dtype=float) @ truth
    case_key = int.from_bytes(
        hashlib.sha256(str(case["case_id"]).encode("utf-8")).digest()[:8], "little"
    )
    eeg_seed, meg_seed = np.random.SeedSequence([ERP_SEED_ROOT, case_key]).spawn(2)
    eeg_target = float(case.get("eeg_snr_db", case["snr_db"]))
    meg_target = float(case.get("meg_snr_db", case["snr_db"]))
    eeg, eeg_actual = _noise(
        clean_eeg,
        shared["noise_factor_eeg"],
        baseline,
        active,
        eeg_target,
        eeg_seed,
    )
    meg, meg_actual = _noise(
        clean_meg,
        shared["noise_factor_meg"],
        baseline,
        active,
        meg_target,
        meg_seed,
    )

    group_waves = [
        truth[group[np.argmax(np.linalg.norm(truth[group], axis=1))]]
        for group in groups
    ]
    actual_correlation = (
        np.corrcoef(np.asarray(group_waves)[:, active]).tolist()
        if len(group_waves) > 1
        else [[1.0]]
    )
    metadata = {
        "case_id": case["case_id"],
        "protocol": "phase-locked-erp-v1",
        "event_sample": EVENT_SAMPLE,
        "n_times": N_TIMES,
        "n_trials": N_TRIALS,
        "active_window_s": list(ACTIVE_LIMITS_S),
        "baseline_window_s": list(BASELINE_LIMITS_S),
        "groups": [group.tolist() for group in groups],
        "correlation_target": correlation,
        "correlation_actual": actual_correlation,
        "waveform_assignment": waveform_assignment.tolist(),
        "deep_surface_ratio_actual": (
            float(np.linalg.norm(deep) / np.linalg.norm(surface))
            if np.linalg.norm(deep) and np.linalg.norm(surface)
            else None
        ),
        "target_snr_db": {"eeg": eeg_target, "meg": meg_target},
        "actual_snr_db": {"eeg": eeg_actual, "meg": meg_actual},
        "implied_single_trial_snr_db": {
            "eeg": eeg_target - 10.0 * np.log10(N_TRIALS),
            "meg": meg_target - 10.0 * np.log10(N_TRIALS),
        },
        "snr_level": "evoked-level",
        "noise_model": "colored Gaussian 40-trial-mean equivalent",
        "baseline_corrected_before_snr_scaling": True,
        "waveform_model": "phase-locked transient ERP without sinusoidal carrier",
        "seed_root": ERP_SEED_ROOT,
        "seed_case_key": case_key,
    }
    return (
        eeg,
        meg,
        truth,
        groups,
        baseline,
        (active,),
        np.flatnonzero(active),
        metadata,
    )
