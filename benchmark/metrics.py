from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from metrics.user_metrics import An_auc, An_cal_AUC, DLE_an, RMSE, SD

DEEP_THRESHOLD_CANDIDATES = (
    0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99
)

AUC_THRESHOLD = 0.01


def _requested_auc(
    truth: np.ndarray,
    estimate: np.ndarray,
    cortex: dict,
    groups: list[np.ndarray],
) -> float:
    groups = [np.asarray(group, dtype=int).ravel() for group in groups if np.asarray(group).size]
    if len(groups) <= 1:
        return float(An_cal_AUC(truth, estimate, cortex, threshold=AUC_THRESHOLD))

    vertices = np.asarray(cortex["Vertices"], dtype=float)
    active = np.concatenate(groups)
    labels = np.concatenate([np.full(group.size, index) for index, group in enumerate(groups)])
    owner = labels[cKDTree(vertices[active]).query(vertices)[1]]
    values = []
    for index, group in enumerate(groups):
        group_truth = np.zeros_like(truth)
        group_truth[group] = truth[group]
        group_estimate = estimate.copy()
        group_estimate[owner != index] = 0.0
        values.append(
            An_cal_AUC(
                group_truth,
                group_estimate,
                cortex,
                threshold=AUC_THRESHOLD,
            )
        )
    return float(np.mean(values))


def source_amplitude(source: np.ndarray, active: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    active = np.asarray(active, dtype=int)
    baseline = np.setdiff1d(np.arange(source.shape[1]), active)
    active_power = np.mean(source[:, active] ** 2, axis=1)
    baseline_power = (
        np.mean(source[:, baseline] ** 2, axis=1) if baseline.size else 0.0
    )
    return np.sqrt(np.maximum(active_power - baseline_power, 0.0))


def _groups_for_layer(
    groups: list[np.ndarray], start: int, stop: int
) -> list[np.ndarray]:
    result = []
    for group in groups:
        group = np.asarray(group, dtype=int).ravel()
        if group.size and np.all((group >= start) & (group < stop)):
            result.append(group - start)
    return result


def _layer_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    positions_m: np.ndarray,
    groups: list[np.ndarray],
    start: int,
    stop: int,
    active: np.ndarray,
    global_support: np.ndarray,
) -> tuple[float, float]:
    local_groups = _groups_for_layer(groups, start, stop)
    if not local_groups:
        return np.nan, np.nan
    keep = global_support[start:stop]
    if not np.any(keep):
        return np.nan, np.nan
    used = np.asarray(estimate[start:stop, :], dtype=float).copy()
    used[~keep] = 0.0
    truth_indices = np.flatnonzero(np.sum(truth[start:stop, active] ** 2, axis=1) > 0)
    sd_mm = float(
        SD(used[:, active], positions_m[start:stop][truth_indices], positions_m[start:stop], 0.0)
        * 1000.0
    )
    dle_mm = float(
        DLE_an(
            used[:, active],
            local_groups,
            positions_m[start:stop],
            index_base=0,
        )
        * 1000.0
    )
    return sd_mm, dle_mm


def evaluate_estimate(
    estimate: np.ndarray,
    truth: np.ndarray,
    positions_m: np.ndarray,
    groups: list[np.ndarray],
    n_surf: int,
    active: np.ndarray,
    cortex: dict,
    *,
    deep_threshold: float = 0.1,
    support_energy_fraction: float = 0.1,
    deep_radius_mm: float = 10.0,
) -> dict[str, float | int]:
    """Call the requested metrics once, with one rule shared by every method."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    positions_m = np.asarray(positions_m, dtype=float)
    active = np.asarray(active, dtype=int).ravel()
    if estimate.shape != truth.shape or estimate.shape[0] != positions_m.shape[0]:
        raise ValueError("estimate, truth and positions must share the source axis")
    if not np.all(np.isfinite(estimate)):
        raise ValueError("estimate contains NaN or infinity")

    active_estimate = estimate[:, active]
    active_truth = truth[:, active]
    energy = np.sum(active_estimate**2, axis=1)
    peak = float(energy.max(initial=0.0))
    support = np.zeros(energy.size, dtype=bool)
    for start, stop in ((0, n_surf), (n_surf, energy.size)):
        layer_peak = float(energy[start:stop].max(initial=0.0))
        if layer_peak > 0:
            support[start:stop] = energy[start:stop] > support_energy_fraction * layer_peak

    try:
        auc = _requested_auc(active_truth, active_estimate, cortex, groups)
    except (ValueError, IndexError, ZeroDivisionError, FloatingPointError):
        auc = np.nan

    # A zero estimate carries no global ranking information. Keep the neutral
    # supplemental AUC and limiting normalized squared error.
    auc_tie_corrected = 0.5
    rmse = 1.0
    if peak > 0:
        target = np.sum(active_truth**2, axis=1) > 0
        positives = int(target.sum())
        negatives = int((~target).sum())
        if positives and negatives:
            auc_tie_corrected = An_auc(np.c_[target, energy])
        rmse = float(RMSE(active_estimate, active_truth))

    surface_sd, surface_dle = _layer_metrics(
        estimate, truth, positions_m, groups, 0, n_surf, active, support
    )
    deep_sd, deep_dle = _layer_metrics(
        estimate, truth, positions_m, groups, n_surf, len(positions_m), active, support
    )

    amplitude = source_amplitude(estimate, active)
    amplitude_peak = float(amplitude.max(initial=0.0))
    deep_peak = float(amplitude[n_surf:].max(initial=0.0))
    deep_score = deep_peak / amplitude_peak if amplitude_peak > 0 else 0.0
    deep_groups = _groups_for_layer(groups, n_surf, len(positions_m))
    has_deep = bool(deep_groups)
    deep_distance = np.nan
    if deep_peak > 0 and has_deep:
        predicted = n_surf + int(np.argmax(amplitude[n_surf:]))
        true_indices = np.concatenate(deep_groups) + n_surf
        deep_distance = float(
            np.min(np.linalg.norm(positions_m[true_indices] - positions_m[predicted], axis=1))
            * 1000.0
        )
    deep_positive = deep_score >= float(deep_threshold)
    detected = bool(
        has_deep
        and deep_positive
        and np.isfinite(deep_distance)
        and deep_distance <= float(deep_radius_mm) + 1e-6
    )
    return {
        # Preserve both established columns: requested parcel AUC and the
        # corrected global tied-rank AUC supplied by the vendored An_auc.
        "auc": auc,
        "auc_tie_corrected": auc_tie_corrected,
        "rmse": rmse,
        "surface_sd_mm": surface_sd,
        "deep_sd_mm": deep_sd,
        "surface_dle_mm": surface_dle,
        "deep_dle_mm": deep_dle,
        "has_deep_true": int(has_deep),
        "deep_score": deep_score,
        "deep_peak_distance_mm": deep_distance,
        "deep_detected": int(detected),
        "deep_false_positive": int((not has_deep) and deep_positive),
        "active_count": int(support.sum()),
    }


def calibrate_deep_threshold(rows: list[dict]) -> float:
    """Pick the conservative best BA threshold, macroing scenario x SNR when known."""
    use_macro = bool(rows) and all(
        row.get("scenario") not in (None, "") and row.get("snr_db") not in (None, "")
        for row in rows
    )
    groups = {}
    if use_macro:
        for row in rows:
            groups.setdefault((str(row["scenario"]), str(row["snr_db"])), []).append(row)

    def rates(selected: list[dict], threshold: float) -> tuple[float, float]:
        positives = [row for row in selected if int(row["has_deep_true"])]
        negatives = [row for row in selected if not int(row["has_deep_true"])]
        sensitivity = (
            float(
                np.mean(
                    [
                        float(row["deep_score"]) >= threshold
                        and float(row["deep_peak_distance_mm"]) <= 10.0 + 1e-6
                        for row in positives
                    ]
                )
            )
            if positives
            else np.nan
        )
        specificity = (
            float(np.mean([float(row["deep_score"]) < threshold for row in negatives]))
            if negatives
            else np.nan
        )
        return sensitivity, specificity

    best = (-np.inf, 0.0)
    for threshold in DEEP_THRESHOLD_CANDIDATES:
        if use_macro:
            cell_rates = [rates(group, threshold) for group in groups.values()]
            sensitivities = [value[0] for value in cell_rates if np.isfinite(value[0])]
            specificities = [value[1] for value in cell_rates if np.isfinite(value[1])]
            sensitivity = float(np.mean(sensitivities)) if sensitivities else np.nan
            specificity = float(np.mean(specificities)) if specificities else np.nan
        else:
            sensitivity, specificity = rates(rows, threshold)
        score = (
            float((sensitivity + specificity) / 2.0)
            if np.isfinite(sensitivity) and np.isfinite(specificity)
            else -np.inf
        )
        best = max(best, (score, threshold))
    return best[1]
