from __future__ import annotations

import numpy as np


def source_amplitude(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    return np.abs(source) if source.ndim == 1 else np.linalg.norm(source, axis=1)


def thresholded_source(source: np.ndarray, relative_threshold: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    amplitude = source_amplitude(source)
    maximum = float(amplitude.max(initial=0.0))
    mask = amplitude >= float(relative_threshold) * maximum if maximum > 0 else np.zeros(amplitude.size, dtype=bool)
    return np.asarray(source, dtype=float) * mask[:, None], mask


def true_source_groups(truth: dict) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    surface_indices = np.asarray(truth["true_surface_indices0"], dtype=int).ravel()
    labels = np.asarray(truth.get("true_surface_patch_labels", np.zeros((1, 0))), dtype=int).ravel()
    if surface_indices.size:
        if labels.size == surface_indices.size and np.any(labels):
            for label in sorted(set(labels.tolist()) - {0}):
                groups.append(surface_indices[labels == label])
        else:
            groups.append(surface_indices)
    has_deep = bool(int(np.asarray(truth.get("has_deep_source", [[1]])).ravel()[0]))
    if has_deep:
        groups.append(np.array([int(np.asarray(truth["true_deep_idx0"]).ravel()[0])], dtype=int))
    return groups


def group_waveform(source: np.ndarray, indices: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    indices = np.asarray(indices, dtype=int).ravel()
    if indices.size == 1:
        return source[int(indices[0])]
    weights = source_amplitude(source[indices])
    if float(weights.sum()) <= 0:
        return np.mean(source[indices], axis=0)
    return np.average(source[indices], axis=0, weights=weights)


def best_estimated_waveform(estimated: np.ndarray, truth_group: np.ndarray) -> tuple[int, np.ndarray]:
    estimated = np.asarray(estimated, dtype=float)
    group = np.asarray(truth_group, dtype=int).ravel()
    amplitude = source_amplitude(estimated)
    if group.size == 0:
        peak = int(np.argmax(amplitude))
    else:
        peak = int(group[np.argmax(amplitude[group])])
        if amplitude[peak] <= 0:
            peak = int(np.argmax(amplitude))
    return peak, estimated[peak]
