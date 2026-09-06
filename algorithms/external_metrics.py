from __future__ import annotations

import numpy as np

from metrics.user_metrics import An_auc, DLE_an, RMSE, SD


def _as_source_matrix(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    return source[:, None] if source.ndim == 1 else source


def external_full_head_metrics(
    estimated_source: np.ndarray,
    truth_source: np.ndarray,
    positions_m: np.ndarray,
    *,
    true_groups: list[np.ndarray] | None = None,
    alpha: float = 0.0,
    n_surf: int | None = None,
) -> dict:
    """Evaluate a whole map and optionally split it at ``n_surf``.

    ``true_groups`` always contains zero-based indices into the full source
    axis; layer-local conversion is performed here before DLE is evaluated.
    """
    estimated = _as_source_matrix(estimated_source)
    truth = _as_source_matrix(truth_source)
    positions = np.asarray(positions_m, dtype=float)
    if estimated.shape != truth.shape:
        raise ValueError("estimated_source and truth_source must align")
    if positions.shape != (estimated.shape[0], 3):
        raise ValueError("positions_m must have shape n_sources x 3")

    truth_energy = np.sum(truth * truth, axis=1)
    active = np.flatnonzero(truth_energy > 0)
    if active.size == 0:
        result = {"auc": np.nan, "rmse": np.nan, "sd_mm": np.nan, "dle_mm": np.nan}
    else:
        groups = [active] if true_groups is None else [np.asarray(group, dtype=int).ravel() for group in true_groups]
        estimated_energy = np.sum(estimated * estimated, axis=1)
        estimated_energy *= estimated_energy > alpha * np.max(estimated_energy)
        labels = truth_energy > 0
        auc = An_auc(np.c_[labels, estimated_energy]) if np.any(~labels) else np.nan
        result = {
            "auc": float(auc),
            "rmse": float(RMSE(estimated, truth)),
            "sd_mm": float(SD(estimated, positions[active], positions, alpha) * 1000.0),
            "dle_mm": float(DLE_an(estimated, groups, positions, index_base=0) * 1000.0),
        }

    if n_surf is None:
        return result

    n_surf = int(n_surf)
    if not 0 <= n_surf <= estimated.shape[0]:
        raise ValueError("n_surf must lie within the source axis")
    groups = [active] if true_groups is None else [np.asarray(group, dtype=int).ravel() for group in true_groups]
    for layer, start, stop in (
        ("surface", 0, n_surf),
        ("deep", n_surf, estimated.shape[0]),
    ):
        local_groups = []
        for group in groups:
            local = group[(group >= start) & (group < stop)] - start
            if local.size:
                local_groups.append(local)
        layer_metrics = external_full_head_metrics(
            estimated[start:stop],
            truth[start:stop],
            positions[start:stop],
            true_groups=local_groups or None,
            alpha=alpha,
        )
        result.update({f"{layer}_{name}": value for name, value in layer_metrics.items()})
    return result
