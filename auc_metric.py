from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from metrics.user_metrics import An_cal_AUC
from metrics.user_metrics.An_cal_AUC import _grow_parcels as _safe_grow_parcels

AUC_THRESHOLD = 0.01
AUC_METRICS_DIR = Path(__file__).resolve().parent / "metrics" / "user_metrics"


def _as_source_matrix(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    return source[:, None] if source.ndim == 1 else source


def auc_cortex(vertices_m: np.ndarray, vert_conn: np.ndarray, n_surf: int) -> dict:
    vertices = np.asarray(vertices_m, dtype=float)
    adjacency = sparse.lil_matrix(np.asarray(vert_conn) != 0, dtype=np.uint8)
    adjacency[:n_surf, n_surf:] = 0
    adjacency[n_surf:, :n_surf] = 0

    # An_cal_AUC assumes a locally connected mesh; fill source-space holes
    # left by forward-model mindist filtering with geometric surface neighbors.
    if n_surf > 1:
        neighbors = cKDTree(vertices[:n_surf]).query(vertices[:n_surf], k=min(7, n_surf))[1]
        for idx, local in enumerate(neighbors):
            adjacency[idx, local] = 1
            adjacency[local, idx] = 1

    n_deep = vertices.shape[0] - int(n_surf)
    if n_deep > 1 and not adjacency[n_surf:, n_surf:].nnz:
        neighbors = cKDTree(vertices[n_surf:]).query(vertices[n_surf:], k=min(7, n_deep))[1]
        for local_idx, local in enumerate(neighbors):
            index = n_surf + local_idx
            deep_neighbors = n_surf + np.asarray(local, dtype=int)
            adjacency[index, deep_neighbors] = 1
            adjacency[deep_neighbors, index] = 1
    return {"Vertices": vertices, "Faces": adjacency.tocsr()}


def an_auc_from_cortex(
    truth: np.ndarray,
    estimated: np.ndarray,
    cortex: dict,
    true_groups: list[np.ndarray] | None = None,
) -> float:
    if not true_groups or len(true_groups) == 1:
        return float(An_cal_AUC(truth, estimated, cortex, threshold=AUC_THRESHOLD))

    truth = _as_source_matrix(truth)
    estimated = _as_source_matrix(estimated)
    vertices = np.asarray(cortex["Vertices"], dtype=float)
    active = np.concatenate([np.asarray(group, dtype=int).ravel() for group in true_groups])
    labels = np.concatenate([np.full(np.asarray(group).size, index) for index, group in enumerate(true_groups)])
    owner = labels[cKDTree(vertices[active]).query(vertices)[1]]
    values = []
    for index, group in enumerate(true_groups):
        group_truth = np.zeros_like(truth)
        group_truth[np.asarray(group, dtype=int)] = truth[np.asarray(group, dtype=int)]
        group_estimated = estimated.copy()
        group_estimated[owner != index] = 0.0
        values.append(An_cal_AUC(group_truth, group_estimated, cortex, threshold=AUC_THRESHOLD))
    return float(np.mean(values))


def an_auc(
    truth: np.ndarray,
    estimated: np.ndarray,
    vertices_m: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    true_groups: list[np.ndarray] | None = None,
) -> float:
    return an_auc_from_cortex(
        truth,
        estimated,
        auc_cortex(vertices_m, vert_conn, n_surf),
        true_groups,
    )
