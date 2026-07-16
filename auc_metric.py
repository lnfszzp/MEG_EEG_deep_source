from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


AUC_THRESHOLD = 0.01
AUC_METRICS_DIR = Path(r"F:\PycharmProjects\meg\function")


def _as_source_matrix(source: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    return source[:, None] if source.ndim == 1 else source


def _install_common() -> None:
    common = types.ModuleType("_common")
    common.as_source_matrix = _as_source_matrix
    common.field = lambda obj, name: obj[name] if isinstance(obj, dict) else getattr(obj, name)

    def matlab_indices(indices: np.ndarray, n: int) -> np.ndarray:
        idx = np.asarray(indices, dtype=int).ravel()
        if idx.size and idx.min() >= 1 and idx.max() <= n:
            idx = idx - 1
        return idx

    def normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        return values / peak if peak > 0 else np.zeros_like(values)

    def mesh_adjacency(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        vertices = np.asarray(vertices)
        if sparse.issparse(faces):
            return faces.tocsr()

        faces = np.asarray(faces)
        if faces.shape == (vertices.shape[0], vertices.shape[0]):
            return sparse.csr_matrix(faces)

        adjacency = sparse.lil_matrix((vertices.shape[0], vertices.shape[0]), dtype=np.uint8)
        face_idx = faces.astype(int)
        if face_idx.size and face_idx.min() == 1:
            face_idx = face_idx - 1
        for tri in face_idx[:, :3]:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                adjacency[a, b] = 1
                adjacency[b, a] = 1
        return adjacency.tocsr()

    def groups_from_cell(groups, n: int) -> list[np.ndarray]:
        if isinstance(groups, np.ndarray) and groups.dtype == object:
            groups = groups.ravel()
        elif not isinstance(groups, (list, tuple)):
            groups = [groups]
        return [matlab_indices(group, n) for group in groups]

    common.matlab_indices = matlab_indices
    common.mesh_adjacency = mesh_adjacency
    common.normalize = normalize
    common.optional_nested_index = lambda *_args, **_kwargs: np.array([], dtype=int)
    common.groups_from_cell = groups_from_cell
    sys.modules["_common"] = common


def _safe_grow_parcels(A, seeds, banned, q_len, count=50):
    """Run the requested parcel growth on source blocks smaller than 50 seeds."""
    seeds = list(seeds)
    if not seeds:
        raise ValueError("An_cal_AUC has no parcel seed in this source block")
    parcels = []
    for index in range(count):
        cursor = index
        parcel = np.array([seeds[cursor % len(seeds)]], dtype=int)
        stalled = 0
        while len(parcel) <= q_len:
            adjacent = np.unique(A[parcel].nonzero()[1])
            allowed = np.setdiff1d(adjacent, banned, assume_unique=False)
            grown = np.unique(np.r_[parcel, allowed])
            if allowed.size and grown.size > parcel.size:
                parcel = grown
                stalled = 0
                continue
            cursor += 1
            stalled += 1
            if stalled >= len(seeds):
                raise ValueError("An_cal_AUC cannot grow a parcel inside this source block")
            parcel = np.array([seeds[cursor % len(seeds)]], dtype=int)
        parcels.append(parcel)
    return parcels


def _load_an_cal_auc_module():
    _install_common()
    if str(AUC_METRICS_DIR) not in sys.path:
        sys.path.append(str(AUC_METRICS_DIR))
    spec = importlib.util.spec_from_file_location("requested_An_cal_AUC", AUC_METRICS_DIR / "An_cal_AUC.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {AUC_METRICS_DIR / 'An_cal_AUC.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._grow_parcels = _safe_grow_parcels
    return module


_an_cal_auc_module = _load_an_cal_auc_module()
An_cal_AUC = _an_cal_auc_module.An_cal_AUC


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
