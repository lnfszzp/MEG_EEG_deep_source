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


def _load_an_cal_auc():
    _install_common()
    if str(AUC_METRICS_DIR) not in sys.path:
        sys.path.append(str(AUC_METRICS_DIR))
    spec = importlib.util.spec_from_file_location("requested_An_cal_AUC", AUC_METRICS_DIR / "An_cal_AUC.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {AUC_METRICS_DIR / 'An_cal_AUC.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.An_cal_AUC


An_cal_AUC = _load_an_cal_auc()


def auc_cortex(vertices_m: np.ndarray, vert_conn: np.ndarray, n_surf: int) -> dict:
    vertices = np.asarray(vertices_m, dtype=float)
    adjacency = sparse.lil_matrix(np.asarray(vert_conn) != 0, dtype=np.uint8)

    # An_cal_AUC assumes a locally connected mesh; fill source-space holes
    # left by forward-model mindist filtering with geometric surface neighbors.
    if n_surf > 1:
        neighbors = cKDTree(vertices[:n_surf]).query(vertices[:n_surf], k=min(7, n_surf))[1]
        for idx, local in enumerate(neighbors):
            adjacency[idx, local] = 1
            adjacency[local, idx] = 1

    # Deep points are not part of the cortical mesh. Give only deep rows a
    # directed kNN escape route so An_cal_AUC can form close/far parcels.
    for idx in range(int(n_surf), vertices.shape[0]):
        dist = np.linalg.norm(vertices - vertices[idx], axis=1)
        adjacency[idx, np.argsort(dist)[1:21]] = 1
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
