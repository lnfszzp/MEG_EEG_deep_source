import numpy as np
from scipy import sparse


def as_source_matrix(value):
    value = np.asarray(value, dtype=float)
    return value.reshape(-1, 1) if value.ndim == 1 else value


def explicit_indices(indices, n, index_base):
    """Convert indices without guessing whether Python or MATLAB supplied them."""
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 (Python) or 1 (MATLAB)")
    result = np.asarray(indices, dtype=int).ravel() - index_base
    if result.size and (result.min() < 0 or result.max() >= n):
        raise IndexError("source index is outside the position array")
    return result


def groups_from_cell(groups, n, index_base):
    if isinstance(groups, np.ndarray) and groups.dtype == object:
        groups = groups.ravel()
    elif not isinstance(groups, (list, tuple)):
        groups = [groups]
    return [explicit_indices(group, n, index_base) for group in groups]


def _canonical_matlab_indices(indices, n):
    """Retain the canonical cortex/pos_lausanne MATLAB import convention."""
    result = np.asarray(indices, dtype=int).ravel()
    if result.size and result.min() >= 1 and result.max() <= n:
        result = result - 1
    return result


def field(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def optional_nested_index(obj, *names, n):
    current = obj
    for name in names:
        if current is None:
            return np.array([], dtype=int)
        current = current.get(name) if isinstance(current, dict) else getattr(current, name, None)
    return np.array([], dtype=int) if current is None else _canonical_matlab_indices(current, n)


def mesh_adjacency(vertices, faces):
    n = len(vertices)
    if sparse.issparse(faces):
        if faces.shape != (n, n):
            raise ValueError("sparse adjacency must have shape n_vertices x n_vertices")
        return faces.tocsr()

    faces = np.asarray(faces)
    if faces.shape == (n, n):
        return sparse.csr_matrix(faces)
    faces = _canonical_matlab_indices(faces, n).reshape(-1, 3)
    rows = np.r_[
        faces[:, 0], faces[:, 1], faces[:, 2],
        faces[:, 1], faces[:, 2], faces[:, 0],
        np.arange(n),
    ]
    cols = np.r_[
        faces[:, 1], faces[:, 2], faces[:, 0],
        faces[:, 0], faces[:, 1], faces[:, 2],
        np.arange(n),
    ]
    return sparse.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))


def normalize(values):
    values = np.asarray(values, dtype=float)
    maximum = np.max(values) if values.size else 0
    return values / maximum if maximum else np.zeros_like(values, dtype=float)
