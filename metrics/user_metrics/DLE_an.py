import numpy as np
from scipy.spatial import cKDTree

from ._common import as_source_matrix, groups_from_cell


def DLE_an(S, S0, pos, *, index_base):
    """Canonical DLE with a required, explicit source-index base.

    Pass ``index_base=0`` for Python indices or ``index_base=1`` for MATLAB
    indices.  No content-based guessing is performed, so a valid zero-based
    group that happens not to contain vertex zero is never shifted.
    """
    source = as_source_matrix(S)
    positions = np.asarray(pos, dtype=float)
    energy = np.sum(source * source, axis=1)
    groups = groups_from_cell(S0, len(positions), index_base)

    centers = np.array([np.mean(positions[group], axis=0) for group in groups])
    active = np.concatenate(groups)
    owner = np.concatenate([np.full(len(group), index) for index, group in enumerate(groups)])
    _, nearest = cKDTree(positions[active]).query(positions)
    vertex_owner = owner[nearest]

    distances = []
    for index, center in enumerate(centers):
        vertices = np.flatnonzero(vertex_owner == index)
        best = vertices[np.argmax(energy[vertices])]
        distances.append(np.linalg.norm(center - positions[best]))
    return float(np.mean(distances))
