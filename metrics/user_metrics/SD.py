import numpy as np
from scipy.spatial import cKDTree

from ._common import as_source_matrix


def SD(S, S0, pos, alpha):
    """Canonical spatial dispersion; distance units follow ``pos``/``S0``."""
    source = as_source_matrix(S)
    true_positions = np.asarray(S0, dtype=float)
    positions = np.asarray(pos, dtype=float)
    energy = np.sum(source * source, axis=1)
    energy = energy * (energy > alpha * np.max(energy))
    distances, _ = cKDTree(true_positions).query(positions)
    return np.sqrt(np.sum((distances * distances) * energy) / np.sum(energy))
