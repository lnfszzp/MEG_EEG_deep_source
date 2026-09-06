import numpy as np

from ._common import as_source_matrix, field, mesh_adjacency, normalize, optional_nested_index
from .An_auc import An_auc


def _neighbors(adjacency, indices):
    return np.unique(adjacency[indices].nonzero()[1])


def _grow_parcels(adjacency, seeds, banned, source_count, count=50):
    """Grow deterministic parcels, cycling small seed sets and failing cleanly."""
    parcels = []
    seeds = list(seeds)
    if not seeds:
        raise ValueError("An_cal_AUC has no parcel seed in this source block")
    for index in range(count):
        for offset in range(len(seeds)):
            parcel = np.array([seeds[(index + offset) % len(seeds)]], dtype=int)
            while len(parcel) <= source_count:
                allowed = np.setdiff1d(
                    _neighbors(adjacency, parcel), banned, assume_unique=False
                )
                grown = np.unique(np.r_[parcel, allowed])
                if grown.size == parcel.size:
                    break
                parcel = grown
            if len(parcel) > source_count:
                parcels.append(parcel)
                break
        else:
            raise ValueError("An_cal_AUC cannot grow a parcel inside this source block")
    return parcels


def _top_or_random(values, indices, count, rng):
    indices = np.asarray(indices, dtype=int)
    if np.any(values[indices]):
        return indices[np.argsort(values[indices])[-count:][::-1]]
    return rng.choice(indices, size=count, replace=False)


def An_cal_AUC(J_simu, J_est, cortex, pos_lausanne=None, threshold=None):
    J_simu = as_source_matrix(J_simu)
    J_est = as_source_matrix(J_est)

    estimated_energy = np.sqrt(np.sum(J_est**2, axis=1))
    if threshold is not None:
        estimated_energy[estimated_energy < np.max(estimated_energy) * threshold] = 0
    estimated_energy = normalize(estimated_energy)
    simulated_energy = normalize(np.sqrt(np.sum(J_simu**2, axis=1)))

    vertices = np.asarray(field(cortex, "Vertices"), dtype=float) * 1000
    adjacency = mesh_adjacency(vertices, field(cortex, "Faces"))

    true_indices = np.flatnonzero(np.sum(np.abs(J_simu), axis=1))
    near = true_indices
    for _ in range(10):
        near = _neighbors(adjacency, near)

    unknown = optional_nested_index(pos_lausanne, "unkown", "index", n=J_simu.shape[0])
    near_candidates = np.setdiff1d(np.setdiff1d(near, true_indices), unknown)
    far_candidates = np.setdiff1d(
        np.setdiff1d(np.arange(J_simu.shape[0]), near), unknown
    )

    rng = np.random.default_rng(0)
    close_parcels = _grow_parcels(
        adjacency,
        rng.permutation(near_candidates),
        np.r_[true_indices, far_candidates],
        len(true_indices),
    )
    far_parcels = _grow_parcels(
        adjacency,
        rng.permutation(far_candidates),
        near,
        len(true_indices),
    )

    close_auc = []
    far_auc = []
    for close, far in zip(close_parcels, far_parcels):
        close_indices = _top_or_random(estimated_energy, close, len(true_indices), rng)
        far_indices = _top_or_random(estimated_energy, far, len(true_indices), rng)
        close_sample = np.r_[close_indices, true_indices]
        far_sample = np.r_[far_indices, true_indices]
        close_auc.append(
            An_auc(np.c_[simulated_energy[close_sample] > 0, estimated_energy[close_sample]], 0.05, "hanley")
        )
        far_auc.append(
            An_auc(np.c_[simulated_energy[far_sample] > 0, estimated_energy[far_sample]], 0.05, "hanley")
        )
    return float(np.mean((np.asarray(close_auc) + np.asarray(far_auc)) / 2))
