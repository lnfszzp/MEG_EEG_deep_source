from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def surface_protection_mask(
    probability: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    n_surf: int,
    *,
    probability_threshold: float = 0.50,
    expansion_hops: int = 0,
) -> np.ndarray:
    """Build a surface-only protection mask from stable MEG detections."""
    probability = np.asarray(probability, dtype=float).ravel()
    graph = sparse.csr_matrix(adjacency)
    if graph.shape != (probability.size, probability.size):
        raise ValueError("adjacency must match probability size")
    surface = np.zeros(probability.size, dtype=bool)
    surface[: int(n_surf)] = True
    protected = surface & (probability >= float(probability_threshold))
    frontier = protected.copy()
    for _ in range(int(expansion_hops)):
        neighbors = np.asarray(graph @ frontier.astype(float)).ravel() > 0
        neighbors &= surface
        new_frontier = neighbors & ~protected
        if not np.any(new_frontier):
            break
        protected |= new_frontier
        frontier = new_frontier
    return protected


def protected_lambda_weights(
    n_sources: int,
    n_surf: int,
    surface_protected: np.ndarray,
    deep_candidates: np.ndarray,
    *,
    protected_surface_scale: float = 0.65,
    outside_surface_scale: float = 1.35,
    deep_scale: float = 1.00,
    other_scale: float = 1.10,
) -> np.ndarray:
    """Assign source-wise sparse-penalty weights for protected fusion."""
    surface_protected = np.asarray(surface_protected, dtype=bool).ravel()
    deep_candidates = np.asarray(deep_candidates, dtype=bool).ravel()
    if surface_protected.size != n_sources or deep_candidates.size != n_sources:
        raise ValueError("masks must match n_sources")
    surface = np.zeros(n_sources, dtype=bool)
    surface[: int(n_surf)] = True
    weights = np.full(n_sources, float(other_scale), dtype=float)
    weights[surface] = float(outside_surface_scale)
    weights[surface_protected & surface] = float(protected_surface_scale)
    weights[deep_candidates & ~surface] = float(deep_scale)
    return weights


def protected_candidate_mask(
    surface_protected: np.ndarray,
    joint_region: np.ndarray,
    joint_stability: np.ndarray,
    n_surf: int,
    *,
    joint_surface_probability_threshold: float = 0.75,
) -> np.ndarray:
    """Combine MEG-protected surface support and stable joint deep support."""
    surface_protected = np.asarray(surface_protected, dtype=bool).ravel()
    joint_region = np.asarray(joint_region, dtype=bool).ravel()
    joint_stability = np.asarray(joint_stability, dtype=float).ravel()
    if not (
        surface_protected.shape == joint_region.shape == joint_stability.shape
    ):
        raise ValueError("all masks must align")
    surface = np.zeros(surface_protected.size, dtype=bool)
    surface[: int(n_surf)] = True
    deep = ~surface
    high_confidence_joint_surface = (
        surface
        & joint_region
        & (joint_stability >= float(joint_surface_probability_threshold))
    )
    joint_deep = deep & joint_region
    return surface_protected | high_confidence_joint_surface | joint_deep


def prune_surface_components(
    source: np.ndarray,
    region_mask: np.ndarray,
    adjacency: np.ndarray | sparse.spmatrix,
    n_surf: int,
    *,
    min_energy_fraction: float = 0.15,
) -> np.ndarray:
    """Drop weak disconnected surface fragments while leaving deep sources unchanged."""
    source = np.asarray(source, dtype=float)
    region = np.asarray(region_mask, dtype=bool).ravel().copy()
    graph = sparse.csr_matrix(adjacency)
    n_surf = int(n_surf)
    if region.size != graph.shape[0]:
        raise ValueError("region_mask must match adjacency size")
    amplitude = np.abs(source) if source.ndim == 1 else np.linalg.norm(source, axis=1)
    selected = np.flatnonzero(region[:n_surf])
    if selected.size == 0:
        return region
    count, labels = connected_components(
        graph[:n_surf, :n_surf][selected][:, selected],
        directed=False,
    )
    energies = []
    members_by_label = []
    for label in range(count):
        members = selected[labels == label]
        members_by_label.append(members)
        energies.append(float(np.sum(amplitude[members] ** 2)))
    maximum = max(energies, default=0.0)
    if maximum <= 0:
        return region
    for members, energy in zip(members_by_label, energies):
        if energy < float(min_energy_fraction) * maximum:
            region[members] = False
    return region
