from __future__ import annotations

import numpy as np


def _sample_pairs(
    left: np.ndarray,
    right: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    pairs = [(int(a), int(b)) for a in left for b in right]
    if len(pairs) <= int(count):
        return pairs
    picks = rng.choice(len(pairs), size=int(count), replace=False)
    return [pairs[int(i)] for i in picks]


def stratified_source_jobs(
    surface_indices: np.ndarray,
    deep_indices: np.ndarray,
    *,
    deep_surface_count: int = 64,
    deep_two_surface_count: int = 64,
    random_state: int = 20260706,
) -> list[dict]:
    """Return deterministic full-head simulation jobs without enumerating all combos."""
    surface = np.asarray(surface_indices, dtype=int).ravel()
    deep = np.asarray(deep_indices, dtype=int).ravel()
    rng = np.random.default_rng(int(random_state))
    jobs: list[dict] = []

    for idx in surface:
        jobs.append(
            {
                "scenario": "surface_only",
                "surface_indices0": np.array([int(idx)], dtype=int),
                "deep_indices0": np.array([], dtype=int),
            }
        )
    for idx in deep:
        jobs.append(
            {
                "scenario": "deep_only",
                "surface_indices0": np.array([], dtype=int),
                "deep_indices0": np.array([int(idx)], dtype=int),
            }
        )
    for d_idx, s_idx in _sample_pairs(deep, surface, deep_surface_count, rng):
        jobs.append(
            {
                "scenario": "deep_plus_surface",
                "surface_indices0": np.array([s_idx], dtype=int),
                "deep_indices0": np.array([d_idx], dtype=int),
            }
        )

    surface_pairs = [
        (int(surface[i]), int(surface[j]))
        for i in range(surface.size)
        for j in range(i + 1, surface.size)
    ]
    if surface_pairs and deep.size:
        all_triples = [(int(d), a, b) for d in deep for a, b in surface_pairs]
        if len(all_triples) > int(deep_two_surface_count):
            picks = rng.choice(
                len(all_triples),
                size=int(deep_two_surface_count),
                replace=False,
            )
            all_triples = [all_triples[int(i)] for i in picks]
        for d_idx, s1, s2 in all_triples:
            jobs.append(
                {
                    "scenario": "deep_plus_two_surface",
                    "surface_indices0": np.array([s1, s2], dtype=int),
                    "deep_indices0": np.array([d_idx], dtype=int),
                }
            )
    return jobs
