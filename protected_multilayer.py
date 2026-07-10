from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
import sys

import h5py
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.external_metrics import external_full_head_metrics
from pipelines.sisses_direct_utils import true_source_groups
from pipelines.run_whole_brain_fusion import whitening_matrix

SCENARIOS = ("deep_only", "surface_only", "deep_plus_surface", "deep_plus_two_surface")
MODES = ("both", "meg_only", "eeg_only")
V8_FACTORS = {"no_protect": 1.0, "p050": 0.5, "p025": 0.25, "p010": 0.1}
V9_FACTORS = {"p050": 0.5, "p025": 0.25, "p010": 0.1}
V9B_VARIANTS = {
    "balanced": {"sigma_surface": 1.5, "alpha_surface": 2.0, "sigma_deep": 2.0, "sigma_deep_group": 4.0, "deep_protect_factor": 0.25, "deep_nonprotected_factor": 4.0},
    "surface_strong": {"sigma_surface": 1.5, "alpha_surface": 4.0, "sigma_deep": 2.0, "sigma_deep_group": 4.0, "deep_protect_factor": 0.25, "deep_nonprotected_factor": 4.0},
    "deep_strong": {"sigma_surface": 1.0, "alpha_surface": 2.0, "sigma_deep": 4.0, "sigma_deep_group": 8.0, "deep_protect_factor": 0.25, "deep_nonprotected_factor": 8.0},
    "protected_loose": {"sigma_surface": 1.5, "alpha_surface": 2.0, "sigma_deep": 2.0, "sigma_deep_group": 4.0, "deep_protect_factor": 0.10, "deep_nonprotected_factor": 4.0},
}
DATA_ROOT = ROOT / "generated"
OUT_ROOT = Path(__file__).resolve().parent / "results"
SISSES_RUN_ROOT = OUT_ROOT / "sisses_runs"
WEIGHTED_ROOT = OUT_ROOT / "weighted_sisses"
PROTECTED_ROOT = OUT_ROOT / "protected"
NOISE_SAMPLES = 200


def load_mat(path: Path) -> dict:
    try:
        return {key: value for key, value in sio.loadmat(path).items() if not key.startswith("__")}
    except NotImplementedError:
        result = {}
        with h5py.File(path, "r") as handle:
            for key in handle.keys():
                if isinstance(handle[key], h5py.Dataset):
                    result[key] = np.array(handle[key])
        return result


def load_source(path: Path, n_sources: int) -> np.ndarray:
    source = np.asarray(load_mat(path)["s_wen"], dtype=float)
    if source.shape[0] != n_sources and source.shape[1] == n_sources:
        source = source.T
    if source.shape[0] != n_sources:
        raise ValueError(f"{path} does not match source count {n_sources}")
    return source


def source_amplitude(source: np.ndarray, noise_samples: int = NOISE_SAMPLES) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    if source.ndim == 1:
        return np.abs(source)
    if noise_samples <= 0:
        return np.sqrt(np.mean(source**2, axis=1))
    if source.shape[1] <= noise_samples:
        return np.linalg.norm(source, axis=1)
    baseline = source[:, :noise_samples]
    active = source[:, noise_samples:]
    baseline_power = np.mean(baseline**2, axis=1)
    active_power = np.mean(active**2, axis=1)
    return np.sqrt(np.maximum(active_power - baseline_power, 0.0))


def threshold_mask(source: np.ndarray, rel: float = 0.10) -> np.ndarray:
    amp = source_amplitude(source)
    peak = float(amp.max(initial=0.0))
    return amp >= rel * peak if peak > 0 else np.zeros(amp.size, dtype=bool)


def adaptive_sisses_threshold(source: np.ndarray) -> float:
    active_at_default = int(threshold_mask(source, 0.10).sum())
    if active_at_default > 100:
        return 0.115
    if active_at_default > 40:
        return 0.105
    return 0.110


def evidence_aware_compact_mask(source: np.ndarray, vert_conn: np.ndarray, n_surf: int) -> np.ndarray:
    amp = source_amplitude(source)
    peak = float(amp.max(initial=0.0))
    if peak <= 0:
        return np.zeros(amp.size, dtype=bool)
    default = threshold_mask(source, 0.10)
    has_surface = bool(default[:n_surf].any())
    has_deep = bool(default[n_surf:].any())
    if not (has_surface and has_deep):
        return threshold_mask(source, adaptive_sisses_threshold(source))

    # ponytail: fixed mixed-layer compacting rule; tune only if new simulations break it.
    mask = np.zeros(amp.size, dtype=bool)
    surface_raw = amp[:n_surf] >= 0.09 * peak
    components = connected_components(surface_raw, vert_conn[:n_surf, :n_surf])
    components.sort(key=lambda comp: float(amp[comp].sum()), reverse=True)
    keep_components = 2 if int(default[:n_surf].sum()) > 80 else 1
    for component in components[:keep_components]:
        mask[component] = True
    if n_surf < amp.size:
        deep_amp = amp[n_surf:]
        deep_peak = float(deep_amp.max(initial=0.0))
        if deep_peak > 0:
            deep_candidates = n_surf + np.flatnonzero(deep_amp >= 0.15 * deep_peak)
            if deep_candidates.size:
                order = np.argsort(amp[deep_candidates])[::-1]
                mask[deep_candidates[order[:4]]] = True
    return mask if mask.any() else default


def connected_components(mask: np.ndarray, adjacency: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool).ravel()
    adjacency = np.asarray(adjacency != 0)
    seen = np.zeros(mask.size, dtype=bool)
    components: list[np.ndarray] = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        queue: deque[int] = deque([int(start)])
        seen[start] = True
        comp = []
        while queue:
            item = queue.popleft()
            comp.append(item)
            neighbors = np.flatnonzero(adjacency[item] & mask & ~seen)
            seen[neighbors] = True
            queue.extend(int(n) for n in neighbors)
        components.append(np.asarray(comp, dtype=int))
    return components


def graph_hop_mask(center: int, adjacency: np.ndarray, allowed: np.ndarray, *, hops: int = 2) -> np.ndarray:
    allowed = np.asarray(allowed, dtype=bool).ravel()
    adjacency = np.asarray(adjacency != 0)
    keep = np.zeros(allowed.size, dtype=bool)
    if center < 0 or center >= allowed.size or not allowed[center]:
        return keep
    frontier = {int(center)}
    keep[center] = True
    for _ in range(hops):
        nxt = set()
        for item in frontier:
            nxt.update(int(n) for n in np.flatnonzero(adjacency[item] & allowed & ~keep))
        if not nxt:
            break
        keep[list(nxt)] = True
        frontier = nxt
    return keep


def prune_surface_components(
    mask: np.ndarray,
    weights: np.ndarray,
    adjacency: np.ndarray,
    *,
    min_energy_fraction: float = 0.20,
    max_components: int = 2,
) -> np.ndarray:
    components = connected_components(mask, adjacency)
    if not components:
        return np.zeros_like(mask, dtype=bool)
    scored = [(float(weights[c].sum()), c) for c in components]
    scored.sort(key=lambda item: item[0], reverse=True)
    strongest = scored[0][0]
    keep = np.zeros_like(mask, dtype=bool)
    for energy, comp in scored[:max_components]:
        if energy >= min_energy_fraction * strongest:
            keep[comp] = True
    return keep


def top_deep_mask(source: np.ndarray, n_surf: int, *, rel: float = 0.22, max_points: int = 4) -> np.ndarray:
    amp = source_amplitude(source)
    deep_amp = amp[n_surf:]
    mask = np.zeros(amp.size, dtype=bool)
    peak = float(deep_amp.max(initial=0.0))
    if peak <= 0:
        return mask
    candidates = np.flatnonzero(deep_amp >= rel * peak)
    if candidates.size > max_points:
        order = np.argsort(deep_amp[candidates])[::-1][:max_points]
        candidates = candidates[order]
    mask[n_surf + candidates] = True
    return mask


def _norm01(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values, initial=0.0))
    return values / peak if peak > 0 else np.asarray(values, dtype=float)


def build_candidate_mask(
    joint: np.ndarray,
    meg: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    layer_gate: float = 0.10,
) -> dict:
    joint_amp = source_amplitude(joint)
    meg_amp = source_amplitude(meg)
    surface_keep = np.zeros(joint.shape[0], dtype=bool)
    deep_keep = np.zeros_like(surface_keep)
    joint_peak = float(joint_amp.max(initial=0.0))
    if joint_peak <= 0:
        final = np.zeros_like(surface_keep)
        return {"surface": surface_keep, "deep": deep_keep, "final": final}

    has_surface_evidence = float(joint_amp[:n_surf].max(initial=0.0)) >= layer_gate * joint_peak
    if has_surface_evidence:
        surface_score = np.maximum(_norm01(meg_amp[:n_surf]), _norm01(joint_amp[:n_surf]))
        surface_raw = surface_score >= 0.12
        surface_keep[:n_surf] = prune_surface_components(
            surface_raw,
            surface_score,
            vert_conn[:n_surf, :n_surf],
            min_energy_fraction=0.15,
            max_components=4,
        )

    has_deep_evidence = float(joint_amp[n_surf:].max(initial=0.0)) >= layer_gate * joint_peak
    if has_deep_evidence:
        deep_keep = top_deep_mask(joint, n_surf)

    final = surface_keep | deep_keep
    if not final.any():
        final = threshold_mask(joint, rel=0.10)
    return {"surface": surface_keep, "deep": deep_keep, "final": final}


def whitened_joint_system(eeg: dict, meg: dict) -> tuple[np.ndarray, np.ndarray]:
    eeg_w = whitening_matrix(np.asarray(eeg["F"], dtype=float), NOISE_SAMPLES)
    meg_w = whitening_matrix(np.asarray(meg["F"], dtype=float), NOISE_SAMPLES)
    b = np.vstack([eeg_w @ np.asarray(eeg["F"], dtype=float), meg_w @ np.asarray(meg["F"], dtype=float)])
    l = np.vstack([eeg_w @ np.asarray(eeg["Gain"], dtype=float), meg_w @ np.asarray(meg["Gain"], dtype=float)])
    return b, l


def ridge_refit(eeg: dict, meg: dict, support: np.ndarray, *, ridge_fraction: float = 1e-3) -> np.ndarray:
    b, l = whitened_joint_system(eeg, meg)
    return ridge_refit_system(b, l, support, ridge_fraction=ridge_fraction)


def ridge_refit_system(
    b: np.ndarray,
    l: np.ndarray,
    support: np.ndarray,
    *,
    ridge_fraction: float = 1e-3,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    support = np.asarray(support, dtype=bool).ravel()
    result = np.zeros((support.size, b.shape[1]), dtype=float)
    indices = np.flatnonzero(support)
    if indices.size == 0:
        return result
    lc = l[:, indices]
    scale = np.linalg.norm(lc, axis=0)
    scale = np.maximum(scale, np.median(scale) * 1e-6)
    ln = lc / scale[None, :]
    gram = ln.T @ ln
    ridge = ridge_fraction * np.trace(gram) / max(1, gram.shape[0])
    rhs = ln.T @ b
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        prior_active = prior[indices]
        if prior_active.ndim == 1:
            prior_active = prior_active[:, None]
        rhs = rhs + ridge * prior_active * scale[:, None]
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    result[indices] = coef / scale[:, None]
    return result


def tbf_selection(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    if b.ndim != 2:
        raise ValueError("b must be channels x time")
    if not np.any(b):
        basis = np.zeros((1, b.shape[1]), dtype=float)
        basis[0, 0] = 1.0
        return basis
    _u, s, vt = np.linalg.svd(b, full_matrices=False)
    eigvals = s**2
    total = float(eigvals.sum())
    keep = eigvals / total >= 1.0 / max(1, b.shape[0])
    k = max(1, int(keep.sum()))
    return vt[:k]


def tbf_ridge_refit_system(
    b: np.ndarray,
    l: np.ndarray,
    support: np.ndarray,
    gb: np.ndarray,
    *,
    ridge_fraction: float = 0.03,
    prior: np.ndarray | None = None,
    prior_weight: float = 0.0,
) -> np.ndarray:
    support = np.asarray(support, dtype=bool).ravel()
    gb = np.asarray(gb, dtype=float)
    if gb.ndim != 2 or gb.shape[1] != b.shape[1]:
        raise ValueError("gb must be basis x time and match b")
    result = np.zeros((support.size, b.shape[1]), dtype=float)
    idx = np.flatnonzero(support)
    if idx.size == 0:
        return result
    lc = l[:, idx]
    b_tbf = b @ gb.T
    scale = np.linalg.norm(lc, axis=0)
    scale = np.maximum(scale, np.median(scale) * 1e-6)
    ln = lc / scale[None, :]
    gram = ln.T @ ln
    ridge = max(float(ridge_fraction * np.trace(gram) / max(1, gram.shape[0])), 1e-12)
    rhs = ln.T @ b_tbf
    if prior is not None and prior_weight > 0:
        prior_a = np.asarray(prior, dtype=float)[idx] @ gb.T
        rhs = rhs + prior_weight * ridge * prior_a * scale[:, None]
        gram = gram + prior_weight * ridge * np.eye(gram.shape[0])
    coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)
    result[idx] = (coef / scale[:, None]) @ gb
    return result


def _component_candidates(
    source: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    loose_rel: float = 0.05,
    src_vertices: np.ndarray | None = None,
    deep_k: int = 1,
) -> list[np.ndarray]:
    amp = source_amplitude(source)
    loose = threshold_mask(source, loose_rel)
    components = connected_components(loose[:n_surf], vert_conn[:n_surf, :n_surf])
    deep_ids = np.flatnonzero(loose[n_surf:])
    if src_vertices is not None and deep_k > 1 and deep_ids.size:
        deep_xyz = np.asarray(src_vertices, dtype=float)[n_surf:]
        expanded = set(int(idx) for idx in deep_ids)
        for idx in deep_ids:
            dist = np.linalg.norm(deep_xyz - deep_xyz[int(idx)], axis=1)
            expanded.update(int(i) for i in np.argsort(dist)[:deep_k])
        deep_ids = np.array(sorted(expanded), dtype=int)
    deep = [np.array([idx + n_surf], dtype=int) for idx in deep_ids]
    candidates = components + deep
    candidates.sort(key=lambda comp: float(amp[comp].sum()), reverse=True)
    return candidates


def residual_deep_scores(
    residual: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    deep_l = np.asarray(leadfield, dtype=float)[:, n_surf:]
    numerator = np.sum((deep_l.T @ np.asarray(residual, dtype=float)) ** 2, axis=1)
    denominator = np.sum(deep_l**2, axis=0) + eps
    return numerator / denominator


def residual_deep_timecourse(
    residual: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    deep_id: int,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    lj = np.asarray(leadfield, dtype=float)[:, n_surf + int(deep_id)]
    return ((lj[:, None].T @ np.asarray(residual, dtype=float)) / (float(lj @ lj) + eps)).ravel()


def numpy_knn_expand_deep(
    seed_deep_ids: np.ndarray,
    src_vertices: np.ndarray,
    n_surf: int,
    *,
    deep_k: int = 8,
) -> np.ndarray:
    seed_deep_ids = np.asarray(seed_deep_ids, dtype=int).ravel()
    if seed_deep_ids.size == 0:
        return seed_deep_ids
    deep_xyz = np.asarray(src_vertices, dtype=float)[n_surf:]
    expanded = set(int(i) for i in seed_deep_ids)
    for idx in seed_deep_ids:
        dist = np.linalg.norm(deep_xyz - deep_xyz[int(idx)], axis=1)
        expanded.update(int(i) for i in np.argsort(dist)[:deep_k])
    return np.array(sorted(expanded), dtype=int)


def _select_deep_by_residual(
    residual: np.ndarray,
    leadfield: np.ndarray,
    n_surf: int,
    deep_ids: np.ndarray,
    *,
    complexity_weight: float = 1.0,
    max_points: int = 4,
) -> np.ndarray:
    deep_ids = np.asarray(deep_ids, dtype=int).ravel()
    if deep_ids.size == 0:
        return deep_ids
    g = np.asarray(leadfield, dtype=float)[:, n_surf + deep_ids]
    selected: list[int] = []
    n_obs = residual.size
    current_rss = float(np.linalg.norm(residual, "fro") ** 2)
    current_score = n_obs * np.log(current_rss / n_obs + np.finfo(float).eps)
    for _ in range(min(max_points, deep_ids.size)):
        best = None
        for col in range(deep_ids.size):
            if col in selected:
                continue
            trial = selected + [col]
            gt = g[:, trial]
            gram = gt.T @ gt
            ridge = 1e-3 * np.trace(gram) / max(1, gram.shape[0])
            ridge = max(float(ridge), 1e-12)
            coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), gt.T @ residual)
            rss = float(np.linalg.norm(residual - gt @ coef, "fro") ** 2)
            score = n_obs * np.log(rss / n_obs + np.finfo(float).eps) + complexity_weight * len(trial) * np.log(n_obs)
            if best is None or score < best[0]:
                best = (score, col)
        if best is None or best[0] >= current_score:
            break
        current_score, best_col = best
        selected.append(best_col)
    return deep_ids[selected]


def shrink_surface_core(
    support: np.ndarray,
    fitted: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    core_rel: float = 0.35,
    min_keep: int = 16,
    max_keep: int = 36,
) -> np.ndarray:
    amp = source_amplitude(fitted)
    new_support = np.asarray(support, dtype=bool).copy()
    new_surface = np.zeros(n_surf, dtype=bool)
    for comp in connected_components(new_support[:n_surf], vert_conn[:n_surf, :n_surf]):
        comp_amp = amp[comp]
        peak = float(comp_amp.max(initial=0.0))
        if peak <= 0:
            continue
        keep = comp[comp_amp >= core_rel * peak]
        if keep.size < min_keep:
            keep = comp[np.argsort(comp_amp)[::-1][: min(min_keep, comp.size)]]
        if keep.size > max_keep:
            keep = keep[np.argsort(amp[keep])[::-1][:max_keep]]
        new_surface[keep] = True
    new_support[:n_surf] = new_surface
    return new_support


def _soft_threshold(values: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _row_group_soft_threshold(values: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    scale = np.maximum(1.0 - threshold[:, None] / np.maximum(norm, 1e-12), 0.0)
    return values * scale


def _support_edges(support_indices: np.ndarray, vert_conn: np.ndarray, n_surf: int) -> list[tuple[int, int]]:
    pos = {int(src): i for i, src in enumerate(support_indices)}
    edges = []
    for a, b in zip(*np.nonzero(np.triu(np.asarray(vert_conn != 0)[:n_surf, :n_surf], 1))):
        if int(a) in pos and int(b) in pos:
            edges.append((pos[int(a)], pos[int(b)]))
    return edges


def _candidate_graph_filter(support_indices: np.ndarray, vert_conn: np.ndarray, n_surf: int, tau: float) -> np.ndarray:
    n = support_indices.size
    adjacency = np.zeros((n, n), dtype=float)
    pos = {int(src): i for i, src in enumerate(support_indices)}
    surface_conn = np.asarray(vert_conn != 0)[:n_surf, :n_surf]
    for a, b in zip(*np.nonzero(surface_conn)):
        if int(a) in pos and int(b) in pos and a != b:
            adjacency[pos[int(a)], pos[int(b)]] = 1.0
    row_sum = adjacency.sum(axis=1)
    normalized = np.divide(adjacency, row_sum[:, None], out=np.zeros_like(adjacency), where=row_sum[:, None] > 0)
    return np.eye(n) - tau * normalized


def _candidate_edge_matrix(support_indices: np.ndarray, vert_conn: np.ndarray, n_surf: int) -> np.ndarray:
    edges = _support_edges(support_indices, vert_conn, n_surf)
    edge_matrix = np.zeros((len(edges), support_indices.size), dtype=float)
    for row, (a, b) in enumerate(edges):
        edge_matrix[row, a] = 1.0
        edge_matrix[row, b] = -1.0
    return edge_matrix


def protected_sisses_admm_refit_system(
    b: np.ndarray,
    l: np.ndarray,
    candidate: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    gb: np.ndarray,
    protected_indices: np.ndarray | None = None,
    sigma: float = 1.0,
    alpha: float = 1.0,
    tau: float = 0.5,
    deep_protect_factor: float = 0.25,
    rho: float = 1.0,
    epsilon: float = 0.05,
    admm_iters: int = 50,
    max_weight_itr: int = 5,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool).ravel()
    gb = np.asarray(gb, dtype=float)
    result = np.zeros((candidate.size, b.shape[1]), dtype=float)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return result

    ls = l[:, idx]
    y_data = b @ gb.T
    mm = _candidate_graph_filter(idx, vert_conn, n_surf, tau)
    edge = _candidate_edge_matrix(idx, vert_conn, n_surf)
    dmm = edge @ mm
    lhs = ls.T @ ls + rho * (mm.T @ mm + dmm.T @ dmm) + 1e-8 * np.eye(idx.size)
    lhs_inv = np.linalg.inv(lhs)
    rhs_data = ls.T @ y_data

    a = np.zeros((idx.size, gb.shape[0]), dtype=float)
    z_amp = np.zeros_like(a)
    y_amp = np.zeros_like(a)
    z_edge = np.zeros((dmm.shape[0], gb.shape[0]), dtype=float)
    y_edge = np.zeros_like(z_edge)
    protected = set() if protected_indices is None else set(int(i) for i in np.asarray(protected_indices).ravel())
    protected_local = np.array([int(global_idx) in protected for global_idx in idx], dtype=bool)
    amp_weight = np.ones(idx.size, dtype=float)
    edge_weight = np.ones(dmm.shape[0], dtype=float)

    for _ in range(max_weight_itr):
        weighted_amp = amp_weight.copy()
        weighted_amp[protected_local] *= deep_protect_factor
        for _ in range(admm_iters):
            rhs = rhs_data + rho * mm.T @ (z_amp - y_amp)
            if dmm.size:
                rhs += rho * dmm.T @ (z_edge - y_edge)
            a = lhs_inv @ rhs
            ma = mm @ a
            z_amp = _soft_threshold(ma + y_amp, (sigma / rho) * weighted_amp[:, None])
            y_amp += ma - z_amp
            if dmm.size:
                da = dmm @ a
                z_edge = _soft_threshold(da + y_edge, (sigma * alpha / rho) * edge_weight[:, None])
                y_edge += da - z_edge

        amp_norm = np.linalg.norm(mm @ a, axis=1)
        scale = max(float(amp_norm.max(initial=0.0)), 1e-12)
        amp_weight = 1.0 / (amp_norm / scale + epsilon)
        if dmm.size:
            edge_norm = np.linalg.norm(dmm @ a, axis=1)
            edge_scale = max(float(edge_norm.max(initial=0.0)), 1e-12)
            edge_weight = 1.0 / (edge_norm / edge_scale + epsilon)

    result[idx] = a @ gb
    return result


def layerwise_sisses_admm_refit_system(
    b: np.ndarray,
    l: np.ndarray,
    candidate: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    gb: np.ndarray,
    protected_indices: np.ndarray | None = None,
    sigma_surface: float = 1.0,
    alpha_surface: float = 1.0,
    sigma_deep: float = 1.0,
    sigma_deep_group: float = 1.0,
    tau: float = 0.5,
    deep_protect_factor: float = 0.25,
    deep_nonprotected_factor: float = 1.0,
    rho: float = 1.0,
    epsilon: float = 0.05,
    admm_iters: int = 50,
    max_weight_itr: int = 5,
) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool).ravel()
    gb = np.asarray(gb, dtype=float)
    result = np.zeros((candidate.size, b.shape[1]), dtype=float)
    idx = np.flatnonzero(candidate)
    if idx.size == 0:
        return result

    ls = l[:, idx]
    y_data = b @ gb.T
    surface_pos = np.flatnonzero(idx < n_surf)
    deep_pos = np.flatnonzero(idx >= n_surf)
    surface_idx = idx[surface_pos]
    mm_s = _candidate_graph_filter(surface_idx, vert_conn, n_surf, tau) if surface_pos.size else np.zeros((0, 0))
    dmm_s = _candidate_edge_matrix(surface_idx, vert_conn, n_surf) @ mm_s if surface_pos.size else np.zeros((0, 0))

    lhs = ls.T @ ls + 1e-8 * np.eye(idx.size)
    if surface_pos.size:
        lhs[np.ix_(surface_pos, surface_pos)] += rho * (mm_s.T @ mm_s + dmm_s.T @ dmm_s)
    if deep_pos.size:
        lhs[np.ix_(deep_pos, deep_pos)] += rho * np.eye(deep_pos.size)
    lhs_inv = np.linalg.inv(lhs)
    rhs_data = ls.T @ y_data

    a = np.zeros((idx.size, gb.shape[0]), dtype=float)
    z_s = np.zeros((surface_pos.size, gb.shape[0]), dtype=float)
    y_s = np.zeros_like(z_s)
    z_edge = np.zeros((dmm_s.shape[0], gb.shape[0]), dtype=float)
    y_edge = np.zeros_like(z_edge)
    z_deep = np.zeros((deep_pos.size, gb.shape[0]), dtype=float)
    y_deep = np.zeros_like(z_deep)

    protected = set() if protected_indices is None else set(int(i) for i in np.asarray(protected_indices).ravel())
    protected_deep = np.array([int(global_idx) in protected for global_idx in idx[deep_pos]], dtype=bool)
    surface_weight = np.ones(surface_pos.size, dtype=float)
    edge_weight = np.ones(dmm_s.shape[0], dtype=float)
    deep_weight = np.ones(deep_pos.size, dtype=float)

    for _ in range(max_weight_itr):
        weighted_deep = deep_weight * deep_nonprotected_factor
        weighted_deep[protected_deep] = deep_weight[protected_deep] * deep_protect_factor
        for _ in range(admm_iters):
            rhs = rhs_data.copy()
            if surface_pos.size:
                rhs[surface_pos] += rho * mm_s.T @ (z_s - y_s)
                if dmm_s.size:
                    rhs[surface_pos] += rho * dmm_s.T @ (z_edge - y_edge)
            if deep_pos.size:
                rhs[deep_pos] += rho * (z_deep - y_deep)
            a = lhs_inv @ rhs

            if surface_pos.size:
                ma = mm_s @ a[surface_pos]
                z_s = _soft_threshold(ma + y_s, (sigma_surface / rho) * surface_weight[:, None])
                y_s += ma - z_s
                if dmm_s.size:
                    da = dmm_s @ a[surface_pos]
                    z_edge = _soft_threshold(da + y_edge, (sigma_surface * alpha_surface / rho) * edge_weight[:, None])
                    y_edge += da - z_edge
            if deep_pos.size:
                deep_trial = _soft_threshold(a[deep_pos] + y_deep, (sigma_deep / rho) * weighted_deep[:, None])
                z_deep = _row_group_soft_threshold(deep_trial, (sigma_deep_group / rho) * weighted_deep)
                y_deep += a[deep_pos] - z_deep

        if surface_pos.size:
            surface_norm = np.linalg.norm(mm_s @ a[surface_pos], axis=1)
            surface_scale = max(float(surface_norm.max(initial=0.0)), 1e-12)
            surface_weight = 1.0 / (surface_norm / surface_scale + epsilon)
        if dmm_s.size:
            edge_norm = np.linalg.norm(dmm_s @ a[surface_pos], axis=1)
            edge_scale = max(float(edge_norm.max(initial=0.0)), 1e-12)
            edge_weight = 1.0 / (edge_norm / edge_scale + epsilon)
        if deep_pos.size:
            deep_norm = np.linalg.norm(a[deep_pos], axis=1)
            deep_scale = max(float(deep_norm.max(initial=0.0)), 1e-12)
            deep_weight = 1.0 / (deep_norm / deep_scale + epsilon)

    result[idx] = a @ gb
    return result


def sisses_style_refit_system(
    b: np.ndarray,
    l: np.ndarray,
    support: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    prior: np.ndarray | None = None,
    lambda_amp: float = 0.05,
    lambda_edge: float = 0.05,
    lambda_prior: float = 0.01,
    max_iter: int = 50,
    weight_iter: int = 5,
) -> np.ndarray:
    support = np.asarray(support, dtype=bool).ravel()
    result = np.zeros((support.size, b.shape[1]), dtype=float)
    idx = np.flatnonzero(support)
    if idx.size == 0:
        return result
    has_prior = prior is not None
    prior = np.zeros_like(result) if prior is None else np.asarray(prior, dtype=float)
    x = prior[idx].copy() if has_prior else ridge_refit_system(b, l, support, ridge_fraction=1e-3)[idx]
    prior_sub = prior[idx]
    ls = l[:, idx]
    lipschitz = float(np.linalg.norm(ls, 2) ** 2 + lambda_prior + 1e-12)
    step = 1.0 / lipschitz
    edges = _support_edges(idx, vert_conn, n_surf)
    amp_weight = np.ones(idx.size)
    for _ in range(weight_iter):
        for _ in range(max_iter):
            grad = ls.T @ (ls @ x - b) + lambda_prior * (x - prior_sub)
            x = _soft_threshold(x - step * grad, step * lambda_amp * amp_weight[:, None])
            for a, b_idx in edges:
                diff = x[a] - x[b_idx]
                shrunk = _soft_threshold(diff, step * lambda_edge)
                delta = 0.5 * (diff - shrunk)
                x[a] -= delta
                x[b_idx] += delta
        amp_weight = 1.0 / (np.sqrt(np.mean(x**2, axis=1)) + 1e-3)
        amp_weight /= max(float(np.median(amp_weight)), 1e-12)
    result[idx] = x
    return result


def _select_from_components(
    b: np.ndarray,
    l: np.ndarray,
    sisses: np.ndarray,
    components: list[np.ndarray],
    *,
    complexity_weight: float,
    max_components: int,
) -> list[np.ndarray]:
    amp = source_amplitude(sisses)
    patterns = []
    kept_components = []
    for component in components:
        weights = amp[component].astype(float)
        norm = float(np.linalg.norm(weights))
        if norm <= 0:
            if len(component) != 1:
                continue
            weights = np.ones(1, dtype=float)
            norm = 1.0
        weights = weights / norm
        patterns.append(l[:, component] @ weights)
        kept_components.append(component)
    if not patterns:
        return []
    g = np.column_stack(patterns)
    selected: list[int] = []
    n_obs = b.size
    current_rss = float(np.linalg.norm(b, "fro") ** 2)
    current_score = n_obs * np.log(current_rss / n_obs + np.finfo(float).eps)

    for _ in range(min(max_components, len(kept_components))):
        best = None
        for idx, component in enumerate(kept_components):
            if idx in selected:
                continue
            trial_ids = selected + [idx]
            gt = g[:, trial_ids]
            gram = gt.T @ gt
            ridge = 1e-3 * np.trace(gram) / max(1, gram.shape[0])
            coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), gt.T @ b)
            rss = float(np.linalg.norm(b - gt @ coef, "fro") ** 2)
            active = sum(len(kept_components[i]) for i in trial_ids)
            score = n_obs * np.log(rss / n_obs + np.finfo(float).eps) + complexity_weight * active * np.log(n_obs)
            if best is None or score < best[0]:
                best = (score, idx)
        if best is None or best[0] >= current_score:
            break
        current_score, best_idx = best
        selected.append(best_idx)
    return [kept_components[i] for i in selected]


def _select_components(
    b: np.ndarray,
    l: np.ndarray,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    loose_rel: float,
    complexity_weight: float,
    max_components: int,
    src_vertices: np.ndarray | None = None,
    deep_k: int = 1,
) -> list[np.ndarray]:
    components = _component_candidates(
        sisses,
        vert_conn,
        n_surf,
        loose_rel=loose_rel,
        src_vertices=src_vertices,
        deep_k=deep_k,
    )
    return _select_from_components(
        b,
        l,
        sisses,
        components,
        complexity_weight=complexity_weight,
        max_components=max_components,
    )


def component_refit_select(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    *,
    loose_rel: float = 0.05,
    complexity_weight: float = 1.0,
    max_components: int = 6,
    final_ridge_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    components = _component_candidates(sisses, vert_conn, n_surf, loose_rel=loose_rel)
    amp = source_amplitude(sisses)
    patterns = []
    kept_components = []
    for component in components:
        weights = amp[component].astype(float)
        norm = float(np.linalg.norm(weights))
        if norm <= 0:
            continue
        weights = weights / norm
        patterns.append(l[:, component] @ weights)
        kept_components.append((component, weights))
    if not patterns:
        support = threshold_mask(sisses, 0.10)
        return sisses * support[:, None], support
    g = np.column_stack(patterns)
    support = np.zeros(sisses.shape[0], dtype=bool)
    selected: list[int] = []
    n_obs = b.size
    current_rss = float(np.linalg.norm(b, "fro") ** 2)
    current_score = n_obs * np.log(current_rss / n_obs + np.finfo(float).eps)

    for _ in range(min(max_components, len(kept_components))):
        best = None
        for idx, (component, _weights) in enumerate(kept_components):
            if idx in selected:
                continue
            trial_ids = selected + [idx]
            gt = g[:, trial_ids]
            gram = gt.T @ gt
            ridge = 1e-3 * np.trace(gram) / max(1, gram.shape[0])
            coef = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), gt.T @ b)
            rss = float(np.linalg.norm(b - gt @ coef, "fro") ** 2)
            active = sum(len(kept_components[i][0]) for i in trial_ids)
            score = n_obs * np.log(rss / n_obs + np.finfo(float).eps) + complexity_weight * active * np.log(n_obs)
            if best is None or score < best[0]:
                best = (score, idx, rss)
        if best is None or best[0] >= current_score:
            break
        current_score, best_idx, current_rss = best
        selected.append(best_idx)

    if not selected:
        support = threshold_mask(sisses, 0.10)
        return sisses * support[:, None], support

    support = np.zeros(sisses.shape[0], dtype=bool)
    for idx in selected:
        component, _weights = kept_components[idx]
        support[component] = True
    fitted = ridge_refit_system(
        b,
        l,
        support,
        ridge_fraction=final_ridge_fraction,
        prior=sisses * support[:, None],
    )
    return fitted, support


def component_refit_select_v3(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    loose_rel: float = 0.05,
    complexity_weight: float = 1.0,
    max_components: int = 6,
    surface_hops: int = 8,
    deep_k: int = 8,
    final_ridge_fraction: float = 0.3,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    components = _select_components(
        b,
        l,
        sisses,
        vert_conn,
        n_surf,
        loose_rel=loose_rel,
        complexity_weight=complexity_weight,
        max_components=max_components,
        src_vertices=src_vertices,
        deep_k=deep_k,
    )
    if not components:
        support = threshold_mask(sisses, 0.10)
        fitted = sisses * support[:, None]
        return (fitted, support, support.copy()) if return_candidate else (fitted, support)

    broad = np.zeros(sisses.shape[0], dtype=bool)
    for component in components:
        broad[component] = True
    broad_fit = ridge_refit_system(b, l, broad, ridge_fraction=final_ridge_fraction, prior=sisses * broad[:, None])
    amp = source_amplitude(broad_fit)

    support = np.zeros_like(broad)
    for component in components:
        if np.all(component < n_surf):
            allowed = np.zeros(n_surf, dtype=bool)
            allowed[component] = True
            peak = int(component[np.argmax(amp[component])])
            support[:n_surf] |= graph_hop_mask(peak, vert_conn[:n_surf, :n_surf], allowed, hops=surface_hops)
        else:
            support[component] = True

    fitted = ridge_refit_system(b, l, support, ridge_fraction=final_ridge_fraction, prior=sisses * support[:, None])
    return (fitted, support, broad) if return_candidate else (fitted, support)


def component_refit_select_v4_deep_rescue(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    loose_rel: float = 0.05,
    complexity_weight: float = 1.0,
    max_surface_components: int = 6,
    surface_hops: int = 8,
    deep_rescue_top: int = 8,
    deep_k: int = 8,
    max_deep_points: int = 1,
    final_ridge_fraction: float = 0.3,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    base_fit, base_support, base_candidate = component_refit_select_v3(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        loose_rel=loose_rel,
        complexity_weight=complexity_weight,
        max_components=max_surface_components,
        surface_hops=surface_hops,
        deep_k=deep_k,
        final_ridge_fraction=final_ridge_fraction,
        return_candidate=True,
    )
    residual = b - l @ base_fit

    seed_deep = set(np.flatnonzero(threshold_mask(sisses, loose_rel)[n_surf:]).astype(int).tolist())
    for extra in (eeg_only_source, meg_only_source):
        if extra is None:
            continue
        deep_amp = source_amplitude(extra)[n_surf:]
        if deep_amp.size:
            seed_deep.update(int(i) for i in np.argsort(deep_amp)[::-1][:deep_rescue_top])

    scores = residual_deep_scores(residual, l, n_surf)
    if scores.size:
        seed_deep.update(int(i) for i in np.argsort(scores)[::-1][:deep_rescue_top])
    expanded_deep = numpy_knn_expand_deep(np.array(sorted(seed_deep), dtype=int), src_vertices, n_surf, deep_k=deep_k)
    if expanded_deep.size > max_deep_points:
        order = np.argsort(scores[expanded_deep])[::-1][:max_deep_points]
        expanded_deep = expanded_deep[order]
    expanded_deep = _select_deep_by_residual(
        residual,
        l,
        n_surf,
        expanded_deep,
        complexity_weight=complexity_weight,
        max_points=max_deep_points,
    )
    candidate = base_support.copy()
    candidate[n_surf + expanded_deep] = True
    if not candidate.any():
        candidate = threshold_mask(sisses, 0.10)
    fitted = ridge_refit_system(b, l, candidate, ridge_fraction=final_ridge_fraction, prior=sisses * candidate[:, None])
    broad_candidate = base_candidate | candidate
    return (fitted, candidate, broad_candidate) if return_candidate else (fitted, candidate)


def component_refit_select_v5_compact_deep_prior(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    loose_rel: float = 0.05,
    complexity_weight: float = 1.0,
    max_surface_components: int = 6,
    surface_hops: int = 8,
    deep_rescue_top: int = 8,
    deep_k: int = 8,
    max_deep_points: int = 1,
    final_ridge_fraction: float = 0.3,
    core_rel: float = 0.03,
    min_keep: int = 24,
    max_keep: int = 160,
    deep_prior_weight: float = 0.05,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    base_fit, base_support, base_candidate = component_refit_select_v3(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        loose_rel=loose_rel,
        complexity_weight=complexity_weight,
        max_components=max_surface_components,
        surface_hops=surface_hops,
        deep_k=deep_k,
        final_ridge_fraction=final_ridge_fraction,
        return_candidate=True,
    )
    residual = b - l @ base_fit
    scores = residual_deep_scores(residual, l, n_surf)
    seed_deep = set(np.flatnonzero(threshold_mask(sisses, loose_rel)[n_surf:]).astype(int).tolist())
    for extra in (eeg_only_source, meg_only_source):
        if extra is not None:
            deep_amp = source_amplitude(extra)[n_surf:]
            seed_deep.update(int(i) for i in np.argsort(deep_amp)[::-1][:deep_rescue_top])
    if scores.size:
        seed_deep.update(int(i) for i in np.argsort(scores)[::-1][:deep_rescue_top])
    rescued = numpy_knn_expand_deep(np.array(sorted(seed_deep), dtype=int), src_vertices, n_surf, deep_k=deep_k)
    if rescued.size > max_deep_points:
        rescued = rescued[np.argsort(scores[rescued])[::-1][:max_deep_points]]
    rescued = _select_deep_by_residual(
        residual,
        l,
        n_surf,
        rescued,
        complexity_weight=complexity_weight,
        max_points=max_deep_points,
    )

    broad_support = base_support.copy()
    broad_support[n_surf + rescued] = True
    prior = sisses * broad_support[:, None]
    for deep_id in rescued:
        idx = n_surf + int(deep_id)
        prior[idx] = (1.0 - deep_prior_weight) * prior[idx] + deep_prior_weight * residual_deep_timecourse(residual, l, n_surf, int(deep_id))
    broad_fit = ridge_refit_system(b, l, broad_support, ridge_fraction=final_ridge_fraction, prior=prior)
    compact_support = shrink_surface_core(
        broad_support,
        broad_fit,
        vert_conn,
        n_surf,
        core_rel=core_rel,
        min_keep=min_keep,
        max_keep=max_keep,
    )
    compact_prior = sisses * compact_support[:, None]
    for deep_id in rescued:
        if compact_support[n_surf + int(deep_id)]:
            idx = n_surf + int(deep_id)
            compact_prior[idx] = (1.0 - deep_prior_weight) * compact_prior[idx] + deep_prior_weight * residual_deep_timecourse(residual, l, n_surf, int(deep_id))
    fitted = ridge_refit_system(b, l, compact_support, ridge_fraction=final_ridge_fraction, prior=compact_prior)
    broad_candidate = base_candidate | broad_support
    return (fitted, compact_support, broad_candidate) if return_candidate else (fitted, compact_support)


def component_refit_select_v6_sisses_refit(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    admm_iters: int = 50,
    max_weight_itr: int = 5,
    return_candidate: bool = False,
) -> dict[str, tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
    b, l = whitened_joint_system(eeg, meg)
    v4_fit, candidate, broad_candidate = component_refit_select_v4_deep_rescue(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        eeg_only_source=eeg_only_source,
        meg_only_source=meg_only_source,
        return_candidate=True,
    )
    variants = {
        "weak": (0.03, 0.015),
        "mid": (0.05, 0.05),
        "strong": (0.08, 0.16),
    }
    out = {}
    for name, (lambda_amp, lambda_edge) in variants.items():
        fitted = sisses_style_refit_system(
            b,
            l,
            candidate,
            vert_conn,
            n_surf,
            prior=v4_fit,
            lambda_amp=lambda_amp,
            lambda_edge=lambda_edge,
            lambda_prior=100.0,
            max_iter=admm_iters,
            weight_iter=max_weight_itr,
        )
        mask = threshold_mask(fitted, 0.05) & candidate
        if not mask.any():
            mask = candidate.copy()
        out[name] = (fitted, mask, broad_candidate) if return_candidate else (fitted, mask)
    return out


def component_refit_select_v7_tbf_refit(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    ridge_fraction: float = 0.003,
    prior_weight: float = 5.0,
    mask_rel: float = 0.05,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    v4_fit, candidate, broad_candidate = component_refit_select_v4_deep_rescue(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        eeg_only_source=eeg_only_source,
        meg_only_source=meg_only_source,
        return_candidate=True,
    )
    gb = tbf_selection(b)
    fitted = tbf_ridge_refit_system(
        b,
        l,
        candidate,
        gb,
        ridge_fraction=ridge_fraction,
        prior=v4_fit,
        prior_weight=prior_weight,
    )
    mask = threshold_mask(fitted, mask_rel) & candidate
    mask[n_surf:] = candidate[n_surf:]
    if not mask.any():
        mask = candidate.copy()
    return (fitted, mask, broad_candidate) if return_candidate else (fitted, mask)


def component_refit_select_v8_protected_sisses(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    sigma: float = 1.0,
    alpha: float = 1.0,
    tau: float = 0.5,
    deep_protect_factor: float = 0.25,
    deep_rescue_top: int = 8,
    deep_k: int = 8,
    max_deep_points: int = 1,
    admm_iters: int = 50,
    max_weight_itr: int = 5,
    metric_rel: float = 0.10,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    _v4_fit, v4_candidate, broad_candidate = component_refit_select_v4_deep_rescue(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        eeg_only_source=eeg_only_source,
        meg_only_source=meg_only_source,
        deep_rescue_top=deep_rescue_top,
        deep_k=deep_k,
        max_deep_points=max_deep_points,
        return_candidate=True,
    )
    candidate = broad_candidate.copy()
    candidate[n_surf:] |= v4_candidate[n_surf:]
    if not candidate.any():
        candidate = threshold_mask(sisses, 0.05)
    protected_deep = np.flatnonzero(v4_candidate[n_surf:]) + n_surf
    gb = tbf_selection(b)
    fitted = protected_sisses_admm_refit_system(
        b,
        l,
        candidate,
        vert_conn,
        n_surf,
        gb=gb,
        protected_indices=protected_deep,
        sigma=sigma,
        alpha=alpha,
        tau=tau,
        deep_protect_factor=deep_protect_factor,
        admm_iters=admm_iters,
        max_weight_itr=max_weight_itr,
    )
    mask = threshold_mask(fitted, metric_rel) & candidate
    mask[protected_deep] = True
    if not mask.any():
        mask = candidate.copy()
    return (fitted, mask, candidate) if return_candidate else (fitted, mask)


def component_refit_select_v9_layerwise_sisses(
    eeg: dict,
    meg: dict,
    sisses: np.ndarray,
    vert_conn: np.ndarray,
    n_surf: int,
    src_vertices: np.ndarray,
    *,
    eeg_only_source: np.ndarray | None = None,
    meg_only_source: np.ndarray | None = None,
    sigma_surface: float = 1.0,
    alpha_surface: float = 1.0,
    sigma_deep: float = 1.0,
    sigma_deep_group: float = 1.0,
    tau: float = 0.5,
    deep_protect_factor: float = 0.25,
    deep_nonprotected_factor: float = 1.0,
    deep_rescue_top: int = 8,
    deep_k: int = 8,
    max_deep_points: int = 1,
    admm_iters: int = 50,
    max_weight_itr: int = 5,
    metric_rel: float = 0.10,
    return_candidate: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, l = whitened_joint_system(eeg, meg)
    _v4_fit, v4_candidate, broad_candidate = component_refit_select_v4_deep_rescue(
        eeg,
        meg,
        sisses,
        vert_conn,
        n_surf,
        src_vertices,
        eeg_only_source=eeg_only_source,
        meg_only_source=meg_only_source,
        deep_rescue_top=deep_rescue_top,
        deep_k=deep_k,
        max_deep_points=max_deep_points,
        return_candidate=True,
    )
    candidate = broad_candidate.copy()
    candidate[n_surf:] |= v4_candidate[n_surf:]
    if not candidate.any():
        candidate = threshold_mask(sisses, 0.05)
    protected_deep = np.flatnonzero(v4_candidate[n_surf:]) + n_surf
    fitted = layerwise_sisses_admm_refit_system(
        b,
        l,
        candidate,
        vert_conn,
        n_surf,
        gb=tbf_selection(b),
        protected_indices=protected_deep,
        sigma_surface=sigma_surface,
        alpha_surface=alpha_surface,
        sigma_deep=sigma_deep,
        sigma_deep_group=sigma_deep_group,
        tau=tau,
        deep_protect_factor=deep_protect_factor,
        deep_nonprotected_factor=deep_nonprotected_factor,
        admm_iters=admm_iters,
        max_weight_itr=max_weight_itr,
    )
    mask = threshold_mask(fitted, metric_rel) & candidate
    mask[protected_deep] = True
    if not mask.any():
        mask = candidate.copy()
    return (fitted, mask, candidate) if return_candidate else (fitted, mask)


def metric_row(scenario: str, method: str, source: np.ndarray, truth: dict, mask: np.ndarray) -> dict:
    true_source = np.asarray(truth["s_true"], dtype=float)
    metrics = external_full_head_metrics(
        source,
        true_source,
        np.asarray(truth["src_vertices"], dtype=float),
        true_groups=true_source_groups(truth),
    )
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    return {
        "scenario": scenario,
        "method": method,
        "active_count": int(mask.sum()),
        "surface_active_count": int(mask[:n_surf].sum()),
        "deep_active_count": int(mask[n_surf:].sum()),
        "auc": metrics["auc"],
        "rmse": metrics["rmse"],
        "sd_mm": metrics["sd_mm"],
        "dle_mm": metrics["dle_mm"],
    }


def save_npz(path: Path, source: np.ndarray, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, S=source, region_mask=mask)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_scenario(scenario: str) -> list[dict]:
    eeg = load_mat(DATA_ROOT / scenario / "sub_EEG.mat")
    meg = load_mat(DATA_ROOT / scenario / "sub_MEG.mat")
    truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
    n_sources = np.asarray(truth["s_true"]).shape[0]
    n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
    vert_conn = np.asarray(eeg["VertConn"], dtype=float)

    sources = {
        mode: load_source(SISSES_RUN_ROOT / scenario / mode / "s_wen.mat", n_sources)
        for mode in MODES
    }
    weighted = load_source(WEIGHTED_ROOT / scenario / "s_wen.mat", n_sources)
    masks = {mode: threshold_mask(source, rel=0.10) for mode, source in sources.items()}
    adaptive_rel = adaptive_sisses_threshold(sources["both"])
    adaptive_mask = threshold_mask(sources["both"], adaptive_rel)
    compact_mask = evidence_aware_compact_mask(sources["both"], vert_conn, n_surf)
    component_refit, component_refit_mask = component_refit_select(eeg, meg, sources["both"], vert_conn, n_surf)
    component_refit_v3, component_refit_v3_mask = component_refit_select_v3(
        eeg,
        meg,
        sources["both"],
        vert_conn,
        n_surf,
        np.asarray(truth["src_vertices"], dtype=float),
    )
    component_refit_v4, component_refit_v4_mask = component_refit_select_v4_deep_rescue(
        eeg,
        meg,
        sources["both"],
        vert_conn,
        n_surf,
        np.asarray(truth["src_vertices"], dtype=float),
        eeg_only_source=sources["eeg_only"],
        meg_only_source=sources["meg_only"],
    )
    component_refit_v5, component_refit_v5_mask = component_refit_select_v5_compact_deep_prior(
        eeg,
        meg,
        sources["both"],
        vert_conn,
        n_surf,
        np.asarray(truth["src_vertices"], dtype=float),
        eeg_only_source=sources["eeg_only"],
        meg_only_source=sources["meg_only"],
    )
    component_refit_v6 = component_refit_select_v6_sisses_refit(
        eeg,
        meg,
        sources["both"],
        vert_conn,
        n_surf,
        np.asarray(truth["src_vertices"], dtype=float),
        eeg_only_source=sources["eeg_only"],
        meg_only_source=sources["meg_only"],
    )
    component_refit_v7, component_refit_v7_mask = component_refit_select_v7_tbf_refit(
        eeg,
        meg,
        sources["both"],
        vert_conn,
        n_surf,
        np.asarray(truth["src_vertices"], dtype=float),
        eeg_only_source=sources["eeg_only"],
        meg_only_source=sources["meg_only"],
    )
    component_refit_v8 = {
        name: component_refit_select_v8_protected_sisses(
            eeg,
            meg,
            sources["both"],
            vert_conn,
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            eeg_only_source=sources["eeg_only"],
            meg_only_source=sources["meg_only"],
            deep_protect_factor=factor,
        )
        for name, factor in V8_FACTORS.items()
    }
    component_refit_v9 = {
        name: component_refit_select_v9_layerwise_sisses(
            eeg,
            meg,
            sources["both"],
            vert_conn,
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            eeg_only_source=sources["eeg_only"],
            meg_only_source=sources["meg_only"],
            deep_protect_factor=factor,
        )
        for name, factor in V9_FACTORS.items()
    }
    component_refit_v9b = {
        name: component_refit_select_v9_layerwise_sisses(
            eeg,
            meg,
            sources["both"],
            vert_conn,
            n_surf,
            np.asarray(truth["src_vertices"], dtype=float),
            eeg_only_source=sources["eeg_only"],
            meg_only_source=sources["meg_only"],
            **params,
        )
        for name, params in V9B_VARIANTS.items()
    }
    weighted_mask = threshold_mask(weighted, rel=0.10)

    candidate = build_candidate_mask(sources["both"], sources["meg_only"], vert_conn, n_surf)
    protected = sources["both"] * candidate["final"][:, None]
    protected_refit = ridge_refit(eeg, meg, candidate["final"])

    scenario_out = PROTECTED_ROOT / scenario
    save_npz(scenario_out / "sisses_eeg_meg_direct.npz", sources["both"], masks["both"])
    save_npz(scenario_out / "sisses_adaptive_threshold.npz", sources["both"], adaptive_mask)
    save_npz(scenario_out / "sisses_evidence_compact.npz", sources["both"], compact_mask)
    save_npz(scenario_out / "sisses_component_refit_v2.npz", component_refit, component_refit_mask)
    save_npz(scenario_out / "sisses_component_refit_v3_localize.npz", component_refit_v3, component_refit_v3_mask)
    save_npz(scenario_out / "sisses_component_refit_v4_deep_rescue.npz", component_refit_v4, component_refit_v4_mask)
    save_npz(scenario_out / "sisses_component_refit_v5_compact_deep_prior.npz", component_refit_v5, component_refit_v5_mask)
    for variant, (source, mask) in component_refit_v6.items():
        save_npz(scenario_out / f"sisses_component_refit_v6_sisses_refit_{variant}.npz", source, mask)
    save_npz(scenario_out / "sisses_component_refit_v7_tbf_refit.npz", component_refit_v7, component_refit_v7_mask)
    for variant, (source, mask) in component_refit_v8.items():
        save_npz(scenario_out / f"sisses_component_refit_v8_protected_sisses_{variant}.npz", source, mask)
    for variant, (source, mask) in component_refit_v9.items():
        save_npz(scenario_out / f"sisses_component_refit_v9_layerwise_sisses_{variant}.npz", source, mask)
    for variant, (source, mask) in component_refit_v9b.items():
        save_npz(scenario_out / f"sisses_component_refit_v9b_layerwise_sisses_{variant}.npz", source, mask)
    save_npz(scenario_out / "sisses_weighted_multilayer.npz", weighted, weighted_mask)
    save_npz(scenario_out / "sisses_meg_only.npz", sources["meg_only"], masks["meg_only"])
    save_npz(scenario_out / "sisses_eeg_only.npz", sources["eeg_only"], masks["eeg_only"])
    save_npz(scenario_out / "sisses_protected_multilayer.npz", protected, candidate["final"])
    save_npz(scenario_out / "sisses_protected_refit.npz", protected_refit, candidate["final"])
    save_npz(scenario_out / "candidate_surface_layer.npz", protected, candidate["surface"])
    save_npz(scenario_out / "candidate_deep_layer.npz", protected, candidate["deep"])

    return [
        metric_row(scenario, "SISSES_EEG_MEG_direct", sources["both"] * masks["both"][:, None], truth, masks["both"]),
        metric_row(scenario, f"SISSES_adaptive_threshold_{adaptive_rel:.3f}", sources["both"] * adaptive_mask[:, None], truth, adaptive_mask),
        metric_row(scenario, "SISSES_evidence_compact", sources["both"] * compact_mask[:, None], truth, compact_mask),
        metric_row(scenario, "SISSES_component_refit_v2", component_refit, truth, component_refit_mask),
        metric_row(scenario, "SISSES_component_refit_v3_localize", component_refit_v3, truth, component_refit_v3_mask),
        metric_row(scenario, "SISSES_component_refit_v4_deep_rescue", component_refit_v4, truth, component_refit_v4_mask),
        metric_row(scenario, "SISSES_component_refit_v5_compact_deep_prior", component_refit_v5, truth, component_refit_v5_mask),
        *[
            metric_row(scenario, f"SISSES_component_refit_v6_sisses_refit_{variant}", source, truth, mask)
            for variant, (source, mask) in component_refit_v6.items()
        ],
        metric_row(scenario, "SISSES_component_refit_v7_tbf_refit", component_refit_v7, truth, component_refit_v7_mask),
        *[
            metric_row(scenario, f"SISSES_component_refit_v8_protected_sisses_{variant}", source, truth, mask)
            for variant, (source, mask) in component_refit_v8.items()
        ],
        *[
            metric_row(scenario, f"SISSES_component_refit_v9_layerwise_sisses_{variant}", source, truth, mask)
            for variant, (source, mask) in component_refit_v9.items()
        ],
        *[
            metric_row(scenario, f"SISSES_component_refit_v9b_layerwise_sisses_{variant}", source, truth, mask)
            for variant, (source, mask) in component_refit_v9b.items()
        ],
        metric_row(scenario, "SISSES_weighted_multilayer", weighted * weighted_mask[:, None], truth, weighted_mask),
        metric_row(scenario, "SISSES_protected_multilayer", protected, truth, candidate["final"]),
        metric_row(scenario, "SISSES_MEG_only", sources["meg_only"] * masks["meg_only"][:, None], truth, masks["meg_only"]),
        metric_row(scenario, "SISSES_EEG_only", sources["eeg_only"] * masks["eeg_only"][:, None], truth, masks["eeg_only"]),
    ]


def main() -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        print("Processing", scenario)
        rows.extend(run_scenario(scenario))
    write_csv(OUT_ROOT / "metrics.csv", rows)
    print("Saved:", OUT_ROOT / "metrics.csv")


if __name__ == "__main__":
    main()
