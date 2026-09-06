"""Frozen, deterministic simulation protocol for the full-head benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    import mne


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("SOURCE_DATA_ROOT", REPO_ROOT / "generated"))
BASE_SEED = 20260715
ACTIVE_START = 200
SCENARIO_CODES = {
    "surface_only": 0,
    "deep_only": 1,
    "deep_plus_surface": 2,
    "deep_plus_two_surface": 3,
}


def _auc_cortex(vertices: np.ndarray, adjacency: np.ndarray, n_surf: int) -> dict:
    """Make the mixed grid usable by the requested cortical-mesh AUC."""
    vertices = np.asarray(vertices, dtype=float)
    n_sources = len(vertices)
    k = min(7, int(n_surf))
    neighbors = cKDTree(vertices[:n_surf]).query(vertices[:n_surf], k=k)[1]
    rows = [np.repeat(np.arange(n_surf), k), neighbors.ravel()]
    cols = [neighbors.ravel(), np.repeat(np.arange(n_surf), k)]
    for index in range(int(n_surf), n_sources):
        nearest = np.argsort(np.linalg.norm(vertices - vertices[index], axis=1))[1:21]
        rows.append(np.full(nearest.size, index))
        cols.append(nearest)
    extra = sparse.csr_matrix(
        (np.ones(sum(map(len, rows)), dtype=np.uint8), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_sources, n_sources),
    )
    graph = ((sparse.csr_matrix(adjacency) + extra) > 0).astype(np.uint8)
    digest = hashlib.sha256(np.asarray(graph.shape, dtype=np.int64).tobytes())
    for values in (graph.indptr, graph.indices, graph.data):
        digest.update(np.ascontiguousarray(values).tobytes())
    return {"Vertices": vertices, "Faces": graph, "GraphSHA256": digest.hexdigest()}


def _covariance(cov: mne.Covariance, names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(cov["names"])}
    missing = set(names) - lookup.keys()
    if missing:
        raise ValueError(f"noise covariance lacks channels: {sorted(missing)}")
    indices = [lookup[name] for name in names]
    data = np.asarray(cov["data"], dtype=float)[np.ix_(indices, indices)]
    return (data + data.T) / 2.0


def _covariance_factor(cov: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(np.asarray(cov, dtype=float))
    floor = max(float(values.max(initial=0.0)) * 1e-12, np.finfo(float).tiny)
    return vectors * np.sqrt(np.maximum(values, floor))


def load_shared(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    sample_path: str | Path | None = None,
) -> dict:
    """Load the one shared coarse mixed source space, MNE metadata, and noise model."""
    import mne

    data_root = Path(data_root)
    case_root = data_root / "deep_plus_two_surface"
    eeg_mat = sio.loadmat(
        case_root / "sub_EEG.mat",
        variable_names=(
            "Gain",
            "Gain3D",
            "VertConn",
            "src_vertices",
            "times",
            "sfreq",
            "n_surf",
            "n_deep",
            "deep_orientations",
        ),
    )
    meg_mat = sio.loadmat(
        case_root / "sub_MEG.mat", variable_names=("Gain", "Gain3D")
    )

    if sample_path is None:
        sample_path = mne.datasets.sample.data_path(download=False)
    sample_path = Path(sample_path)
    sample_dir = sample_path / "MEG" / "sample"
    raw = mne.io.read_raw_fif(
        sample_dir / "sample_audvis_raw.fif", preload=False, verbose=False
    )
    forward = mne.read_forward_solution(
        sample_dir / "sample_audvis-meg-eeg-oct-6-fwd.fif", verbose=False
    )
    fixed = mne.convert_forward_solution(
        forward, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
    )
    fwd_meg = mne.pick_types_forward(fixed, meg="mag", eeg=False)
    fwd_eeg = mne.pick_types_forward(fixed, meg=False, eeg=True)
    eeg_picks = mne.pick_types(
        raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
    )
    good_eeg = {raw.info["ch_names"][index] for index in eeg_picks}
    eeg_names = [name for name in fwd_eeg["info"]["ch_names"] if name in good_eeg]
    fwd_eeg = mne.pick_channels_forward(fwd_eeg, eeg_names, ordered=True)
    eeg_names = list(fwd_eeg["info"]["ch_names"])
    meg_names = list(fwd_meg["info"]["ch_names"])

    cov_eeg = _covariance(
        mne.read_cov(sample_dir / "sample_audvis-cov.fif", verbose=False), eeg_names
    )
    cov_meg = _covariance(
        mne.read_cov(sample_dir / "ernoise-cov.fif", verbose=False), meg_names
    )
    n_surf = int(np.asarray(eeg_mat["n_surf"]).ravel()[0])
    n_deep = int(np.asarray(eeg_mat["n_deep"]).ravel()[0])
    gain_eeg = np.asarray(eeg_mat["Gain"], dtype=float)
    gain_meg = np.asarray(meg_mat["Gain"], dtype=float)
    if gain_eeg.shape != (len(eeg_names), n_surf + n_deep):
        raise ValueError("generated EEG gain and MNE channel/source order disagree")
    if gain_meg.shape != (len(meg_names), n_surf + n_deep):
        raise ValueError("generated MEG gain and MNE channel/source order disagree")

    vertices = np.asarray(eeg_mat["src_vertices"], dtype=float)
    adjacency = np.asarray(eeg_mat["VertConn"], dtype=np.uint8)
    return {
        "gain_eeg": gain_eeg,
        "gain_meg": gain_meg,
        "gain3d_eeg": np.asarray(eeg_mat["Gain3D"], dtype=float),
        "gain3d_meg": np.asarray(meg_mat["Gain3D"], dtype=float),
        "vertices": vertices,
        "adjacency": adjacency,
        "auc_cortex": _auc_cortex(vertices, adjacency, n_surf),
        "times": np.asarray(eeg_mat["times"], dtype=float).ravel(),
        "sfreq": float(np.asarray(eeg_mat["sfreq"]).ravel()[0]),
        "active_start": ACTIVE_START,
        "n_surf": n_surf,
        "n_deep": n_deep,
        "deep_orientations": np.asarray(eeg_mat["deep_orientations"], dtype=float),
        "noise_cov_eeg": cov_eeg,
        "noise_cov_meg": cov_meg,
        "noise_factor_eeg": _covariance_factor(cov_eeg),
        "noise_factor_meg": _covariance_factor(cov_meg),
        "info_eeg": fwd_eeg["info"].copy(),
        "info_meg": fwd_meg["info"].copy(),
        "fwd_surface_eeg": fwd_eeg,
        "fwd_surface_meg": fwd_meg,
        "src_surface": fwd_meg["src"],
        "subject": "sample",
        "subjects_dir": sample_path / "subjects",
    }


def _farthest_three(indices: list[int], vertices: np.ndarray) -> list[int]:
    indices = sorted(map(int, indices))
    if len(indices) < 3:
        raise ValueError("each aparc parcel needs at least three coarse-grid vertices")
    xyz = vertices[indices]
    first = int(np.argmin(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1)))
    selected = [first]
    while len(selected) < 3:
        distance = np.min(
            np.linalg.norm(xyz[:, None, :] - xyz[selected][None, :, :], axis=2),
            axis=1,
        )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return [indices[index] for index in selected]


def _parcels(shared: dict) -> dict[str, list[int]]:
    if "parcels" in shared:
        return {name: list(map(int, values)) for name, values in shared["parcels"].items()}
    import mne

    labels = mne.read_labels_from_annot(
        shared.get("subject", "sample"),
        parc="aparc",
        subjects_dir=shared["subjects_dir"],
        verbose=False,
    )
    parcels: dict[str, list[int]] = {}
    offset = 0
    for hemi, src in zip(("lh", "rh"), shared["src_surface"]):
        lookup = {int(vertex): offset + index for index, vertex in enumerate(src["vertno"])}
        for label in labels:
            if label.hemi == hemi:
                parcels[label.name] = [lookup[int(v)] for v in label.vertices if int(v) in lookup]
        offset += len(src["vertno"])
    return parcels


def _farthest_pairs(left: list[int], right: list[int], vertices: np.ndarray) -> list[list[int]]:
    if len(left) != len(right):
        raise ValueError("left/right aparc center counts differ")
    available = set(map(int, right))
    pairs = []
    for lhs in left:
        rhs = max(
            available,
            key=lambda index: (float(np.linalg.norm(vertices[lhs] - vertices[index])), -index),
        )
        pairs.append([int(lhs), int(rhs)])
        available.remove(rhs)
    return pairs


def build_split(shared: dict) -> dict:
    """Freeze one development and two held-out centers in every aparc parcel."""
    vertices = np.asarray(shared["vertices"], dtype=float)
    records = []
    for name, indices in sorted(_parcels(shared).items()):
        dev, test_a, test_b = _farthest_three(indices, vertices)
        hemi = "lh" if name.endswith("-lh") else "rh" if name.endswith("-rh") else ""
        records.append(
            {
                "name": name,
                "hemi": hemi,
                "dev_center": dev,
                "test_centers": [test_a, test_b],
            }
        )
    if len(records) != 68 or {record["hemi"] for record in records} != {"lh", "rh"}:
        raise ValueError("the frozen protocol requires the 68 bilateral aparc parcels")

    dev = [record["dev_center"] for record in records]
    test = [center for record in records for center in record["test_centers"]]
    dev_lh = [record["dev_center"] for record in records if record["hemi"] == "lh"]
    dev_rh = [record["dev_center"] for record in records if record["hemi"] == "rh"]
    test_lh = [center for record in records if record["hemi"] == "lh" for center in record["test_centers"]]
    test_rh = [center for record in records if record["hemi"] == "rh" for center in record["test_centers"]]
    deep_dev = [0, 3, 6, 9, 12]
    n_deep = int(shared["n_deep"])
    if n_deep != 15:
        raise ValueError("the frozen coarse thalamic grid must contain 15 points")
    deep_test = [index for index in range(n_deep) if index not in deep_dev]
    return {
        "protocol": "full-head-coarse-v1",
        "n_surf": int(shared["n_surf"]),
        "n_deep": n_deep,
        "parcels": records,
        "surface_dev": dev,
        "surface_test": test,
        "surface_dev_pairs": _farthest_pairs(dev_lh, dev_rh, vertices),
        "surface_test_pairs": _farthest_pairs(test_lh, test_rh, vertices),
        "deep_dev_local": deep_dev,
        "deep_test_local": deep_test,
    }


def _base_cases(split: dict, panel: str) -> list[dict]:
    surface = split[f"surface_{panel}"]
    pairs = split[f"surface_{panel}_pairs"]
    deep = split[f"deep_{panel}_local"]
    cases = []
    for location, center in enumerate(surface):
        cases.append(
            dict(
                scenario="surface_only",
                location=location,
                replicate=0,
                surface_centers=[center],
                deep_local=None,
            )
        )
    for deep_local in deep:
        for replicate in range(3):
            cases.append(
                dict(
                    scenario="deep_only",
                    location=deep_local,
                    replicate=replicate,
                    surface_centers=[],
                    deep_local=deep_local,
                )
            )
    for location, center in enumerate(surface):
        cases.append(
            dict(
                scenario="deep_plus_surface",
                location=location,
                replicate=0,
                surface_centers=[center],
                deep_local=deep[location % len(deep)],
            )
        )
    for location, pair in enumerate(pairs):
        cases.append(
            dict(
                scenario="deep_plus_two_surface",
                location=location,
                replicate=0,
                surface_centers=pair,
                deep_local=deep[location % len(deep)],
            )
        )
    return cases


def make_manifest(
    split: dict,
    panel: str = "test",
    snrs: tuple[int, ...] = (0, 10, 20),
) -> list[dict]:
    """Create the frozen 185-case development or 1,110-case final manifest."""
    if panel not in {"dev", "test"}:
        raise ValueError("panel must be 'dev' or 'test'")
    snrs = tuple(int(snr) for snr in snrs)
    if not snrs or any(snr < 0 for snr in snrs):
        raise ValueError("snrs must be non-negative integer dB values")
    base = _base_cases(split, panel)
    scheduled = (
        [(snrs[index % len(snrs)], case) for index, case in enumerate(base)]
        if panel == "dev"
        else [(snr, case) for snr in snrs for case in base]
    )
    manifest = []
    for number, (snr, base_case) in enumerate(scheduled):
        case = dict(base_case)
        scenario = case["scenario"]
        code = SCENARIO_CODES[scenario]
        deep_local = case["deep_local"]
        multi_source = scenario in {"deep_plus_surface", "deep_plus_two_surface"}
        case.update(
            {
                "case_id": f"{panel}-{number:04d}-{scenario}-snr{snr:02d}",
                "case_number": number,
                "panel": panel,
                "scenario_code": code,
                "snr_db": snr,
                "deep_index": None
                if deep_local is None
                else int(split["n_surf"] + deep_local),
                "deep_surface_ratio": (0.5, 1.0)[case["location"] % 2]
                if multi_source
                else None,
                "correlation": (0.0, 0.5, 0.9)[case["location"] % 3]
                if multi_source
                else None,
                "seed": [
                    BASE_SEED,
                    code,
                    snr,
                    int(case["location"]),
                    int(case["replicate"]),
                ],
            }
        )
        manifest.append(case)
    return manifest


def _neighbors(adjacency, index: int, n_surf: int) -> np.ndarray:
    if hasattr(adjacency, "getrow"):
        return adjacency.getrow(index)[:, :n_surf].indices
    return np.flatnonzero(np.asarray(adjacency[index, :n_surf]))


def _surface_patch(shared: dict, center: int) -> tuple[np.ndarray, np.ndarray]:
    n_surf = int(shared["n_surf"])
    seen, frontier = {int(center)}, {int(center)}
    for _ in range(2):
        frontier = {
            int(neighbor)
            for index in frontier
            for neighbor in _neighbors(shared["adjacency"], index, n_surf)
            if int(neighbor) not in seen
        }
        seen.update(frontier)
    indices = np.array(sorted(seen), dtype=int)
    distance = np.linalg.norm(
        np.asarray(shared["vertices"])[indices] - np.asarray(shared["vertices"])[center],
        axis=1,
    )
    weights = np.exp(-(distance**2) / (2 * 0.010**2))
    weights /= np.linalg.norm(weights)
    return indices, weights


def _waveforms(times: np.ndarray, count: int, correlation: float) -> np.ndarray:
    active_times = times[ACTIVE_START:] - times[ACTIVE_START]
    envelope = np.sin(np.linspace(0.0, np.pi, active_times.size))
    candidates = np.column_stack(
        [envelope * np.sin(2 * np.pi * (11 + 3 * index) * active_times + index / 3) for index in range(count)]
    )
    candidates -= candidates.mean(axis=0, keepdims=True)
    basis, _ = np.linalg.qr(candidates)
    target = np.full((count, count), float(correlation))
    np.fill_diagonal(target, 1.0)
    active = np.linalg.cholesky(target) @ basis[:, :count].T
    waves = np.zeros((count, times.size), dtype=float)
    waves[:, ACTIVE_START:] = active
    return waves


def _add_noise(
    clean: np.ndarray,
    factor: np.ndarray,
    snr_db: float,
    seed: np.random.SeedSequence,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    noise = np.asarray(factor) @ rng.standard_normal(clean.shape)
    clean_energy = float(np.sum(clean[:, ACTIVE_START:] ** 2))
    noise_energy = float(np.sum(noise[:, ACTIVE_START:] ** 2))
    if clean_energy == 0.0 or noise_energy == 0.0:
        raise ValueError("active clean signal and colored noise must have non-zero energy")
    noise *= np.sqrt(clean_energy / (noise_energy * 10 ** (snr_db / 10.0)))
    actual = 10 * np.log10(clean_energy / np.sum(noise[:, ACTIVE_START:] ** 2))
    return clean + noise, float(actual)


def truth_for_case(
    shared: dict, case: dict
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Build a case's frozen source truth without regenerating its observations."""
    times = np.asarray(shared["times"], dtype=float)
    if int(shared.get("active_start", ACTIVE_START)) != ACTIVE_START or times.size <= ACTIVE_START:
        raise ValueError("the frozen protocol requires 200 baseline samples and an active tail")
    surface_parts = [_surface_patch(shared, center) for center in case["surface_centers"]]
    groups = [indices for indices, _ in surface_parts]
    if case.get("deep_index") is not None:
        groups.append(np.array([int(case["deep_index"])], dtype=int))
    correlation = float(case.get("correlation") or 0.0)
    waves = _waveforms(times, len(groups), correlation)
    components = []
    for (indices, weights), wave in zip(surface_parts, waves):
        component = np.zeros((int(shared["n_surf"]) + int(shared["n_deep"]), times.size))
        component[indices] = weights[:, None] * wave
        components.append(("surface", component))
    if case.get("deep_index") is not None:
        component = np.zeros((int(shared["n_surf"]) + int(shared["n_deep"]), times.size))
        component[int(case["deep_index"])] = waves[len(surface_parts)]
        components.append(("deep", component))

    if surface_parts and case.get("deep_index") is not None:
        surface_norm = np.linalg.norm(sum(part for layer, part in components if layer == "surface"))
        ratio = float(case["deep_surface_ratio"])
        components[-1] = ("deep", components[-1][1] * ratio * surface_norm)
    truth = sum((component for _, component in components), start=np.zeros_like(components[0][1]))
    surface_truth = sum(
        (part for layer, part in components if layer == "surface"),
        start=np.zeros_like(truth),
    )
    deep_truth = sum(
        (part for layer, part in components if layer == "deep"),
        start=np.zeros_like(truth),
    )
    meta = {
        "case_id": case["case_id"],
        "active_start": ACTIVE_START,
        "groups": [group.tolist() for group in groups],
        "component_frobenius": [float(np.linalg.norm(part)) for _, part in components],
        "deep_surface_ratio_actual": (
            float(np.linalg.norm(deep_truth) / np.linalg.norm(surface_truth))
            if np.linalg.norm(surface_truth) and np.linalg.norm(deep_truth)
            else None
        ),
    }
    return truth, groups, meta


def simulate_case(shared: dict, case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], dict]:
    """Stream one case: EEG, MEG, dense truth, source groups, and audit metadata."""
    truth, groups, meta = truth_for_case(shared, case)
    clean_eeg = np.asarray(shared["gain_eeg"]) @ truth
    clean_meg = np.asarray(shared["gain_meg"]) @ truth
    seeds = np.random.SeedSequence(case["seed"]).spawn(2)
    eeg_target_snr = case.get("eeg_snr_db", case["snr_db"])
    meg_target_snr = case.get("meg_snr_db", case["snr_db"])
    eeg, eeg_snr = _add_noise(
        clean_eeg, shared["noise_factor_eeg"], eeg_target_snr, seeds[0]
    )
    meg, meg_snr = _add_noise(
        clean_meg, shared["noise_factor_meg"], meg_target_snr, seeds[1]
    )
    meta["actual_snr_db"] = {"eeg": eeg_snr, "meg": meg_snr}
    return eeg, meg, truth, groups, meta


def save_manifest(
    path: str | Path,
    manifest: list[dict],
    *,
    expected_digest: str | None = None,
) -> str:
    """Atomically write canonical JSON after optional frozen-digest validation."""
    path = Path(path)
    payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_digest is not None and digest != expected_digest:
        raise RuntimeError(
            f"refusing to overwrite frozen manifest: expected {expected_digest}, got {digest}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_paths: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            stream.write(payload)
            temporary_paths.append(Path(stream.name))
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="ascii", dir=path.parent, delete=False
        ) as stream:
            stream.write(f"{digest}  {path.name}\n")
            temporary_paths.append(Path(stream.name))
        os.replace(temporary_paths[0], path)
        os.replace(temporary_paths[1], sidecar)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return digest
