"""Run rebuilt OASTER directly on the immutable strict-blind MAT chunks.

This runner never synthesizes observations.  It reads ``F_EEG`` and ``F_MEG``
from the archived chunks, reconstructs truth only for scoring via the frozen
manifest protocol, and checkpoints one atomically replaced CSV per chunk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Each Python worker is already a case-level thread.  Prevent BLAS from nesting
# another full thread pool under every case unless the caller explicitly chose
# different limits before starting Python.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy.io as sio
from scipy import sparse

import protected_multilayer as protected
from benchmark import metrics as benchmark_metrics
from benchmark import protocol
from candidates import oaster_rebuilt as oaster


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = Path(
    os.environ.get("STRICT_INPUT_ROOT", r"D:\oaster_strict_blind_sisses\matlab_input")
)
DEFAULT_MANIFEST = ROOT / "results" / "strict_blind" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "strict_blind" / "oaster_rebuilt"
LEGACY_RESULTS_ROOT = ROOT / "results" / "strict_blind"
EXPECTED_MANIFEST_SHA256 = (
    "3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76"
)
EXPECTED_CASES = 9065
FORMAT_VERSION = "strict_blind_sisses_input_v2"
METHOD = "OASTER_rebuilt"
CHUNK_PATTERN = re.compile(r"^strict_(\d+)_(\d+)\.mat$")

METRIC_FIELDS = (
    "auc",
    "auc_tie_corrected",
    "rmse",
    "surface_sd_mm",
    "surface_dle_mm",
    "deep_sd_mm",
    "deep_dle_mm",
    "has_deep_true",
    "deep_score",
    "deep_peak_distance_mm",
    "deep_detected",
    "deep_false_positive",
    "active_count",
)
ROW_FIELDS = (
    "manifest_sha256",
    "chunk_index",
    "chunk_file",
    "case_number",
    "case_id",
    "configuration_number",
    "configuration_id",
    "pair_index",
    "eeg_snr_db",
    "meg_snr_db",
    "scenario",
    "surface_centers",
    "deep_index",
    "deep_surface_ratio",
    "correlation",
    "method",
    "status",
    "error",
    "elapsed_seconds",
    "temporal_rank",
    "selected_templates",
    "has_surface_true",
    *METRIC_FIELDS,
)
SUMMARY_METRICS = (
    "auc",
    "auc_tie_corrected",
    "rmse",
    "surface_sd_mm",
    "surface_dle_mm",
    "deep_sd_mm",
    "deep_dle_mm",
    "surface_sd_mm_penalized",
    "surface_dle_mm_penalized",
    "deep_sd_mm_penalized",
    "deep_dle_mm_penalized",
    "deep_score",
    "deep_peak_distance_mm",
    "active_count",
)


@dataclass(frozen=True)
class Chunk:
    index: int
    path: Path
    start: int
    stop: int

    @property
    def case_count(self) -> int:
        return self.stop - self.start


def _text(value: object) -> str:
    """Normalize a scalar MATLAB string/cell to Python text."""
    item = value
    while isinstance(item, np.ndarray) and item.size == 1:
        item = item.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _string_list(value: object) -> list[str]:
    return [_text(item) for item in np.atleast_1d(value).ravel()]


def _load_manifest(path: Path) -> tuple[list[dict], str]:
    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    is_legacy_default = path.resolve() == DEFAULT_MANIFEST.resolve()
    if is_legacy_default and digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"strict manifest SHA-256 mismatch: expected "
            f"{EXPECTED_MANIFEST_SHA256}, got {digest}"
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_fields = (
        sidecar.read_text(encoding="ascii").split() if sidecar.exists() else []
    )
    if sidecar_fields[:2] != [digest, path.name]:
        raise RuntimeError(f"manifest sidecar checksum mismatch: {sidecar}")
    cases = json.loads(payload)
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("strict manifest must contain a non-empty case list")
    if is_legacy_default and len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"strict manifest must contain {EXPECTED_CASES} cases")
    numbers = [int(case["case_number"]) for case in cases]
    ids = [str(case["case_id"]) for case in cases]
    if numbers != list(range(len(cases))) or len(set(ids)) != len(ids):
        raise RuntimeError("manifest case_number order or case_id uniqueness is invalid")
    return cases, digest


def _guard_explicit_manifest_paths(
    manifest_path: Path, input_root: Path, data_root: Path, output: Path
) -> None:
    if Path(manifest_path).resolve() == DEFAULT_MANIFEST.resolve():
        return
    resolved_output = Path(output).resolve()
    legacy_results = LEGACY_RESULTS_ROOT.resolve()
    unsafe = []
    if resolved_output == legacy_results or legacy_results in resolved_output.parents:
        unsafe.append("legacy output tree")
    if Path(input_root).resolve() == DEFAULT_INPUT_ROOT.resolve():
        unsafe.append("legacy input archive")
    if Path(data_root).resolve() == protocol.DEFAULT_DATA_ROOT.resolve():
        unsafe.append("default legacy data root")
    if unsafe:
        raise ValueError(
            "an explicit non-default manifest requires isolated explicit paths; "
            + ", ".join(unsafe)
        )


def _discover_chunks(input_root: Path, case_count: int) -> list[Chunk]:
    found: list[tuple[int, int, Path]] = []
    for path in input_root.glob("strict_*.mat"):
        match = CHUNK_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    found.sort(key=lambda item: (item[0], item[1], item[2].name))
    if not found:
        raise FileNotFoundError(f"no strict_*.mat chunks in {input_root}")
    expected_start = 0
    chunks = []
    for index, (start, stop, path) in enumerate(found):
        if start != expected_start or stop <= start or stop > case_count:
            raise RuntimeError(
                f"chunk ranges must partition [0,{case_count}); invalid {path.name}"
            )
        chunks.append(Chunk(index, path, start, stop))
        expected_start = stop
    if expected_start != case_count:
        raise RuntimeError(
            f"chunk ranges stop at {expected_start}, not manifest size {case_count}"
        )
    return chunks


def _load_geometry(data_root: Path) -> dict:
    generated = (
        data_root
        if (data_root / "deep_plus_two_surface" / "sub_EEG.mat").exists()
        else data_root / "generated"
    )
    path = generated / "deep_plus_two_surface" / "sub_EEG.mat"
    if not path.exists():
        raise FileNotFoundError(f"geometry reference not found: {path}")
    values = sio.loadmat(
        path,
        variable_names=("src_vertices", "times", "n_surf", "n_deep"),
        simplify_cells=True,
    )
    geometry = {
        "path": path,
        "vertices": np.asarray(values["src_vertices"], dtype=float),
        "times": np.asarray(values["times"], dtype=float).ravel(),
        "n_surf": int(np.asarray(values["n_surf"]).ravel()[0]),
        "n_deep": int(np.asarray(values["n_deep"]).ravel()[0]),
    }
    n_sources = geometry["n_surf"] + geometry["n_deep"]
    if geometry["vertices"].shape != (n_sources, 3):
        raise RuntimeError("geometry source count does not match n_surf + n_deep")
    if geometry["times"].size <= protocol.ACTIVE_START:
        raise RuntimeError("geometry does not contain the frozen 200-sample baseline")
    if not np.all(np.isfinite(geometry["vertices"])):
        raise RuntimeError("geometry contains NaN or infinity")
    return geometry


def _canonical_adjacency(value: object) -> sparse.csr_matrix:
    graph = sparse.csr_matrix(value)
    graph.sum_duplicates()
    graph.eliminate_zeros()
    graph.sort_indices()
    if graph.nnz:
        graph.data = np.ones(graph.nnz, dtype=np.uint8)
    return graph.astype(np.uint8)


def _dense_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _graph_digest(value: sparse.csr_matrix) -> str:
    graph = _canonical_adjacency(value)
    digest = hashlib.sha256(np.asarray(graph.shape, dtype=np.int64).tobytes())
    for array in (graph.indptr, graph.indices, graph.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _load_chunk(path: Path, *, observations: bool) -> dict:
    names = [
        "Gain_EEG",
        "Gain_MEG",
        "VertConn",
        "case_ids",
        "manifest_sha256",
        "format_version",
    ]
    if observations:
        names += ["F_EEG", "F_MEG"]
    values = sio.loadmat(path, variable_names=names, simplify_cells=True)
    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError(f"{path.name} lacks variables: {missing}")
    result = {
        "gain_eeg": np.asarray(values["Gain_EEG"], dtype=float),
        "gain_meg": np.asarray(values["Gain_MEG"], dtype=float),
        "adjacency": _canonical_adjacency(values["VertConn"]),
        "case_ids": _string_list(values["case_ids"]),
        "manifest_sha256": _text(values["manifest_sha256"]),
        "format_version": _text(values["format_version"]),
    }
    if observations:
        result["f_eeg"] = np.asarray(values["F_EEG"], dtype=float)
        result["f_meg"] = np.asarray(values["F_MEG"], dtype=float)
    return result


def _fingerprints(chunk: dict) -> dict[str, str]:
    return {
        "Gain_EEG": _dense_digest(chunk["gain_eeg"]),
        "Gain_MEG": _dense_digest(chunk["gain_meg"]),
        "VertConn": _graph_digest(chunk["adjacency"]),
    }


def _validate_chunk(
    spec: Chunk,
    chunk: dict,
    cases: list[dict],
    manifest_sha256: str,
    geometry: dict,
    reference_fingerprints: dict[str, str],
) -> None:
    expected = cases[spec.start : spec.stop]
    expected_ids = [str(case["case_id"]) for case in expected]
    expected_numbers = list(range(spec.start, spec.stop))
    if [int(case["case_number"]) for case in expected] != expected_numbers:
        raise RuntimeError(f"manifest range disagrees with {spec.path.name}")
    if chunk["case_ids"] != expected_ids:
        raise RuntimeError(f"case_ids disagree with manifest range in {spec.path.name}")
    if chunk["manifest_sha256"] != manifest_sha256:
        raise RuntimeError(f"embedded manifest SHA-256 mismatch in {spec.path.name}")
    if chunk["format_version"] != FORMAT_VERSION:
        raise RuntimeError(f"unsupported format_version in {spec.path.name}")

    n_sources = int(geometry["n_surf"] + geometry["n_deep"])
    if chunk["gain_eeg"].ndim != 2 or chunk["gain_eeg"].shape[1] != n_sources:
        raise RuntimeError(f"Gain_EEG source dimension mismatch in {spec.path.name}")
    if chunk["gain_meg"].ndim != 2 or chunk["gain_meg"].shape[1] != n_sources:
        raise RuntimeError(f"Gain_MEG source dimension mismatch in {spec.path.name}")
    if chunk["adjacency"].shape != (n_sources, n_sources):
        raise RuntimeError(f"VertConn source dimension mismatch in {spec.path.name}")
    for name, digest in _fingerprints(chunk).items():
        if digest != reference_fingerprints[name]:
            raise RuntimeError(f"{name} differs across strict chunks: {spec.path.name}")

    if "f_eeg" not in chunk:
        return
    expected_shape_eeg = (
        chunk["gain_eeg"].shape[0],
        geometry["times"].size,
        spec.case_count,
    )
    expected_shape_meg = (
        chunk["gain_meg"].shape[0],
        geometry["times"].size,
        spec.case_count,
    )
    if spec.case_count == 1 and chunk["f_eeg"].ndim == 2:
        chunk["f_eeg"] = chunk["f_eeg"][:, :, None]
    if spec.case_count == 1 and chunk["f_meg"].ndim == 2:
        chunk["f_meg"] = chunk["f_meg"][:, :, None]
    if chunk["f_eeg"].shape != expected_shape_eeg:
        raise RuntimeError(
            f"F_EEG shape {chunk['f_eeg'].shape} != {expected_shape_eeg} in {spec.path.name}"
        )
    if chunk["f_meg"].shape != expected_shape_meg:
        raise RuntimeError(
            f"F_MEG shape {chunk['f_meg'].shape} != {expected_shape_meg} in {spec.path.name}"
        )
    if not np.all(np.isfinite(chunk["f_eeg"])) or not np.all(np.isfinite(chunk["f_meg"])):
        raise RuntimeError(f"observations contain NaN or infinity in {spec.path.name}")


def _runtime(geometry: dict, reference: dict) -> dict:
    adjacency_dense = reference["adjacency"].toarray().astype(np.uint8, copy=False)
    kernels = protected.connected_euclidean_surface_kernels(
        geometry["vertices"],
        adjacency_dense,
        geometry["n_surf"],
        scales_mm=oaster.SURFACE_SCALES_MM,
    )
    shared = {
        "times": geometry["times"],
        "vertices": geometry["vertices"],
        "adjacency": reference["adjacency"],
        "n_surf": geometry["n_surf"],
        "n_deep": geometry["n_deep"],
        "active_start": protocol.ACTIVE_START,
    }
    return {
        "shared": shared,
        "kernels": kernels,
        "cortex": protocol._auc_cortex(
            geometry["vertices"], adjacency_dense, geometry["n_surf"]
        ),
        "active": np.arange(protocol.ACTIVE_START, geometry["times"].size),
        "penalty_mm": float(np.linalg.norm(np.ptp(geometry["vertices"], axis=0)) * 1000.0),
    }


def _case_fields(
    case: dict, spec: Chunk, manifest_sha256: str, has_surface_true: int
) -> dict:
    return {
        "manifest_sha256": manifest_sha256,
        "chunk_index": spec.index,
        "chunk_file": spec.path.name,
        "case_number": int(case["case_number"]),
        "case_id": str(case["case_id"]),
        "configuration_number": case.get("configuration_number", ""),
        "configuration_id": case.get("configuration_id", ""),
        "pair_index": case.get("pair_index", ""),
        "eeg_snr_db": int(case["eeg_snr_db"]),
        "meg_snr_db": int(case["meg_snr_db"]),
        "scenario": str(case["scenario"]),
        "surface_centers": json.dumps(case.get("surface_centers", [])),
        "deep_index": "" if case.get("deep_index") is None else int(case["deep_index"]),
        "deep_surface_ratio": (
            "" if case.get("deep_surface_ratio") is None else case["deep_surface_ratio"]
        ),
        "correlation": "" if case.get("correlation") is None else case["correlation"],
        "method": METHOD,
        "has_surface_true": has_surface_true,
    }


def _score_case(
    case: dict,
    local_index: int,
    spec: Chunk,
    chunk: dict,
    runtime: dict,
    manifest_sha256: str,
) -> dict:
    began = time.perf_counter()
    base = _case_fields(
        case,
        spec,
        manifest_sha256,
        int(bool(case.get("surface_centers"))),
    )
    try:
        # This is the sole truth construction call.  Observations remain the
        # archived F_EEG/F_MEG arrays and are never regenerated from truth.
        truth, groups, _truth_meta = protocol.truth_for_case(runtime["shared"], case)
        estimate, diagnostics = oaster.reconstruct(
            chunk["f_eeg"][:, :, local_index],
            chunk["f_meg"][:, :, local_index],
            chunk["gain_eeg"],
            chunk["gain_meg"],
            runtime["shared"]["n_surf"],
            runtime["kernels"],
        )
        metrics = benchmark_metrics.evaluate_estimate(
            estimate,
            truth,
            runtime["shared"]["vertices"],
            groups,
            runtime["shared"]["n_surf"],
            runtime["active"],
            runtime["cortex"],
        )
        base.update(
            status="ok",
            error="",
            elapsed_seconds=time.perf_counter() - began,
            temporal_rank=int(diagnostics["temporal_rank"]),
            selected_templates=int(diagnostics["selected_templates"]),
            **metrics,
        )
    except Exception as exc:  # Preserve a complete, auditable failed checkpoint.
        base.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - began,
            temporal_rank="",
            selected_templates="",
            **{field: np.nan for field in METRIC_FIELDS},
        )
    return base


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _atomic_csv(path: Path, rows: list[dict], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_completion(
    output: Path,
    rows: list[dict],
    *,
    expected_row_count: int,
    manifest_sha256: str,
    chunk_count: int,
    methods: Iterable[str],
) -> None:
    """Atomically mark a fully validated run complete."""
    failed = sum(row.get("status") != "ok" for row in rows)
    if len(rows) != expected_row_count or failed:
        raise RuntimeError(
            f"refusing completion marker: rows={len(rows)}/{expected_row_count}, "
            f"errors={failed}"
        )
    method_names = list(methods)
    payload = {
        "status": "complete",
        "row_count": len(rows),
        "expected_row_count": expected_row_count,
        "error_count": failed,
        "manifest_sha256": manifest_sha256,
        "chunk_count": chunk_count,
        "method_count": len(method_names),
        "methods": method_names,
    }
    path = output / "completion.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _provenance(files: dict[str, str | Path]) -> dict:
    """Record the exact code and numerical environment used for a final merge."""
    packages = ("numpy", "scipy", "h5py", "matplotlib", "mne", "nibabel")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "git_commit_at_merge": commit,
        "python": platform.python_version(),
        "packages": versions,
        "code_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in files.items()
        },
        "bitwise_cross_environment_reproducibility_claimed": False,
    }


def _valid_part(
    rows: list[dict], spec: Chunk, cases: list[dict], manifest_sha256: str
) -> bool:
    expected = cases[spec.start : spec.stop]
    try:
        return (
            len(rows) == spec.case_count
            and [row["case_id"] for row in rows]
            == [str(case["case_id"]) for case in expected]
            and [int(row["case_number"]) for row in rows]
            == list(range(spec.start, spec.stop))
            and all(row["manifest_sha256"] == manifest_sha256 for row in rows)
            and all(row["chunk_file"] == spec.path.name for row in rows)
            and all(row["method"] == METHOD and row["status"] == "ok" for row in rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _aggregate(rows: list[dict], penalty_mm: float) -> dict:
    result = {
        "case_count": len(rows),
        "scenario_count": len({row["scenario"] for row in rows}),
    }
    for name in (
        "auc",
        "auc_tie_corrected",
        "rmse",
        "surface_sd_mm",
        "surface_dle_mm",
        "deep_sd_mm",
        "deep_dle_mm",
        "deep_score",
        "deep_peak_distance_mm",
        "active_count",
    ):
        result[name] = _mean(_number(row, name) for row in rows)

    for layer in ("surface", "deep"):
        expected_key = "has_surface_true" if layer == "surface" else "has_deep_true"
        expected_rows = [row for row in rows if int(_number(row, expected_key)) == 1]
        for metric in ("sd_mm", "dle_mm"):
            name = f"{layer}_{metric}"
            values = [_number(row, name) for row in expected_rows]
            result[name + "_penalized"] = (
                float(np.mean([value if np.isfinite(value) else penalty_mm for value in values]))
                if values
                else np.nan
            )

    positives = [row for row in rows if int(_number(row, "has_deep_true")) == 1]
    negatives = [row for row in rows if int(_number(row, "has_deep_true")) == 0]
    sensitivity = _mean(_number(row, "deep_detected") for row in positives)
    specificity = _mean(1.0 - _number(row, "deep_false_positive") for row in negatives)
    result["deep_sensitivity"] = sensitivity
    result["deep_specificity"] = specificity
    result["deep_balanced_accuracy"] = (
        (sensitivity + specificity) / 2.0
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else np.nan
    )
    return result


def _summaries(rows: list[dict], output: Path, penalty_mm: float) -> None:
    cells: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(int(row["eeg_snr_db"]), int(row["meg_snr_db"]), row["scenario"])].append(row)

    cell_rows = []
    for (eeg_snr, meg_snr, scenario), group in sorted(cells.items()):
        cell_rows.append(
            {
                "eeg_snr_db": eeg_snr,
                "meg_snr_db": meg_snr,
                "scenario": scenario,
                "aggregation": "case_weighted_within_scenario",
                **_aggregate(group, penalty_mm),
            }
        )
    _atomic_csv(
        output / "summary_by_snr_scenario.csv", cell_rows, cell_rows[0].keys()
    )

    pair_cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    pair_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in cell_rows:
        pair_cells[(int(row["eeg_snr_db"]), int(row["meg_snr_db"]))].append(row)
    for row in rows:
        pair_rows[(int(row["eeg_snr_db"]), int(row["meg_snr_db"]))].append(row)

    macro_rows = []
    weighted_rows = []
    for pair in sorted(pair_rows):
        eeg_snr, meg_snr = pair
        summaries = pair_cells[pair]
        macro = {
            "eeg_snr_db": eeg_snr,
            "meg_snr_db": meg_snr,
            "aggregation": "scenario_macro",
            "case_count": sum(int(row["case_count"]) for row in summaries),
            "scenario_count": len(summaries),
        }
        for name in SUMMARY_METRICS:
            macro[name] = _mean(_number(row, name) for row in summaries)
        macro["deep_sensitivity"] = _mean(
            _number(row, "deep_sensitivity") for row in summaries
        )
        macro["deep_specificity"] = _mean(
            _number(row, "deep_specificity") for row in summaries
        )
        macro["deep_balanced_accuracy"] = (
            (macro["deep_sensitivity"] + macro["deep_specificity"]) / 2.0
            if np.isfinite(macro["deep_sensitivity"])
            and np.isfinite(macro["deep_specificity"])
            else np.nan
        )
        macro_rows.append(macro)

        weighted_rows.append(
            {
                "eeg_snr_db": eeg_snr,
                "meg_snr_db": meg_snr,
                "aggregation": "case_weighted",
                **_aggregate(pair_rows[pair], penalty_mm),
            }
        )

    _atomic_csv(
        output / "summary_by_snr_pair_scenario_macro.csv",
        macro_rows,
        macro_rows[0].keys(),
    )
    _atomic_csv(
        output / "summary_by_snr_pair_case_weighted.csv",
        weighted_rows,
        weighted_rows[0].keys(),
    )


def _metadata(
    output: Path,
    manifest_path: Path,
    manifest_sha256: str,
    input_root: Path,
    geometry: dict,
    chunks: list[Chunk],
    start_chunk: int,
    limit_chunks: int | None,
    workers: int,
    reference_fingerprints: dict[str, str],
) -> None:
    payload = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "input_root": str(input_root.resolve()),
        "geometry_reference": str(geometry["path"].resolve()),
        "method": METHOD,
        "adapter": "candidates.oaster_rebuilt",
        "observations": "archived F_EEG/F_MEG; never regenerated",
        "truth": "benchmark.protocol.truth_for_case; scoring only",
        "metrics": {
            "auc": "raw requested An_cal_AUC",
            "auc_tie_corrected": "corrected tied-rank An_auc",
            "spatial": "surface/deep SD and DLE",
            "deep_balanced_accuracy": "mean(deep sensitivity, surface-only specificity)",
        },
        "surface_scales_mm": list(oaster.SURFACE_SCALES_MM),
        "ridge_fraction": oaster.RIDGE_FRACTION,
        "spectral_fraction": oaster.SPECTRAL_FRACTION,
        "deep_rescue_tau": oaster.DEEP_RESCUE_TAU,
        "maximum_deep_rescues": oaster.MAX_DEEP_RESCUES,
        "deep_threshold": 0.1,
        "threshold_recalibrated_on_strict_blind": False,
        "checkpoint": "one atomically replaced CSV per immutable input chunk",
        "chunk_count": len(chunks),
        "start_chunk": start_chunk,
        "limit_chunks": limit_chunks,
        "workers": workers,
        "cross_chunk_fingerprints_sha256": reference_fingerprints,
        "provenance": _provenance(
            {
                "strict_runner": __file__,
                "oaster_algorithm": oaster.__file__,
                "metrics": benchmark_metrics.__file__,
            }
        ),
    }
    path = output / "metadata.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(
    manifest_path: Path = DEFAULT_MANIFEST,
    input_root: Path = DEFAULT_INPUT_ROOT,
    data_root: Path = protocol.DEFAULT_DATA_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    workers: int = 1,
    start_chunk: int = 0,
    limit_chunks: int | None = None,
    force: bool = False,
) -> None:
    """Evaluate selected immutable chunks and merge only a complete run."""
    _guard_explicit_manifest_paths(manifest_path, input_root, data_root, output)
    cases, manifest_sha256 = _load_manifest(Path(manifest_path))
    chunks = _discover_chunks(Path(input_root), len(cases))
    if workers < 1 or start_chunk < 0 or start_chunk >= len(chunks):
        raise ValueError("workers and start_chunk are outside their valid range")
    if limit_chunks is not None and limit_chunks < 1:
        raise ValueError("limit_chunks must be positive")
    selected = chunks[start_chunk:]
    if limit_chunks is not None:
        selected = selected[:limit_chunks]

    geometry = _load_geometry(Path(data_root))
    reference = _load_chunk(chunks[0].path, observations=False)
    reference_fingerprints = _fingerprints(reference)
    _validate_chunk(
        chunks[0],
        reference,
        cases,
        manifest_sha256,
        geometry,
        reference_fingerprints,
    )

    output = Path(output)
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    full_run = start_chunk == 0 and limit_chunks is None
    if full_run:
        (output / "completion.json").unlink(missing_ok=True)
    runtime = None
    for spec in selected:
        part_path = parts / f"{spec.path.stem}.csv"
        existing = _read_csv(part_path)
        if not force and _valid_part(existing, spec, cases, manifest_sha256):
            header = _load_chunk(spec.path, observations=False)
            _validate_chunk(
                spec,
                header,
                cases,
                manifest_sha256,
                geometry,
                reference_fingerprints,
            )
            print(f"chunk {spec.index}: verified checkpoint {part_path.name}", flush=True)
            continue

        chunk = _load_chunk(spec.path, observations=True)
        _validate_chunk(
            spec,
            chunk,
            cases,
            manifest_sha256,
            geometry,
            reference_fingerprints,
        )
        if runtime is None:
            runtime = _runtime(geometry, reference)
        batch = cases[spec.start : spec.stop]
        arguments = [
            (case, index, spec, chunk, runtime, manifest_sha256)
            for index, case in enumerate(batch)
        ]
        if workers == 1:
            rows = [_score_case(*argument) for argument in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rows = list(pool.map(lambda argument: _score_case(*argument), arguments))
        _atomic_csv(part_path, rows, ROW_FIELDS)
        failed = [row for row in rows if row["status"] != "ok"]
        if failed:
            raise RuntimeError(
                f"{spec.path.name} has {len(failed)} failures; first={failed[0]['error']}"
            )
        if not _valid_part(rows, spec, cases, manifest_sha256):
            raise RuntimeError(f"invalid checkpoint produced for {spec.path.name}")
        print(f"chunk {spec.index}: scored {len(rows)} cases", flush=True)

    combined = None
    if full_run:
        combined = []
        for spec in chunks:
            rows = _read_csv(parts / f"{spec.path.stem}.csv")
            if not _valid_part(rows, spec, cases, manifest_sha256):
                raise RuntimeError(f"incomplete or invalid checkpoint for {spec.path.name}")
            combined.extend(rows)
        _atomic_csv(output / "rows.csv", combined, ROW_FIELDS)
        penalty_mm = float(np.linalg.norm(np.ptp(geometry["vertices"], axis=0)) * 1000.0)
        _summaries(combined, output, penalty_mm)

    _metadata(
        output,
        Path(manifest_path),
        manifest_sha256,
        Path(input_root),
        geometry,
        chunks,
        start_chunk,
        limit_chunks,
        workers,
        reference_fingerprints,
    )
    if full_run:
        _write_completion(
            output,
            combined,
            expected_row_count=len(cases),
            manifest_sha256=manifest_sha256,
            chunk_count=len(chunks),
            methods=(METHOD,),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=protocol.DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.start_chunk < 0
        or (args.limit_chunks is not None and args.limit_chunks < 1)
    ):
        parser.error("workers/limit-chunks must be positive and start-chunk non-negative")
    run(
        args.manifest,
        args.input_root,
        args.data_root,
        args.output,
        workers=args.workers,
        start_chunk=args.start_chunk,
        limit_chunks=args.limit_chunks,
        force=args.force,
    )


if __name__ == "__main__":
    main()
