"""Run the phase-locked ERP whole-head benchmark on one shared observation.

Each manifest case is simulated once, independently whitened once per modality,
and then sent to OASTER-ERP and the same seven fixed-grid comparators. One atomic
checkpoint is written per EEG/MEG SNR pair so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

import protected_multilayer as protected
import run_strict_comparators as comparators
import run_strict_oaster as archive
from benchmark import erp_protocol, methods as comparator_methods
from benchmark import metrics as benchmark_metrics
from benchmark import protocol
from candidates import oaster_rebuilt as oaster


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
DEFAULT_DATA_ROOT = ROOT / "corrected_v2" / "generated"
DEFAULT_SAMPLE_PATH = Path(
    os.environ.get("MNE_SAMPLE_PATH", r"D:\mne_data\MNE-sample-data")
)
DEFAULT_OUTPUT = ROOT / "results" / "erp_whole_head" / "current_method_v1"
DEFAULT_V4_OUTPUT = ROOT / "results" / "erp_whole_head" / "adaptive_v4" / "all_methods"
DEFAULT_V5_OUTPUT = ROOT / "results" / "erp_whole_head" / "adaptive_v5" / "pilot_five_snr_all_methods"
EXPECTED_CASES = 9114
METHODS = ("OASTER-ERP",) + comparators.METHODS
SCENARIOS = (
    "surface_only",
    "deep_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)
ALGORITHM_VERSIONS = ("v1", "v2", "v3", "v4", "v5")
ADAPTIVE_VERSIONS = ("v4", "v5")
ADAPTIVE_CODE = {
    "v4": {
        "oaster_adaptive": ROOT / "candidates" / "oaster_adaptive.py",
        "graph_solver_helpers": ROOT / "algorithms" / "spatial_fused_fusion.py",
    },
    "v5": {
        "oaster_balanced": ROOT / "candidates" / "oaster_balanced.py",
        "graph_reweight_solver": ROOT / "candidates" / "graph_reweight_solver.py",
        "graph_irls": ROOT / "candidates" / "graph_irls.py",
        "graph_solver_helpers": ROOT / "algorithms" / "spatial_fused_fusion.py",
    },
}
MODALITY_WEIGHTINGS = ("equal", "evidence")


def _code_fingerprint(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return digest.hexdigest()


def _shared_fingerprint(shared: dict) -> str:
    """Bind checkpoints to every numeric input used to simulate or reconstruct."""
    digest = hashlib.sha256()
    for name in (
        "gain_eeg",
        "gain_meg",
        "vertices",
        "adjacency",
        "times",
        "noise_factor_eeg",
        "noise_factor_meg",
        "n_surf",
        "n_deep",
    ):
        values = np.ascontiguousarray(np.asarray(shared[name]))
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _environment_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in ("numpy", "scipy", "mne")
        },
        **{
            name: os.environ.get(name, "")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _checkpoint_fingerprint(
    manifest_sha256: str,
    code_fingerprint: str,
    shared_fingerprint: str,
    environment: dict[str, str],
    configuration: dict | None = None,
) -> str:
    payload = {
        "manifest": manifest_sha256,
        "code": code_fingerprint,
        "shared": shared_fingerprint,
        "environment": environment,
        "configuration": configuration or {},
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _oaster_method(algorithm_version: str) -> str:
    return {
        "v1": "OASTER-ERP",
        "v2": "OASTER-ERP-v2",
        "v3": "OASTER-ERP-v3",
        "v4": "OASTER-ERP-v4",
        "v5": "OASTER-ERP-v5",
    }[algorithm_version]


def _resolve_oaster(algorithm_version: str):
    if algorithm_version not in ALGORITHM_VERSIONS:
        raise ValueError(f"algorithm_version must be one of {ALGORITHM_VERSIONS}")
    if algorithm_version == "v4":
        from candidates.oaster_adaptive import reconstruct_evoked_oaster_v4_from_whitened

        return reconstruct_evoked_oaster_v4_from_whitened
    if algorithm_version == "v5":
        from candidates.oaster_balanced import reconstruct_evoked_oaster_v5_from_whitened

        return reconstruct_evoked_oaster_v5_from_whitened
    name = {
        "v1": "reconstruct_evoked_oaster_from_whitened",
        "v2": "reconstruct_evoked_oaster_v2_from_whitened",
        "v3": "reconstruct_evoked_oaster_v3_from_whitened",
    }[algorithm_version]
    solver = getattr(oaster, name, None)
    if solver is None:
        raise RuntimeError(
            f"OASTER {algorithm_version} entry point is unavailable: {name}"
        )
    return solver


def _modality_evidence_weights(
    data_blocks: tuple[np.ndarray, ...], baseline: np.ndarray, active: np.ndarray
) -> np.ndarray:
    """Return scale-free active-over-baseline weights from observations only."""
    ratios = np.asarray(
        [
            np.mean(block[:, active] ** 2)
            / max(float(np.mean(block[:, baseline] ** 2)), np.finfo(float).tiny)
            for block in data_blocks
        ]
    )
    excess = np.maximum(ratios - 1.0, 0.0)
    return (
        np.sqrt(excess / excess.max())
        if excess.max() > 0.0
        else np.ones(len(data_blocks))
    )


def _pair_path(parts: Path, pair: tuple[int, int]) -> Path:
    return parts / f"eeg_{pair[0]:+03d}_meg_{pair[1]:+03d}.csv"


def _select_pairs(
    cases: list[dict],
    snr_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
    cases_per_scenario: int | None,
) -> list[tuple[tuple[int, int], list[dict]]]:
    if cases_per_scenario is not None and cases_per_scenario < 1:
        raise ValueError("cases_per_scenario must be positive")
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for case in cases:
        grouped[(int(case["eeg_snr_db"]), int(case["meg_snr_db"]))].append(case)
    requested = list(dict.fromkeys(snr_pairs or grouped.keys()))
    missing = [pair for pair in requested if pair not in grouped]
    if not requested or missing:
        raise ValueError(f"unknown or empty SNR selection: {missing}")

    selected = []
    for pair in requested:
        pair_cases = grouped[pair]
        if cases_per_scenario is not None:
            counts: dict[str, int] = defaultdict(int)
            limited = []
            for case in pair_cases:
                scenario = str(case["scenario"])
                if counts[scenario] < cases_per_scenario:
                    limited.append(case)
                    counts[scenario] += 1
            pair_cases = limited
        found = {str(case["scenario"]) for case in pair_cases}
        if found != set(SCENARIOS):
            raise ValueError(f"SNR pair {pair} lacks scenarios: {sorted(set(SCENARIOS) - found)}")
        selected.append((pair, pair_cases))
    return selected


def _base_row(case: dict, manifest_sha256: str) -> dict:
    pair = (int(case["eeg_snr_db"]), int(case["meg_snr_db"]))
    return {
        "manifest_sha256": manifest_sha256,
        "chunk_index": int(case.get("pair_index", 0)),
        "chunk_file": _pair_path(Path(), pair).name,
        "case_number": int(case["case_number"]),
        "case_id": str(case["case_id"]),
        "configuration_number": case.get("configuration_number", ""),
        "configuration_id": case.get("configuration_id", ""),
        "pair_index": case.get("pair_index", ""),
        "eeg_snr_db": pair[0],
        "meg_snr_db": pair[1],
        "scenario": str(case["scenario"]),
        "surface_centers": json.dumps(case.get("surface_centers", [])),
        "deep_index": "" if case.get("deep_index") is None else int(case["deep_index"]),
        "deep_surface_ratio": (
            "" if case.get("deep_surface_ratio") is None else case["deep_surface_ratio"]
        ),
        "correlation": "" if case.get("correlation") is None else case["correlation"],
        "has_surface_true": int(bool(case.get("surface_centers"))),
    }


def _error_row(base: dict, method: str, error: Exception, elapsed: float) -> dict:
    return {
        **base,
        "method": method,
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "elapsed_seconds": elapsed,
        "temporal_rank": "",
        "selected_templates": "",
        **{field: np.nan for field in archive.METRIC_FIELDS},
    }


def _success_row(
    base: dict,
    method: str,
    estimate: np.ndarray,
    truth: np.ndarray,
    groups: list[np.ndarray],
    active: np.ndarray,
    baseline: np.ndarray,
    runtime: dict,
    elapsed: float,
    *,
    temporal_rank: int | str = "",
    selected_templates: int | str = "",
) -> dict:
    score_started = time.perf_counter()
    metrics = benchmark_metrics.evaluate_estimate(
        estimate,
        truth,
        runtime["shared"]["vertices"],
        groups,
        runtime["shared"]["n_surf"],
        active,
        runtime["shared"]["auc_cortex"],
        baseline=baseline,
    )
    return {
        **base,
        "method": method,
        "status": "ok",
        "error": "",
        "elapsed_seconds": elapsed + time.perf_counter() - score_started,
        "temporal_rank": temporal_rank,
        "selected_templates": selected_templates,
        **metrics,
    }


def _score_case(case: dict, runtime: dict, manifest_sha256: str) -> dict[str, dict]:
    base = _base_row(case, manifest_sha256)
    algorithm_version = str(runtime.get("algorithm_version", "v1"))
    oaster_method = _oaster_method(algorithm_version)
    methods = tuple(runtime.get("methods", (oaster_method,) + comparators.METHODS))
    modality_weighting = str(runtime.get("modality_weighting", "equal"))
    setup_started = time.perf_counter()
    try:
        (
            eeg,
            meg,
            truth,
            groups,
            baseline,
            active_windows,
            active,
            _simulation_metadata,
        ) = erp_protocol.simulate_case(
            runtime["shared"],
            case,
            seed_root=int(runtime.get("seed_root", erp_protocol.ERP_SEED_ROOT)),
        )
        if modality_weighting == "evidence":
            eeg_white, eeg_gain = comparator_methods.whiten(
                eeg, runtime["shared"]["gain_eeg"]
            )
            meg_white, meg_gain = comparator_methods.whiten(
                meg, runtime["shared"]["gain_meg"]
            )
            data = np.vstack((eeg_white, meg_white))
            gain = np.vstack((eeg_gain, meg_gain))
            retained_rows = (eeg_white.shape[0], meg_white.shape[0])
        elif modality_weighting == "equal":
            data, gain = comparator_methods.joint_whiten(
                eeg,
                meg,
                runtime["shared"]["gain_eeg"],
                runtime["shared"]["gain_meg"],
            )
            retained_rows = None
        else:
            raise ValueError(
                f"modality_weighting must be one of {MODALITY_WEIGHTINGS}"
            )
        data -= data[:, baseline].mean(axis=1, keepdims=True)
        window_channel_weights = None
        if retained_rows is not None:
            split = retained_rows[0]
            blocks = (data[:split], data[split:])
            window_channel_weights = []
            for window in active_windows:
                weights = _modality_evidence_weights(blocks, baseline, window)
                window_channel_weights.append(
                    np.r_[
                        np.full(retained_rows[0], weights[0]),
                        np.full(retained_rows[1], weights[1]),
                    ]
                )
    except Exception as exc:
        elapsed = time.perf_counter() - setup_started
        return {method: _error_row(base, method, exc, elapsed) for method in methods}
    setup_elapsed = time.perf_counter() - setup_started
    rows: dict[str, dict] = {}

    started = time.perf_counter()
    try:
        solver = runtime.get("oaster_solver") or _resolve_oaster(algorithm_version)
        estimate, diagnostics = solver(
            data,
            gain,
            runtime["shared"]["n_surf"],
            runtime["kernels"],
            baseline=baseline,
            active_windows=active_windows,
            window_channel_weights=window_channel_weights,
            require_one=False,
            **({"adjacency": runtime["shared"]["adjacency"]} if algorithm_version in ADAPTIVE_VERSIONS else {}),
            **runtime.get("oaster_kwargs", {}),
        )
        if runtime.get("diagnostics_dir") is not None:
            path = Path(runtime["diagnostics_dir"]) / f"case_{int(case['case_number']):05d}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"case_id": case["case_id"], "method": oaster_method,
                     "manifest_sha256": manifest_sha256,
                     "parameters": runtime.get("oaster_kwargs", {}),
                     "diagnostics": diagnostics},
                    ensure_ascii=False, indent=2, allow_nan=False,
                ) + "\n", encoding="utf-8",
            )
            os.replace(temporary, path)
        rows[oaster_method] = _success_row(
            base,
            oaster_method,
            estimate,
            truth,
            groups,
            active,
            baseline,
            runtime,
            setup_elapsed + time.perf_counter() - started,
            temporal_rank=sum(
                int(window.get("temporal_rank", 0)) for window in diagnostics["windows"]
            ),
            selected_templates=(
                len(diagnostics["selected_templates"])
                if isinstance(diagnostics["selected_templates"], (list, tuple))
                else int(diagnostics["selected_templates"])
            ),
        )
    except Exception as exc:
        rows[oaster_method] = _error_row(
            base, oaster_method, exc, setup_elapsed + time.perf_counter() - started
        )

    if methods == (oaster_method,):
        return rows

    started = time.perf_counter()
    try:
        estimates = comparator_methods.minimum_norm_family(data, gain)
        elapsed = setup_elapsed + time.perf_counter() - started
        for method in comparators.MINIMUM_NORM_METHODS:
            rows[method] = _success_row(
                base,
                method,
                estimates[method],
                truth,
                groups,
                active,
                baseline,
                runtime,
                elapsed,
            )
    except Exception as exc:
        elapsed = setup_elapsed + time.perf_counter() - started
        for method in comparators.MINIMUM_NORM_METHODS:
            rows[method] = _error_row(base, method, exc, elapsed)

    calls = {
        "LCMV": lambda: comparator_methods.lcmv(data, gain, active),
        "Dipole fitting (grid)": lambda: comparator_methods.dipole_fit(data, gain, active),
        "RAP-MUSIC": lambda: comparator_methods.rap_music(data, gain, active),
    }
    for method, call in calls.items():
        started = time.perf_counter()
        try:
            estimate = call()
            rows[method] = _success_row(
                base,
                method,
                estimate,
                truth,
                groups,
                active,
                baseline,
                runtime,
                setup_elapsed + time.perf_counter() - started,
            )
        except Exception as exc:
            rows[method] = _error_row(
                base, method, exc, setup_elapsed + time.perf_counter() - started
            )
    return rows


def _valid_pair(
    rows: list[dict],
    cases: list[dict],
    manifest_sha256: str,
    methods: tuple[str, ...] = METHODS,
    diagnostics_dir: Path | None = None,
) -> bool:
    try:
        if diagnostics_dir is not None:
            for case in cases:
                path = diagnostics_dir / f"case_{int(case['case_number']):05d}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (payload["case_id"] != case["case_id"]
                        or payload["manifest_sha256"] != manifest_sha256
                        or payload["method"] != methods[0]
                        or not isinstance(payload["diagnostics"]["windows"], list)):
                    return False
        expected = [
            (str(case["case_id"]), method)
            for case in cases
            for method in methods
        ]
        return (
            len(rows) == len(expected)
            and [(row["case_id"], row["method"]) for row in rows] == expected
            and all(row["manifest_sha256"] == manifest_sha256 for row in rows)
            and all(row["status"] == "ok" for row in rows)
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _write_metadata(
    output: Path,
    manifest_path: Path,
    manifest_sha256: str,
    data_root: Path,
    sample_path: Path | None,
    selected: list[tuple[tuple[int, int], list[dict]]],
    cases_per_scenario: int | None,
    workers: int,
    algorithm_version: str,
    modality_weighting: str,
    seed_root: int,
    methods: tuple[str, ...],
    oaster_kwargs: dict,
    code_fingerprint: str,
    shared_fingerprint: str,
    environment: dict[str, str],
    checkpoint_fingerprint: str,
) -> None:
    payload = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "data_root": str(data_root.resolve()),
        "sample_path": None if sample_path is None else str(sample_path.resolve()),
        "protocol": "phase-locked-erp-v1",
        "methods": methods,
        "oaster_algorithm_version": algorithm_version,
        "oaster_modality_weighting": modality_weighting,
        "oaster_modality_weighting_rule": (
            "equal weights"
            if modality_weighting == "equal"
            else (
                "sqrt(normalized positive active/baseline power-ratio excess) "
                "from retained whitened EEG/MAG observation rows"
            )
        ),
        "erp_seed_root": seed_root,
        "oaster_kwargs": oaster_kwargs,
        "checkpoint_code_fingerprint": code_fingerprint,
        "checkpoint_shared_fingerprint": shared_fingerprint,
        "checkpoint_environment": environment,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "observation_rule": (
            "simulate once and independently whiten EEG/MAG once; comparators use "
            "the same unweighted stacked data and gain"
        ),
        "window_rule": (
            "one fixed baseline and active ERP window shared by all methods; "
            "joint whitened observations are baseline-corrected before dispatch"
        ),
        "snr_level": "evoked-level after a 40-trial-mean-equivalent noise draw",
        "deep_detection_rule": {
            "relative_amplitude_threshold": benchmark_metrics.DEEP_DETECTION_THRESHOLD,
            "maximum_peak_distance_mm": 10.0,
        },
        "snr_pairs": [list(pair) for pair, _cases in selected],
        "cases_per_scenario": cases_per_scenario,
        "selected_case_count": sum(len(cases) for _pair, cases in selected),
        "checkpoint": "one atomically replaced CSV per EEG/MEG SNR pair",
        "solver_diagnostics": (
            "Per-case JSON in active parts_*/diagnostics; status=ok means output was "
            "computed, not that all spatial solver subproblems converged. Inspect "
            "diagnostics.windows[].solver.history and .converged."
            if algorithm_version in ADAPTIVE_VERSIONS else None
        ),
        "workers": workers,
        "blas_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "provenance": archive._provenance(
            {
                "runner": __file__,
                "erp_protocol": erp_protocol.__file__,
                "oaster_algorithm": oaster.__file__,
                **ADAPTIVE_CODE.get(algorithm_version, {}),
                "comparator_algorithms": comparator_methods.__file__,
                "metrics": benchmark_metrics.__file__,
                "archive_io": archive.__file__,
                "comparator_summaries": comparators.__file__,
            }
        ),
    }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output / "metadata.json")


def run(
    manifest_path: Path = DEFAULT_MANIFEST,
    data_root: Path = DEFAULT_DATA_ROOT,
    sample_path: Path | None = DEFAULT_SAMPLE_PATH,
    output: Path | None = None,
    *,
    snr_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    cases_per_scenario: int | None = None,
    workers: int = 1,
    force: bool = False,
    algorithm_version: str = "v1",
    modality_weighting: str = "equal",
    seed_root: int = erp_protocol.ERP_SEED_ROOT,
    oaster_only: bool = False,
    v2_deep_rescue_delta: float = oaster.ERP_V2_DEEP_RESCUE_DELTA,
    v4_mrf_strength: float = 0.5,
    v4_edge_fraction: float = 0.5,
    v4_noise_multiplier: float = 1.0,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    if modality_weighting not in MODALITY_WEIGHTINGS:
        raise ValueError(f"modality_weighting must be one of {MODALITY_WEIGHTINGS}")
    seed_root = int(seed_root)
    if seed_root < 0:
        raise ValueError("seed_root must be non-negative")
    if not np.isfinite(v2_deep_rescue_delta):
        raise ValueError("v2_deep_rescue_delta must be finite")
    if not np.isfinite(v4_mrf_strength) or not 0 <= v4_mrf_strength < 1:
        raise ValueError("v4_mrf_strength must be finite and in [0, 1)")
    if not np.isfinite(v4_edge_fraction) or v4_edge_fraction < 0:
        raise ValueError("v4_edge_fraction must be finite and non-negative")
    if not np.isfinite(v4_noise_multiplier) or v4_noise_multiplier <= 0:
        raise ValueError("v4_noise_multiplier must be finite and positive")
    solver = _resolve_oaster(algorithm_version)
    oaster_method = _oaster_method(algorithm_version)
    methods = (oaster_method,) if oaster_only else (oaster_method,) + comparators.METHODS
    oaster_kwargs = (
        {"deep_rescue_delta": float(v2_deep_rescue_delta)}
        if algorithm_version in {"v2", "v3"}
        else {}
    )
    if algorithm_version in ADAPTIVE_VERSIONS:
        oaster_kwargs = {
            "mrf_strength": float(v4_mrf_strength),
            "edge_fraction": float(v4_edge_fraction),
            "noise_multiplier": float(v4_noise_multiplier),
        }
    if algorithm_version == "v5":
        oaster_kwargs.update(calibration="layer", temporal_mode="smooth", solver_kind="admm")
    if output is None:
        output = {"v4": DEFAULT_V4_OUTPUT, "v5": DEFAULT_V5_OUTPUT}.get(algorithm_version, DEFAULT_OUTPUT)
    manifest_path, data_root, output = map(Path, (manifest_path, data_root, output))
    sample_path = None if sample_path is None else Path(sample_path)
    cases, manifest_sha256 = archive._load_manifest(manifest_path)
    if manifest_path.resolve() == DEFAULT_MANIFEST.resolve() and len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"corrected-v2 manifest must contain {EXPECTED_CASES} cases, got {len(cases)}"
        )
    selected = _select_pairs(cases, snr_pairs, cases_per_scenario)
    shared = protocol.load_shared(data_root, sample_path)
    kernels = (
        () if algorithm_version in ADAPTIVE_VERSIONS
        else protected.connected_euclidean_surface_kernels(
            shared["vertices"],
            shared["adjacency"],
            shared["n_surf"],
            scales_mm=oaster.SURFACE_SCALES_MM,
        )
    )
    runtime = {
        "shared": shared,
        "kernels": kernels,
        "algorithm_version": algorithm_version,
        "oaster_solver": solver,
        "modality_weighting": modality_weighting,
        "seed_root": seed_root,
        "methods": methods,
        "oaster_kwargs": oaster_kwargs,
    }
    penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000.0)
    configuration = f"{algorithm_version}_{modality_weighting}_seed_{seed_root}"
    if algorithm_version in {"v2", "v3"}:
        rescue_tag = f"{float(v2_deep_rescue_delta):g}".replace("-", "m").replace(".", "p")
        rescue_sha = hashlib.sha256(
            repr(float(v2_deep_rescue_delta)).encode("ascii")
        ).hexdigest()[:8]
        configuration += f"_rescue_{rescue_tag}_{rescue_sha}"
    if algorithm_version in ADAPTIVE_VERSIONS:
        configuration += (
            f"_mrf_{v4_mrf_strength:g}_edge_{v4_edge_fraction:g}_noise_{v4_noise_multiplier:g}"
        ).replace(".", "p")
    fingerprint_paths = (
        __file__,
        erp_protocol.__file__,
        protocol.__file__,
        protected.__file__,
        oaster.__file__,
        comparator_methods.__file__,
        benchmark_metrics.__file__,
        archive.__file__,
        comparators.__file__,
        *ADAPTIVE_CODE.get(algorithm_version, {}).values(),
    )
    code_fingerprint = _code_fingerprint(fingerprint_paths)
    shared_fingerprint = _shared_fingerprint(shared)
    environment = _environment_versions()
    checkpoint_fingerprint = _checkpoint_fingerprint(
        manifest_sha256,
        code_fingerprint,
        shared_fingerprint,
        environment,
        {"algorithm_version": algorithm_version, "modality_weighting": modality_weighting,
         "seed_root": seed_root, "methods": methods, "oaster_kwargs": oaster_kwargs},
    )
    parts = output / f"parts_{configuration}_run_{checkpoint_fingerprint[:12]}"
    parts.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = parts / "diagnostics" if algorithm_version in ADAPTIVE_VERSIONS else None
    if diagnostics_dir is not None:
        diagnostics_dir.mkdir(exist_ok=True)
        runtime["diagnostics_dir"] = diagnostics_dir
    (output / "completion.json").unlink(missing_ok=True)

    for pair, pair_cases in selected:
        path = _pair_path(parts, pair)
        existing = archive._read_csv(path)
        if not force and _valid_pair(existing, pair_cases, manifest_sha256, methods, diagnostics_dir):
            print(f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: verified checkpoint", flush=True)
            continue
        arguments = [(case, runtime, manifest_sha256) for case in pair_cases]
        if workers == 1:
            scored = [_score_case(*argument) for argument in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(lambda argument: _score_case(*argument), arguments))
        rows = [result[method] for result in scored for method in methods]
        archive._atomic_csv(path, rows, archive.ROW_FIELDS)
        if not _valid_pair(rows, pair_cases, manifest_sha256, methods, diagnostics_dir):
            failures = [row for row in rows if row["status"] != "ok"]
            detail = failures[0]["error"] if failures else "checkpoint ordering mismatch"
            raise RuntimeError(f"SNR pair {pair} failed: {detail}")
        print(
            f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: "
            f"scored {len(pair_cases)} cases x {len(methods)} methods",
            flush=True,
        )

    if _code_fingerprint(fingerprint_paths) != code_fingerprint:
        raise RuntimeError("benchmark code changed while the run was in progress")
    combined = []
    for pair, pair_cases in selected:
        rows = archive._read_csv(_pair_path(parts, pair))
        if not _valid_pair(rows, pair_cases, manifest_sha256, methods, diagnostics_dir):
            raise RuntimeError(f"incomplete checkpoint for SNR pair {pair}")
        combined.extend(rows)
    combined.sort(key=lambda row: (int(row["case_number"]), methods.index(row["method"])))
    archive._atomic_csv(output / "rows.csv", combined, archive.ROW_FIELDS)
    comparators._summaries(combined, output, penalty_mm)
    _write_metadata(
        output,
        manifest_path,
        manifest_sha256,
        data_root,
        sample_path,
        selected,
        cases_per_scenario,
        workers,
        algorithm_version,
        modality_weighting,
        seed_root,
        methods,
        oaster_kwargs,
        code_fingerprint,
        shared_fingerprint,
        environment,
        checkpoint_fingerprint,
    )
    archive._write_completion(
        output,
        combined,
        expected_row_count=sum(len(cases) for _pair, cases in selected) * len(methods),
        manifest_sha256=manifest_sha256,
        chunk_count=len(selected),
        methods=methods,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--snr-pair",
        nargs=2,
        action="append",
        type=int,
        metavar=("EEG_DB", "MEG_DB"),
        help="repeat to run selected EEG/MEG SNR pairs; default is all 49",
    )
    parser.add_argument("--cases-per-scenario", type=int)
    version_selection = parser.add_mutually_exclusive_group()
    version_selection.add_argument(
        "--algorithm-version", choices=ALGORITHM_VERSIONS, default="v1"
    )
    version_selection.add_argument(
        "--method",
        choices=tuple(_oaster_method(version) for version in ALGORITHM_VERSIONS),
        help="choose OASTER version; the seven comparators still run unless --oaster-only",
    )
    parser.add_argument(
        "--modality-weighting", choices=MODALITY_WEIGHTINGS, default="equal"
    )
    parser.add_argument("--seed-root", type=int, default=erp_protocol.ERP_SEED_ROOT)
    parser.add_argument("--oaster-only", action="store_true")
    parser.add_argument("--v4-mrf-strength", type=float, default=0.5, help="shared adaptive parameter for v4/v5")
    parser.add_argument("--v4-edge-fraction", type=float, default=0.5, help="shared adaptive parameter for v4/v5")
    parser.add_argument("--v4-noise-multiplier", type=float, default=1.0, help="shared adaptive parameter for v4/v5")
    parser.add_argument(
        "--deep-rescue-delta",
        "--v2-deep-rescue-delta",
        dest="deep_rescue_delta",
        type=float,
        default=oaster.ERP_V2_DEEP_RESCUE_DELTA,
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, min(4, os.cpu_count() or 1))
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        run(
            args.manifest,
            args.data_root,
            args.sample_path,
            args.output,
            snr_pairs=[tuple(pair) for pair in args.snr_pair] if args.snr_pair else None,
            cases_per_scenario=args.cases_per_scenario,
            workers=args.workers,
            force=args.force,
            algorithm_version=(
                next(version for version in ALGORITHM_VERSIONS if _oaster_method(version) == args.method)
                if args.method else args.algorithm_version
            ),
            modality_weighting=args.modality_weighting,
            seed_root=args.seed_root,
            oaster_only=args.oaster_only,
            v2_deep_rescue_delta=args.deep_rescue_delta,
            v4_mrf_strength=args.v4_mrf_strength,
            v4_edge_fraction=args.v4_edge_fraction,
            v4_noise_multiplier=args.v4_noise_multiplier,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
