"""Run the phase-locked ERP whole-head benchmark on one shared observation.

Each manifest case is simulated once, jointly whitened once, and then sent to
OASTER-ERP and the same seven fixed-grid comparators.  One atomic checkpoint is
written per EEG/MEG SNR pair so interrupted runs resume without recomputation.
"""

from __future__ import annotations

import argparse
import json
import os
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
EXPECTED_CASES = 9114
METHODS = ("OASTER-ERP",) + comparators.METHODS
SCENARIOS = (
    "surface_only",
    "deep_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
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
        ) = erp_protocol.simulate_case(runtime["shared"], case)
        data, gain = comparator_methods.joint_whiten(
            eeg,
            meg,
            runtime["shared"]["gain_eeg"],
            runtime["shared"]["gain_meg"],
        )
        data -= data[:, baseline].mean(axis=1, keepdims=True)
    except Exception as exc:
        elapsed = time.perf_counter() - setup_started
        return {method: _error_row(base, method, exc, elapsed) for method in METHODS}
    setup_elapsed = time.perf_counter() - setup_started
    rows: dict[str, dict] = {}

    started = time.perf_counter()
    try:
        estimate, diagnostics = oaster.reconstruct_evoked_oaster_from_whitened(
            data,
            gain,
            runtime["shared"]["n_surf"],
            runtime["kernels"],
            baseline=baseline,
            active_windows=active_windows,
            require_one=False,
        )
        rows["OASTER-ERP"] = _success_row(
            base,
            "OASTER-ERP",
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
            selected_templates=int(diagnostics["selected_templates"]),
        )
    except Exception as exc:
        rows["OASTER-ERP"] = _error_row(
            base, "OASTER-ERP", exc, setup_elapsed + time.perf_counter() - started
        )

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
    rows: list[dict], cases: list[dict], manifest_sha256: str
) -> bool:
    try:
        expected = [
            (str(case["case_id"]), method)
            for case in cases
            for method in METHODS
        ]
        return (
            len(rows) == len(expected)
            and [(row["case_id"], row["method"]) for row in rows] == expected
            and all(row["manifest_sha256"] == manifest_sha256 for row in rows)
            and all(row["status"] == "ok" for row in rows)
        )
    except (KeyError, TypeError, ValueError):
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
) -> None:
    payload = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "data_root": str(data_root.resolve()),
        "sample_path": None if sample_path is None else str(sample_path.resolve()),
        "protocol": "phase-locked-erp-v1",
        "methods": METHODS,
        "observation_rule": "simulate once and joint_whiten once per case for all methods",
        "window_rule": (
            "one fixed baseline and active ERP window shared by all methods; "
            "joint whitened observations are baseline-corrected before dispatch"
        ),
        "snr_level": "evoked-level after a 40-trial-mean-equivalent noise draw",
        "snr_pairs": [list(pair) for pair, _cases in selected],
        "cases_per_scenario": cases_per_scenario,
        "selected_case_count": sum(len(cases) for _pair, cases in selected),
        "checkpoint": "one atomically replaced CSV per EEG/MEG SNR pair",
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
                "comparator_algorithms": comparator_methods.__file__,
                "metrics": benchmark_metrics.__file__,
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
    output: Path = DEFAULT_OUTPUT,
    *,
    snr_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
    cases_per_scenario: int | None = None,
    workers: int = 1,
    force: bool = False,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest_path, data_root, output = map(Path, (manifest_path, data_root, output))
    sample_path = None if sample_path is None else Path(sample_path)
    cases, manifest_sha256 = archive._load_manifest(manifest_path)
    if manifest_path.resolve() == DEFAULT_MANIFEST.resolve() and len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"corrected-v2 manifest must contain {EXPECTED_CASES} cases, got {len(cases)}"
        )
    selected = _select_pairs(cases, snr_pairs, cases_per_scenario)
    shared = protocol.load_shared(data_root, sample_path)
    kernels = protected.connected_euclidean_surface_kernels(
        shared["vertices"],
        shared["adjacency"],
        shared["n_surf"],
        scales_mm=oaster.SURFACE_SCALES_MM,
    )
    runtime = {"shared": shared, "kernels": kernels}
    penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000.0)
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (output / "completion.json").unlink(missing_ok=True)

    for pair, pair_cases in selected:
        path = _pair_path(parts, pair)
        existing = archive._read_csv(path)
        if not force and _valid_pair(existing, pair_cases, manifest_sha256):
            print(f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: verified checkpoint", flush=True)
            continue
        arguments = [(case, runtime, manifest_sha256) for case in pair_cases]
        if workers == 1:
            scored = [_score_case(*argument) for argument in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(lambda argument: _score_case(*argument), arguments))
        rows = [result[method] for result in scored for method in METHODS]
        archive._atomic_csv(path, rows, archive.ROW_FIELDS)
        if not _valid_pair(rows, pair_cases, manifest_sha256):
            failures = [row for row in rows if row["status"] != "ok"]
            detail = failures[0]["error"] if failures else "checkpoint ordering mismatch"
            raise RuntimeError(f"SNR pair {pair} failed: {detail}")
        print(
            f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: "
            f"scored {len(pair_cases)} cases x {len(METHODS)} methods",
            flush=True,
        )

    combined = []
    for pair, pair_cases in selected:
        rows = archive._read_csv(_pair_path(parts, pair))
        if not _valid_pair(rows, pair_cases, manifest_sha256):
            raise RuntimeError(f"incomplete checkpoint for SNR pair {pair}")
        combined.extend(rows)
    combined.sort(key=lambda row: (int(row["case_number"]), METHODS.index(row["method"])))
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
    )
    archive._write_completion(
        output,
        combined,
        expected_row_count=sum(len(cases) for _pair, cases in selected) * len(METHODS),
        manifest_sha256=manifest_sha256,
        chunk_count=len(selected),
        methods=METHODS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--snr-pair",
        nargs=2,
        action="append",
        type=int,
        metavar=("EEG_DB", "MEG_DB"),
        help="repeat to run selected EEG/MEG SNR pairs; default is all 49",
    )
    parser.add_argument("--cases-per-scenario", type=int)
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
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
