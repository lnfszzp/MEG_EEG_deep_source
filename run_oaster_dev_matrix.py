"""Evaluate rebuilt OASTER on a reproducible simulated development SNR matrix.

Unlike ``run_strict_oaster.py``, this development-only harness deliberately
calls ``benchmark.protocol.simulate_case``.  It never reads the strict archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import protected_multilayer as protected
from benchmark import metrics as benchmark_metrics
from benchmark import protocol
from candidates import oaster_rebuilt as oaster
from run_strict_oaster import METRIC_FIELDS, _atomic_csv, _summaries


ROOT = Path(__file__).resolve().parent
BASE_MANIFEST = ROOT / "benchmark" / "results" / "protocol" / "dev_manifest.json"
DEFAULT_OUTPUT = ROOT / "results" / "dev_matrix" / "oaster_rebuilt"
SEED_ROOT = 20260906
LEVELS = (-10, -5, 0, 5, 10, 15, 20)
PAIRS = tuple(product(LEVELS, repeat=2))
EXPECTED_CONFIG_COUNTS = {
    "surface_only": 68,
    "deep_only": 15,
    "deep_plus_surface": 68,
    "deep_plus_two_surface": 34,
}
METHOD = "OASTER_rebuilt"
ROW_FIELDS = (
    "base_manifest_sha256",
    "matrix_manifest_sha256",
    "case_number",
    "case_id",
    "configuration_number",
    "configuration_id",
    "source_case_id",
    "pair_index",
    "eeg_snr_db",
    "meg_snr_db",
    "actual_eeg_snr_db",
    "actual_meg_snr_db",
    "scenario",
    "surface_centers",
    "deep_index",
    "deep_surface_ratio",
    "correlation",
    "seed",
    "method",
    "status",
    "error",
    "elapsed_seconds",
    "temporal_rank",
    "selected_templates",
    "has_surface_true",
    *METRIC_FIELDS,
)


def _load_base_manifest(path: Path) -> tuple[list[dict], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="ascii").split()[0] != digest:
        raise RuntimeError(f"development manifest checksum mismatch: {path}")
    cases = json.loads(payload)
    if not isinstance(cases, list) or len(cases) != sum(EXPECTED_CONFIG_COUNTS.values()):
        raise RuntimeError("development manifest must contain 185 base configurations")
    if [int(case["case_number"]) for case in cases] != list(range(len(cases))):
        raise RuntimeError("development base case_number order is invalid")
    if Counter(case["scenario"] for case in cases) != Counter(EXPECTED_CONFIG_COUNTS):
        raise RuntimeError("development base scenario counts are invalid")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("development base case_id values are not unique")
    return cases, digest


def _even_subset(cases: list[dict], count: int | None) -> list[dict]:
    ordered = sorted(cases, key=lambda case: int(case["case_number"]))
    if count is None or count >= len(ordered):
        return ordered
    if count < 1:
        raise ValueError("cases_per_scenario must be positive")
    indices = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    return [ordered[int(index)] for index in indices]


def make_matrix(
    base: list[dict], *, cases_per_scenario: int | None, diagonal: bool
) -> list[dict]:
    """Expand fixed base source configurations over independent EEG/MEG SNR."""
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for case in base:
        by_scenario[str(case["scenario"])].append(case)
    selected = sorted(
        (
            case
            for scenario in EXPECTED_CONFIG_COUNTS
            for case in _even_subset(by_scenario[scenario], cases_per_scenario)
        ),
        key=lambda case: int(case["case_number"]),
    )
    matrix = []
    for pair_index, (eeg_snr, meg_snr) in enumerate(PAIRS):
        if diagonal and eeg_snr != meg_snr:
            continue
        for source in selected:
            case = dict(source)
            configuration_number = int(source["case_number"])
            scenario = str(source["scenario"])
            case.update(
                case_id=(
                    f"dev-matrix-{pair_index:02d}-{configuration_number:03d}-{scenario}-"
                    f"eeg{eeg_snr:+03d}-meg{meg_snr:+03d}"
                ),
                case_number=pair_index * len(base) + configuration_number,
                configuration_id=f"dev-source-{configuration_number:03d}-{scenario}",
                configuration_number=configuration_number,
                source_case_id=str(source["case_id"]),
                panel="dev_matrix",
                pair_index=pair_index,
                snr_db=eeg_snr,
                eeg_snr_db=eeg_snr,
                meg_snr_db=meg_snr,
                seed=[
                    SEED_ROOT,
                    pair_index,
                    int(source["scenario_code"]),
                    configuration_number,
                ],
            )
            matrix.append(case)
    if len({case["case_id"] for case in matrix}) != len(matrix):
        raise RuntimeError("development matrix case IDs are not unique")
    if len({tuple(case["seed"]) for case in matrix}) != len(matrix):
        raise RuntimeError("development matrix seeds are not unique")
    return matrix


def _runtime(shared: dict) -> dict:
    kernels = protected.connected_euclidean_surface_kernels(
        shared["vertices"],
        shared["adjacency"],
        int(shared["n_surf"]),
        scales_mm=oaster.SURFACE_SCALES_MM,
    )
    return {
        "kernels": kernels,
        "active": np.arange(protocol.ACTIVE_START, len(shared["times"])),
        "penalty_mm": float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000.0),
    }


def _base_row(case: dict, base_sha: str, matrix_sha: str) -> dict:
    return {
        "base_manifest_sha256": base_sha,
        "matrix_manifest_sha256": matrix_sha,
        "case_number": int(case["case_number"]),
        "case_id": str(case["case_id"]),
        "configuration_number": int(case["configuration_number"]),
        "configuration_id": str(case["configuration_id"]),
        "source_case_id": str(case["source_case_id"]),
        "pair_index": int(case["pair_index"]),
        "eeg_snr_db": int(case["eeg_snr_db"]),
        "meg_snr_db": int(case["meg_snr_db"]),
        "scenario": str(case["scenario"]),
        "surface_centers": json.dumps(case.get("surface_centers", [])),
        "deep_index": "" if case.get("deep_index") is None else int(case["deep_index"]),
        "deep_surface_ratio": (
            "" if case.get("deep_surface_ratio") is None else case["deep_surface_ratio"]
        ),
        "correlation": "" if case.get("correlation") is None else case["correlation"],
        "seed": json.dumps(case["seed"]),
        "method": METHOD,
        "has_surface_true": int(bool(case.get("surface_centers"))),
    }


def _score_case(
    case: dict, shared: dict, runtime: dict, base_sha: str, matrix_sha: str
) -> dict:
    began = time.perf_counter()
    row = _base_row(case, base_sha, matrix_sha)
    try:
        eeg, meg, truth, groups, meta = protocol.simulate_case(shared, case)
        estimate, diagnostics = oaster.reconstruct(
            eeg,
            meg,
            shared["gain_eeg"],
            shared["gain_meg"],
            int(shared["n_surf"]),
            runtime["kernels"],
        )
        metrics = benchmark_metrics.evaluate_estimate(
            estimate,
            truth,
            shared["vertices"],
            groups,
            int(shared["n_surf"]),
            runtime["active"],
            shared["auc_cortex"],
        )
        actual = meta["actual_snr_db"]
        row.update(
            actual_eeg_snr_db=float(actual["eeg"]),
            actual_meg_snr_db=float(actual["meg"]),
            status="ok",
            error="",
            elapsed_seconds=time.perf_counter() - began,
            temporal_rank=int(diagnostics["temporal_rank"]),
            selected_templates=int(diagnostics["selected_templates"]),
            **metrics,
        )
    except Exception as exc:
        row.update(
            actual_eeg_snr_db=np.nan,
            actual_meg_snr_db=np.nan,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.perf_counter() - began,
            temporal_rank="",
            selected_templates="",
            **{field: np.nan for field in METRIC_FIELDS},
        )
    return row


def _write_metadata(
    output: Path,
    base_manifest: Path,
    base_sha: str,
    matrix_sha: str,
    cases: list[dict],
    data_root: Path,
    cases_per_scenario: int | None,
    diagonal: bool,
    workers: int,
) -> None:
    payload = {
        "purpose": "development simulation only; strict archive was not read",
        "base_manifest": str(base_manifest.resolve()),
        "base_manifest_sha256": base_sha,
        "matrix_manifest_sha256": matrix_sha,
        "data_root": str(data_root.resolve()),
        "method": METHOD,
        "adapter": "candidates.oaster_rebuilt",
        "seed_root": SEED_ROOT,
        "levels_db": list(LEVELS),
        "diagonal": diagonal,
        "cases_per_scenario": cases_per_scenario,
        "case_count": len(cases),
        "workers": workers,
        "simulator": "benchmark.protocol.simulate_case",
        "surface_scales_mm": list(oaster.SURFACE_SCALES_MM),
        "ridge_fraction": oaster.RIDGE_FRACTION,
        "spectral_fraction": oaster.SPECTRAL_FRACTION,
        "strict_results_used_for_tuning": False,
    }
    path = output / "metadata.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(
    base_manifest: Path = BASE_MANIFEST,
    data_root: Path = protocol.DEFAULT_DATA_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    cases_per_scenario: int | None = None,
    diagonal: bool = False,
    workers: int = 1,
    sample_path: Path | None = None,
) -> list[dict]:
    if workers < 1 or (cases_per_scenario is not None and cases_per_scenario < 1):
        raise ValueError("workers and cases_per_scenario must be positive")
    base, base_sha = _load_base_manifest(Path(base_manifest))
    cases = make_matrix(
        base, cases_per_scenario=cases_per_scenario, diagonal=diagonal
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matrix_sha = protocol.save_manifest(output / "manifest.json", cases)
    kwargs = {"data_root": Path(data_root)}
    if sample_path is not None:
        kwargs["sample_path"] = Path(sample_path)
    shared = protocol.load_shared(**kwargs)
    runtime = _runtime(shared)
    score = lambda case: _score_case(case, shared, runtime, base_sha, matrix_sha)
    if workers == 1:
        rows = [score(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(score, cases))
    _atomic_csv(output / "rows.csv", rows, ROW_FIELDS)
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} development cases failed; first={failed[0]['error']}")
    _summaries(rows, output, runtime["penalty_mm"])
    _write_metadata(
        output,
        Path(base_manifest),
        base_sha,
        matrix_sha,
        cases,
        Path(data_root),
        cases_per_scenario,
        diagonal,
        workers,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=protocol.DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases-per-scenario", type=int)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    if args.workers < 1 or (
        args.cases_per_scenario is not None and args.cases_per_scenario < 1
    ):
        parser.error("workers and cases-per-scenario must be positive")
    run(
        args.base_manifest,
        args.data_root,
        args.output,
        cases_per_scenario=args.cases_per_scenario,
        diagonal=args.diagonal,
        workers=args.workers,
        sample_path=args.sample_path,
    )


if __name__ == "__main__":
    main()
