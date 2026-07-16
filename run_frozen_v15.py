from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(os.environ.get("V15_BENCHMARK_ROOT", ROOT.parent / "benchmark"))
PARENT_ROOT = Path(
    os.environ.get(
        "V15_PARENT_ROOT",
        r"F:\博士\工作＆汇报\源定位\codex\roi_deep_multimethod_comparison",
    )
)
for _path in (BENCHMARK_ROOT, PARENT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import auc_metric
import metrics as benchmark_metrics
import protected_multilayer as protected
from protocol import load_shared, simulate_case
from refined_grid import refined_deep_evidence


METHODS = ("V14-common-grid", "V15-support-rescue", "V16-evidence-rescue")
V16_JOINT_DEEP_EXCESS = float(os.environ.get("V16_JOINT_DEEP_EXCESS", "0.06"))
V16_MODALITY_FLOOR = float(os.environ.get("V16_MODALITY_FLOOR", "0.0"))
V16_SURFACE_EVIDENCE_POWER = float(os.environ.get("V16_SURFACE_EVIDENCE_POWER", "2.0"))
FIELDS = (
    "panel", "case_id", "scenario", "snr_db", "location", "method", "status", "error",
    "elapsed_sec", "support_count", "deep_evidence", "eeg_evidence", "meg_evidence",
    "auc", "auc_tie_corrected", "rmse", "surface_sd_mm", "deep_sd_mm",
    "surface_dle_mm", "deep_dle_mm", "has_deep_true", "deep_score",
    "deep_peak_distance_mm", "deep_detected", "deep_false_positive", "active_count",
)
SURFACE_SCENARIOS = {"surface_only", "deep_plus_surface", "deep_plus_two_surface"}
DEEP_SCENARIOS = {"deep_only", "deep_plus_surface", "deep_plus_two_surface"}
_WORK: dict = {}


def _load_sisses(path: str | Path) -> np.ndarray:
    result = sio.loadmat(path, simplify_cells=True)
    names = [str(value) for value in np.atleast_1d(result["method_names"]).tolist()]
    estimates = np.asarray(result["source_estimates"], dtype=float)
    if estimates.ndim == 2:
        return estimates
    if estimates.shape[-1] != len(names) and estimates.shape[0] == len(names):
        estimates = np.moveaxis(estimates, 0, -1)
    return estimates[:, :, names.index("SISSES")]


def _file_map(panel: str) -> dict[str, str]:
    with (BENCHMARK_ROOT / "results" / "scores.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            row["case_id"]: str(BENCHMARK_ROOT / json.loads(row["metadata_json"])["file"])
            for row in rows
            if row["panel"] == panel and row["method"] == "SISSES" and row["status"] == "ok"
        }


def _init_worker(files: dict[str, str]) -> None:
    benchmark_metrics._an_cal_auc_module._grow_parcels = auc_metric._safe_grow_parcels
    shared = load_shared()
    _WORK.update(
        shared=shared,
        files=files,
        cortex=auc_metric.auc_cortex(
            shared["vertices"], shared["adjacency"], int(shared["n_surf"])
        ),
    )


def _old_v14(eeg: dict, meg: dict, v11: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict | None]:
    shared = _WORK["shared"]
    n_surf = int(shared["n_surf"])
    mask = protected.threshold_mask(v11, 0.50) & np.asarray(candidate, dtype=bool)
    mask[n_surf:] = False
    seeded = v11 * mask[:, None]
    empty = {
        "n_surf": np.array([0], dtype=np.int64),
        "gain_eeg": np.empty((eeg["F"].shape[0], 0)),
        "gain_meg": np.empty((meg["F"].shape[0], 0)),
        "vertices": np.empty((0, 3)),
    }
    deep = refined_deep_evidence(eeg, meg, v11, shared["vertices"], n_surf, empty)
    if deep is not None:
        index = int(deep["coarse_index"])
        seeded[index] = np.asarray(deep["timecourse"], dtype=float)
        mask[index] = True
    fitted, final_mask = protected.component_refit_select_v14_local_evidence(
        eeg, meg, seeded, mask, shared["adjacency"], n_surf
    )
    return fitted, final_mask, deep


def _row(case: dict, method: str, source: np.ndarray, mask: np.ndarray, truth: np.ndarray, groups: list[np.ndarray], deep: dict | None, elapsed: float) -> dict:
    shared = _WORK["shared"]
    metrics = benchmark_metrics.evaluate_estimate(
        source * mask[:, None],
        truth,
        shared["vertices"],
        groups,
        int(shared["n_surf"]),
        np.arange(int(shared["active_start"]), truth.shape[1]),
        _WORK["cortex"],
    )
    if method == METHODS[0]:
        eeg_evidence = np.nan if deep is None else float(deep["eeg_drop"])
        meg_evidence = np.nan if deep is None else float(deep["meg_drop"])
    else:
        eeg_evidence = np.nan if deep is None else float(deep["eeg_excess"])
        meg_evidence = np.nan if deep is None else float(deep["meg_excess"])
    return {
        "panel": case["case_id"].split("-", 1)[0],
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "snr_db": int(case["snr_db"]),
        "location": int(case["location"]),
        "method": method,
        "status": "ok",
        "error": "",
        "elapsed_sec": elapsed,
        "support_count": int(np.count_nonzero(mask)),
        "deep_evidence": int(deep is not None),
        "eeg_evidence": eeg_evidence,
        "meg_evidence": meg_evidence,
        **metrics,
    }


def _score_case(case: dict) -> list[dict]:
    started = time.perf_counter()
    try:
        shared = _WORK["shared"]
        eeg_data, meg_data, truth, groups, _meta = simulate_case(shared, case)
        sisses = _load_sisses(_WORK["files"][case["case_id"]])
        eeg = {"F": eeg_data, "Gain": shared["gain_eeg"]}
        meg = {"F": meg_data, "Gain": shared["gain_meg"]}
        v11, range_mask, candidate = protected.component_refit_select_v11_compactness_sisses(
            eeg,
            meg,
            sisses,
            shared["adjacency"],
            int(shared["n_surf"]),
            shared["vertices"],
            surface_compactness=0.10,
            sigma_surface_component=0.10,
            deep_compactness=0.50,
            return_candidate=True,
        )
        old_source, old_mask, old_deep = _old_v14(eeg, meg, v11, candidate)
        split = time.perf_counter()
        new_source, new_mask, new_deep = protected.component_refit_select_v15_from_v11(
            eeg,
            meg,
            v11,
            range_mask,
            shared["adjacency"],
            int(shared["n_surf"]),
        )
        middle = time.perf_counter()
        v16_source, v16_mask, v16_deep = protected.component_refit_select_v16_evidence_rescue_from_v11(
            eeg,
            meg,
            v11,
            range_mask,
            shared["adjacency"],
            int(shared["n_surf"]),
            joint_deep_excess=V16_JOINT_DEEP_EXCESS,
            modality_floor=V16_MODALITY_FLOOR,
            surface_evidence_power=V16_SURFACE_EVIDENCE_POWER,
        )
        finished = time.perf_counter()
        return [
            _row(case, METHODS[0], old_source, old_mask, truth, groups, old_deep, split - started),
            _row(case, METHODS[1], new_source, new_mask, truth, groups, new_deep, middle - split),
            _row(case, METHODS[2], v16_source, v16_mask, truth, groups, v16_deep, finished - middle),
        ]
    except Exception as exc:
        return [
            {
                **{field: np.nan for field in FIELDS},
                "panel": case["case_id"].split("-", 1)[0],
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "snr_db": int(case["snr_db"]),
                "location": int(case["location"]),
                "method": method,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            for method in METHODS
        ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["case_id"], row["method"])))


def _float(row: dict, field: str) -> float:
    try:
        return float(row[field])
    except (TypeError, ValueError):
        return np.nan


def _macro(rows: list[dict], field: str, scenarios: set[str], penalty: float | None = None) -> float:
    cells = []
    for scenario in sorted(scenarios):
        for snr in (0, 10, 20):
            values = [_float(row, field) for row in rows if row["scenario"] == scenario and int(row["snr_db"]) == snr]
            if penalty is not None:
                values = [value if np.isfinite(value) else penalty for value in values]
            values = [value for value in values if np.isfinite(value)]
            if values:
                cells.append(float(np.mean(values)))
    return float(np.mean(cells)) if cells else np.nan


def _summaries(rows: list[dict], vertices: np.ndarray) -> list[dict]:
    penalty = float(np.linalg.norm(np.ptp(vertices, axis=0)) * 1000.0)
    result = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method and row["status"] == "ok"]
        positives = [row for row in selected if int(float(row["has_deep_true"]))]
        negatives = [row for row in selected if not int(float(row["has_deep_true"]))]
        sensitivity = float(np.mean([int(float(row["deep_detected"])) for row in positives])) if positives else np.nan
        specificity = float(np.mean([not int(float(row["deep_false_positive"])) for row in negatives])) if negatives else np.nan
        tp = sum(int(float(row["deep_detected"])) for row in positives)
        fp = sum(int(float(row["deep_false_positive"])) for row in negatives)
        fn = len(positives) - tp
        sensitivity_cells = []
        for scenario in sorted(DEEP_SCENARIOS):
            for snr in (0, 10, 20):
                cell = [row for row in positives if row["scenario"] == scenario and int(row["snr_db"]) == snr]
                if cell:
                    sensitivity_cells.append(float(np.mean([int(float(row["deep_detected"])) for row in cell])))
        specificity_cells = []
        for snr in (0, 10, 20):
            cell = [row for row in negatives if int(row["snr_db"]) == snr]
            if cell:
                specificity_cells.append(float(np.mean([not int(float(row["deep_false_positive"])) for row in cell])))
        sensitivity_macro = float(np.mean(sensitivity_cells))
        specificity_macro = float(np.mean(specificity_cells))
        summary = {
            "method": method,
            "n": len(selected),
            "auc": _macro(selected, "auc", SURFACE_SCENARIOS | DEEP_SCENARIOS),
            "rmse": _macro(selected, "rmse", SURFACE_SCENARIOS | DEEP_SCENARIOS),
            "deep_sensitivity": sensitivity,
            "deep_specificity": specificity,
            "deep_balanced_accuracy": (sensitivity + specificity) / 2.0,
            "deep_f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "deep_sensitivity_macro_scenario_snr": sensitivity_macro,
            "deep_specificity_macro_scenario_snr": specificity_macro,
            "deep_balanced_accuracy_macro_scenario_snr": (sensitivity_macro + specificity_macro) / 2.0,
            "miss_penalty_mm": penalty,
        }
        for field, scenarios in (
            ("surface_sd_mm", SURFACE_SCENARIOS),
            ("surface_dle_mm", SURFACE_SCENARIOS),
            ("deep_sd_mm", DEEP_SCENARIOS),
            ("deep_dle_mm", DEEP_SCENARIOS),
        ):
            applicable = [row for row in selected if row["scenario"] in scenarios]
            summary[field] = _macro(selected, field, scenarios)
            summary[f"{field}_penalized"] = _macro(selected, field, scenarios, penalty)
            summary[f"{field}_valid"] = sum(np.isfinite(_float(row, field)) for row in applicable)
            summary[f"{field}_total"] = len(applicable)
        result.append(summary)
    return result


def _cell_summaries(rows: list[dict], vertices: np.ndarray) -> list[dict]:
    penalty = float(np.linalg.norm(np.ptp(vertices, axis=0)) * 1000.0)
    result = []
    for method in METHODS:
        for scenario in sorted(SURFACE_SCENARIOS | DEEP_SCENARIOS):
            for snr in (0, 10, 20):
                selected = [
                    row for row in rows
                    if row["method"] == method
                    and row["status"] == "ok"
                    and row["scenario"] == scenario
                    and int(row["snr_db"]) == snr
                ]
                item = {
                    "method": method,
                    "scenario": scenario,
                    "snr_db": snr,
                    "n": len(selected),
                    "auc": float(np.mean([_float(row, "auc") for row in selected])),
                    "rmse": float(np.mean([_float(row, "rmse") for row in selected])),
                    "deep_detected_rate": (
                        float(np.mean([int(float(row["deep_detected"])) for row in selected]))
                        if scenario in DEEP_SCENARIOS else np.nan
                    ),
                    "deep_false_positive_rate": (
                        float(np.mean([int(float(row["deep_false_positive"])) for row in selected]))
                        if scenario == "surface_only" else np.nan
                    ),
                }
                for field, scenarios in (
                    ("surface_sd_mm", SURFACE_SCENARIOS),
                    ("surface_dle_mm", SURFACE_SCENARIOS),
                    ("deep_sd_mm", DEEP_SCENARIOS),
                    ("deep_dle_mm", DEEP_SCENARIOS),
                ):
                    values = [_float(row, field) for row in selected]
                    finite = [value for value in values if np.isfinite(value)]
                    item[field] = float(np.mean(finite)) if finite and scenario in scenarios else np.nan
                    item[f"{field}_penalized"] = (
                        float(np.mean([value if np.isfinite(value) else penalty for value in values]))
                        if scenario in scenarios else np.nan
                    )
                    item[f"{field}_valid"] = sum(np.isfinite(value) for value in values) if scenario in scenarios else 0
                result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen V14/V15/V16 benchmark")
    parser.add_argument("--panel", choices=("dev", "test"), required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v16_frozen")
    args = parser.parse_args()

    manifest = json.loads(
        (BENCHMARK_ROOT / "results" / "protocol" / f"{args.panel}_manifest.json").read_text(encoding="utf-8")
    )
    if args.limit is not None:
        manifest = manifest[: args.limit]
    files = _file_map(args.panel)
    missing = [case["case_id"] for case in manifest if case["case_id"] not in files]
    if missing:
        raise FileNotFoundError(f"missing SISSES output for {missing[0]} ({len(missing)} cases)")

    score_path = args.output / f"{args.panel}_scores.csv"
    existing = []
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    done = {(row["case_id"], row["method"]) for row in existing if row["status"] == "ok"}
    pending = [case for case in manifest if any((case["case_id"], method) not in done for method in METHODS)]
    rows = existing
    began = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(files,)) as pool:
            for index, pair in enumerate(pool.map(_score_case, pending), 1):
                keys = {(row["case_id"], row["method"]) for row in pair}
                rows = [row for row in rows if (row["case_id"], row["method"]) not in keys] + pair
                if index % 10 == 0 or index == len(pending):
                    _write(score_path, rows)
                    print(f"[{index}/{len(pending)}] {pair[0]['case_id']} elapsed={time.perf_counter() - began:.1f}s", flush=True)
    _write(score_path, rows)

    _init_worker(files)
    summaries = _summaries(rows, _WORK["shared"]["vertices"])
    summary_path = args.output / f"{args.panel}_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    cells = _cell_summaries(rows, _WORK["shared"]["vertices"])
    cell_path = args.output / f"{args.panel}_by_cell.csv"
    with cell_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    metadata = {
        "panel": args.panel,
        "cases": len(manifest),
        "auc": "An_cal_AUC(threshold=0.01), block-diagonal surface/deep graph",
        "surface_deep_cross_edges": 0,
        "v15_surface_seed": "V11 broad 0.10 range; no 0.50 post-mask",
        "v15_deep_scan": "all 15 deep points; active-minus-baseline; both modalities >= 0.02",
        "v16_joint_deep_excess": V16_JOINT_DEEP_EXCESS,
        "v16_modality_floor": V16_MODALITY_FLOOR,
        "v16_surface_evidence_power": V16_SURFACE_EVIDENCE_POWER,
    }
    (args.output / f"{args.panel}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(score_path)
    print(summary_path)
    print(cell_path)


if __name__ == "__main__":
    main()
