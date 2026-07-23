from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

import protected_multilayer as protected
import run_frozen_v15 as frozen
from protocol import simulate_case


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = frozen.BENCHMARK_ROOT
METHODS = ("V16-evidence-rescue", "V17-data-driven-surface")
FIELDS = frozen.FIELDS + (
    "surface_centers_json",
    "surface_scales_json",
    "surface_excess_json",
    "surface_timecourse_correlations_json",
)


def _init_worker(files: dict[str, str]) -> None:
    frozen._init_worker(files)
    shared = frozen._WORK["shared"]
    frozen._WORK["surface_kernels"] = protected.connected_euclidean_surface_kernels(
        shared["vertices"],
        shared["adjacency"],
        int(shared["n_surf"]),
    )


def _score_case(case: dict) -> list[dict]:
    started = time.perf_counter()
    try:
        shared = frozen._WORK["shared"]
        n_surf = int(shared["n_surf"])
        eeg_data, meg_data, truth, groups, _meta = simulate_case(shared, case)
        sisses = frozen._load_sisses(frozen._WORK["files"][case["case_id"]])
        eeg = {"F": eeg_data, "Gain": shared["gain_eeg"]}
        meg = {"F": meg_data, "Gain": shared["gain_meg"]}
        v11, range_mask, _candidate = protected.component_refit_select_v11_compactness_sisses(
            eeg,
            meg,
            sisses,
            shared["adjacency"],
            n_surf,
            shared["vertices"],
            surface_compactness=0.10,
            sigma_surface_component=0.10,
            deep_compactness=0.50,
            return_candidate=True,
        )
        v16, v16_mask, deep = protected.component_refit_select_v16_evidence_rescue_from_v11(
            eeg,
            meg,
            v11,
            range_mask,
            shared["adjacency"],
            n_surf,
            joint_deep_excess=frozen.V16_JOINT_DEEP_EXCESS,
            modality_floor=frozen.V16_MODALITY_FLOOR,
            surface_evidence_power=frozen.V16_SURFACE_EVIDENCE_POWER,
        )
        middle = time.perf_counter()
        b, l = protected.whitened_joint_system(eeg, meg)
        v17, v17_mask, details = protected.multiscale_surface_point_refit_system(
            b,
            l,
            v16,
            v16_mask,
            n_surf,
            frozen._WORK["surface_kernels"],
        )
        finished = time.perf_counter()
        v16_row = frozen._row(
            case,
            METHODS[0],
            v16,
            v16_mask,
            truth,
            groups,
            deep,
            middle - started,
        )
        v17_row = frozen._row(
            case,
            METHODS[1],
            v17,
            v17_mask,
            truth,
            groups,
            deep,
            finished - middle,
        )
        for row in (v16_row, v17_row):
            row.update(
                surface_centers_json="[]",
                surface_scales_json="[]",
                surface_excess_json="[]",
                surface_timecourse_correlations_json="[]",
            )
        v17_row.update(
            surface_centers_json=json.dumps(details["centers"]),
            surface_scales_json=json.dumps(details["scales_mm"]),
            surface_excess_json=json.dumps(details["active_excess"]),
            surface_timecourse_correlations_json=json.dumps(
                details["timecourse_correlations"]
            ),
        )
        return [v16_row, v17_row]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen no-oracle V16/V17 benchmark")
    parser.add_argument("--panel", choices=("dev", "test"), required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v17_no_oracle")
    args = parser.parse_args()

    manifest = json.loads(
        (BENCHMARK_ROOT / "results" / "protocol" / f"{args.panel}_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if args.limit is not None:
        manifest = manifest[: args.limit]
    files = frozen._file_map(args.panel)
    missing = [case["case_id"] for case in manifest if case["case_id"] not in files]
    if missing:
        raise FileNotFoundError(f"missing SISSES output for {missing[0]} ({len(missing)} cases)")

    score_path = args.output / f"{args.panel}_scores.csv"
    existing = []
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    done = {(row["case_id"], row["method"]) for row in existing if row["status"] == "ok"}
    pending = [
        case
        for case in manifest
        if any((case["case_id"], method) not in done for method in METHODS)
    ]
    rows = existing
    began = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(files,),
        ) as pool:
            for index, pair in enumerate(pool.map(_score_case, pending), 1):
                keys = {(row["case_id"], row["method"]) for row in pair}
                rows = [row for row in rows if (row["case_id"], row["method"]) not in keys] + pair
                if index % 10 == 0 or index == len(pending):
                    _write(score_path, rows)
                    print(
                        f"[{index}/{len(pending)}] {pair[0]['case_id']} "
                        f"elapsed={time.perf_counter() - began:.1f}s",
                        flush=True,
                    )
    _write(score_path, rows)

    _init_worker(files)
    summaries = frozen._summaries(rows, frozen._WORK["shared"]["vertices"], METHODS)
    summary_path = args.output / f"{args.panel}_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    cells = frozen._cell_summaries(rows, frozen._WORK["shared"]["vertices"], METHODS)
    cell_path = args.output / f"{args.panel}_by_cell.csv"
    with cell_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    metadata = {
        "panel": args.panel,
        "cases": len(manifest),
        "method": METHODS[1],
        "surface_scales_mm": [0.0, 4.0, 7.0],
        "surface_kernel": "connected Euclidean Gaussian atoms; no fixed graph hops",
        "surface_time_basis": "none; active-minus-baseline residual power from observed EEG+MEG",
        "surface_source_count": "residual evidence stopping; no fixed count",
        "additional_surface_gate": {
            "active_excess": 0.10,
            "ratio_to_first": 0.15,
            "max_observed_timecourse_correlation": 0.98,
        },
        "surface_output": "localized centers plus a 0.15 tail from the data-selected continuous physical kernel",
        "simulator_private_helpers": [],
    }
    (args.output / f"{args.panel}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
