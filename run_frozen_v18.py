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
import run_frozen_v17 as v17
from protocol import simulate_case


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = frozen.BENCHMARK_ROOT
FIELDS = v17.FIELDS + ("extent_fraction",)
_FRACTIONS: tuple[float, ...] = ()


def _method(fraction: float) -> str:
    return f"V18-extent-{fraction:.3f}"


def _init_worker(files: dict[str, str], fractions: tuple[float, ...]) -> None:
    global _FRACTIONS
    _FRACTIONS = fractions
    v17._init_worker(files)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["case_id"], row["method"])))


def _score_case(case: dict) -> list[dict]:
    started = time.perf_counter()
    methods = ("V17-data-driven-surface", *map(_method, _FRACTIONS))
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
        b, l = protected.whitened_joint_system(eeg, meg)
        sparse, sparse_mask, details = protected.multiscale_surface_point_refit_system(
            b,
            l,
            v16,
            v16_mask,
            n_surf,
            frozen._WORK["surface_kernels"],
        )
        rows = [
            frozen._row(
                case,
                methods[0],
                sparse,
                sparse_mask,
                truth,
                groups,
                deep,
                time.perf_counter() - started,
            )
        ]
        for fraction, method in zip(_FRACTIONS, methods[1:]):
            estimate = protected.add_scaled_surface_extent(sparse, sisses, n_surf, fraction)
            mask = sparse_mask.copy()
            if np.linalg.norm(sparse[:n_surf, protected.NOISE_SAMPLES:]) > 0:
                mask[:n_surf] |= np.linalg.norm(sisses[:n_surf], axis=1) > 0
            rows.append(
                frozen._row(
                    case,
                    method,
                    estimate,
                    mask,
                    truth,
                    groups,
                    deep,
                    time.perf_counter() - started,
                )
            )
        for row, fraction in zip(rows, (0.0, *_FRACTIONS)):
            row.update(
                surface_centers_json=json.dumps(details["centers"]),
                surface_scales_json=json.dumps(details["scales_mm"]),
                surface_excess_json=json.dumps(details["active_excess"]),
                surface_timecourse_correlations_json=json.dumps(
                    details["timecourse_correlations"]
                ),
                extent_fraction=fraction,
            )
        return rows
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
                "extent_fraction": fraction,
            }
            for method, fraction in zip(methods, (0.0, *_FRACTIONS))
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrained V18 surface-extent benchmark")
    parser.add_argument("--panel", choices=("dev", "test"), required=True)
    parser.add_argument("--fractions", default="0.25")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "v18_no_oracle")
    args = parser.parse_args()
    fractions = tuple(float(value) for value in args.fractions.split(","))
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        parser.error("fractions must be comma-separated values in (0, 1]")
    methods = ("V17-data-driven-surface", *map(_method, fractions))

    manifest = json.loads(
        (BENCHMARK_ROOT / "results" / "protocol" / f"{args.panel}_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    files = frozen._file_map(args.panel)
    score_path = args.output / f"{args.panel}_scores.csv"
    existing = []
    if score_path.exists():
        with score_path.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
    done = {(row["case_id"], row["method"]) for row in existing if row["status"] == "ok"}
    pending = [
        case for case in manifest if any((case["case_id"], method) not in done for method in methods)
    ]
    rows = existing
    began = time.perf_counter()
    if pending:
        args.output.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(files, fractions),
        ) as pool:
            for index, batch in enumerate(pool.map(_score_case, pending), 1):
                keys = {(row["case_id"], row["method"]) for row in batch}
                rows = [row for row in rows if (row["case_id"], row["method"]) not in keys] + batch
                if index % 10 == 0 or index == len(pending):
                    _write(score_path, rows)
                    print(
                        f"[{index}/{len(pending)}] {batch[0]['case_id']} "
                        f"elapsed={time.perf_counter() - began:.1f}s",
                        flush=True,
                    )
    _write(score_path, rows)

    _init_worker(files, fractions)
    summaries = frozen._summaries(rows, frozen._WORK["shared"]["vertices"], methods)
    summary_path = args.output / f"{args.panel}_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    cells = frozen._cell_summaries(rows, frozen._WORK["shared"]["vertices"], methods)
    cell_path = args.output / f"{args.panel}_by_cell.csv"
    with cell_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    metadata = {
        "panel": args.panel,
        "cases": len(manifest),
        "fractions": fractions,
        "extent": "raw SISSES surface estimate scaled to a fraction of the V17 active surface peak",
        "deep_branch": "unchanged V16 evidence rescue",
        "simulator_private_helpers": [],
    }
    (args.output / f"{args.panel}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary_path)


if __name__ == "__main__":
    main()
