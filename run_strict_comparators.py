"""Run the seven Python comparators on immutable strict-blind observations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Case workers are already threads; avoid a nested BLAS pool by default.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

import run_strict_oaster as archive
from benchmark import methods as comparator_methods
from benchmark import metrics as benchmark_metrics


METHODS = (
    "MNE",
    "dSPM",
    "sLORETA",
    "eLORETA",
    "LCMV",
    "Dipole fitting (grid)",
    "RAP-MUSIC",
)
MINIMUM_NORM_METHODS = METHODS[:4]
METHOD_SLUGS = {
    "MNE": "mne",
    "dSPM": "dspm",
    "sLORETA": "sloreta",
    "eLORETA": "eloreta",
    "LCMV": "lcmv",
    "Dipole fitting (grid)": "dipole_fit_grid",
    "RAP-MUSIC": "rap_music",
}
DEFAULT_OUTPUT = archive.ROOT / "results" / "strict_blind" / "comparators"


def _selected_methods(methods: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(methods))
    unknown = sorted(set(selected) - set(METHODS))
    if not selected or unknown:
        raise ValueError(f"methods must be chosen from {METHODS}; unknown={unknown}")
    return tuple(method for method in METHODS if method in selected)


def _part_path(parts: Path, spec: archive.Chunk, method: str) -> Path:
    return parts / f"{spec.path.stem}__{METHOD_SLUGS[method]}.csv"


def _valid_part(
    rows: list[dict],
    spec: archive.Chunk,
    cases: list[dict],
    manifest_sha256: str,
    method: str,
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
            and all(row["method"] == method and row["status"] == "ok" for row in rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _runtime(geometry: dict, reference: dict) -> dict:
    adjacency = reference["adjacency"].toarray().astype(np.uint8, copy=False)
    shared = {
        "times": geometry["times"],
        "vertices": geometry["vertices"],
        "adjacency": reference["adjacency"],
        "n_surf": geometry["n_surf"],
        "n_deep": geometry["n_deep"],
        "active_start": archive.protocol.ACTIVE_START,
    }
    return {
        "shared": shared,
        "active": np.arange(archive.protocol.ACTIVE_START, geometry["times"].size),
        "cortex": archive.protocol._auc_cortex(
            geometry["vertices"], adjacency, geometry["n_surf"]
        ),
        "penalty_mm": float(
            np.linalg.norm(np.ptp(geometry["vertices"], axis=0)) * 1000.0
        ),
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
    runtime: dict,
    elapsed_before_score: float,
) -> dict:
    score_started = time.perf_counter()
    metrics = benchmark_metrics.evaluate_estimate(
        estimate,
        truth,
        runtime["shared"]["vertices"],
        groups,
        runtime["shared"]["n_surf"],
        runtime["active"],
        runtime["cortex"],
    )
    return {
        **base,
        "method": method,
        "status": "ok",
        "error": "",
        "elapsed_seconds": elapsed_before_score + time.perf_counter() - score_started,
        "temporal_rank": "",
        "selected_templates": "",
        **metrics,
    }


def _score_case(
    case: dict,
    local_index: int,
    spec: archive.Chunk,
    chunk: dict,
    runtime: dict,
    manifest_sha256: str,
    methods: tuple[str, ...],
) -> dict[str, dict]:
    base = archive._case_fields(
        case, spec, manifest_sha256, int(bool(case.get("surface_centers")))
    )
    setup_started = time.perf_counter()
    try:
        truth, groups, _ = archive.protocol.truth_for_case(runtime["shared"], case)
        data, gain = comparator_methods.joint_whiten(
            chunk["f_eeg"][:, :, local_index],
            chunk["f_meg"][:, :, local_index],
            chunk["gain_eeg"],
            chunk["gain_meg"],
        )
    except Exception as exc:
        elapsed = time.perf_counter() - setup_started
        return {method: _error_row(base, method, exc, elapsed) for method in methods}
    setup_elapsed = time.perf_counter() - setup_started
    rows: dict[str, dict] = {}

    family = tuple(method for method in methods if method in MINIMUM_NORM_METHODS)
    if family:
        started = time.perf_counter()
        try:
            estimates = comparator_methods.minimum_norm_family(data, gain)
            method_elapsed = time.perf_counter() - started
            for method in family:
                rows[method] = _success_row(
                    base,
                    method,
                    estimates[method],
                    truth,
                    groups,
                    runtime,
                    setup_elapsed + method_elapsed,
                )
        except Exception as exc:
            elapsed = setup_elapsed + time.perf_counter() - started
            rows.update({method: _error_row(base, method, exc, elapsed) for method in family})

    calls = {
        "LCMV": lambda: comparator_methods.lcmv(data, gain, runtime["active"]),
        "Dipole fitting (grid)": lambda: comparator_methods.dipole_fit(
            data, gain, runtime["active"]
        ),
        "RAP-MUSIC": lambda: comparator_methods.rap_music(
            data, gain, runtime["active"]
        ),
    }
    for method in methods:
        if method not in calls:
            continue
        started = time.perf_counter()
        try:
            estimate = calls[method]()
            method_elapsed = time.perf_counter() - started
            rows[method] = _success_row(
                base,
                method,
                estimate,
                truth,
                groups,
                runtime,
                setup_elapsed + method_elapsed,
            )
        except Exception as exc:
            rows[method] = _error_row(
                base, method, exc, setup_elapsed + time.perf_counter() - started
            )
    return rows


def _summaries(rows: list[dict], output: Path, penalty_mm: float) -> None:
    cells: dict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    pairs: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["method"], int(row["eeg_snr_db"]), int(row["meg_snr_db"]))
        cells[(*key, row["scenario"])].append(row)
        pairs[key].append(row)

    cell_rows = [
        {
            "method": method,
            "eeg_snr_db": eeg_snr,
            "meg_snr_db": meg_snr,
            "scenario": scenario,
            "aggregation": "case_weighted_within_scenario",
            **archive._aggregate(group, penalty_mm),
        }
        for (method, eeg_snr, meg_snr, scenario), group in sorted(cells.items())
    ]
    archive._atomic_csv(
        output / "summary_by_snr_scenario.csv", cell_rows, cell_rows[0].keys()
    )

    macro_rows, weighted_rows = [], []
    for (method, eeg_snr, meg_snr), group in sorted(pairs.items()):
        scenarios = [
            row
            for row in cell_rows
            if row["method"] == method
            and int(row["eeg_snr_db"]) == eeg_snr
            and int(row["meg_snr_db"]) == meg_snr
        ]
        macro = {
            "method": method,
            "eeg_snr_db": eeg_snr,
            "meg_snr_db": meg_snr,
            "aggregation": "scenario_macro",
            "case_count": sum(int(row["case_count"]) for row in scenarios),
            "scenario_count": len(scenarios),
        }
        for name in archive.SUMMARY_METRICS:
            macro[name] = archive._mean(archive._number(row, name) for row in scenarios)
        macro["deep_sensitivity"] = archive._mean(
            archive._number(row, "deep_sensitivity") for row in scenarios
        )
        macro["deep_specificity"] = archive._mean(
            archive._number(row, "deep_specificity") for row in scenarios
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
                "method": method,
                "eeg_snr_db": eeg_snr,
                "meg_snr_db": meg_snr,
                "aggregation": "case_weighted",
                **archive._aggregate(group, penalty_mm),
            }
        )
    archive._atomic_csv(
        output / "summary_by_snr_pair_scenario_macro.csv",
        macro_rows,
        macro_rows[0].keys(),
    )
    archive._atomic_csv(
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
    chunks: list[archive.Chunk],
    methods: tuple[str, ...],
    workers: int,
    start_chunk: int,
    limit_chunks: int | None,
    fingerprints: dict[str, str],
) -> None:
    payload = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "input_root": str(input_root.resolve()),
        "geometry_reference": str(geometry["path"].resolve()),
        "methods": methods,
        "adapter": "benchmark.methods after joint_whiten",
        "implementation_scope": (
            "repository fixed-grid numerical implementations; not asserted to be "
            "value-identical to third-party package defaults"
        ),
        "fixed_parameters": {
            "minimum_norm_family": {
                "lambda2": 1.0 / 9.0,
                "depth": 0.8,
                "depth_limit": 10.0,
                "eloreta_max_iter": 20,
                "eloreta_tolerance": 1e-6,
            },
            "lcmv": {"reg": 0.05},
            "dipole_fit": {"max_dipoles": 3, "max_temporal_rank": 12},
            "rap_music": {"max_sources": 3, "max_temporal_rank": 12},
        },
        "observations": "archived F_EEG/F_MEG; never regenerated",
        "truth": "benchmark.protocol.truth_for_case; scoring only",
        "checkpoint": "one atomic CSV per immutable input chunk and method",
        "metrics": "raw/corrected AUC, RMSE, layer SD/DLE, deep balanced accuracy",
        "chunk_count": len(chunks),
        "start_chunk": start_chunk,
        "limit_chunks": limit_chunks,
        "workers": workers,
        "blas_threads": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "cross_chunk_fingerprints_sha256": fingerprints,
        "provenance": archive._provenance(
            {
                "strict_runner": __file__,
                "comparator_algorithms": comparator_methods.__file__,
                "metrics": benchmark_metrics.__file__,
            }
        ),
    }
    path = output / "metadata.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output, suffix=".metadata.tmp", delete=False
        ) as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(
    manifest_path: Path = archive.DEFAULT_MANIFEST,
    input_root: Path = archive.DEFAULT_INPUT_ROOT,
    data_root: Path = archive.protocol.DEFAULT_DATA_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    methods: tuple[str, ...] | list[str] = METHODS,
    workers: int = 1,
    start_chunk: int = 0,
    limit_chunks: int | None = None,
    force: bool = False,
) -> None:
    methods = _selected_methods(methods)
    cases, manifest_sha256 = archive._load_manifest(Path(manifest_path))
    chunks = archive._discover_chunks(Path(input_root), len(cases))
    if workers < 1 or start_chunk < 0 or start_chunk >= len(chunks):
        raise ValueError("workers and start_chunk are outside their valid range")
    if limit_chunks is not None and limit_chunks < 1:
        raise ValueError("limit_chunks must be positive")
    selected = chunks[start_chunk:]
    if limit_chunks is not None:
        selected = selected[:limit_chunks]

    geometry = archive._load_geometry(Path(data_root))
    reference = archive._load_chunk(chunks[0].path, observations=False)
    fingerprints = archive._fingerprints(reference)
    archive._validate_chunk(
        chunks[0], reference, cases, manifest_sha256, geometry, fingerprints
    )
    output, input_root = Path(output), Path(input_root)
    parts = output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    full_run = start_chunk == 0 and limit_chunks is None
    if full_run:
        (output / "completion.json").unlink(missing_ok=True)
    runtime = None

    for spec in selected:
        pending = tuple(
            method
            for method in methods
            if force
            or not _valid_part(
                archive._read_csv(_part_path(parts, spec, method)),
                spec,
                cases,
                manifest_sha256,
                method,
            )
        )
        if not pending:
            header = archive._load_chunk(spec.path, observations=False)
            archive._validate_chunk(
                spec, header, cases, manifest_sha256, geometry, fingerprints
            )
            print(f"chunk {spec.index}: verified {len(methods)} method checkpoints", flush=True)
            continue

        chunk = archive._load_chunk(spec.path, observations=True)
        archive._validate_chunk(
            spec, chunk, cases, manifest_sha256, geometry, fingerprints
        )
        if runtime is None:
            runtime = _runtime(geometry, reference)
        arguments = [
            (case, index, spec, chunk, runtime, manifest_sha256, pending)
            for index, case in enumerate(cases[spec.start : spec.stop])
        ]
        if workers == 1:
            scored = [_score_case(*argument) for argument in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(lambda argument: _score_case(*argument), arguments))
        failures = []
        for method in pending:
            rows = [result[method] for result in scored]
            archive._atomic_csv(
                _part_path(parts, spec, method), rows, archive.ROW_FIELDS
            )
            if not _valid_part(rows, spec, cases, manifest_sha256, method):
                failed_rows = [row for row in rows if row["status"] != "ok"]
                if not failed_rows:
                    raise RuntimeError(
                        f"invalid checkpoint produced: {spec.path.name} / {method}"
                    )
                failures.extend(failed_rows)
        if failures:
            raise RuntimeError(
                f"{spec.path.name} has {len(failures)} comparator failures; "
                f"first={failures[0]['method']}: {failures[0]['error']}"
            )
        print(f"chunk {spec.index}: scored {spec.case_count} cases x {len(pending)} methods", flush=True)

    combined = None
    if full_run:
        combined = []
        for spec in chunks:
            for method in methods:
                rows = archive._read_csv(_part_path(parts, spec, method))
                if not _valid_part(rows, spec, cases, manifest_sha256, method):
                    raise RuntimeError(
                        f"incomplete checkpoint: {spec.path.name} / {method}"
                    )
                combined.extend(rows)
        combined.sort(key=lambda row: (int(row["case_number"]), METHODS.index(row["method"])))
        archive._atomic_csv(output / "rows.csv", combined, archive.ROW_FIELDS)
        penalty_mm = float(np.linalg.norm(np.ptp(geometry["vertices"], axis=0)) * 1000.0)
        _summaries(combined, output, penalty_mm)

    _metadata(
        output,
        Path(manifest_path),
        manifest_sha256,
        input_root,
        geometry,
        chunks,
        methods,
        workers,
        start_chunk,
        limit_chunks,
        fingerprints,
    )
    if full_run:
        archive._write_completion(
            output,
            combined,
            expected_row_count=len(cases) * len(methods),
            manifest_sha256=manifest_sha256,
            chunk_count=len(chunks),
            methods=methods,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=archive.DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=archive.DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--data-root", type=Path, default=archive.protocol.DEFAULT_DATA_ROOT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        run(
            args.manifest,
            args.input_root,
            args.data_root,
            args.output,
            methods=args.methods,
            workers=args.workers,
            start_chunk=args.start_chunk,
            limit_chunks=args.limit_chunks,
            force=args.force,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
