"""Run the 2x2 OASTER observation-by-core crossover experiment.

The two algorithms always receive the same jointly whitened observation in a
cell.  Truth is kept outside ``_reconstruct_observation`` and is used only by
the existing scorer.
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
import run_erp_whole_head_matrix as erp_runner
import run_strict_comparators as comparators
import run_strict_oaster as archive
from benchmark import erp_protocol, methods, protocol
from candidates import oaster_rebuilt as oaster


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
DEFAULT_INPUT_ROOT = Path(r"D:\oaster_corrected_v2_sisses\matlab_input")
DEFAULT_DATA_ROOT = ROOT / "corrected_v2" / "generated"
DEFAULT_SAMPLE_PATH = Path(os.environ.get("MNE_SAMPLE_PATH", r"D:\mne_data\MNE-sample-data"))
DEFAULT_OUTPUT = ROOT / "results" / "erp_crossover"

SPECTRAL_OBSERVATION = "spectral_obs"
ERP_OBSERVATION = "erp_obs"
SPECTRAL_CORE = "spectral_core"
ERP_CORE = "erp_core"
METHODS = (
    f"{SPECTRAL_OBSERVATION}__{SPECTRAL_CORE}",
    f"{SPECTRAL_OBSERVATION}__{ERP_CORE}",
    f"{ERP_OBSERVATION}__{SPECTRAL_CORE}",
    f"{ERP_OBSERVATION}__{ERP_CORE}",
)


def _spectral_masks(n_times: int) -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    if n_times != 601:
        raise ValueError("archived spectral observations must contain 601 samples")
    baseline = np.zeros(n_times, dtype=bool)
    baseline[: protocol.ACTIVE_START] = True
    windows = []
    for start, stop in ((200, 400), (400, 600)):
        window = np.zeros(n_times, dtype=bool)
        window[start:stop] = True
        windows.append(window)
    return baseline, tuple(windows), np.arange(200, 601)


def _reconstruct_observation(
    eeg: np.ndarray,
    meg: np.ndarray,
    runtime: dict,
    baseline: np.ndarray,
    active_windows: tuple[np.ndarray, ...],
) -> dict[str, tuple[np.ndarray | None, dict, float, Exception | None]]:
    """Reconstruct one observation without accepting or reading source truth."""
    started = time.perf_counter()
    try:
        data, gain = methods.joint_whiten(
            eeg,
            meg,
            runtime["shared"]["gain_eeg"],
            runtime["shared"]["gain_meg"],
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {core: (None, {}, elapsed, exc) for core in (SPECTRAL_CORE, ERP_CORE)}
    setup_elapsed = time.perf_counter() - started

    result = {}
    for core in (SPECTRAL_CORE, ERP_CORE):
        began = time.perf_counter()
        try:
            if core == SPECTRAL_CORE:
                estimate, diagnostics = oaster.reconstruct_from_whitened(
                    data, gain, runtime["shared"]["n_surf"], runtime["kernels"]
                )
            else:
                estimate, diagnostics = oaster.reconstruct_evoked_oaster_from_whitened(
                    data,
                    gain,
                    runtime["shared"]["n_surf"],
                    runtime["kernels"],
                    baseline=baseline,
                    active_windows=active_windows,
                    require_one=False,
                )
            result[core] = (
                estimate,
                diagnostics,
                setup_elapsed + time.perf_counter() - began,
                None,
            )
        except Exception as exc:
            result[core] = (
                None,
                {},
                setup_elapsed + time.perf_counter() - began,
                exc,
            )
    return result


def _rows_for_observation(
    base: dict,
    observation: str,
    reconstructed: dict,
    truth: np.ndarray,
    groups: list[np.ndarray],
    active: np.ndarray,
    baseline: np.ndarray,
    runtime: dict,
) -> dict[str, dict]:
    rows = {}
    for core in (SPECTRAL_CORE, ERP_CORE):
        method = f"{observation}__{core}"
        estimate, diagnostics, elapsed, error = reconstructed[core]
        if error is not None:
            rows[method] = erp_runner._error_row(base, method, error, elapsed)
            continue
        try:
            temporal_rank = (
                int(diagnostics.get("temporal_rank", 0))
                if core == SPECTRAL_CORE
                else sum(int(item.get("temporal_rank", 0)) for item in diagnostics["windows"])
            )
            rows[method] = erp_runner._success_row(
                base,
                method,
                estimate,
                truth,
                groups,
                active,
                baseline,
                runtime,
                elapsed,
                temporal_rank=temporal_rank,
                selected_templates=int(diagnostics.get("selected_templates", 0)),
            )
        except Exception as exc:
            rows[method] = erp_runner._error_row(base, method, exc, elapsed)
    return rows


def _score_case(
    case: dict,
    archived_observation: tuple[np.ndarray, np.ndarray],
    runtime: dict,
    manifest_sha256: str,
) -> dict[str, dict]:
    base = erp_runner._base_row(case, manifest_sha256)
    spectral_baseline, spectral_windows, spectral_active = _spectral_masks(
        archived_observation[0].shape[1]
    )
    # Reconstruct first; archived truth is deliberately constructed afterwards.
    spectral_reconstructed = _reconstruct_observation(
        *archived_observation,
        runtime,
        spectral_baseline,
        spectral_windows,
    )
    spectral_truth, spectral_groups, _ = protocol.truth_for_case(runtime["shared"], case)
    rows = _rows_for_observation(
        base,
        SPECTRAL_OBSERVATION,
        spectral_reconstructed,
        spectral_truth,
        spectral_groups,
        spectral_active,
        spectral_baseline,
        runtime,
    )

    try:
        (
            eeg,
            meg,
            erp_truth,
            erp_groups,
            erp_baseline,
            erp_windows,
            erp_active,
            _metadata,
        ) = erp_protocol.simulate_case(runtime["shared"], case)
        erp_reconstructed = _reconstruct_observation(
            eeg, meg, runtime, erp_baseline, erp_windows
        )
        rows.update(
            _rows_for_observation(
                base,
                ERP_OBSERVATION,
                erp_reconstructed,
                erp_truth,
                erp_groups,
                erp_active,
                erp_baseline,
                runtime,
            )
        )
    except Exception as exc:
        rows.update(
            {
                method: erp_runner._error_row(base, method, exc, 0.0)
                for method in METHODS
                if method.startswith(ERP_OBSERVATION + "__")
            }
        )
    return rows


def _archive_observations(
    selected_cases: list[dict],
    chunks: list[archive.Chunk],
    all_cases: list[dict],
    manifest_sha256: str,
    geometry: dict,
    fingerprints: dict[str, str],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    by_number = {
        number: spec for spec in chunks for number in range(spec.start, spec.stop)
    }
    needed: dict[int, list[dict]] = defaultdict(list)
    for case in selected_cases:
        needed[by_number[int(case["case_number"])].index].append(case)
    observations = {}
    for chunk_index, cases in needed.items():
        spec = chunks[chunk_index]
        chunk = archive._load_chunk(spec.path, observations=True)
        archive._validate_chunk(
            spec, chunk, all_cases, manifest_sha256, geometry, fingerprints
        )
        for case in cases:
            number = int(case["case_number"])
            local = number - spec.start
            observations[number] = (
                chunk["f_eeg"][:, :, local].copy(),
                chunk["f_meg"][:, :, local].copy(),
            )
    return observations


def _valid_pair(rows: list[dict], cases: list[dict], manifest_sha256: str) -> bool:
    expected = [(str(case["case_id"]), method) for case in cases for method in METHODS]
    try:
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
    input_root: Path,
    data_root: Path,
    sample_path: Path | None,
    selected: list[tuple[tuple[int, int], list[dict]]],
    cases_per_scenario: int | None,
    workers: int,
    fingerprints: dict[str, str],
    code_fingerprint: str,
) -> None:
    payload = {
        "design": "2x2 observation protocol by OASTER core crossover",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "archived_input_root": str(input_root.resolve()),
        "data_root": str(data_root.resolve()),
        "sample_path": None if sample_path is None else str(sample_path.resolve()),
        "methods": METHODS,
        "algorithm_axis": [SPECTRAL_CORE, ERP_CORE],
        "observation_axis": [SPECTRAL_OBSERVATION, ERP_OBSERVATION],
        "whitening": "one joint-whitened observation shared by both cores in each cell",
        "spectral_windows": ["200:400", "400:600"],
        "spectral_scoring_active": "200:601",
        "erp_windows": "registered benchmark.erp_protocol masks",
        "truth_boundary": "truth is never passed to _reconstruct_observation; scoring only",
        "snr_pairs": [list(pair) for pair, _cases in selected],
        "cases_per_scenario": cases_per_scenario,
        "selected_case_count": sum(len(cases) for _pair, cases in selected),
        "checkpoint": "one atomically replaced CSV per EEG/MEG SNR pair",
        "workers": workers,
        "cross_chunk_fingerprints_sha256": fingerprints,
        "checkpoint_code_fingerprint": code_fingerprint,
        "provenance": archive._provenance(
            {
                "runner": __file__,
                "spectral_protocol": protocol.__file__,
                "erp_protocol": erp_protocol.__file__,
                "oaster_algorithm": oaster.__file__,
            }
        ),
    }
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "metadata.json")


def run(
    manifest_path: Path = DEFAULT_MANIFEST,
    input_root: Path = DEFAULT_INPUT_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    sample_path: Path | None = DEFAULT_SAMPLE_PATH,
    output: Path = DEFAULT_OUTPUT,
    *,
    snr_pairs: list[tuple[int, int]] | None = None,
    cases_per_scenario: int | None = None,
    workers: int = 1,
    force: bool = False,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    manifest_path, input_root, data_root, output = map(
        Path, (manifest_path, input_root, data_root, output)
    )
    sample_path = None if sample_path is None else Path(sample_path)
    cases, manifest_sha256 = archive._load_manifest(manifest_path)
    selected = erp_runner._select_pairs(cases, snr_pairs, cases_per_scenario)
    chunks = archive._discover_chunks(input_root, len(cases))
    shared = protocol.load_shared(data_root, sample_path)
    geometry = {
        "path": data_root / "deep_plus_two_surface" / "sub_EEG.mat",
        "vertices": shared["vertices"],
        "times": shared["times"],
        "n_surf": shared["n_surf"],
        "n_deep": shared["n_deep"],
    }
    reference = archive._load_chunk(chunks[0].path, observations=False)
    fingerprints = archive._fingerprints(reference)
    archive._validate_chunk(
        chunks[0], reference, cases, manifest_sha256, geometry, fingerprints
    )
    if not (
        np.array_equal(reference["gain_eeg"], shared["gain_eeg"])
        and np.array_equal(reference["gain_meg"], shared["gain_meg"])
        and np.array_equal(reference["adjacency"].toarray(), shared["adjacency"])
    ):
        raise RuntimeError("archived and ERP protocols do not share the same forward grid")
    kernels = protected.connected_euclidean_surface_kernels(
        shared["vertices"],
        shared["adjacency"],
        shared["n_surf"],
        scales_mm=oaster.SURFACE_SCALES_MM,
    )
    runtime = {"shared": shared, "kernels": kernels}
    penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000.0)
    fingerprint_paths = (
        __file__,
        erp_runner.__file__,
        protocol.__file__,
        erp_protocol.__file__,
        protected.__file__,
        oaster.__file__,
        methods.__file__,
    )
    code_fingerprint = erp_runner._code_fingerprint(fingerprint_paths)
    parts = output / f"parts_code_{code_fingerprint[:12]}"
    parts.mkdir(parents=True, exist_ok=True)
    (output / "completion.json").unlink(missing_ok=True)

    for pair, pair_cases in selected:
        path = erp_runner._pair_path(parts, pair)
        existing = archive._read_csv(path)
        if not force and _valid_pair(existing, pair_cases, manifest_sha256):
            print(f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: verified checkpoint", flush=True)
            continue
        archived = _archive_observations(
            pair_cases, chunks, cases, manifest_sha256, geometry, fingerprints
        )
        arguments = [
            (case, archived[int(case["case_number"])], runtime, manifest_sha256)
            for case in pair_cases
        ]
        if workers == 1:
            scored = [_score_case(*argument) for argument in arguments]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                scored = list(pool.map(lambda argument: _score_case(*argument), arguments))
        rows = [result[method] for result in scored for method in METHODS]
        archive._atomic_csv(path, rows, archive.ROW_FIELDS)
        if not _valid_pair(rows, pair_cases, manifest_sha256):
            failed = [row for row in rows if row["status"] != "ok"]
            detail = failed[0]["error"] if failed else "checkpoint ordering mismatch"
            raise RuntimeError(f"SNR pair {pair} failed: {detail}")
        print(
            f"SNR EEG={pair[0]:+d}, MEG={pair[1]:+d}: "
            f"scored {len(pair_cases)} cases x 4 crossover cells",
            flush=True,
        )

    if erp_runner._code_fingerprint(fingerprint_paths) != code_fingerprint:
        raise RuntimeError("crossover code changed while the run was in progress")
    combined = []
    for pair, pair_cases in selected:
        rows = archive._read_csv(erp_runner._pair_path(parts, pair))
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
        input_root,
        data_root,
        sample_path,
        selected,
        cases_per_scenario,
        workers,
        fingerprints,
        code_fingerprint,
    )
    archive._write_completion(
        output,
        combined,
        expected_row_count=sum(len(pair_cases) for _pair, pair_cases in selected) * len(METHODS),
        manifest_sha256=manifest_sha256,
        chunk_count=len(selected),
        methods=METHODS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
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
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        run(
            args.manifest,
            args.input_root,
            args.data_root,
            args.sample_path,
            args.output,
            snr_pairs=[tuple(pair) for pair in args.snr_pair] if args.snr_pair else None,
            cases_per_scenario=args.cases_per_scenario,
            workers=args.workers,
            force=args.force,
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
