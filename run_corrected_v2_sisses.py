"""Run external SISSES on corrected-v2 without vendoring third-party code.

With no stage this command only checks paths and the first frozen input header.
Use ``smoke`` for one case, or explicitly pass ``--limit-chunks``/``--all-chunks``
to ``run`` and ``score``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.io as sio

import run_strict_comparators as comparators
import run_strict_oaster as strict


ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "corrected_v2_sisses_adapter.m"
DEFAULT_MANIFEST = ROOT / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
DEFAULT_INPUT = Path(r"D:\oaster_corrected_v2_sisses\matlab_input")
DEFAULT_DATA = ROOT / "corrected_v2" / "generated"
DEFAULT_OUTPUT = ROOT / "results" / "corrected_v2" / "strict_blind" / "sisses_final"
METHOD = "SISSES"
OUTPUT_FORMAT = "corrected_v2_sisses_output_v1"
CORE_FILES = ("wen_sisses.m", "wen_mrf_admm.m", "VariationEdge.m", "TBFSelection.m")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solver_info(sisses_root: Path | None, fast_root: Path | None) -> dict:
    if sisses_root is None:
        raise ValueError("set SISSES_ROOT or pass --sisses-root")
    root = Path(sisses_root).resolve()
    files = {name: root / name for name in CORE_FILES}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"external SISSES core is incomplete: {missing}")
    fast = None if fast_root is None else Path(fast_root).resolve()
    if fast is not None:
        files["wen_sisses_fast.m"] = fast / "wen_sisses_fast.m"
        if not files["wen_sisses_fast.m"].is_file():
            raise FileNotFoundError(f"SISSES_FAST_ROOT lacks wen_sisses_fast.m: {fast}")
    digest = hashlib.sha256(("fast" if fast else "upstream").encode())
    digest.update(ADAPTER.read_bytes())
    hashes = {}
    for name, path in files.items():
        hashes[name] = _sha256(path)
        digest.update(name.encode())
        digest.update(bytes.fromhex(hashes[name]))
    return {
        "sisses_root": str(root),
        "fast_root": "" if fast is None else str(fast),
        "backend": "wen_sisses" if fast is None else "wen_sisses_fast",
        "external_file_sha256": hashes,
        "solver_fingerprint": digest.hexdigest(),
    }


def _matlab_executable(value: str) -> str:
    path = Path(value)
    found = str(path.resolve()) if path.is_file() else shutil.which(value)
    if not found:
        raise FileNotFoundError(f"MATLAB executable was not found: {value}")
    return found


def _context(manifest: Path, input_root: Path, data_root: Path, output: Path) -> dict:
    strict._guard_explicit_manifest_paths(manifest, input_root, data_root, output)
    cases, manifest_sha256 = strict._load_manifest(manifest)
    chunks = strict._discover_chunks(input_root, len(cases))
    geometry = strict._load_geometry(data_root)
    reference = strict._load_chunk(chunks[0].path, observations=False)
    fingerprints = strict._fingerprints(reference)
    strict._validate_chunk(
        chunks[0], reference, cases, manifest_sha256, geometry, fingerprints
    )
    return {
        "cases": cases,
        "manifest_sha256": manifest_sha256,
        "chunks": chunks,
        "geometry": geometry,
        "reference": reference,
        "fingerprints": fingerprints,
    }


def select_chunks(
    chunks: list[strict.Chunk], stage: str, start: int, limit: int | None, all_chunks: bool
) -> list[strict.Chunk]:
    if start < 0 or start >= len(chunks):
        raise ValueError("--start-chunk is outside the archive")
    if all_chunks and (start or limit is not None):
        raise ValueError("--all-chunks cannot be combined with a range")
    if limit is not None and limit < 1:
        raise ValueError("--limit-chunks must be positive")
    if stage in {"run", "score"} and not all_chunks and limit is None:
        raise ValueError(f"{stage} requires --limit-chunks or --all-chunks")
    if all_chunks:
        return chunks
    if stage == "smoke":
        return [chunks[start]]
    return chunks[start : start + (limit or 1)]


def _validate_headers(selected: list[strict.Chunk], context: dict) -> None:
    for spec in selected:
        chunk = strict._load_chunk(spec.path, observations=False)
        strict._validate_chunk(
            spec,
            chunk,
            context["cases"],
            context["manifest_sha256"],
            context["geometry"],
            context["fingerprints"],
        )


def _output_path(output: Path, spec: strict.Chunk, local_index: int) -> Path:
    return output / "matlab_output" / spec.path.stem / f"case_{local_index + 1:04d}.mat"


def _small_output_fields(path: Path) -> dict:
    return sio.loadmat(
        path,
        variable_names=(
            "case_id",
            "method_names",
            "source_shape",
            "metadata",
            "success",
            "errors",
            "preprocessing_elapsed_sec",
            "format_version",
            "manifest_sha256",
            "solver_fingerprint",
        ),
        simplify_cells=True,
    )


def validate_output(
    path: Path,
    case_id: str,
    manifest_sha256: str,
    solver_fingerprint: str,
    expected_shape: tuple[int, int],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    fields = _small_output_fields(path)
    shapes = {name: shape for name, shape, _kind in sio.whosmat(path)}
    source_shape = shapes.get("source_estimates")
    recorded = tuple(int(value) for value in np.asarray(fields.get("source_shape", ())).ravel())
    if (
        source_shape is None
        or tuple(source_shape[:2]) != expected_shape
        or recorded[:2] != expected_shape
        or strict._text(fields.get("case_id")) != case_id
        or strict._string_list(fields.get("method_names", [])) != [METHOD]
        or not bool(np.asarray(fields.get("success", False)).item())
        or strict._text(fields.get("format_version")) != OUTPUT_FORMAT
        or strict._text(fields.get("manifest_sha256")) != manifest_sha256
        or strict._text(fields.get("solver_fingerprint")) != solver_fingerprint
    ):
        raise ValueError(f"invalid or stale SISSES output: {path}")
    return fields


def load_estimate(
    path: Path,
    case_id: str,
    manifest_sha256: str,
    solver_fingerprint: str,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    fields = validate_output(
        path, case_id, manifest_sha256, solver_fingerprint, expected_shape
    )
    source = sio.loadmat(
        path, variable_names=("source_estimates",), simplify_cells=True
    )["source_estimates"]
    estimate = np.asarray(source, dtype=float).squeeze()
    if estimate.shape == expected_shape[::-1]:
        estimate = estimate.T
    if estimate.shape != expected_shape or not np.all(np.isfinite(estimate)):
        raise ValueError(f"invalid SISSES numerical estimate: {path}")
    return estimate, fields


def _matlab_quote(value: object) -> str:
    return str(value).replace("'", "''")


def _run_chunk(
    spec: strict.Chunk,
    context: dict,
    output: Path,
    solver: dict,
    matlab: str,
    case_limit: int,
) -> str:
    chunk_output = output / "matlab_output" / spec.path.stem
    chunk_output.mkdir(parents=True, exist_ok=True)
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    expression = (
        "maxNumCompThreads(1);"
        f"addpath('{_matlab_quote(ROOT)}');"
        f"corrected_v2_sisses_adapter('{_matlab_quote(spec.path.resolve())}',"
        f"'{_matlab_quote(chunk_output.resolve())}',"
        f"'{context['manifest_sha256']}',{case_limit},"
        f"'{solver['solver_fingerprint']}');"
    )
    environment = os.environ.copy()
    environment["SISSES_ROOT"] = solver["sisses_root"]
    if solver["fast_root"]:
        environment["SISSES_FAST_ROOT"] = solver["fast_root"]
    else:
        environment.pop("SISSES_FAST_ROOT", None)
    with (log_root / f"{spec.path.stem}.out.log").open("w", encoding="utf-8") as stdout, (
        log_root / f"{spec.path.stem}.err.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            (matlab, "-batch", expression),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"MATLAB failed for {spec.path.stem}; see {log_root / (spec.path.stem + '.err.log')}"
        )
    expected_shape = (
        int(context["geometry"]["n_surf"] + context["geometry"]["n_deep"]),
        int(context["geometry"]["times"].size),
    )
    count = spec.case_count if case_limit <= 0 else min(case_limit, spec.case_count)
    for local_index, case in enumerate(context["cases"][spec.start : spec.start + count]):
        validate_output(
            _output_path(output, spec, local_index),
            str(case["case_id"]),
            context["manifest_sha256"],
            solver["solver_fingerprint"],
            expected_shape,
        )
    return f"{spec.path.stem}: {count} valid output(s)"


def _valid_part(rows: list[dict], spec: strict.Chunk, cases: list[dict], digest: str) -> bool:
    expected = cases[spec.start : spec.stop]
    try:
        return (
            len(rows) == spec.case_count
            and [row["case_id"] for row in rows] == [case["case_id"] for case in expected]
            and all(row["manifest_sha256"] == digest for row in rows)
            and all(row["chunk_file"] == spec.path.name for row in rows)
            and all(row["method"] == METHOD and row["status"] == "ok" for row in rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _score_chunk(
    spec: strict.Chunk, context: dict, runtime: dict, output: Path, solver: dict, rescore: bool
) -> str:
    part = output / "parts" / f"{spec.path.stem}.csv"
    existing = _read_csv(part)
    if not rescore and _valid_part(
        existing, spec, context["cases"], context["manifest_sha256"]
    ):
        return f"{spec.path.stem}: verified score checkpoint"
    expected_shape = (
        int(context["geometry"]["n_surf"] + context["geometry"]["n_deep"]),
        int(context["geometry"]["times"].size),
    )
    rows = []
    for local_index, case in enumerate(context["cases"][spec.start : spec.stop]):
        base = strict._case_fields(
            case, spec, context["manifest_sha256"], int(bool(case.get("surface_centers")))
        )
        try:
            estimate, fields = load_estimate(
                _output_path(output, spec, local_index),
                str(case["case_id"]),
                context["manifest_sha256"],
                solver["solver_fingerprint"],
                expected_shape,
            )
            truth, groups, _ = strict.protocol.truth_for_case(runtime["shared"], case)
            metadata = fields.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            elapsed = float(np.asarray(fields.get("preprocessing_elapsed_sec", 0)).item())
            elapsed += float(metadata.get("elapsed_sec", 0) or 0)
            row = comparators._success_row(
                base, METHOD, estimate, truth, groups, runtime, elapsed
            )
            row["temporal_rank"] = metadata.get("temporal_rank", "")
        except Exception as exc:
            row = comparators._error_row(base, METHOD, exc, 0.0)
        rows.append(row)
    strict._atomic_csv(part, rows, strict.ROW_FIELDS)
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{spec.path.stem} has {len(failed)} scoring failure(s): {failed[0]['error']}")
    if not _valid_part(rows, spec, context["cases"], context["manifest_sha256"]):
        raise RuntimeError(f"invalid score checkpoint: {part}")
    return f"{spec.path.stem}: scored {len(rows)} case(s)"


def _write_metadata(output: Path, context: dict, solver: dict, stage: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": METHOD,
        "stage": stage,
        "manifest_sha256": context["manifest_sha256"],
        "case_count": len(context["cases"]),
        "input_root": str(context["chunks"][0].path.parent.resolve()),
        "adapter": str(ADAPTER),
        "adapter_sha256": _sha256(ADAPTER),
        **solver,
        "third_party_source_copied": False,
        "parameters": [[0.05, 0.01, 0.0], [0.1, 0.05, 0.2]],
        "selection": "per-case observation-only BIC; truth is used only by score",
    }
    path = output / "metadata.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _merge(context: dict, output: Path) -> None:
    rows = []
    for spec in context["chunks"]:
        part = _read_csv(output / "parts" / f"{spec.path.stem}.csv")
        if not _valid_part(part, spec, context["cases"], context["manifest_sha256"]):
            raise RuntimeError(f"missing or invalid score checkpoint for {spec.path.stem}")
        rows.extend(part)
    strict._atomic_csv(output / "rows.csv", rows, strict.ROW_FIELDS)
    penalty_mm = float(
        np.linalg.norm(np.ptp(context["geometry"]["vertices"], axis=0)) * 1000.0
    )
    comparators._summaries(rows, output, penalty_mm)
    strict._write_completion(
        output,
        rows,
        expected_row_count=len(context["cases"]),
        manifest_sha256=context["manifest_sha256"],
        chunk_count=len(context["chunks"]),
        methods=(METHOD,),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="check", choices=("check", "smoke", "run", "score"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sisses-root", type=Path, default=os.environ.get("SISSES_ROOT"))
    parser.add_argument("--sisses-fast-root", type=Path, default=os.environ.get("SISSES_FAST_ROOT"))
    parser.add_argument("--matlab", default=os.environ.get("MATLAB_EXE", "matlab"))
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--all-chunks", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rescore", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        context = _context(args.manifest, args.input_root, args.data_root, args.output)
        solver = solver_info(args.sisses_root, args.sisses_fast_root)
        matlab = _matlab_executable(args.matlab)
        selected = select_chunks(
            context["chunks"], args.stage, args.start_chunk, args.limit_chunks, args.all_chunks
        )
        _validate_headers(selected, context)
        if args.stage == "check":
            print(json.dumps({
                "status": "ok",
                "manifest_sha256": context["manifest_sha256"],
                "cases": len(context["cases"]),
                "chunks": len(context["chunks"]),
                "checked_chunks": len(selected),
                "matlab": matlab,
                **solver,
            }, ensure_ascii=False, indent=2))
            return
        _write_metadata(args.output, context, solver, args.stage)
        if args.stage in {"smoke", "run"}:
            case_limit = 1 if args.stage == "smoke" else 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for message in pool.map(
                    lambda spec: _run_chunk(spec, context, args.output, solver, matlab, case_limit),
                    selected,
                ):
                    print(message, flush=True)
            return
        runtime = comparators._runtime(context["geometry"], context["reference"])
        for spec in selected:
            print(_score_chunk(spec, context, runtime, args.output, solver, args.rescore), flush=True)
        if args.all_chunks:
            _merge(context, args.output)
            _write_metadata(args.output, context, solver, "complete")
            print(f"complete: {args.output}")
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
