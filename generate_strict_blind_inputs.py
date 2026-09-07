"""Generate resumable immutable MAT chunks for an explicit strict manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse

from benchmark import protocol
import run_strict_oaster as strict


FORMAT_VERSION = strict.FORMAT_VERSION
LEGACY_ARCHIVE = Path(r"D:\oaster_strict_blind_sisses").resolve()


def _safe_output(path: Path) -> Path:
    path = Path(path).resolve()
    if path == LEGACY_ARCHIVE or LEGACY_ARCHIVE in path.parents:
        raise ValueError(f"refusing to write inside the preserved legacy archive: {path}")
    return path


def _runtime_geometry(shared: dict) -> tuple[dict, dict, dict[str, str]]:
    geometry = {
        "vertices": np.asarray(shared["vertices"], dtype=float),
        "times": np.asarray(shared["times"], dtype=float).ravel(),
        "n_surf": int(shared["n_surf"]),
        "n_deep": int(shared["n_deep"]),
    }
    reference = {
        "gain_eeg": np.asarray(shared["gain_eeg"], dtype=float),
        "gain_meg": np.asarray(shared["gain_meg"], dtype=float),
        "adjacency": strict._canonical_adjacency(shared["adjacency"]),
    }
    return geometry, reference, strict._fingerprints(reference)


def _load_valid_chunk(
    spec: strict.Chunk,
    cases: list[dict],
    manifest_sha256: str,
    geometry: dict,
    fingerprints: dict[str, str],
) -> None:
    chunk = strict._load_chunk(spec.path, observations=True)
    strict._validate_chunk(
        spec, chunk, cases, manifest_sha256, geometry, fingerprints
    )


def _observations(
    shared: dict,
    case: dict,
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    key = str(
        case.get("configuration_id")
        or case.get("source_case_id")
        or case["case_id"]
    )
    if key not in clean_cache:
        truth, _groups, _meta = protocol.truth_for_case(shared, case)
        clean_cache[key] = (
            np.asarray(shared["gain_eeg"]) @ truth,
            np.asarray(shared["gain_meg"]) @ truth,
        )
    clean_eeg, clean_meg = clean_cache[key]
    eeg_seed, meg_seed = np.random.SeedSequence(case["seed"]).spawn(2)
    eeg, _ = protocol._add_noise(
        clean_eeg,
        shared["noise_factor_eeg"],
        case.get("eeg_snr_db", case["snr_db"]),
        eeg_seed,
    )
    meg, _ = protocol._add_noise(
        clean_meg,
        shared["noise_factor_meg"],
        case.get("meg_snr_db", case["snr_db"]),
        meg_seed,
    )
    return eeg, meg


def _write_chunk(
    spec: strict.Chunk,
    cases: list[dict],
    manifest_sha256: str,
    shared: dict,
    geometry: dict,
    reference: dict,
    fingerprints: dict[str, str],
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    eeg, meg = [], []
    for case in cases[spec.start : spec.stop]:
        observed_eeg, observed_meg = _observations(shared, case, clean_cache)
        eeg.append(observed_eeg)
        meg.append(observed_meg)
    payload = {
        "F_EEG": np.stack(eeg, axis=2),
        "F_MEG": np.stack(meg, axis=2),
        "Gain_EEG": reference["gain_eeg"],
        "Gain_MEG": reference["gain_meg"],
        "VertConn": sparse.csc_matrix(reference["adjacency"]),
        "case_ids": np.asarray(
            [case["case_id"] for case in cases[spec.start : spec.stop]], dtype=object
        )[:, None],
        "manifest_sha256": manifest_sha256,
        "format_version": FORMAT_VERSION,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=spec.path.parent, prefix=".strict-", suffix=".tmp.mat", delete=False
        ) as stream:
            temporary = Path(stream.name)
        sio.savemat(
            temporary,
            payload,
            appendmat=False,
            do_compression=True,
            oned_as="column",
        )
        temporary_spec = strict.Chunk(spec.index, temporary, spec.start, spec.stop)
        _load_valid_chunk(
            temporary_spec, cases, manifest_sha256, geometry, fingerprints
        )
        os.replace(temporary, spec.path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def generate(
    manifest_path: Path,
    data_root: Path,
    output: Path,
    *,
    chunk_size: int = 20,
    sample_path: Path | None = None,
    force: bool = False,
    shared: dict | None = None,
) -> dict:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    output = _safe_output(output)
    cases, manifest_sha256 = strict._load_manifest(Path(manifest_path))
    if shared is None:
        kwargs = {"data_root": Path(data_root)}
        if sample_path is not None:
            kwargs["sample_path"] = Path(sample_path)
        shared = protocol.load_shared(**kwargs)
    geometry, reference, fingerprints = _runtime_geometry(shared)
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        strict.Chunk(index, output / f"strict_{start:05d}_{stop:05d}.mat", start, stop)
        for index, start in enumerate(range(0, len(cases), chunk_size))
        for stop in (min(start + chunk_size, len(cases)),)
    ]
    expected = {spec.path for spec in specs}
    unexpected = sorted(
        path.name for path in output.glob("strict_*.mat") if path not in expected
    )
    if unexpected:
        raise RuntimeError(f"unexpected strict MAT files in output: {unexpected[:3]}")

    written = resumed = 0
    clean_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for spec in specs:
        if spec.path.exists() and not force:
            try:
                _load_valid_chunk(
                    spec, cases, manifest_sha256, geometry, fingerprints
                )
            except Exception as exc:
                raise RuntimeError(
                    f"existing chunk is invalid; rerun with --force to replace it: {spec.path.name}"
                ) from exc
            resumed += 1
            print(f"verified {spec.path.stem}", flush=True)
            continue
        _write_chunk(
            spec,
            cases,
            manifest_sha256,
            shared,
            geometry,
            reference,
            fingerprints,
            clean_cache,
        )
        written += 1
        print(f"wrote {spec.path.stem}", flush=True)
    return {
        "status": "ok",
        "manifest_sha256": manifest_sha256,
        "format_version": FORMAT_VERSION,
        "cases": len(cases),
        "chunks": len(specs),
        "written": written,
        "resumed": resumed,
        "output": str(output),
        "cross_chunk_fingerprints_sha256": fingerprints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = generate(
        args.manifest,
        args.data_root,
        args.output,
        chunk_size=args.chunk_size,
        sample_path=args.sample_path,
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
