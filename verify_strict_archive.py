"""Verify the preserved strict-blind SISSES input archive without scanning outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

from benchmark import protocol


EXPECTED_SHA256 = "3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76"
FORMAT_VERSION = "strict_blind_sisses_input_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _group_sha256(groups: list[np.ndarray]) -> str:
    payload = json.dumps(
        [np.asarray(group, dtype=int).tolist() for group in groups],
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _snr(clean: np.ndarray, observed: np.ndarray, active_start: int) -> float:
    clean_energy = float(np.sum(clean[:, active_start:] ** 2))
    residual_energy = float(np.sum((observed[:, active_start:] - clean[:, active_start:]) ** 2))
    if clean_energy == 0.0 or residual_energy == 0.0:
        raise ValueError("sampled clean signal and archive residual must have non-zero energy")
    return float(10.0 * np.log10(clean_energy / residual_energy))


def verify_archive(
    manifest_path: str | Path,
    input_dir: str | Path,
    *,
    expected_sha256: str = EXPECTED_SHA256,
    chunk_size: int = 20,
    sample_cases: tuple[int, ...] | list[int] = (),
    data_root: str | Path | None = None,
    mne_sample_path: str | Path | None = None,
    shared: dict | None = None,
) -> dict:
    """Validate chunk identity; optionally confirm truth/groups and archived SNR."""
    manifest_path, input_dir = Path(manifest_path), Path(input_dir)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = _sha256(manifest_path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError(f"manifest SHA-256 mismatch: {digest}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if [int(case["case_number"]) for case in manifest] != list(range(len(manifest))):
        raise ValueError("manifest case_number values are not contiguous and ordered")
    case_ids = [str(case["case_id"]) for case in manifest]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("manifest case_id values are not unique")

    ranges = [(start, min(start + chunk_size, len(manifest))) for start in range(0, len(manifest), chunk_size)]
    expected_files = {
        input_dir / f"strict_{start:05d}_{end:05d}.mat" for start, end in ranges
    }
    actual_files = set(input_dir.glob("strict_*.mat"))
    missing = sorted(path.name for path in expected_files - actual_files)
    unexpected = sorted(path.name for path in actual_files - expected_files)
    if missing or unexpected:
        raise ValueError(f"chunk set mismatch; missing={missing[:3]}, unexpected={unexpected[:3]}")

    locations: dict[int, tuple[Path, int]] = {}
    archived_ids = 0
    for start, end in ranges:
        path = input_dir / f"strict_{start:05d}_{end:05d}.mat"
        metadata = sio.loadmat(
            path,
            variable_names=("case_ids", "manifest_sha256", "format_version"),
            simplify_cells=True,
        )
        ids = [str(value) for value in np.atleast_1d(metadata["case_ids"])]
        if ids != case_ids[start:end]:
            raise ValueError(f"case_ids disagree with manifest: {path.name}")
        if str(metadata["manifest_sha256"]) != digest:
            raise ValueError(f"embedded manifest SHA-256 mismatch: {path.name}")
        if str(metadata.get("format_version", "")) != FORMAT_VERSION:
            raise ValueError(f"unexpected format_version: {path.name}")
        archived_ids += len(ids)
        locations.update({number: (path, number - start) for number in range(start, end)})
    if archived_ids != len(manifest):
        raise ValueError(f"archive contains {archived_ids} case IDs, expected {len(manifest)}")

    requested = list(dict.fromkeys(map(int, sample_cases)))
    if any(number < 0 or number >= len(manifest) for number in requested):
        raise ValueError("sample case number is outside the manifest")
    samples = []
    if requested:
        if shared is None:
            kwargs = {}
            if data_root is not None:
                kwargs["data_root"] = data_root
            if mne_sample_path is not None:
                kwargs["sample_path"] = mne_sample_path
            shared = protocol.load_shared(**kwargs)
        by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)
        for number in requested:
            path, offset = locations[number]
            by_file[path].append((number, offset))
        repeated: dict[str, tuple[str, str]] = {}
        for path, entries in by_file.items():
            chunk = sio.loadmat(
                path,
                variable_names=("F_EEG", "F_MEG", "Gain_EEG", "Gain_MEG"),
            )
            for key, shared_key in (("Gain_EEG", "gain_eeg"), ("Gain_MEG", "gain_meg")):
                if not np.array_equal(np.asarray(chunk[key]), np.asarray(shared[shared_key])):
                    raise ValueError(f"{key} disagrees with shared forward model: {path.name}")
            for number, offset in entries:
                case = manifest[number]
                truth, groups, _ = protocol.truth_for_case(shared, case)
                expected_groups = len(case["surface_centers"]) + int(case.get("deep_index") is not None)
                if len(groups) != expected_groups:
                    raise ValueError(f"unexpected source group count: {case['case_id']}")
                for center, group in zip(case["surface_centers"], groups):
                    if int(center) not in np.asarray(group, dtype=int):
                        raise ValueError(f"surface center absent from its group: {case['case_id']}")
                if case.get("deep_index") is not None and np.asarray(groups[-1]).tolist() != [int(case["deep_index"])]:
                    raise ValueError(f"deep source group mismatch: {case['case_id']}")
                support = np.flatnonzero(np.linalg.norm(truth[:, protocol.ACTIVE_START :], axis=1) > 0)
                grouped = np.unique(np.concatenate(groups))
                if not np.array_equal(support, grouped):
                    raise ValueError(f"truth support disagrees with groups: {case['case_id']}")

                truth_sha, groups_sha = _array_sha256(truth), _group_sha256(groups)
                configuration = str(case.get("configuration_id", case.get("source_case_id", case["case_id"])))
                pair = (truth_sha, groups_sha)
                if configuration in repeated and repeated[configuration] != pair:
                    raise ValueError(f"truth/groups changed across SNR pairs: {configuration}")
                repeated[configuration] = pair
                observed_eeg = np.asarray(chunk["F_EEG"], dtype=float)[:, :, offset]
                observed_meg = np.asarray(chunk["F_MEG"], dtype=float)[:, :, offset]
                clean_eeg = np.asarray(shared["gain_eeg"]) @ truth
                clean_meg = np.asarray(shared["gain_meg"]) @ truth
                actual_eeg = _snr(clean_eeg, observed_eeg, protocol.ACTIVE_START)
                actual_meg = _snr(clean_meg, observed_meg, protocol.ACTIVE_START)
                target_eeg = float(case.get("eeg_snr_db", case["snr_db"]))
                target_meg = float(case.get("meg_snr_db", case["snr_db"]))
                if not np.isclose(actual_eeg, target_eeg, rtol=0.0, atol=1e-9) or not np.isclose(
                    actual_meg, target_meg, rtol=0.0, atol=1e-9
                ):
                    raise ValueError(f"archived observation SNR mismatch: {case['case_id']}")
                samples.append(
                    {
                        "case_number": number,
                        "case_id": case["case_id"],
                        "chunk": path.name,
                        "truth_sha256": truth_sha,
                        "groups_sha256": groups_sha,
                        "eeg_snr_db": actual_eeg,
                        "meg_snr_db": actual_meg,
                    }
                )

    return {
        "status": "ok",
        "manifest_sha256": digest,
        "cases": len(manifest),
        "chunks": len(expected_files),
        "archived_case_ids": archived_ids,
        "format_version": FORMAT_VERSION,
        "sampled_cases": sorted(samples, key=lambda item: item["case_number"]),
        "output_tree_scanned": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parent / "results" / "strict_blind" / "manifest.json")
    parser.add_argument("--input-dir", type=Path, required=True, help="Archive matlab_input directory")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--sample-cases", nargs="+", type=int, metavar="CASE_NUMBER")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--mne-sample-path", type=Path)
    args = parser.parse_args()
    try:
        report = verify_archive(
            args.manifest,
            args.input_dir,
            expected_sha256=args.expected_sha256,
            chunk_size=args.chunk_size,
            sample_cases=args.sample_cases or (),
            data_root=args.data_root,
            mne_sample_path=args.mne_sample_path,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
