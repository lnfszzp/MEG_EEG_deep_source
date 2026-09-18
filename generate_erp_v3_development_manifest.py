"""Generate the full development-position ERP SNR matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from benchmark import protocol
import generate_corrected_v2_manifest as corrected
import generate_strict_blind_manifest as strict


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "corrected_v2" / "generated"
OUTPUT = ROOT / "results" / "erp_whole_head" / "development_full_v3" / "manifest.json"
ID_PREFIX = "erp-v3-development"
PANEL = "erp_v3_development"
SEED_ROOT = 20260921
EXPECTED_CONFIG_COUNTS = {
    "surface_only": 68,
    "deep_only": 18,
    "deep_plus_surface": 68,
    "deep_plus_two_surface": 34,
}
EXPECTED_SHA256 = "0989782ea769bc0a698e8e6f250795ee829398afc5984a904eeb9544a32432b5"


def make_manifest(shared: dict) -> list[dict]:
    split = protocol.build_split(shared)
    base = protocol.make_manifest(split, panel="dev", snrs=(0,))
    if Counter(case["scenario"] for case in base) != Counter(EXPECTED_CONFIG_COUNTS):
        raise RuntimeError("ERP-v3 development scenario counts are invalid")

    manifest = []
    for pair_index, (eeg_snr, meg_snr) in enumerate(strict.PAIRS):
        for configuration_number, source in enumerate(base):
            case = dict(source)
            scenario = str(case["scenario"])
            case.update(
                case_id=(
                    f"{ID_PREFIX}-{len(manifest):05d}-{scenario}-"
                    f"eeg{eeg_snr:+03d}-meg{meg_snr:+03d}"
                ),
                case_number=len(manifest),
                configuration_id=(
                    f"{ID_PREFIX}-source-{configuration_number:03d}-{scenario}"
                ),
                configuration_number=configuration_number,
                source_case_id=str(source["case_id"]),
                panel=PANEL,
                pair_index=pair_index,
                snr_db=eeg_snr,
                eeg_snr_db=eeg_snr,
                meg_snr_db=meg_snr,
                seed=[
                    SEED_ROOT,
                    pair_index,
                    int(case["scenario_code"]),
                    configuration_number,
                ],
            )
            manifest.append(case)
    return manifest


def _check(
    shared: dict,
    manifest: list[dict],
    digest: str,
    *,
    expected_digest: str | None = EXPECTED_SHA256,
) -> None:
    split = protocol.build_split(shared)
    surface_dev = set(map(int, split["surface_dev"]))
    surface_test = set(map(int, split["surface_test"]))
    deep_dev = set(map(int, split["deep_dev_local"]))
    deep_test = set(map(int, split["deep_test_local"]))
    assert not (surface_dev & surface_test)
    assert not (deep_dev & deep_test)

    cell_size = sum(EXPECTED_CONFIG_COUNTS.values())
    expected_cases = len(strict.PAIRS) * cell_size
    assert len(manifest) == expected_cases == 9212
    assert [int(case["case_number"]) for case in manifest] == list(range(expected_cases))
    assert len({case["case_id"] for case in manifest}) == expected_cases
    assert len({tuple(case["seed"]) for case in manifest}) == expected_cases
    assert all(case["panel"] == PANEL and case["seed"][0] == SEED_ROOT for case in manifest)
    assert {
        int(center) for case in manifest for center in case["surface_centers"]
    } <= surface_dev
    assert {
        int(case["deep_local"])
        for case in manifest
        if case["deep_local"] is not None
    } <= deep_dev

    for pair_index, pair in enumerate(strict.PAIRS):
        cell = manifest[pair_index * cell_size : (pair_index + 1) * cell_size]
        assert Counter(case["scenario"] for case in cell) == Counter(
            EXPECTED_CONFIG_COUNTS
        )
        assert {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cell} == {
            pair
        }
        assert all(case["pair_index"] == pair_index for case in cell)
    if expected_digest is not None:
        assert digest == expected_digest


def generate(
    path: Path = OUTPUT,
    *,
    data_root: Path = DATA_ROOT,
    sample_path: Path | None = None,
    expected_digest: str | None = EXPECTED_SHA256,
) -> str:
    shared = corrected._load_shared(Path(data_root), sample_path)
    manifest = make_manifest(shared)
    digest = strict._manifest_digest(manifest)
    _check(shared, manifest, digest, expected_digest=expected_digest)
    return protocol.save_manifest(path, manifest, expected_digest=expected_digest)


def self_check(
    path: Path = OUTPUT,
    *,
    data_root: Path = DATA_ROOT,
    sample_path: Path | None = None,
    expected_digest: str | None = EXPECTED_SHA256,
) -> str:
    shared = corrected._load_shared(Path(data_root), sample_path)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _check(shared, manifest, digest, expected_digest=expected_digest)
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii")
    assert sidecar == f"{digest}  {path.name}\n"
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "self-check"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--sample-path", type=Path)
    args = parser.parse_args()
    action = generate if args.stage == "generate" else self_check
    print(action(args.output, data_root=args.data_root, sample_path=args.sample_path))


if __name__ == "__main__":
    main()
