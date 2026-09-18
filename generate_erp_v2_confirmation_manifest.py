"""Generate the independent ERP-v2 confirmation SNR matrix."""

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
OUTPUT = ROOT / "results" / "erp_whole_head" / "confirmation_v3" / "manifest.json"
ID_PREFIX = "erp-v3-confirmation"
PANEL = "erp_v3_confirmation"
SEED_ROOT = 20261002
# Configuration 302 was rendered during tooling QA, so it is excluded before locking v3.
EXCLUDED_CONFIGURATION_NUMBERS = (302,)
EXPECTED_CONFIG_COUNTS = {
    "surface_only": 136,
    "deep_only": 30,
    "deep_plus_surface": 136,
    "deep_plus_two_surface": 67,
}
EXPECTED_SHA256 = "929b6f2b227ac6230883275cfff20d671e2d5f9549e6c87973ebc9479b5d37b9"


def make_manifest(shared: dict) -> list[dict]:
    """Expand 369 untouched source configurations over 49 modality SNR pairs."""
    split = protocol.build_split(shared)
    all_base = protocol.make_manifest(split, panel="test", snrs=(0,))
    base = [
        (number, case)
        for number, case in enumerate(all_base)
        if number not in EXCLUDED_CONFIGURATION_NUMBERS
    ]
    if Counter(case["scenario"] for _number, case in base) != Counter(
        EXPECTED_CONFIG_COUNTS
    ):
        raise RuntimeError("ERP-v2 confirmation base scenario counts are invalid")

    manifest = []
    for pair_index, (eeg_snr, meg_snr) in enumerate(strict.PAIRS):
        for configuration_number, source in base:
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

    expected_cases = len(strict.PAIRS) * sum(EXPECTED_CONFIG_COUNTS.values())
    assert len(manifest) == expected_cases == 18081
    assert [int(case["case_number"]) for case in manifest] == list(range(expected_cases))
    assert len({case["case_id"] for case in manifest}) == expected_cases
    assert len({tuple(case["seed"]) for case in manifest}) == expected_cases
    assert all(case["panel"] == PANEL and case["seed"][0] == SEED_ROOT for case in manifest)
    assert not (
        {int(case["configuration_number"]) for case in manifest}
        & set(EXCLUDED_CONFIGURATION_NUMBERS)
    )
    assert {
        int(center) for case in manifest for center in case["surface_centers"]
    } <= surface_test
    assert {
        int(case["deep_local"])
        for case in manifest
        if case["deep_local"] is not None
    } <= deep_test

    cell_size = sum(EXPECTED_CONFIG_COUNTS.values())
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
