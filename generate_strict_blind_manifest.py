"""Generate the deterministic, center-disjoint OASTER strict-blind manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BENCHMARK = Path(os.environ.get("V15_BENCHMARK_ROOT", ROOT / "benchmark"))

from benchmark import protocol


OUTPUT = ROOT / "results" / "strict_blind" / "manifest.json"
OLD_MANIFESTS = (
    BENCHMARK / "results" / "protocol" / "dev_manifest.json",
    BENCHMARK / "results" / "protocol" / "test_manifest.json",
)
LEVELS = (-10, -5, 0, 5, 10, 15, 20)
PAIRS = tuple(product(LEVELS, repeat=2))
SEED_ROOT = 20260901
EXPECTED_CONFIG_COUNTS = {
    "surface_only": 68,
    "deep_only": 15,
    "deep_plus_surface": 68,
    "deep_plus_two_surface": 34,
}
EXPECTED_SHA256 = "3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76"


def _old_surface_centers() -> set[int]:
    return {
        int(center)
        for path in OLD_MANIFESTS
        for case in json.loads(path.read_text(encoding="utf-8"))
        for center in case["surface_centers"]
    }


def _blind_centers(shared: dict) -> list[dict]:
    """Select one unseen maximin center in each of the same 68 aparc parcels."""
    vertices = np.asarray(shared["vertices"], dtype=float)
    old = _old_surface_centers()
    result = []
    for name, indices in sorted(protocol._parcels(shared).items()):
        indices = sorted(map(int, indices))
        used = sorted(set(indices) & old)
        unseen = sorted(set(indices) - old)
        if not unseen:
            raise ValueError(f"no blind coarse-grid center remains in {name}")
        center = max(
            unseen,
            key=lambda index: (
                min(float(np.linalg.norm(vertices[index] - vertices[old_index])) for old_index in used),
                -index,
            ),
        )
        result.append({
            "parcel": name,
            "hemi": "lh" if name.endswith("-lh") else "rh",
            "center": int(center),
        })
    if len(result) != 68 or {row["hemi"] for row in result} != {"lh", "rh"}:
        raise ValueError("strict blind protocol requires 68 bilateral aparc parcels")
    return result


def _base_configurations(shared: dict) -> list[dict]:
    records = _blind_centers(shared)
    vertices = np.asarray(shared["vertices"], dtype=float)
    left = [row["center"] for row in records if row["hemi"] == "lh"]
    right = [row["center"] for row in records if row["hemi"] == "rh"]
    pairs = protocol._farthest_pairs(left, right, vertices)
    parcel_by_center = {row["center"]: row["parcel"] for row in records}
    n_surf = int(shared["n_surf"])
    n_deep = int(shared["n_deep"])
    if n_deep != 15:
        raise ValueError("strict blind protocol expects the frozen 15-point deep grid")

    base = []
    for location, row in enumerate(records):
        base.append(dict(
            scenario="surface_only", location=location, replicate=0,
            surface_centers=[row["center"]], surface_parcels=[row["parcel"]],
            deep_local=None,
        ))
    for deep_local in range(n_deep):
        base.append(dict(
            scenario="deep_only", location=deep_local, replicate=0,
            surface_centers=[], surface_parcels=[], deep_local=deep_local,
        ))
    for location, row in enumerate(records):
        base.append(dict(
            scenario="deep_plus_surface", location=location, replicate=0,
            surface_centers=[row["center"]], surface_parcels=[row["parcel"]],
            deep_local=location % n_deep,
        ))
    for location, centers in enumerate(pairs):
        base.append(dict(
            scenario="deep_plus_two_surface", location=location, replicate=0,
            surface_centers=centers,
            surface_parcels=[parcel_by_center[center] for center in centers],
            deep_local=location % n_deep,
        ))

    for index, case in enumerate(base):
        scenario = case["scenario"]
        deep_local = case["deep_local"]
        multi = scenario in {"deep_plus_surface", "deep_plus_two_surface"}
        case.update(
            configuration_id=f"strict-blind-source-{index:03d}-{scenario}",
            configuration_number=index,
            scenario_code=protocol.SCENARIO_CODES[scenario],
            deep_index=None if deep_local is None else n_surf + deep_local,
            deep_surface_ratio=(0.5, 1.0)[case["location"] % 2] if multi else None,
            correlation=(0.0, 0.5, 0.9)[case["location"] % 3] if multi else None,
        )
    return base


def make_manifest(shared: dict) -> list[dict]:
    base = _base_configurations(shared)
    manifest = []
    for pair_index, (eeg_snr, meg_snr) in enumerate(PAIRS):
        for configuration in base:
            case = dict(configuration)
            number = len(manifest)
            scenario = case["scenario"]
            case.update(
                case_id=(
                    f"strict-blind-{number:05d}-{scenario}-"
                    f"eeg{eeg_snr:+03d}-meg{meg_snr:+03d}"
                ),
                case_number=number,
                panel="strict_blind",
                source_case_id=case["configuration_id"],
                pair_index=pair_index,
                snr_db=eeg_snr,
                eeg_snr_db=eeg_snr,
                meg_snr_db=meg_snr,
                seed=[
                    SEED_ROOT,
                    pair_index,
                    int(case["scenario_code"]),
                    int(case["configuration_number"]),
                ],
            )
            manifest.append(case)
    return manifest


def _check(shared: dict, manifest: list[dict], digest: str) -> None:
    old = _old_surface_centers()
    new = {int(center) for case in manifest for center in case["surface_centers"]}
    assert len(new) == 68 and not (new & old)

    by_source = {case["source_case_id"]: case for case in manifest}
    assert len(by_source) == 185
    assert Counter(case["scenario"] for case in by_source.values()) == EXPECTED_CONFIG_COUNTS
    assert {case["deep_local"] for case in by_source.values() if case["deep_local"] is not None} == set(range(15))

    assert len(manifest) == 49 * 185
    assert len({case["case_id"] for case in manifest}) == len(manifest)
    assert len({tuple(case["seed"]) for case in manifest}) == len(manifest)
    for pair_index, pair in enumerate(PAIRS):
        cell = [case for case in manifest if int(case["pair_index"]) == pair_index]
        assert len(cell) == 185
        assert {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cell} == {pair}
        assert all(case["seed"][1] == pair_index for case in cell)

    parcels = protocol._parcels(shared)
    parcel_hits = Counter()
    for case in by_source.values():
        if case["scenario"] != "surface_only":
            continue
        center = int(case["surface_centers"][0])
        parcel = case["surface_parcels"][0]
        assert center in set(map(int, parcels[parcel]))
        parcel_hits[parcel] += 1
    assert len(parcel_hits) == 68 and set(parcel_hits.values()) == {1}
    assert digest == EXPECTED_SHA256


def _load_shared(data_root: Path | None, sample_path: Path | None) -> dict:
    kwargs = {}
    if data_root is not None:
        kwargs["data_root"] = data_root
    if sample_path is not None:
        kwargs["sample_path"] = sample_path
    return protocol.load_shared(**kwargs)


def generate(
    path: Path = OUTPUT,
    *,
    data_root: Path | None = None,
    sample_path: Path | None = None,
) -> str:
    shared = _load_shared(data_root, sample_path)
    manifest = make_manifest(shared)
    _check(shared, manifest, EXPECTED_SHA256)
    digest = protocol.save_manifest(path, manifest, expected_digest=EXPECTED_SHA256)
    _check(shared, manifest, digest)
    return digest


def self_check(
    path: Path = OUTPUT,
    *,
    data_root: Path | None = None,
    sample_path: Path | None = None,
) -> str:
    shared = _load_shared(data_root, sample_path)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _check(shared, manifest, digest)
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii")
    assert sidecar == f"{digest}  {path.name}\n"
    print(f"strict blind manifest self-check passed: {digest}")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "self-check"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-path", type=Path)
    args = parser.parse_args()
    if args.stage == "generate":
        print(generate(args.output, data_root=args.data_root, sample_path=args.sample_path))
    else:
        self_check(args.output, data_root=args.data_root, sample_path=args.sample_path)


if __name__ == "__main__":
    main()
