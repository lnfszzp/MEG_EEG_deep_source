from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pytest

from benchmark import protocol
import generate_corrected_v2_manifest as corrected
import generate_strict_blind_manifest as strict
import run_oaster_dev_matrix as dev_runner


ROOT = Path(__file__).resolve().parents[1]


def _split_shared(n_deep: int, *, corrected_geometry: bool = False) -> dict:
    parcels = {}
    for hemi_index, hemi in enumerate(("lh", "rh")):
        offset = hemi_index * 102
        for index in range(34):
            parcels[f"parcel-{index:02d}-{hemi}"] = list(
                range(offset + 3 * index, offset + 3 * index + 3)
            )
    rng = np.random.default_rng(7)
    shared = {
        "vertices": rng.normal(size=(204 + n_deep, 3)),
        "n_surf": 204,
        "n_deep": n_deep,
        "parcels": parcels,
    }
    if corrected_geometry:
        shared["deep_rr_mri"] = np.zeros((n_deep, 3))
    return shared


def _strict_shared(n_deep: int) -> tuple[dict, list[dict]]:
    records = [
        {
            "parcel": f"parcel-{index:02d}-{'lh' if index < 34 else 'rh'}",
            "hemi": "lh" if index < 34 else "rh",
            "center": index,
        }
        for index in range(68)
    ]
    vertices = np.column_stack(
        (np.arange(68 + n_deep), np.zeros(68 + n_deep), np.zeros(68 + n_deep))
    ).astype(float)
    return {"vertices": vertices, "n_surf": 68, "n_deep": n_deep}, records


def test_dynamic_deep_split_preserves_legacy(tmp_path: Path) -> None:
    legacy_shared = _split_shared(15)
    corrected_shared = _split_shared(16, corrected_geometry=True)
    legacy = protocol.build_split(legacy_shared)
    corrected_split = protocol.build_split(corrected_shared)

    assert legacy["deep_dev_local"] == [0, 3, 6, 9, 12]
    assert legacy["deep_test_local"] == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
    assert legacy["protocol"] == "full-head-coarse-v1"
    assert corrected_split["deep_dev_local"] == [0, 3, 6, 9, 12, 15]
    assert len(corrected_split["deep_test_local"]) == 10
    assert corrected_split["protocol"] == "full-head-coarse-v2"
    legacy_manifest = protocol.make_manifest(legacy, panel="dev")
    corrected_manifest = protocol.make_manifest(corrected_split, panel="dev")
    assert len(legacy_manifest) == 185
    assert len(corrected_manifest) == 188
    path = tmp_path / "corrected_dev.json"
    protocol.save_manifest(path, corrected_manifest)
    assert len(dev_runner._load_base_manifest(path)[0]) == 188


def test_strict_manifest_defaults_and_corrected_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_shared, records = _strict_shared(15)
    monkeypatch.setattr(strict, "_blind_centers", lambda _shared: records)
    legacy = strict.make_manifest(legacy_shared)
    explicit_legacy = strict.make_manifest(
        legacy_shared,
        id_prefix="strict-blind",
        panel="strict_blind",
        seed_root=strict.SEED_ROOT,
    )
    assert legacy == explicit_legacy
    assert len(legacy) == 9065

    corrected_shared, records = _strict_shared(16)
    monkeypatch.setattr(strict, "_blind_centers", lambda _shared: records)
    manifest = strict.make_manifest(
        corrected_shared,
        id_prefix=corrected.ID_PREFIX,
        panel=corrected.PANEL,
        seed_root=corrected.SEED_ROOT,
    )
    one_cell = manifest[:186]
    assert len(manifest) == 9114
    assert Counter(case["scenario"] for case in one_cell) == {
        "surface_only": 68,
        "deep_only": 16,
        "deep_plus_surface": 68,
        "deep_plus_two_surface": 34,
    }
    assert {case["deep_local"] for case in one_cell if case["deep_local"] is not None} == set(
        range(16)
    )
    assert all(case["case_id"].startswith(corrected.ID_PREFIX) for case in manifest)
    assert all(case["seed"][0] == corrected.SEED_ROOT for case in manifest)
    monkeypatch.setattr(strict, "_old_surface_centers", lambda: set())
    monkeypatch.setattr(
        protocol,
        "_parcels",
        lambda _shared: {row["parcel"]: [row["center"]] for row in records},
    )
    digest = strict._manifest_digest(manifest)
    strict._check(
        corrected_shared,
        manifest,
        digest,
        expected_digest=digest,
        id_prefix=corrected.ID_PREFIX,
        panel=corrected.PANEL,
        seed_root=corrected.SEED_ROOT,
    )


def test_legacy_manifest_and_output_tree_are_protected() -> None:
    payload = (ROOT / "results" / "strict_blind" / "manifest.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == strict.EXPECTED_SHA256
    with pytest.raises(ValueError, match="legacy result tree"):
        corrected._guard_output(strict.OUTPUT)
