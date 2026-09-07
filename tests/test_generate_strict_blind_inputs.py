from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

import generate_strict_blind_inputs as generator
from benchmark import protocol
from verify_strict_archive import verify_archive


def _fixture(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    case = {
        "case_number": 0,
        "case_id": "corrected-mini-0",
        "configuration_id": "corrected-source-0",
        "scenario": "deep_only",
        "surface_centers": [],
        "deep_index": 2,
        "deep_surface_ratio": None,
        "correlation": None,
        "snr_db": -5,
        "eeg_snr_db": -5,
        "meg_snr_db": 10,
        "seed": [20260908, 0],
    }
    second = {
        **case,
        "case_number": 1,
        "case_id": "corrected-mini-1",
        "snr_db": 20,
        "eeg_snr_db": 20,
        "meg_snr_db": -10,
        "seed": [20260908, 1],
    }
    cases = [case, second]
    manifest = tmp_path / "manifest.json"
    payload = (
        json.dumps(cases, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )
    shared = {
        "times": np.linspace(0.0, 0.219, 220),
        "active_start": protocol.ACTIVE_START,
        "n_surf": 2,
        "n_deep": 1,
        "vertices": np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.02]]
        ),
        "adjacency": np.array(
            [[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8
        ),
        "gain_eeg": np.array([[1.0, 0.2, 0.5], [0.1, 0.8, 0.3]]),
        "gain_meg": np.array([[0.4, 0.6, 1.0], [0.7, 0.1, 0.9]]),
        "noise_factor_eeg": np.eye(2),
        "noise_factor_meg": np.eye(2),
    }
    return manifest, shared, cases


def test_generate_verify_and_resume(tmp_path: Path, monkeypatch) -> None:
    manifest, shared, cases = _fixture(tmp_path)
    expected = [protocol.simulate_case(shared, case)[:2] for case in cases]
    original = protocol.truth_for_case
    truth_calls = 0

    def counted_truth(*args, **kwargs):
        nonlocal truth_calls
        truth_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(generator.protocol, "truth_for_case", counted_truth)
    output = tmp_path / "corrected" / "matlab_input"
    first = generator.generate(manifest, tmp_path, output, chunk_size=2, shared=shared)
    chunk = output / "strict_00000_00002.mat"
    assert truth_calls == 1
    values = sio.loadmat(chunk, variable_names=("F_EEG", "F_MEG"))
    for index, (eeg, meg) in enumerate(expected):
        np.testing.assert_array_equal(values["F_EEG"][:, :, index], eeg)
        np.testing.assert_array_equal(values["F_MEG"][:, :, index], meg)
    before = (chunk.stat().st_size, chunk.stat().st_mtime_ns)
    second = generator.generate(manifest, tmp_path, output, chunk_size=2, shared=shared)

    assert (first["written"], first["resumed"]) == (1, 0)
    assert (second["written"], second["resumed"]) == (0, 1)
    assert (chunk.stat().st_size, chunk.stat().st_mtime_ns) == before
    report = verify_archive(
        manifest,
        output,
        expected_sha256=first["manifest_sha256"],
        chunk_size=2,
        sample_cases=(0, 1),
        shared=shared,
    )
    assert (report["cases"], report["chunks"]) == (2, 1)
    np.testing.assert_allclose(
        [
            report["sampled_cases"][0]["eeg_snr_db"],
            report["sampled_cases"][0]["meg_snr_db"],
        ],
        [-5.0, 10.0],
        atol=1e-12,
    )


def test_preserved_legacy_archive_is_rejected(tmp_path: Path, monkeypatch) -> None:
    legacy = (tmp_path / "legacy").resolve()
    monkeypatch.setattr(generator, "LEGACY_ARCHIVE", legacy)
    try:
        generator.generate(
            tmp_path / "missing-manifest.json",
            tmp_path,
            legacy / "matlab_input",
        )
    except ValueError as exc:
        assert "preserved legacy archive" in str(exc)
    else:
        raise AssertionError("legacy archive path was accepted")
