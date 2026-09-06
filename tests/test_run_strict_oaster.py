from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from scipy import sparse

import run_strict_oaster as runner


def _case(number: int, scenario: str) -> dict:
    has_deep = scenario != "surface_only"
    return {
        "case_number": number,
        "case_id": f"mini-{number}",
        "configuration_number": number,
        "configuration_id": f"config-{number}",
        "pair_index": 0,
        "eeg_snr_db": -10,
        "meg_snr_db": 5,
        "scenario": scenario,
        "surface_centers": [] if scenario == "deep_only" else [0],
        "deep_index": 2 if has_deep else None,
        "deep_surface_ratio": None,
        "correlation": None,
    }


def _fixture(tmp_path: Path, chunks: list[tuple[int, int]], *, bad_ids=False, bad_gain=False):
    cases = [_case(0, "surface_only"), _case(1, "deep_only")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(cases, separators=(",", ":")), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    data_root = tmp_path / "data"
    geometry_dir = data_root / "generated" / "deep_plus_two_surface"
    geometry_dir.mkdir(parents=True)
    times = np.arange(201, dtype=float) / 200.0
    vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    sio.savemat(
        geometry_dir / "sub_EEG.mat",
        {"src_vertices": vertices, "times": times, "n_surf": 2, "n_deep": 1},
    )

    input_root = tmp_path / "matlab_input"
    input_root.mkdir()
    gain_eeg = np.array([[1.0, 0.2, 0.1], [0.1, 1.0, 0.3]])
    gain_meg = np.array([[0.8, 0.4, 0.2], [0.2, 0.5, 1.0]])
    adjacency = sparse.csr_matrix(
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    )
    for chunk_number, (start, stop) in enumerate(chunks):
        count = stop - start
        ids = [cases[index]["case_id"] for index in range(start, stop)]
        if bad_ids and chunk_number == 0:
            ids = list(reversed(ids))
        chunk_meg = gain_meg.copy()
        if bad_gain and chunk_number == 1:
            chunk_meg[0, 0] += 0.01
        sio.savemat(
            input_root / f"strict_{start:05d}_{stop:05d}.mat",
            {
                "F_EEG": np.zeros((2, times.size, count)),
                "F_MEG": np.zeros((2, times.size, count)),
                "Gain_EEG": gain_eeg,
                "Gain_MEG": chunk_meg,
                "VertConn": adjacency,
                "case_ids": np.asarray(ids, dtype=object)[:, None],
                "manifest_sha256": digest,
                "format_version": runner.FORMAT_VERSION,
            },
        )
    return cases, manifest, digest, data_root, input_root


def _patch_small_run(monkeypatch: pytest.MonkeyPatch, digest: str, calls: list[str]) -> None:
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(runner, "EXPECTED_CASES", 2)
    identity = sparse.eye(2, format="csc")
    monkeypatch.setattr(
        runner.protected,
        "connected_euclidean_surface_kernels",
        lambda *_args, **_kwargs: tuple(
            (scale, identity.copy()) for scale in runner.oaster.SURFACE_SCALES_MM
        ),
    )

    def truth_for_case(shared, case):
        truth = np.zeros((3, len(shared["times"])))
        if case["scenario"] == "surface_only":
            groups = [np.array([0])]
            truth[0, 200:] = 1.0
        else:
            groups = [np.array([2])]
            truth[2, 200:] = 1.0
        return truth, groups, {}

    def reconstruct(eeg, meg, gain_eeg, gain_meg, n_surf, kernels):
        calls.append("reconstruct")
        return np.zeros((gain_eeg.shape[1], eeg.shape[1])), {
            "temporal_rank": 0,
            "selected_templates": 0,
        }

    def evaluate(_estimate, _truth, _vertices, groups, n_surf, _active, _cortex):
        has_deep = int(any(np.any(np.asarray(group) >= n_surf) for group in groups))
        return {
            "auc": 0.8,
            "auc_tie_corrected": 0.9,
            "rmse": 0.2,
            "surface_sd_mm": np.nan if has_deep else 2.0,
            "surface_dle_mm": np.nan if has_deep else 3.0,
            "deep_sd_mm": 4.0 if has_deep else np.nan,
            "deep_dle_mm": 5.0 if has_deep else np.nan,
            "has_deep_true": has_deep,
            "deep_score": 0.2 if has_deep else 0.0,
            "deep_peak_distance_mm": 1.0 if has_deep else np.nan,
            "deep_detected": has_deep,
            "deep_false_positive": 0,
            "active_count": 1,
        }

    monkeypatch.setattr(runner.protocol, "truth_for_case", truth_for_case)
    monkeypatch.setattr(runner.oaster, "reconstruct", reconstruct)
    monkeypatch.setattr(runner.benchmark_metrics, "evaluate_estimate", evaluate)


def test_direct_chunk_run_checkpoints_and_summarizes(tmp_path, monkeypatch) -> None:
    _cases, manifest, digest, data_root, input_root = _fixture(tmp_path, [(0, 2)])
    calls: list[str] = []
    _patch_small_run(monkeypatch, digest, calls)
    output = tmp_path / "output"

    runner.run(manifest, input_root, data_root, output, workers=2)

    assert calls == ["reconstruct", "reconstruct"]
    with (output / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["case_id"] for row in rows] == ["mini-0", "mini-1"]
    assert {row["auc"] for row in rows} == {"0.8"}
    assert {row["auc_tie_corrected"] for row in rows} == {"0.9"}
    assert (output / "parts" / "strict_00000_00002.csv").exists()
    with (output / "summary_by_snr_pair_case_weighted.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        pair = next(csv.DictReader(stream))
    assert float(pair["deep_balanced_accuracy"]) == 1.0
    assert float(pair["surface_sd_mm"]) == 2.0
    assert float(pair["deep_dle_mm"]) == 5.0
    assert (output / "summary_by_snr_pair_scenario_macro.csv").exists()
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert completion == {
        "status": "complete",
        "row_count": 2,
        "expected_row_count": 2,
        "error_count": 0,
        "manifest_sha256": digest,
        "chunk_count": 1,
        "method_count": 1,
        "methods": [runner.METHOD],
    }
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["provenance"]["code_sha256"]["oaster_algorithm"]) == 64
    assert metadata["provenance"]["packages"]["numpy"]

    # A valid checkpoint is verified against its source chunk but not recomputed.
    runner.run(manifest, input_root, data_root / "generated", output, workers=1)
    assert calls == ["reconstruct", "reconstruct"]


def test_case_ids_must_match_manifest_range(tmp_path, monkeypatch) -> None:
    _cases, manifest, digest, data_root, input_root = _fixture(
        tmp_path, [(0, 2)], bad_ids=True
    )
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(runner, "EXPECTED_CASES", 2)

    with pytest.raises(RuntimeError, match="case_ids disagree"):
        runner.run(manifest, input_root, data_root, tmp_path / "output")


def test_gain_is_checked_across_chunks(tmp_path, monkeypatch) -> None:
    _cases, manifest, digest, data_root, input_root = _fixture(
        tmp_path, [(0, 1), (1, 2)], bad_gain=True
    )
    monkeypatch.setattr(runner, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(runner, "EXPECTED_CASES", 2)

    with pytest.raises(RuntimeError, match="Gain_MEG differs across strict chunks"):
        runner.run(
            manifest,
            input_root,
            data_root,
            tmp_path / "output",
            start_chunk=1,
            limit_chunks=1,
        )
