from __future__ import annotations

import csv
from collections import Counter

import numpy as np
from scipy import sparse

import run_oaster_dev_matrix as runner


def test_matrix_is_even_stable_and_independent_by_modality() -> None:
    base, _digest = runner._load_base_manifest(runner.BASE_MANIFEST)

    matrix = runner.make_matrix(base, cases_per_scenario=5, diagonal=False)

    assert len(matrix) == 5 * 4 * 49
    assert Counter(case["scenario"] for case in matrix) == {
        scenario: 5 * 49 for scenario in runner.EXPECTED_CONFIG_COUNTS
    }
    assert {case["eeg_snr_db"] for case in matrix} == set(runner.LEVELS)
    assert {case["meg_snr_db"] for case in matrix} == set(runner.LEVELS)
    assert all(case["case_id"].startswith("dev-matrix-") for case in matrix)
    assert all(case["seed"][0] == runner.SEED_ROOT for case in matrix)
    assert len({tuple(case["seed"]) for case in matrix}) == len(matrix)

    for scenario in runner.EXPECTED_CONFIG_COUNTS:
        available = [case for case in base if case["scenario"] == scenario]
        chosen = [
            case["configuration_number"]
            for case in matrix
            if case["pair_index"] == 0 and case["scenario"] == scenario
        ]
        assert chosen[0] == available[0]["case_number"]
        assert chosen[-1] == available[-1]["case_number"]

    diagonal = runner.make_matrix(base, cases_per_scenario=2, diagonal=True)
    assert len(diagonal) == 2 * 4 * 7
    assert all(case["eeg_snr_db"] == case["meg_snr_db"] for case in diagonal)


def test_dev_run_uses_simulator_and_writes_shared_summaries(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    times = np.arange(201, dtype=float) / 200.0
    shared = {
        "times": times,
        "vertices": np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
        ),
        "adjacency": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8),
        "n_surf": 2,
        "n_deep": 1,
        "gain_eeg": np.ones((2, 3)),
        "gain_meg": np.ones((2, 3)),
        "auc_cortex": {},
    }
    monkeypatch.setattr(runner.protocol, "load_shared", lambda **_kwargs: shared)
    identity = sparse.eye(2, format="csc")
    monkeypatch.setattr(
        runner.protected,
        "connected_euclidean_surface_kernels",
        lambda *_args, **_kwargs: tuple(
            (scale, identity.copy()) for scale in runner.oaster.SURFACE_SCALES_MM
        ),
    )

    def simulate_case(_shared, case):
        calls.append(case["case_id"])
        truth = np.zeros((3, times.size))
        has_deep = case["scenario"] != "surface_only"
        groups = [np.array([2 if has_deep else 0])]
        return (
            np.zeros((2, times.size)),
            np.zeros((2, times.size)),
            truth,
            groups,
            {"actual_snr_db": {"eeg": case["eeg_snr_db"], "meg": case["meg_snr_db"]}},
        )

    monkeypatch.setattr(runner.protocol, "simulate_case", simulate_case)
    monkeypatch.setattr(
        runner.oaster,
        "reconstruct",
        lambda eeg, _meg, gain_eeg, *_args: (
            np.zeros((gain_eeg.shape[1], eeg.shape[1])),
            {"temporal_rank": 0, "selected_templates": 0},
        ),
    )

    def evaluate(_estimate, _truth, _vertices, groups, n_surf, _active, _cortex):
        has_deep = int(np.asarray(groups[0]).ravel()[0] >= n_surf)
        return {
            "auc": 0.75,
            "auc_tie_corrected": 0.85,
            "rmse": 0.25,
            "surface_sd_mm": np.nan if has_deep else 2.0,
            "surface_dle_mm": np.nan if has_deep else 3.0,
            "deep_sd_mm": 4.0 if has_deep else np.nan,
            "deep_dle_mm": 5.0 if has_deep else np.nan,
            "has_deep_true": has_deep,
            "deep_score": float(has_deep),
            "deep_peak_distance_mm": 1.0 if has_deep else np.nan,
            "deep_detected": has_deep,
            "deep_false_positive": 0,
            "active_count": 1,
        }

    monkeypatch.setattr(runner.benchmark_metrics, "evaluate_estimate", evaluate)
    output = tmp_path / "dev"

    rows = runner.run(
        runner.BASE_MANIFEST,
        tmp_path / "unused-generated",
        output,
        cases_per_scenario=1,
        diagonal=True,
        workers=2,
    )

    assert len(rows) == 4 * 7 == len(calls)
    assert all(row["status"] == "ok" for row in rows)
    assert (output / "manifest.json.sha256").exists()
    with (output / "summary_by_snr_scenario.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        assert len(list(csv.DictReader(stream))) == 4 * 7
    with (output / "summary_by_snr_pair_case_weighted.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 7
    assert {float(row["deep_balanced_accuracy"]) for row in summaries} == {1.0}
