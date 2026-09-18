import hashlib
import json

import numpy as np

import run_erp_whole_head_matrix as runner


def _case(number, scenario="surface_only"):
    deep = scenario != "surface_only"
    surface = scenario != "deep_only"
    return {
        "case_number": number,
        "case_id": f"case-{number}",
        "configuration_number": number,
        "configuration_id": f"source-{number}",
        "pair_index": 0,
        "eeg_snr_db": -10,
        "meg_snr_db": -10,
        "scenario": scenario,
        "surface_centers": [0] if surface else [],
        "deep_index": 1 if deep else None,
        "deep_surface_ratio": 1.0 if deep and surface else None,
        "correlation": 0.0 if deep and surface else None,
    }


def _metrics(case):
    deep = case["deep_index"] is not None
    return {
        "auc": 0.9,
        "auc_tie_corrected": 0.9,
        "rmse": 0.1,
        "surface_sd_mm": 1.0,
        "surface_dle_mm": 1.0,
        "deep_sd_mm": 1.0,
        "deep_dle_mm": 1.0,
        "has_deep_true": int(deep),
        "deep_score": float(deep),
        "deep_peak_distance_mm": 1.0 if deep else np.nan,
        "deep_detected": int(deep),
        "deep_false_positive": 0,
        "active_count": 1,
    }


def test_snr_pair_and_smoke_selection():
    cases = []
    for pair_index, pair in enumerate(((-10, -10), (0, 5))):
        for scenario in runner.SCENARIOS:
            for repeat in range(2):
                case = _case(len(cases), scenario)
                case.update(
                    pair_index=pair_index,
                    eeg_snr_db=pair[0],
                    meg_snr_db=pair[1],
                    case_id=f"{pair_index}-{scenario}-{repeat}",
                )
                cases.append(case)

    selected = runner._select_pairs(cases, [(0, 5)], cases_per_scenario=1)

    assert [pair for pair, _cases in selected] == [(0, 5)]
    assert [case["scenario"] for case in selected[0][1]] == list(runner.SCENARIOS)


def test_one_simulation_and_one_window_are_shared(monkeypatch):
    case = _case(0)
    baseline_mask = np.array([1, 1, 0, 0, 0, 0], dtype=bool)
    window = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    active = np.flatnonzero(window)
    truth = np.zeros((2, 6))
    observations = {"simulate": 0, "active": [], "score_active": []}

    def simulate(shared, received):
        assert received is case
        observations["simulate"] += 1
        return (
            np.ones((2, 6)),
            np.ones((1, 6)),
            truth,
            [np.array([0])],
            baseline_mask,
            (window,),
            active,
            {},
        )

    def joint(eeg, meg, gain_eeg, gain_meg):
        assert eeg.shape == (2, 6) and meg.shape == (1, 6)
        return np.ones((3, 6)), np.ones((3, 2))

    def oaster(data, gain, n_surf, kernels, **kwargs):
        assert np.array_equal(kwargs["baseline"], baseline_mask)
        assert len(kwargs["active_windows"]) == 1
        assert np.array_equal(kwargs["active_windows"][0], window)
        return np.zeros_like(truth), {
            "selected_templates": 0,
            "windows": [{"temporal_rank": 1}],
        }

    def family(data, gain):
        return {name: np.zeros_like(truth) for name in runner.comparators.MINIMUM_NORM_METHODS}

    def active_method(data, gain, received):
        observations["active"].append(np.asarray(received).copy())
        return np.zeros_like(truth)

    def score(
        estimate,
        truth_,
        vertices,
        groups,
        n_surf,
        received,
        cortex,
        *,
        baseline,
    ):
        observations["score_active"].append(np.asarray(received).copy())
        assert np.array_equal(baseline, baseline_mask)
        return _metrics(case)

    monkeypatch.setattr(runner.erp_protocol, "simulate_case", simulate)
    monkeypatch.setattr(runner.comparator_methods, "joint_whiten", joint)
    monkeypatch.setattr(
        runner.oaster, "reconstruct_evoked_oaster_from_whitened", oaster
    )
    monkeypatch.setattr(runner.comparator_methods, "minimum_norm_family", family)
    monkeypatch.setattr(runner.comparator_methods, "lcmv", active_method)
    monkeypatch.setattr(runner.comparator_methods, "dipole_fit", active_method)
    monkeypatch.setattr(runner.comparator_methods, "rap_music", active_method)
    monkeypatch.setattr(runner.benchmark_metrics, "evaluate_estimate", score)
    runtime = {
        "shared": {
            "gain_eeg": np.ones((2, 2)),
            "gain_meg": np.ones((1, 2)),
            "vertices": np.zeros((2, 3)),
            "n_surf": 1,
            "auc_cortex": {},
        },
        "kernels": (),
    }

    rows = runner._score_case(case, runtime, "digest")

    assert observations["simulate"] == 1
    assert tuple(rows) == runner.METHODS
    assert all(row["status"] == "ok" for row in rows.values())
    assert len(observations["active"]) == 3
    assert len(observations["score_active"]) == len(runner.METHODS)
    assert all(np.array_equal(value, active) for value in observations["active"])
    assert all(np.array_equal(value, active) for value in observations["score_active"])


def test_pair_checkpoint_resumes_without_rescoring(tmp_path, monkeypatch):
    cases = [_case(index, scenario) for index, scenario in enumerate(runner.SCENARIOS)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(cases), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(
        f"{digest} {manifest.name}\n", encoding="ascii"
    )
    shared = {
        "vertices": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "adjacency": np.eye(2),
        "n_surf": 1,
        "n_deep": 1,
        "auc_cortex": {},
    }
    monkeypatch.setattr(runner.protocol, "load_shared", lambda *_args: shared)
    monkeypatch.setattr(
        runner.protected,
        "connected_euclidean_surface_kernels",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(runner.archive, "_provenance", lambda *_args: {})
    scored = []

    def score(case, runtime, manifest_sha256):
        scored.append(case["case_id"])
        base = runner._base_row(case, manifest_sha256)
        return {
            method: {
                **base,
                "method": method,
                "status": "ok",
                "error": "",
                "elapsed_seconds": 0.01,
                "temporal_rank": 1 if method == "OASTER-ERP" else "",
                "selected_templates": 1 if method == "OASTER-ERP" else "",
                **_metrics(case),
            }
            for method in runner.METHODS
        }

    monkeypatch.setattr(runner, "_score_case", score)
    output = tmp_path / "output"
    runner.run(manifest, tmp_path, None, output, workers=1)
    assert scored == [case["case_id"] for case in cases]
    assert len(runner.archive._read_csv(output / "rows.csv")) == 4 * len(runner.METHODS)
    assert (output / "completion.json").exists()
    for name in (
        "summary_by_snr_scenario.csv",
        "summary_by_snr_pair_scenario_macro.csv",
        "summary_by_snr_pair_case_weighted.csv",
    ):
        assert (output / name).exists()

    scored.clear()
    runner.run(manifest, tmp_path, None, output, workers=1)
    assert scored == []
