import hashlib
import json

import numpy as np
import pytest

import run_erp_whole_head_matrix as runner


def test_v3_entry_point_is_explicitly_versioned() -> None:
    assert runner._oaster_method("v3") == "OASTER-ERP-v3"
    assert (
        runner._resolve_oaster("v3")
        is runner.oaster.reconstruct_evoked_oaster_v3_from_whitened
    )


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
    surface = bool(case["surface_centers"])
    return {
        "auc": 0.9,
        "auc_tie_corrected": 0.9,
        "surface_auc_tie_corrected": 0.9,
        "deep_auc_tie_corrected": 0.9 if deep else np.nan,
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
        "surface_active_count": int(not deep or surface),
        "deep_active_count": int(deep),
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

    def simulate(shared, received, *, seed_root):
        assert received is case
        assert seed_root == runner.erp_protocol.ERP_SEED_ROOT
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
        assert kwargs["window_channel_weights"] is None
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
        runner.oaster,
        "reconstruct_evoked_oaster_from_whitened",
        oaster,
        raising=False,
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


def test_observation_evidence_weights_are_scale_free():
    baseline = np.array([1, 1, 0, 0], dtype=bool)
    active = ~baseline
    eeg = np.array([[1.0, 1.0, np.sqrt(5.0), np.sqrt(5.0)]])
    mag = np.array([[1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0)]])

    weights = runner._modality_evidence_weights((eeg, mag * 1e-12), baseline, active)

    np.testing.assert_allclose(weights, [1.0, 0.5])


def test_v2_evidence_dispatch_uses_retained_whitened_rows(monkeypatch):
    case = _case(0)
    baseline = np.array([1, 1, 0, 0], dtype=bool)
    active_mask = ~baseline
    truth = np.zeros((2, 4))
    captured = {}

    monkeypatch.setattr(
        runner.erp_protocol,
        "simulate_case",
        lambda _shared, _case, *, seed_root: (
            np.ones((3, 4)),
            np.ones((4, 4)),
            truth,
            [np.array([0])],
            baseline,
            (active_mask,),
            np.flatnonzero(active_mask),
            {"seed_root": seed_root},
        ),
    )

    calls = iter(
        (
            (np.array([[-1.0, 1.0, 3.0, -3.0]]), np.ones((1, 2))),
            (np.array([[-1.0, 1.0, 2.0, -2.0]] * 2), np.ones((2, 2))),
        )
    )
    monkeypatch.setattr(
        runner.comparator_methods, "whiten", lambda _data, _gain: next(calls)
    )

    def v2(data, gain, n_surf, kernels, **kwargs):
        captured["weights"] = kwargs["window_channel_weights"][0]
        assert kwargs["deep_rescue_delta"] == -2.0
        return np.zeros_like(truth), {
            "selected_templates": 0,
            "windows": [{"temporal_rank": 1}],
        }

    monkeypatch.setattr(
        runner.comparator_methods,
        "minimum_norm_family",
        lambda _data, _gain: {
            name: np.zeros_like(truth)
            for name in runner.comparators.MINIMUM_NORM_METHODS
        },
    )
    for name in ("lcmv", "dipole_fit", "rap_music"):
        monkeypatch.setattr(
            runner.comparator_methods,
            name,
            lambda _data, _gain, _active: np.zeros_like(truth),
        )
    monkeypatch.setattr(
        runner.benchmark_metrics,
        "evaluate_estimate",
        lambda *_args, **_kwargs: _metrics(case),
    )
    runtime = {
        "shared": {
            "gain_eeg": np.ones((3, 2)),
            "gain_meg": np.ones((4, 2)),
            "vertices": np.zeros((2, 3)),
            "n_surf": 1,
            "auc_cortex": {},
        },
        "kernels": (),
        "algorithm_version": "v2",
        "oaster_solver": v2,
        "modality_weighting": "evidence",
        "seed_root": 321,
        "methods": ("OASTER-ERP-v2",) + runner.comparators.METHODS,
        "oaster_kwargs": {"deep_rescue_delta": -2.0},
    }

    rows = runner._score_case(case, runtime, "digest")

    assert tuple(rows) == runtime["methods"]
    assert all(row["status"] == "ok" for row in rows.values())
    assert captured["weights"].shape == (3,)
    np.testing.assert_allclose(captured["weights"], [1.0, 0.61237244, 0.61237244])


@pytest.mark.parametrize("version", ("v1", "v4"))
def test_oaster_only_returns_before_comparators(monkeypatch, tmp_path, version):
    case = _case(0)
    baseline = np.array([1, 1, 0, 0], dtype=bool)
    active = ~baseline
    truth = np.zeros((2, 4))
    monkeypatch.setattr(
        runner.erp_protocol,
        "simulate_case",
        lambda *_args, **_kwargs: (
            np.ones((1, 4)), np.ones((1, 4)), truth, [np.array([0])],
            baseline, (active,), np.flatnonzero(active), {},
        ),
    )
    monkeypatch.setattr(
        runner.comparator_methods,
        "joint_whiten",
        lambda *_args: (np.ones((2, 4)), np.ones((2, 2))),
    )
    monkeypatch.setattr(
        runner.comparator_methods,
        "minimum_norm_family",
        lambda *_args: (_ for _ in ()).throw(AssertionError("comparators ran")),
    )
    monkeypatch.setattr(
        runner.benchmark_metrics,
        "evaluate_estimate",
        lambda *_args, **_kwargs: _metrics(case),
    )
    adjacency = np.eye(2)

    def solver(_data, _gain, _n_surf, kernels, **kwargs):
        assert kernels == ()
        if version == "v4":
            assert kwargs["adjacency"] is adjacency
        else:
            assert "adjacency" not in kwargs
        return truth, {
            "selected_templates": [],
            "windows": [{"temporal_rank": 1, "solver": {"converged": False,
                         "history": [{"outer": 0, "iterations": 300}]}}],
        }

    method = runner._oaster_method(version)
    runtime = {
        "shared": {
            "gain_eeg": np.ones((1, 2)),
            "gain_meg": np.ones((1, 2)),
            "vertices": np.zeros((2, 3)),
            "adjacency": adjacency,
            "n_surf": 1,
            "auc_cortex": {},
        },
        "kernels": (),
        "algorithm_version": version,
        "oaster_solver": solver,
        "methods": (method,),
    }
    if version == "v4":
        runtime["diagnostics_dir"] = tmp_path

    rows = runner._score_case(case, runtime, "digest")

    assert tuple(rows) == (method,)
    assert rows[method]["status"] == "ok"
    if version == "v4":
        diagnostics = json.loads((tmp_path / "case_00000.json").read_text())
        assert diagnostics["case_id"] == case["case_id"]
        assert diagnostics["diagnostics"]["windows"][0]["solver"]["converged"] is False
        assert diagnostics["diagnostics"]["windows"][0]["solver"]["history"][0]["iterations"] == 300


@pytest.mark.parametrize("version", ("v1", "v4"))
def test_pair_checkpoint_resumes_without_rescoring(tmp_path, monkeypatch, version):
    cases = [_case(index, scenario) for index, scenario in enumerate(runner.SCENARIOS)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(cases), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(
        f"{digest} {manifest.name}\n", encoding="ascii"
    )
    shared = {
        "gain_eeg": np.ones((1, 2)),
        "gain_meg": np.ones((1, 2)),
        "vertices": np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        "adjacency": np.eye(2),
        "times": np.arange(4, dtype=float),
        "noise_factor_eeg": np.eye(1),
        "noise_factor_meg": np.eye(1),
        "n_surf": 1,
        "n_deep": 1,
        "auc_cortex": {},
    }
    monkeypatch.setattr(runner.protocol, "load_shared", lambda *_args: shared)
    def kernels(*_args, **_kwargs):
        assert version != "v4", "v4 must not build Gaussian templates"
        return ()

    monkeypatch.setattr(
        runner.protected,
        "connected_euclidean_surface_kernels",
        kernels,
    )
    monkeypatch.setattr(runner.archive, "_provenance", lambda *_args: {})
    fingerprint = ["a" * 64]
    monkeypatch.setattr(runner, "_code_fingerprint", lambda _paths: fingerprint[0])
    monkeypatch.setattr(runner, "_resolve_oaster", lambda _version: object())
    scored = []

    def score(case, runtime, manifest_sha256):
        scored.append(case["case_id"])
        base = runner._base_row(case, manifest_sha256)
        if runtime.get("diagnostics_dir") is not None:
            path = runtime["diagnostics_dir"] / f"case_{int(case['case_number']):05d}.json"
            path.write_text(json.dumps({"case_id": case["case_id"],
                "method": runtime["methods"][0], "manifest_sha256": manifest_sha256,
                "diagnostics": {"windows": []}}), encoding="utf-8")
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
            for method in runtime["methods"]
        }

    monkeypatch.setattr(runner, "_score_case", score)
    output = tmp_path / "output"
    runner.run(manifest, tmp_path, None, output, workers=1, algorithm_version=version)
    assert scored == [case["case_id"] for case in cases]
    assert len(runner.archive._read_csv(output / "rows.csv")) == 4 * len(runner.METHODS)
    assert (output / "completion.json").exists()
    for name in (
        "summary_by_snr_scenario.csv",
        "summary_by_snr_pair_scenario_macro.csv",
        "summary_by_snr_pair_case_weighted.csv",
    ):
        assert (output / name).exists()
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["oaster_algorithm_version"] == version
    assert metadata["oaster_modality_weighting"] == "equal"
    assert metadata["erp_seed_root"] == runner.erp_protocol.ERP_SEED_ROOT
    assert len(metadata["checkpoint_shared_fingerprint"]) == 64
    assert len(metadata["checkpoint_fingerprint"]) == 64
    assert {"python", "numpy", "scipy", "mne"} <= set(
        metadata["checkpoint_environment"]
    )

    scored.clear()
    runner.run(manifest, tmp_path, None, output, workers=1, algorithm_version=version)
    assert scored == []

    shared["gain_eeg"][0, 0] = 2.0
    runner.run(manifest, tmp_path, None, output, workers=1, algorithm_version=version)
    assert scored == [case["case_id"] for case in cases]

    scored.clear()
    fingerprint[0] = "b" * 64
    runner.run(manifest, tmp_path, None, output, workers=1, algorithm_version=version)
    assert scored == [case["case_id"] for case in cases]
    if version == "v4":
        for path in output.glob("parts_*/diagnostics/case_00000.json"):
            path.unlink()
        scored.clear()
        runner.run(manifest, tmp_path, None, output, workers=1, algorithm_version=version)
        assert scored == [case["case_id"] for case in cases]


def test_method_cli_selects_v4_and_preserves_comparators(monkeypatch):
    calls = []
    monkeypatch.setattr("sys.argv", ["run_erp_whole_head_matrix.py", "--method", "OASTER-ERP-v4"])
    monkeypatch.setattr(runner, "run", lambda *_args, **kwargs: calls.append(kwargs))
    runner.main()
    assert calls[0]["algorithm_version"] == "v4"
    assert calls[0]["oaster_only"] is False
