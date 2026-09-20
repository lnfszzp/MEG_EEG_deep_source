from __future__ import annotations

import numpy as np

import plot_erp_brain_maps as maps


def test_defaults_use_v3_full_development_only() -> None:
    assert maps.DEFAULT_MANIFEST.parts[-2:] == (
        "development_full_v3",
        "manifest.json",
    )
    assert "confirmation" not in str(maps.DEFAULT_MANIFEST)
    assert maps.OASTER_METHOD == "OASTER-ERP-v3"


def test_reconstruct_captures_the_exact_runner_observation_and_estimates(monkeypatch) -> None:
    eeg = np.arange(10.0).reshape(2, 5)
    meg = eeg + 20.0
    truth = np.zeros((3, 5))
    baseline = np.array([True, True, False, False, False])
    active = np.array([3, 4])
    calls = {}

    def fake_simulate(shared, case, *, seed_root):
        calls["seed_root"] = seed_root
        return eeg, meg, truth, [np.array([1])], baseline, (active,), active, {
            "active_window_s": [0.02, 0.12]
        }

    def fake_success(_base, method, _estimate, *_args, **_kwargs):
        return {"method": method, "status": "ok", "error": ""}

    def fake_score(case, runtime, manifest_sha256):
        simulated = maps.erp_protocol.simulate_case(
            runtime["shared"], case, seed_root=runtime["seed_root"]
        )
        calls["runtime"] = runtime
        calls["manifest_sha256"] = manifest_sha256
        return {
            method: maps.erp_run._success_row(
                {},
                method,
                np.full_like(simulated[2], number),
                simulated[2],
                simulated[3],
                simulated[6],
                simulated[4],
                runtime,
                0.0,
            )
            for number, method in enumerate(maps.METHODS, start=1)
        }

    monkeypatch.setattr(maps.erp_protocol, "simulate_case", fake_simulate)
    monkeypatch.setattr(maps.erp_run, "_success_row", fake_success)
    monkeypatch.setattr(maps.erp_run, "_score_case", fake_score)

    result = maps.reconstruct_case(
        {"n_surf": 2}, (), {"case_id": "example"}, "manifest-sha", 20261001, -6.0
    )

    assert result["eeg"] is eeg and result["meg"] is meg
    assert result["active"] is active
    assert tuple(result["estimates"]) == maps.METHODS
    assert calls["seed_root"] == 20261001
    assert calls["runtime"]["algorithm_version"] == "v3"
    assert calls["runtime"]["oaster_solver"] is maps.erp_run._resolve_oaster("v3")
    assert calls["runtime"]["modality_weighting"] == "evidence"
    assert calls["runtime"]["oaster_kwargs"] == {"deep_rescue_delta": -6.0}
    assert calls["manifest_sha256"] == "manifest-sha"
    assert "ERP window 20–120 ms" in maps.brain_maps._case_title(
        {
            "case_number": 1,
            "scenario": "deep_only",
            "eeg_snr_db": -10,
            "meg_snr_db": 5,
            "active_window_s": [0.02, 0.12],
        }
    )


def test_algorithm_display_uses_one_global_amplitude_floor() -> None:
    source = np.zeros((20, 4))
    source[0, 2:] = 1.0
    source[1:17, 2:] = 0.05
    source[17, 2:] = 0.20

    displayed = maps._global_threshold_source(source, np.array([2, 3]), 0.10)

    assert np.array_equal(np.flatnonzero(np.any(displayed, axis=1)), [0, 17])
