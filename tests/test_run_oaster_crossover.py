import inspect

import numpy as np

import run_oaster_crossover as runner


def test_crossover_shares_observations_and_keeps_truth_outside_reconstruction(monkeypatch):
    n_times = 601
    case = {
        "case_number": 0,
        "case_id": "case-0",
        "configuration_number": 0,
        "configuration_id": "source-0",
        "pair_index": 0,
        "eeg_snr_db": 0,
        "meg_snr_db": 0,
        "scenario": "surface_only",
        "surface_centers": [0],
        "deep_index": None,
        "deep_surface_ratio": None,
        "correlation": None,
    }
    archived = (np.full((1, n_times), 10.0), np.full((1, n_times), 11.0))
    erp_eeg = np.full((1, n_times), 30.0)
    erp_meg = np.full((1, n_times), 31.0)
    truth = np.full((2, n_times), 99.0)
    baseline = np.zeros(n_times, dtype=bool)
    baseline[:200] = True
    erp_window = np.zeros(n_times, dtype=bool)
    erp_window[220:230] = True
    events = []
    core_inputs = []
    scored = []

    def joint(eeg, meg, gain_eeg, gain_meg):
        marker = float(eeg[0, 0])
        events.append(("whiten", marker))
        return np.full((2, n_times), marker), np.ones((2, 2))

    def spectral(data, gain, n_surf, kernels):
        marker = float(data[0, 0])
        events.append(("spectral", marker))
        core_inputs.append((marker, data.copy()))
        return np.zeros((2, n_times)), {"temporal_rank": 2, "selected_templates": 1}

    def erp(data, gain, n_surf, kernels, *, baseline, active_windows, require_one):
        marker = float(data[0, 0])
        events.append(("erp", marker))
        core_inputs.append((marker, data.copy()))
        if marker == 10.0:
            assert [int(window.sum()) for window in active_windows] == [200, 200]
            assert not np.any(active_windows[0] & active_windows[1])
            assert np.array_equal(
                np.flatnonzero(active_windows[0] | active_windows[1]), np.arange(200, 600)
            )
        else:
            assert len(active_windows) == 1
            assert np.array_equal(active_windows[0], erp_window)
        return np.zeros((2, n_times)), {
            "windows": [{"temporal_rank": 3} for _window in active_windows],
            "selected_templates": 1,
        }

    def spectral_truth(shared, received):
        assert received is case
        events.append(("truth", None))
        return truth, [np.array([0])], {}

    def simulate(shared, received):
        assert received is case
        return (
            erp_eeg,
            erp_meg,
            truth,
            [np.array([0])],
            baseline,
            (erp_window,),
            np.flatnonzero(erp_window),
            {},
        )

    def success(
        base,
        method,
        estimate,
        received_truth,
        groups,
        active,
        received_baseline,
        runtime,
        elapsed,
        **diagnostics,
    ):
        assert received_truth is truth
        scored.append((method, np.asarray(active).copy()))
        return {**base, "method": method, "status": "ok"}

    monkeypatch.setattr(runner.methods, "joint_whiten", joint)
    monkeypatch.setattr(runner.oaster, "reconstruct_from_whitened", spectral)
    monkeypatch.setattr(runner.oaster, "reconstruct_evoked_oaster_from_whitened", erp)
    monkeypatch.setattr(runner.protocol, "truth_for_case", spectral_truth)
    monkeypatch.setattr(runner.erp_protocol, "simulate_case", simulate)
    monkeypatch.setattr(runner.erp_runner, "_success_row", success)
    runtime = {
        "shared": {
            "gain_eeg": np.ones((1, 2)),
            "gain_meg": np.ones((1, 2)),
            "n_surf": 1,
        },
        "kernels": (),
    }

    rows = runner._score_case(case, archived, runtime, "digest")

    assert "truth" not in inspect.signature(runner._reconstruct_observation).parameters
    assert tuple(rows) == runner.METHODS
    assert events[:4] == [("whiten", 10.0), ("spectral", 10.0), ("erp", 10.0), ("truth", None)]
    assert events[4:] == [("whiten", 30.0), ("spectral", 30.0), ("erp", 30.0)]
    assert np.array_equal(core_inputs[0][1], core_inputs[1][1])
    assert np.array_equal(core_inputs[2][1], core_inputs[3][1])
    assert all(np.array_equal(active, np.arange(200, 601)) for _, active in scored[:2])
    assert all(np.array_equal(active, np.arange(220, 230)) for _, active in scored[2:])


def test_spectral_masks_are_disjoint_and_leave_only_sample_600_for_scoring():
    baseline, windows, active = runner._spectral_masks(601)

    assert np.array_equal(np.flatnonzero(baseline), np.arange(200))
    assert [int(window.sum()) for window in windows] == [200, 200]
    assert not np.any(windows[0] & windows[1])
    assert np.array_equal(np.flatnonzero(windows[0] | windows[1]), np.arange(200, 600))
    assert np.array_equal(active, np.arange(200, 601))
