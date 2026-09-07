from __future__ import annotations

import inspect

import numpy as np

import experiment_oaster_deep_rescue as experiment
from candidates import oaster_rebuilt as oaster


def test_deep_rescue_is_observation_only_and_uses_deep_universe() -> None:
    assert experiment.deep_rescue_trial is oaster.deep_rescue_trial
    assert list(inspect.signature(experiment.deep_rescue_trial).parameters) == [
        "data",
        "leadfield",
        "primary",
        "n_surf",
        "basis",
    ]
    basis = np.zeros((1, 400))
    basis[0, 200:] = 1.0 / np.sqrt(200.0)
    leadfield = np.eye(5)
    primary = np.zeros((5, 400))
    primary[0] = 2.0 * basis[0]
    data = leadfield @ primary + 20.0 * leadfield[:, [3]] @ basis

    rescue, diagnostics = experiment.deep_rescue_trial(
        data, leadfield, primary, 2, basis
    )

    assert diagnostics["universe"] == 3
    assert diagnostics["deep_local"] == 1
    assert diagnostics["ebic_delta_without_tau"] < 0.0
    assert np.linalg.norm(rescue[3]) > 0.0
    assert np.array_equal(rescue[[0, 1, 2, 4]], np.zeros((4, 400)))

    empty, rejected = experiment.deep_rescue_trial(
        leadfield @ primary, leadfield, primary, 2, basis
    )
    assert not rejected["accepted_without_tau"]
    assert np.array_equal(empty, np.zeros_like(primary))
