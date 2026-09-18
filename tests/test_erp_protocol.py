from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from benchmark import erp_protocol, metrics


@pytest.fixture
def shared() -> dict:
    rng = np.random.default_rng(4)
    n_surf, n_deep = 30, 2
    adjacency = np.zeros((n_surf + n_deep, n_surf + n_deep), dtype=np.uint8)
    for index in range(n_surf - 1):
        adjacency[index, index + 1] = adjacency[index + 1, index] = 1
    eeg_channels, meg_channels = 8, 10
    eeg_color = np.linalg.cholesky(0.3 ** np.abs(np.subtract.outer(range(eeg_channels), range(eeg_channels))))
    meg_color = np.linalg.cholesky(0.4 ** np.abs(np.subtract.outer(range(meg_channels), range(meg_channels))))
    return {
        "times": np.arange(601) / 1000.0 - 0.2,
        "vertices": np.column_stack((np.arange(n_surf + n_deep) * 0.02, np.zeros(n_surf + n_deep), np.zeros(n_surf + n_deep))),
        "adjacency": adjacency,
        "n_surf": n_surf,
        "n_deep": n_deep,
        "gain_eeg": rng.normal(size=(eeg_channels, n_surf + n_deep)),
        "gain_meg": rng.normal(size=(meg_channels, n_surf + n_deep)),
        "noise_factor_eeg": eeg_color,
        "noise_factor_meg": meg_color,
    }


def _case(scenario: str, correlation: float = 0.5, ratio: float = 0.5) -> dict:
    surfaces = {
        "surface_only": [4],
        "deep_only": [],
        "deep_plus_surface": [4],
        "deep_plus_two_surface": [4, 24],
    }[scenario]
    deep = None if scenario == "surface_only" else 30
    return {
        "case_id": f"erp-{scenario}-{correlation}-{ratio}",
        "scenario": scenario,
        "surface_centers": surfaces,
        "deep_index": deep,
        "deep_surface_ratio": ratio if surfaces and deep is not None else None,
        "correlation": correlation if len(surfaces) + (deep is not None) > 1 else None,
        "snr_db": 0,
        "eeg_snr_db": -5,
        "meg_snr_db": 10,
    }


def test_masks_exact_snr_and_reproducibility(shared: dict) -> None:
    case = _case("deep_plus_two_surface", correlation=0.9, ratio=0.5)
    first = erp_protocol.simulate_case(shared, case)
    second = erp_protocol.simulate_case(shared, case)
    eeg, meg, truth, _groups, baseline, windows, active_indices, metadata = first
    active = windows[0]

    assert truth.shape[1] == 601 and np.all(truth[:, ~active] == 0.0)
    assert np.all(truth[:, np.flatnonzero(active)[[0, -1]]] == pytest.approx(0.0))
    assert active_indices.tolist() == np.flatnonzero(active).tolist()
    assert active.sum() <= baseline.sum() and not np.any(active & baseline)
    np.testing.assert_allclose(eeg[:, baseline].mean(axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(meg[:, baseline].mean(axis=1), 0.0, atol=1e-15)
    assert metadata["event_sample"] == 200
    assert metadata["n_trials"] == 40 and metadata["snr_level"] == "evoked-level"
    assert metadata["baseline_corrected_before_snr_scaling"] is True
    for actual, repeated in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(actual, repeated)

    clean_eeg = shared["gain_eeg"] @ truth
    clean_meg = shared["gain_meg"] @ truth
    eeg_snr = 10 * np.log10(np.sum(clean_eeg[:, active] ** 2) / np.sum((eeg - clean_eeg)[:, active] ** 2))
    meg_snr = 10 * np.log10(np.sum(clean_meg[:, active] ** 2) / np.sum((meg - clean_meg)[:, active] ** 2))
    assert eeg_snr == pytest.approx(-5.0, abs=1e-12)
    assert meg_snr == pytest.approx(10.0, abs=1e-12)


def test_seed_root_can_create_an_independent_noise_confirmation(shared: dict) -> None:
    case = _case("surface_only")
    original = erp_protocol.simulate_case(shared, case)
    explicit_default = erp_protocol.simulate_case(
        shared, case, seed_root=erp_protocol.ERP_SEED_ROOT
    )
    development = erp_protocol.simulate_case(shared, case, seed_root=123)

    np.testing.assert_array_equal(original[0], explicit_default[0])
    np.testing.assert_array_equal(original[1], explicit_default[1])
    np.testing.assert_array_equal(original[2], development[2])
    assert not np.array_equal(original[0], development[0])
    assert not np.array_equal(original[1], development[1])
    assert original[7]["seed_root"] == erp_protocol.ERP_SEED_ROOT
    assert development[7]["seed_root"] == 123
    with pytest.raises(ValueError, match="seed_root"):
        erp_protocol.simulate_case(shared, case, seed_root=-1)


def test_source_amplitude_uses_the_registered_baseline_only() -> None:
    source = np.zeros((1, 10))
    source[:, :2] = 2.0
    source[:, 4:6] = 3.0

    requested = metrics.source_amplitude(source, np.array([4, 5]), np.array([0, 1]))
    complement = metrics.source_amplitude(source, np.array([4, 5]))

    assert requested[0] == pytest.approx(np.sqrt(5.0))
    assert complement[0] > requested[0]


@pytest.mark.parametrize(
    ("scenario", "locations", "wave_count"),
    (("deep_plus_surface", 68, 2), ("deep_plus_two_surface", 34, 3)),
)
def test_waveform_assignment_is_balanced_inside_ratio_correlation_cells(
    shared: dict, scenario: str, locations: int, wave_count: int
) -> None:
    counts = defaultdict(Counter)
    for location in range(locations):
        ratio = (0.5, 1.0)[location % 2]
        correlation = (0.0, 0.5, 0.9)[location % 3]
        case = _case(scenario, correlation, ratio)
        case.update(case_id=f"{scenario}-{location}", location=location)
        metadata = erp_protocol.simulate_case(shared, case)[7]
        deep_wave = metadata["waveform_assignment"][-1]
        counts[(ratio, correlation)][deep_wave] += 1

    assert len(counts) == 6
    for cell in counts.values():
        frequencies = [cell[index] for index in range(wave_count)]
        assert max(frequencies) - min(frequencies) <= 1


@pytest.mark.parametrize(
    ("scenario", "expected_groups"),
    [("surface_only", 1), ("deep_only", 1), ("deep_plus_surface", 2), ("deep_plus_two_surface", 3)],
)
def test_scenario_group_count(shared: dict, scenario: str, expected_groups: int) -> None:
    assert len(erp_protocol.simulate_case(shared, _case(scenario))[3]) == expected_groups


@pytest.mark.parametrize("correlation", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("ratio", [0.5, 1.0])
def test_correlation_and_deep_surface_ratio(
    shared: dict, correlation: float, ratio: float
) -> None:
    result = erp_protocol.simulate_case(
        shared, _case("deep_plus_two_surface", correlation, ratio)
    )
    truth, groups, active, metadata = result[2], result[3], result[5][0], result[7]
    waves = [
        truth[group[np.argmax(np.linalg.norm(truth[group], axis=1))], active]
        for group in groups
    ]
    actual = np.corrcoef(waves)
    np.testing.assert_allclose(actual[np.triu_indices(3, 1)], correlation, atol=1e-12)
    assert metadata["deep_surface_ratio_actual"] == pytest.approx(ratio, abs=1e-12)
