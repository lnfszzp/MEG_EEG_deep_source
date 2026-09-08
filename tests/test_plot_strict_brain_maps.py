from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io as sio

import plot_strict_brain_maps as brain_maps


def test_sisses_estimate_is_read_only_and_case_bound(tmp_path: Path) -> None:
    source = np.arange(30, dtype=float).reshape(3, 10)
    loaded = {
        "truth": np.zeros_like(source),
        "case": {"case_id": "strict-test-22", "case_number": 22},
        "chunk": SimpleNamespace(start=20, path=Path("strict_00020_00040.mat")),
    }
    path = (
        tmp_path
        / "matlab_output"
        / "strict_00020_00040"
        / "case_0003.mat"
    )
    path.parent.mkdir(parents=True)
    sio.savemat(
        path,
        {
            "source_estimates": source,
            "method_names": "SISSES",
            "case_id": "strict-test-22",
            "success": True,
            "source_shape": [3, 10, 1],
            "format_version": "test-v1",
            "metadata": {"variant": "test"},
        },
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    actual, actual_path, metadata = brain_maps.load_sisses_estimate(loaded, tmp_path)

    assert np.array_equal(actual, source)
    assert actual_path == path
    assert metadata["format_version"] == "test-v1"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_method_brain_map_renders_truth_and_estimate(tmp_path: Path) -> None:
    times = np.arange(220, dtype=float) / 200.0
    truth = np.zeros((3, times.size))
    truth[0, 200:] = 1.0
    truth[2, 200:] = 0.7
    estimate = truth.copy()
    loaded = {
        "case": {
            "case_id": "mini",
            "case_number": 0,
            "scenario": "deep_plus_surface",
            "surface_centers": [0],
            "deep_index": 2,
            "eeg_snr_db": 0,
            "meg_snr_db": 0,
        },
        "truth": truth,
        "groups": [np.array([0]), np.array([2])],
        "geometry": {
            "vertices": np.array(
                [[0.020, 0.020, 0.020], [0.026, 0.020, 0.020], [0.020, 0.026, 0.026]]
            ),
            "n_surf": 2,
            "n_deep": 1,
            "times": times,
        },
    }
    volume = np.indices((48, 48, 48), dtype=float).sum(axis=0)
    anatomy = {
        "volume": volume,
        "vox2ras_tkr": np.eye(4),
        "head_to_mri": np.eye(4),
    }
    metrics = {
        "auc_tie_corrected": 1.0,
        "auc": 1.0,
        "rmse": 0.0,
        "surface_dle_mm": 0.0,
        "deep_dle_mm": 0.0,
    }

    output = brain_maps.render_method(
        "OASTER V20",
        estimate,
        metrics,
        loaded,
        anatomy,
        tmp_path / "brain.png",
        0.10,
        80,
    )

    assert len(brain_maps.METHOD_ORDER) == 9
    assert [handle.get_label() for handle in brain_maps._legend_handles()] == [
        "Estimated source energy",
    ]
    assert output.is_file() and output.stat().st_size > 10_000


def test_output_cannot_be_written_into_sisses_archive(tmp_path: Path) -> None:
    archive = tmp_path / "sisses"
    archive.mkdir()
    try:
        brain_maps.plot_brain_maps(
            sisses_root=archive,
            output_root=archive / "new-results",
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("archive output guard did not run")


def test_corrected_deep_focus_is_labeled_thalamic() -> None:
    loaded = {
        "case": {"surface_centers": [], "deep_index": 2},
        "geometry": {"deep_aseg_labels": np.array([10, 49])},
    }
    assert brain_maps._focuses(loaded) == [("Thalamic source", 2)]


def test_completed_oaster_label_follows_selected_result_version(
    tmp_path: Path, monkeypatch
) -> None:
    for directory, label in (
        ("oaster_v20_final", "OASTER V20"),
        ("oaster_v19_final", "OASTER V19"),
    ):
        availability = [
            {
                "method": "OASTER",
                "source": str(tmp_path / directory / brain_maps.metric_plots.SUMMARY_NAME),
            },
            {"method": "SISSES", "source": "", "status": "N/A"},
        ]
        monkeypatch.setattr(
            brain_maps.metric_plots,
            "load_results_with_availability",
            lambda _root: ({"OASTER": [{}], "MNE": [{}]}, availability),
        )

        methods, actual_availability = brain_maps._completed_method_order(tmp_path)

        assert methods == (label, "MNE")
        assert actual_availability is availability


def test_slice_draws_only_continuous_heat_without_truth_markers(monkeypatch) -> None:
    images = []

    class Axes:
        def imshow(self, *_args, **_kwargs):
            images.append((_args, _kwargs))

        def scatter(self, *_args, **_kwargs):
            raise AssertionError("truth markers must not be drawn")

        def set_axis_off(self):
            pass

    monkeypatch.setattr(
        brain_maps,
        "plane_image",
        lambda _volume, _plane, _center: (
            np.zeros((8, 8)),
            2,
            4,
            lambda voxel: np.asarray(voxel)[:2],
        ),
    )
    center = np.array([3.0, 4.0, 4.0])
    brain_maps._draw_slice(
        Axes(),
        "axial",
        center,
        np.array([[3.0, 4.0, 4.0]]),
        np.array([1.0]),
        np.zeros((8, 8, 8)),
    )
    assert len(images) == 2


def test_case_titles_are_human_readable_and_complete() -> None:
    title = brain_maps._case_title(
        {
            "case_number": 3264,
            "scenario": "deep_plus_surface",
            "eeg_snr_db": 0,
            "meg_snr_db": 5,
        }
    )
    assert title == (
        "Case 03264 | Thalamic + one cortical source | "
        "EEG SNR +0 dB | MEG SNR +5 dB"
    )


def test_surface_forward_mapping(tmp_path: Path) -> None:
    subjects_dir = tmp_path / "subjects"
    for hemi in ("lh", "rh"):
        path = subjects_dir / "sample" / "surf" / f"{hemi}.pial"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    left_rr = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    right_rr = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    src = [
        {"vertno": np.array([1, 2]), "rr": left_rr, "subject_his_id": "sample"},
        {"vertno": np.array([0, 2]), "rr": right_rr, "subject_his_id": "sample"},
    ]
    geometry = {
        "n_surf": 4,
        "vertices": np.vstack([left_rr[[1, 2]], right_rr[[0, 2]], [[0.0, 0.0, 1.0]]]),
        "deep_aseg_labels": np.array([10]),
    }

    surface = brain_maps._validate_surface_source_space(geometry, src, subjects_dir)
    assert [part.tolist() for part in surface["vertices"]] == [[1, 2], [0, 2]]
    bad_geometry = {**geometry, "vertices": geometry["vertices"].copy()}
    bad_geometry["vertices"][0, 0] += 1e-6
    with pytest.raises(ValueError, match="source order"):
        brain_maps._validate_surface_source_space(bad_geometry, src, subjects_dir)


def test_surface_render_uses_mne_brain_without_projecting_deep_truth(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {"plots": []}

    class Brain:
        def __init__(self):
            self.closed = False

        def add_label(self, *_args, **_kwargs):
            raise AssertionError("truth labels must not be drawn")

        def add_foci(self, *_args, **_kwargs):
            raise AssertionError("truth foci must not be drawn")

        def screenshot(self, **kwargs):
            calls["screenshot"] = kwargs
            return np.full((20, 30, 3), 255, dtype=np.uint8)

        def close(self):
            self.closed = True

    def fake_plot(stc, **kwargs):
        calls["stc"] = stc
        calls["plot"] = kwargs
        brain = Brain()
        calls["plots"].append(brain)
        return brain

    monkeypatch.setattr(brain_maps.mne.SourceEstimate, "plot", fake_plot)
    estimate = np.zeros((5, 220))
    estimate[1, 200:] = 1.0
    estimate[4, 200:] = 2.0
    loaded = {
        "case": {
            "case_number": 3264,
            "scenario": "deep_plus_surface",
            "surface_centers": [1],
            "deep_index": 4,
            "eeg_snr_db": 0,
            "meg_snr_db": 5,
        },
        "groups": [np.array([0, 1, 4]), np.array([4])],
        "geometry": {"n_surf": 4},
    }
    surface = {
        "subject": "sample",
        "subjects_dir": tmp_path,
        "vertices": (np.array([10, 11]), np.array([20, 21])),
    }

    output = brain_maps.render_surface_method(
        "OASTER V20", estimate, loaded, surface, tmp_path / "surface.png"
    )

    assert output.is_file() and output.stat().st_size > 1_000
    assert calls["plot"]["surface"] == "inflated"
    assert calls["plot"]["hemi"] == "split"
    assert calls["plot"]["views"] == ("lateral", "medial")
    assert calls["plot"]["view_layout"] == "horizontal"
    assert calls["plot"]["colormap"] == "inferno"
    assert calls["plot"]["background"] == "white"
    assert calls["plot"]["cortex"] == "classic"
    assert calls["plot"]["clim"]["kind"] == "value"
    assert calls["plot"]["clim"]["lims"] == pytest.approx([0.85, 0.91, 0.97])
    assert calls["plot"]["brain_kwargs"] == {"show": False, "theme": "light"}
    assert calls["screenshot"] == {"mode": "rgb", "time_viewer": False}
    assert calls["stc"].data.shape == (4, 1)
    assert calls["stc"].data.max() == 1.0
    assert calls["plots"][0].closed

    truth_output = brain_maps.render_surface_method(
        "Simulated truth",
        estimate,
        loaded,
        surface,
        tmp_path / "simulation_truth_surface.png",
        is_truth=True,
    )

    assert truth_output.is_file() and truth_output.stat().st_size > 1_000
    assert calls["plot"]["clim"]["kind"] == "value"
    assert calls["plots"][1].closed


def test_display_cutoffs_keep_truth_and_fallback_for_sparse_surface() -> None:
    amplitude = np.arange(1.0, 101.0)
    _relative, algorithm = brain_maps._display_mask(amplitude, 0.10, 95)
    _relative, truth = brain_maps._display_mask(amplitude, 0.10, 0)
    assert algorithm.sum() == 5
    assert truth.sum() == 91

    displayed, clim = brain_maps._surface_display(
        np.r_[1.0, np.zeros(999)], 0.10, False
    )
    assert np.count_nonzero(displayed) == 1
    assert clim["kind"] == "value"


def test_deep_case_builds_truth_method_and_all_combined_maps(
    tmp_path: Path, monkeypatch
) -> None:
    case = {
        "case_id": "deep-mini",
        "case_number": 0,
        "scenario": "deep_plus_surface",
        "surface_centers": [0],
        "deep_index": 2,
        "eeg_snr_db": -5,
        "meg_snr_db": 10,
    }
    source = np.zeros((3, 220))
    loaded = {
        "case": case,
        "truth": source,
        "groups": [np.array([0]), np.array([2])],
        "geometry": {
            "n_surf": 2,
            "n_deep": 1,
            "deep_aseg_labels": np.array([10]),
        },
        "reference": {},
    }
    montages = []

    def fake_image(*args, **_kwargs):
        output = Path(args[5] if len(args) >= 8 else args[4])
        output.write_bytes(b"image")
        return output

    def fake_montage(paths, output, _dpi, title):
        output.write_bytes(b"montage")
        montages.append(([label for label, _path in paths], output.name, title))
        return output

    def fake_pair(_mri, _surface, output, _dpi, title):
        output.write_bytes(b"pair")
        montages.append((["Anatomical MRI", "Cortical surface"], output.name, title))
        return output

    monkeypatch.setattr(brain_maps.strict_plot, "load_strict_case", lambda *_a, **_k: loaded)
    monkeypatch.setattr(
        brain_maps,
        "reconstruct_all",
        lambda *_a, **_k: ({"MNE": source}, {}),
    )
    monkeypatch.setattr(brain_maps, "evaluate_all", lambda *_a: [{"method": "MNE"}])
    monkeypatch.setattr(
        brain_maps, "load_anatomy", lambda *_a: {"sample_path": "sample"}
    )
    monkeypatch.setattr(brain_maps, "load_surface_source_space", lambda *_a: {})
    monkeypatch.setattr(brain_maps, "render_method", fake_image)
    monkeypatch.setattr(brain_maps, "render_surface_method", fake_image)
    monkeypatch.setattr(brain_maps, "render_anatomy_surface_pair", fake_pair)
    monkeypatch.setattr(brain_maps, "render_montage", fake_montage)

    case_dir = brain_maps.plot_brain_maps(
        output_root=tmp_path / "out",
        sisses_root=tmp_path / "archive",
        case_number=0,
        surface_maps=True,
    )

    assert case_dir.name == "case_00000"
    assert [name for _labels, name, _title in montages] == [
        "simulation_truth_mri_surface.png",
        "mne_mri_surface.png",
        "all_methods_brain_mri.png",
        "all_methods_brain_surface.png",
        "all_methods_brain_combined.png",
    ]
    assert montages[-1][0] == ["Simulated truth", "MNE"]
    assert "EEG SNR -5 dB" in montages[-1][2]
    assert "P95" in montages[-1][2]
