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
        "Simulated source center",
        "Simulated source parcel",
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


def test_slice_draws_truth_patch_ring_and_exact_center_star(monkeypatch) -> None:
    calls = []

    class Axes:
        def imshow(self, *_args, **_kwargs):
            pass

        def scatter(self, *args, **kwargs):
            calls.append((args, kwargs))

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
        np.empty((0, 3)),
        np.empty(0),
        np.array([[2.0, 3.0, 4.0]]),
        np.zeros((8, 8, 8)),
    )

    assert any(
        call[1].get("facecolors") == "none"
        and call[1].get("edgecolors") == brain_maps.TRUTH_COLOR
        for call in calls
    )
    star_args, star_style = calls[-1]
    assert star_style["marker"] == "*"
    assert star_style["facecolor"] == brain_maps.TRUTH_COLOR
    assert np.array_equal(star_args[0], [center[0]])
    assert np.array_equal(star_args[1], [center[1]])


def test_truth_rings_are_limited_to_the_current_source() -> None:
    loaded = {
        "groups": [np.array([0, 1]), np.array([2])],
        "geometry": {"vertices": np.eye(3)},
    }
    anatomy = {"head_to_mri": np.eye(4), "vox2ras_tkr": np.eye(4)}

    assert brain_maps._truth_voxels(loaded, anatomy, 0).shape == (2, 3)
    assert brain_maps._truth_voxels(loaded, anatomy, 2).shape == (1, 3)


def test_surface_forward_mapping_and_deep_truth_filter(tmp_path: Path) -> None:
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
    overlays = brain_maps._surface_truth_overlays(
        {
            "case": {"surface_centers": [1, 2], "deep_index": 4},
            "groups": [np.array([0, 1, 4]), np.array([2, 3]), np.array([4])],
            "geometry": geometry,
        },
        surface,
    )

    assert [part.tolist() for part in surface["vertices"]] == [[1, 2], [0, 2]]
    assert [(item["hemi"], item["center_vertex"]) for item in overlays] == [
        ("lh", 2),
        ("rh", 0),
    ]
    assert [item["patch_vertices"].tolist() for item in overlays] == [[1, 2], [0, 2]]
    bad_geometry = {**geometry, "vertices": geometry["vertices"].copy()}
    bad_geometry["vertices"][0, 0] += 1e-6
    with pytest.raises(ValueError, match="source order"):
        brain_maps._validate_surface_source_space(bad_geometry, src, subjects_dir)


def test_surface_render_uses_mne_brain_without_projecting_deep_truth(
    tmp_path: Path, monkeypatch
) -> None:
    calls = {"labels": [], "foci": []}

    class Brain:
        closed = False

        def add_label(self, label, **kwargs):
            calls["labels"].append((label, kwargs))

        def add_foci(self, coords, **kwargs):
            calls["foci"].append((list(coords), kwargs))

        def save_image(self, filename):
            Path(filename).write_bytes(b"surface")

        def close(self):
            self.closed = True

    fake_brain = Brain()

    def fake_plot(stc, **kwargs):
        calls["stc"] = stc
        calls["plot"] = kwargs
        return fake_brain

    monkeypatch.setattr(brain_maps.mne.SourceEstimate, "plot", fake_plot)
    estimate = np.zeros((5, 220))
    estimate[1, 200:] = 1.0
    estimate[4, 200:] = 2.0
    loaded = {
        "case": {
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

    assert output.read_bytes() == b"surface"
    assert calls["plot"]["surface"] == "inflated"
    assert calls["plot"]["hemi"] == "split"
    assert calls["plot"]["views"] == ("lateral", "medial")
    assert calls["plot"]["view_layout"] == "horizontal"
    assert calls["plot"]["colormap"] == "inferno"
    assert calls["plot"]["background"] == "white"
    assert calls["plot"]["cortex"] == "classic"
    assert calls["plot"]["brain_kwargs"] == {"show": False, "theme": "light"}
    assert calls["labels"][0][0].vertices.tolist() == [10, 11]
    assert calls["foci"] == [
        (
            [11],
            {
                "coords_as_verts": True,
                "hemi": "lh",
                "scale_factor": 0.7,
                "color": brain_maps.TRUTH_COLOR,
                "name": "simulated_surface_1_center",
            },
        )
    ]
    assert calls["stc"].data.shape == (4, 1)
    assert calls["stc"].data.max() == 1.0
    assert fake_brain.closed
