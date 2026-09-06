from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse

import plot_strict_case as plotter


def test_saved_estimate_plots_archived_case_without_running_oaster(
    tmp_path: Path, monkeypatch
) -> None:
    times = np.arange(220, dtype=float) / 200.0
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, -0.01]]
    )
    case = {
        "case_number": 0,
        "case_id": "plot-mini-0",
        "configuration_number": 0,
        "configuration_id": "config-0",
        "pair_index": 0,
        "eeg_snr_db": -10,
        "meg_snr_db": 5,
        "scenario": "deep_plus_surface",
        "surface_centers": [0],
        "deep_index": 2,
        "deep_surface_ratio": 0.5,
        "correlation": 0.0,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([case], separators=(",", ":")), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.with_suffix(".json.sha256").write_text(
        f"{digest}  manifest.json\n", encoding="ascii"
    )

    data_root = tmp_path / "data"
    geometry_dir = data_root / "generated" / "deep_plus_two_surface"
    geometry_dir.mkdir(parents=True)
    sio.savemat(
        geometry_dir / "sub_EEG.mat",
        {"src_vertices": vertices, "times": times, "n_surf": 2, "n_deep": 1},
    )
    input_root = tmp_path / "input"
    input_root.mkdir()
    gain_eeg = np.array([[1.0, 0.2, 0.1], [0.1, 1.0, 0.3]])
    gain_meg = np.array([[0.8, 0.4, 0.2], [0.2, 0.5, 1.0]])
    eeg = np.arange(2 * times.size, dtype=float).reshape(2, times.size)
    meg = -eeg
    sio.savemat(
        input_root / "strict_00000_00001.mat",
        {
            "F_EEG": eeg[:, :, None],
            "F_MEG": meg[:, :, None],
            "Gain_EEG": gain_eeg,
            "Gain_MEG": gain_meg,
            "VertConn": sparse.csr_matrix(
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
            ),
            "case_ids": np.array([[case["case_id"]]], dtype=object),
            "manifest_sha256": digest,
            "format_version": plotter.strict.FORMAT_VERSION,
        },
    )

    estimate = np.zeros((3, times.size))
    estimate[0, 200:] = np.sin(np.linspace(0, np.pi, times.size - 200))
    estimate[2, 200:] = 0.5 * estimate[0, 200:]
    estimate_path = tmp_path / "estimate.npz"
    np.savez_compressed(estimate_path, estimate=estimate)
    monkeypatch.setattr(plotter.strict, "EXPECTED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(plotter.strict, "EXPECTED_CASES", 1)
    monkeypatch.setattr(
        plotter.strict.protocol,
        "simulate_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("observations must not be regenerated")
        ),
    )
    monkeypatch.setattr(
        plotter.strict.oaster,
        "reconstruct",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("saved estimate must bypass OASTER")
        ),
    )

    output = plotter.plot_case(
        manifest,
        input_root,
        data_root,
        case_id=case["case_id"],
        estimate_path=estimate_path,
        output=tmp_path / "case.png",
    )

    assert output.is_file() and output.stat().st_size > 10_000
