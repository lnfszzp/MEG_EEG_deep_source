from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
from scipy.spatial import cKDTree

from protected_multilayer import (
    DATA_ROOT,
    NOISE_SAMPLES,
    OUT_ROOT,
    load_mat,
    residual_deep_scores,
)
from pipelines.run_whole_brain_fusion import whitening_matrix


CACHE = OUT_ROOT / "refined_forward_oct7_5mm.npz"


def build_refined_forward(path: Path = CACHE) -> Path:
    """Build a real oct-7 cortical and 5 mm thalamic MNE forward model."""
    import mne
    import nibabel as nib

    mne.set_log_level("warning")
    data_path = mne.datasets.sample.data_path(download=False)
    subjects_dir = data_path / "subjects"
    subject = "sample"
    raw_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    trans_file = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
    bem_file = subjects_dir / subject / "bem" / "sample-5120-5120-5120-bem-sol.fif"
    inner_skull = subjects_dir / subject / "bem" / "inner_skull.surf"
    aseg_file = subjects_dir / subject / "mri" / "aseg.mgz"

    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    src_surf = mne.setup_source_space(
        subject,
        spacing="oct7",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_surf = mne.make_forward_solution(
            raw.info,
            trans=trans_file,
            src=src_surf,
            bem=bem_file,
            meg=True,
            eeg=True,
            mindist=5.0,
            n_jobs=4,
            verbose=False,
        )
    fixed = mne.convert_forward_solution(
        fwd_surf,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=False,
    )
    surf_meg = mne.pick_types_forward(fixed, meg="mag", eeg=False)
    surf_eeg = mne.pick_types_forward(fixed, meg=False, eeg=True)
    raw_eeg = raw.copy().pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    common_eeg = [name for name in surf_eeg["info"]["ch_names"] if name in raw_eeg.info["ch_names"]]
    surf_eeg = mne.pick_channels_forward(surf_eeg, include=common_eeg)
    surf_xyz = np.vstack([src["rr"][src["vertno"]] for src in surf_meg["src"]])
    surf_edges = np.column_stack(
        mne.spatial_src_adjacency(surf_meg["src"]).tocsr().nonzero()
    )
    surf_edges = surf_edges[surf_edges[:, 0] < surf_edges[:, 1]]

    volume = mne.setup_volume_source_space(
        subject,
        pos=5.0,
        subjects_dir=subjects_dir,
        surface=inner_skull,
        add_interpolator=False,
        verbose=False,
    )
    volume_xyz = volume[0]["rr"][volume[0]["vertno"]]
    aseg = nib.load(str(aseg_file))
    aseg_data = np.asarray(aseg.get_fdata(), dtype=int)
    inverse = np.linalg.inv(aseg.header.get_vox2ras_tkr())
    vox = np.round(
        (inverse @ np.column_stack([volume_xyz * 1000.0, np.ones(volume_xyz.shape[0])]).T).T[:, :3]
    ).astype(int)
    inside = np.all((vox >= 0) & (vox < np.asarray(aseg_data.shape)), axis=1)
    labels = np.zeros(volume_xyz.shape[0], dtype=int)
    labels[inside] = aseg_data[tuple(vox[inside].T)]
    deep_xyz = volume_xyz[np.isin(labels, (10, 49))]
    deep_src = mne.setup_volume_source_space(
        pos={"rr": deep_xyz, "nn": np.tile([0.0, 0.0, 1.0], (deep_xyz.shape[0], 1))},
        add_interpolator=False,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fwd_deep = mne.make_forward_solution(
            raw.info,
            trans=trans_file,
            src=deep_src,
            bem=bem_file,
            meg=True,
            eeg=True,
            mindist=0.0,
            n_jobs=4,
            verbose=False,
        )
    deep_meg = mne.pick_types_forward(fwd_deep, meg="mag", eeg=False)
    deep_eeg = mne.pick_types_forward(fwd_deep, meg=False, eeg=True)
    deep_eeg = mne.pick_channels_forward(deep_eeg, include=common_eeg)
    gm3 = deep_meg["sol"]["data"].reshape(deep_meg["sol"]["data"].shape[0], deep_xyz.shape[0], 3)
    ge3 = deep_eeg["sol"]["data"].reshape(deep_eeg["sol"]["data"].shape[0], deep_xyz.shape[0], 3)
    gm = np.zeros(gm3.shape[:2])
    ge = np.zeros(ge3.shape[:2])
    for index in range(deep_xyz.shape[0]):
        joint = np.vstack(
            [gm3[:, index] / max(np.linalg.norm(gm3[:, index]), 1e-20), ge3[:, index] / max(np.linalg.norm(ge3[:, index]), 1e-20)]
        )
        orientation = np.linalg.svd(joint, full_matrices=False)[2][0]
        gm[:, index] = gm3[:, index] @ orientation
        ge[:, index] = ge3[:, index] @ orientation

    coarse_eeg = load_mat(DATA_ROOT / "deep_plus_two_surface" / "sub_EEG.mat")
    coarse_meg = load_mat(DATA_ROOT / "deep_plus_two_surface" / "sub_MEG.mat")
    if surf_eeg["sol"]["data"].shape[0] != coarse_eeg["F"].shape[0] or surf_meg["sol"]["data"].shape[0] != coarse_meg["F"].shape[0]:
        raise ValueError("refined forward channels do not match the simulation data")
    surface_spacing = np.median(np.linalg.norm(surf_xyz[surf_edges[:, 0]] - surf_xyz[surf_edges[:, 1]], axis=1))
    deep_spacing = np.median(cKDTree(deep_xyz).query(deep_xyz, k=2)[0][:, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        gain_eeg=np.hstack([surf_eeg["sol"]["data"], ge]),
        gain_meg=np.hstack([surf_meg["sol"]["data"], gm]),
        vertices=np.vstack([surf_xyz, deep_xyz]),
        surface_edges=surf_edges.astype(np.int32),
        n_surf=np.array([surf_xyz.shape[0]], dtype=np.int64),
        n_deep=np.array([deep_xyz.shape[0]], dtype=np.int64),
        surface_spacing_mm=np.array([surface_spacing * 1000.0]),
        deep_spacing_mm=np.array([deep_spacing * 1000.0]),
    )
    return path


def load_refined_forward(path: Path = CACHE) -> dict[str, np.ndarray]:
    if not path.exists():
        build_refined_forward(path)
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def refined_sd_dle(
    source: np.ndarray,
    mask: np.ndarray,
    vertices: np.ndarray,
    truth_vertices: np.ndarray,
    true_groups: list[np.ndarray],
) -> tuple[float, float]:
    """Use the requested SD/DLE functions when estimate and truth grids differ."""
    from DLE_an import DLE_an
    from SD import SD

    used = np.asarray(source, dtype=float) * np.asarray(mask, dtype=bool)[:, None]
    truth_vertices = np.asarray(truth_vertices, dtype=float)
    true_positions = np.vstack([truth_vertices[np.asarray(group, dtype=int)] for group in true_groups])
    sd_mm = float(SD(used, true_positions, vertices, 0.0) * 1000.0)

    positions = np.vstack([vertices, true_positions])
    padded = np.vstack([used, np.zeros((true_positions.shape[0], used.shape[1]))])
    groups_1based = []
    start = vertices.shape[0]
    for group in true_groups:
        size = np.asarray(group).size
        groups_1based.append(np.arange(start, start + size) + 1)
        start += size
    dle_mm = float(DLE_an(padded, groups_1based, positions) * 1000.0)
    return sd_mm, dle_mm


def refined_deep_evidence(
    eeg: dict,
    meg: dict,
    coarse_source: np.ndarray,
    coarse_vertices: np.ndarray,
    coarse_n_surf: int,
    refined: dict[str, np.ndarray],
    *,
    min_modality_drop: float = 0.05,
) -> dict | None:
    """Choose one coarse/fine deep point only when both modalities support it."""
    fine_n_surf = int(np.asarray(refined["n_surf"]).ravel()[0])
    eeg_w = whitening_matrix(np.asarray(eeg["F"], dtype=float), NOISE_SAMPLES)
    meg_w = whitening_matrix(np.asarray(meg["F"], dtype=float), NOISE_SAMPLES)
    eeg_gain = eeg_w @ np.asarray(eeg["Gain"], dtype=float)
    meg_gain = meg_w @ np.asarray(meg["Gain"], dtype=float)
    eeg_residual = eeg_w @ np.asarray(eeg["F"], dtype=float) - eeg_gain[:, :coarse_n_surf] @ coarse_source[:coarse_n_surf]
    meg_residual = meg_w @ np.asarray(meg["F"], dtype=float) - meg_gain[:, :coarse_n_surf] @ coarse_source[:coarse_n_surf]
    eeg_deep = np.hstack([eeg_gain[:, coarse_n_surf:], eeg_w @ refined["gain_eeg"][:, fine_n_surf:]])
    meg_deep = np.hstack([meg_gain[:, coarse_n_surf:], meg_w @ refined["gain_meg"][:, fine_n_surf:]])
    eeg_score = residual_deep_scores(eeg_residual, eeg_deep, 0)
    meg_score = residual_deep_scores(meg_residual, meg_deep, 0)
    combined = eeg_score / max(float(eeg_score.max()), 1e-12) + meg_score / max(float(meg_score.max()), 1e-12)
    index = int(np.argmax(combined))

    drops = []
    for residual, leadfield in ((eeg_residual, eeg_deep), (meg_residual, meg_deep)):
        column = leadfield[:, index, None]
        coefficient = (column.T @ residual) / (float((column.T @ column)[0, 0]) + 1e-12)
        drops.append(1.0 - float(np.linalg.norm(residual - column @ coefficient, "fro") ** 2 / np.linalg.norm(residual, "fro") ** 2))
    if min(drops) < min_modality_drop:
        return None

    joint_column = np.r_[eeg_deep[:, index], meg_deep[:, index]][:, None]
    joint_residual = np.vstack([eeg_residual, meg_residual])
    timecourse = ((joint_column.T @ joint_residual) / (float((joint_column.T @ joint_column)[0, 0]) + 1e-12)).ravel()
    coarse_deep = coarse_vertices.shape[0] - coarse_n_surf
    if index < coarse_deep:
        position = np.asarray(coarse_vertices, dtype=float)[coarse_n_surf + index]
        coarse_index = coarse_n_surf + index
        grid = "coarse"
    else:
        position = np.asarray(refined["vertices"], dtype=float)[fine_n_surf + index - coarse_deep]
        coarse_index = coarse_n_surf + int(cKDTree(coarse_vertices[coarse_n_surf:]).query(position)[1])
        grid = "fine"
    return {
        "position": position,
        "timecourse": timecourse,
        "coarse_index": coarse_index,
        "grid": grid,
        "eeg_drop": drops[0],
        "meg_drop": drops[1],
    }
