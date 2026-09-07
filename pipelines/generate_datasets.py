# %%
# ============================================================
# 生成解剖 ROI 深部源多方法对比数据
# 场景：仅深部、仅皮层、深部+单皮层 patch、深部+双皮层 patch
#
# 输出目录：
#   roi_deep_multimethod_comparison/generated/deep_only
#   roi_deep_multimethod_comparison/generated/surface_only
#   roi_deep_multimethod_comparison/generated/deep_plus_surface
#   roi_deep_multimethod_comparison/generated/deep_plus_two_surface
# ============================================================

from pathlib import Path
import os
import warnings

import mne
import nibabel as nib
import numpy as np
import scipy.io as sio
from mne.io.constants import FIFF
from scipy import sparse


mne.set_log_level("warning")


# %%
# ==================== 1. 参数设置 ====================

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(
    os.environ.get("CORRECTED_DATA_ROOT", ROOT / "generated_corrected_v2")
).resolve()
if OUT_ROOT == (ROOT / "generated").resolve():
    raise RuntimeError("corrected-v2 must not overwrite the frozen legacy geometry")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

subject = "sample"
snr_meg = 10
snr_eeg = 10
source_amp = 1.0
rng = np.random.RandomState(20260623)

# FreeSurfer aseg 标签：双侧丘脑
roi_name = "bilateral_thalamus"
roi_label_ids = np.array([10, 49], dtype=np.int64)  # Left/Right-Thalamus-Proper

# 体积源空间分辨率。10 mm 下 sample 里丘脑候选点大约 16 个。
deep_pos_mm = 10.0
deep_neighbor_k = 4

# 皮层 patch 设置。
surface_patch_n_hops = 2
surface_patch_amp = 0.70
surface_center_target_m = np.array([0.045, -0.020, 0.060])
surface_center_target_2_m = np.array([-0.045, -0.020, 0.060])


# %%
# ==================== 2. 小工具函数 ====================

def awgn(signal, snr_db):
    power = np.mean(np.abs(signal) ** 2)
    noise_power = power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise


def mri_pos_to_aseg_label(pos_mri_m, aseg_data, vox2ras_tkr_inv):
    pos_mm = np.asarray(pos_mri_m) * 1000.0
    vox = np.round((vox2ras_tkr_inv @ np.r_[pos_mm, 1.0])[:3]).astype(int)
    if np.any(vox < 0) or np.any(vox >= np.array(aseg_data.shape)):
        return 0
    return int(aseg_data[tuple(vox)])


def bfs_patch_indices(adj, center_idx, n_hops):
    adj = adj.tocsr()
    visited = {int(center_idx)}
    frontier = {int(center_idx)}
    for _ in range(n_hops):
        new_frontier = set()
        for idx in frontier:
            neigh = adj[idx, :].nonzero()[-1]
            new_frontier.update(int(x) for x in neigh)
        new_frontier -= visited
        visited |= new_frontier
        frontier = new_frontier
    return np.array(sorted(visited), dtype=int)


def save_dataset(out_dir, F_meg, F_eeg, gain_mixed_meg, gain_mixed_eeg,
                 gain_mixed_meg_3d, gain_mixed_eeg_3d, VertConn, src_vertices,
                 times, sfreq, s_true, s_true_deep, s_true_surface, true_deep_idx,
                 true_surface_indices, true_surface_patch_labels, true_surface_centers,
                 has_deep_source, n_surf, n_deep, deep_rr, deep_rr_mri,
                 deep_aseg_labels, deep_orientations):
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "VertConn": VertConn,
        "sfreq": np.array([[sfreq]]),
        "times": times[np.newaxis, :],
        "n_surf": np.array([[n_surf]], dtype=np.int64),
        "n_deep": np.array([[n_deep]], dtype=np.int64),
        "roi_label_ids": roi_label_ids[np.newaxis, :],
        "true_deep_idx0": np.array([[true_deep_idx]], dtype=np.int64),
        "true_deep_idx1": np.array([[true_deep_idx + 1]], dtype=np.int64),
        "has_deep_source": np.array([[int(has_deep_source)]], dtype=np.int64),
        "true_surface_indices0": true_surface_indices.astype(np.int64)[np.newaxis, :],
        "true_surface_indices1": (true_surface_indices + 1).astype(np.int64)[np.newaxis, :],
        "true_surface_patch_labels": true_surface_patch_labels.astype(np.int64)[np.newaxis, :],
        "true_surface_centers0": true_surface_centers.astype(np.int64)[np.newaxis, :],
        "true_surface_centers1": (true_surface_centers + 1).astype(np.int64)[np.newaxis, :],
        "deep_rr": deep_rr,
        "deep_rr_mri": deep_rr_mri,
        "deep_aseg_labels": deep_aseg_labels.astype(np.int64)[np.newaxis, :],
        "deep_orientations": deep_orientations,
        "src_vertices": src_vertices,
    }
    sio.savemat(
        out_dir / "sub_MEG.mat",
        {"F": F_meg, "Gain": gain_mixed_meg, "Gain3D": gain_mixed_meg_3d, **common},
        do_compression=True,
    )
    sio.savemat(
        out_dir / "sub_EEG.mat",
        {"F": F_eeg, "Gain": gain_mixed_eeg, "Gain3D": gain_mixed_eeg_3d, **common},
        do_compression=True,
    )
    sio.savemat(
        out_dir / "s_true.mat",
        {
            "s_true": s_true,
            "s_true_deep": s_true_deep,
            "s_true_surface": s_true_surface,
            "src_vertices": src_vertices,
            "sfreq": np.array([[sfreq]]),
            "times": times[np.newaxis, :],
            "t_idx0": np.where(np.sqrt(np.sum(s_true ** 2, axis=0)) > 0)[0][np.newaxis, :],
            "n_surf": np.array([[n_surf]], dtype=np.int64),
            "n_deep": np.array([[n_deep]], dtype=np.int64),
            "roi_label_ids": roi_label_ids[np.newaxis, :],
            "true_deep_idx0": np.array([[true_deep_idx]], dtype=np.int64),
            "true_deep_idx1": np.array([[true_deep_idx + 1]], dtype=np.int64),
            "has_deep_source": np.array([[int(has_deep_source)]], dtype=np.int64),
            "true_surface_indices0": true_surface_indices.astype(np.int64)[np.newaxis, :],
            "true_surface_indices1": (true_surface_indices + 1).astype(np.int64)[np.newaxis, :],
            "true_surface_patch_labels": true_surface_patch_labels.astype(np.int64)[np.newaxis, :],
            "true_surface_centers0": true_surface_centers.astype(np.int64)[np.newaxis, :],
            "true_surface_centers1": (true_surface_centers + 1).astype(np.int64)[np.newaxis, :],
            "deep_rr": deep_rr,
            "deep_rr_mri": deep_rr_mri,
            "deep_aseg_labels": deep_aseg_labels.astype(np.int64)[np.newaxis, :],
        },
        do_compression=True,
    )


# %%
# ==================== 3. 读取 MNE sample 和表面 forward ====================

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / "subjects"
raw_file = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
fwd_file = data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"
trans_file = data_path / "MEG" / "sample" / "sample_audvis_raw-trans.fif"
inner_skull = subjects_dir / subject / "bem" / "inner_skull.surf"
bem_file = subjects_dir / subject / "bem" / "sample-5120-5120-5120-bem-sol.fif"
aseg_file = subjects_dir / subject / "mri" / "aseg.mgz"

raw = mne.io.read_raw_fif(raw_file, preload=True)
sfreq = float(raw.info["sfreq"])

fwd_surf = mne.read_forward_solution(fwd_file)
fwd_surf_free = mne.convert_forward_solution(fwd_surf, surf_ori=True, force_fixed=False, use_cps=True)
fwd_surf_fixed = mne.convert_forward_solution(fwd_surf, surf_ori=True, force_fixed=True, use_cps=True)
fwd_surf_meg = mne.pick_types_forward(fwd_surf_fixed, meg="mag", eeg=False)
fwd_surf_eeg = mne.pick_types_forward(fwd_surf_fixed, meg=False, eeg=True)
fwd_surf_free_meg = mne.pick_types_forward(fwd_surf_free, meg="mag", eeg=False)
fwd_surf_free_eeg = mne.pick_types_forward(fwd_surf_free, meg=False, eeg=True)

raw_eeg = raw.copy().pick_types(meg=False, eeg=True, stim=False, eog=False, exclude="bads")
common_eeg = [ch for ch in fwd_surf_eeg["info"]["ch_names"] if ch in raw_eeg.info["ch_names"]]
fwd_surf_eeg = mne.pick_channels_forward(fwd_surf_eeg, include=common_eeg)
fwd_surf_free_eeg = mne.pick_channels_forward(fwd_surf_free_eeg, include=common_eeg)

gain_surf_meg = fwd_surf_meg["sol"]["data"].astype(float)
gain_surf_eeg = fwd_surf_eeg["sol"]["data"].astype(float)
src_surf = fwd_surf_meg["src"]
n_surf = gain_surf_meg.shape[1]
gain_surf_meg_3d = fwd_surf_free_meg["sol"]["data"].astype(float).reshape(gain_surf_meg.shape[0], n_surf, 3)
gain_surf_eeg_3d = fwd_surf_free_eeg["sol"]["data"].astype(float).reshape(gain_surf_eeg.shape[0], n_surf, 3)

src_vertices_surf = np.vstack(
    [
        src_surf[0]["rr"][src_surf[0]["vertno"]],
        src_surf[1]["rr"][src_surf[1]["vertno"]],
    ]
)

print("surface sources:", n_surf)
print("surface MEG gain:", gain_surf_meg.shape)
print("surface EEG gain:", gain_surf_eeg.shape)
print("surface MEG free-orientation gain:", gain_surf_meg_3d.shape)
print("surface EEG free-orientation gain:", gain_surf_eeg_3d.shape)


# %%
# ==================== 4. 生成体积 forward，并按 aseg ROI 选深部候选点 ====================

vol_src = mne.setup_volume_source_space(
    subject=subject,
    pos=deep_pos_mm,
    subjects_dir=subjects_dir,
    surface=inner_skull,
    add_interpolator=False,
    verbose=True,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fwd_vol = mne.make_forward_solution(
        raw.info,
        trans=trans_file,
        src=vol_src,
        bem=bem_file,
        meg=True,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=True,
    )

fwd_vol_meg = mne.pick_types_forward(fwd_vol, meg="mag", eeg=False)
fwd_vol_eeg = mne.pick_types_forward(fwd_vol, meg=False, eeg=True)
fwd_vol_eeg = mne.pick_channels_forward(fwd_vol_eeg, include=common_eeg)

aseg = nib.load(str(aseg_file))
aseg_data = np.asarray(aseg.get_fdata(), dtype=int)
vox2ras_tkr_inv = np.linalg.inv(aseg.header.get_vox2ras_tkr())

source_rr_head = fwd_vol_meg["source_rr"]
head_to_mri = mne.transforms.invert_transform(fwd_vol_meg["mri_head_t"])
if (
    int(head_to_mri["from"]) != int(FIFF.FIFFV_COORD_HEAD)
    or int(head_to_mri["to"]) != int(FIFF.FIFFV_COORD_MRI)
):
    raise RuntimeError("volume forward does not provide a HEAD-to-MRI transform")
source_rr_mri = mne.transforms.apply_trans(head_to_mri, source_rr_head)
roundtrip = mne.transforms.apply_trans(fwd_vol_meg["mri_head_t"], source_rr_mri)
if not np.allclose(roundtrip, source_rr_head, rtol=0.0, atol=1e-10):
    raise RuntimeError("HEAD-to-MRI coordinate transform failed its round-trip check")
labels = np.array(
    [mri_pos_to_aseg_label(p, aseg_data, vox2ras_tkr_inv) for p in source_rr_mri],
    dtype=int,
)
deep_sel = np.where(np.isin(labels, roi_label_ids))[0]
deep_rr = source_rr_head[deep_sel]
deep_rr_mri = source_rr_mri[deep_sel]
n_deep = len(deep_sel)

if n_deep < 4:
    raise RuntimeError(f"ROI 候选点太少：{n_deep}")
label_values, label_counts = np.unique(labels[deep_sel], return_counts=True)
actual_label_counts = dict(zip(label_values.tolist(), label_counts.tolist()))
if actual_label_counts != {10: 9, 49: 7}:
    raise RuntimeError(
        f"corrected-v2 requires the frozen 16-point thalamic grid; got {actual_label_counts}"
    )

print("volume candidate sources:", fwd_vol_meg["nsource"])
print("ROI:", roi_name, "label ids:", roi_label_ids.tolist())
print("selected ROI deep sources:", n_deep)
print("selected labels:", labels[deep_sel].tolist())
print("selected MRI RAS coordinates (m):", deep_rr_mri.tolist())


# %%
# ==================== 5. 把体积源三方向压成单方向 leadfield ====================

gain_deep_meg = np.zeros((gain_surf_meg.shape[0], n_deep))
gain_deep_eeg = np.zeros((gain_surf_eeg.shape[0], n_deep))
gain_deep_meg_3d = np.zeros((gain_surf_meg.shape[0], n_deep, 3))
gain_deep_eeg_3d = np.zeros((gain_surf_eeg.shape[0], n_deep, 3))
deep_orientations = np.zeros((n_deep, 3))

for out_i, src_i in enumerate(deep_sel):
    cols = slice(3 * src_i, 3 * src_i + 3)
    Gm = fwd_vol_meg["sol"]["data"][:, cols].astype(float)
    Ge = fwd_vol_eeg["sol"]["data"][:, cols].astype(float)
    gain_deep_meg_3d[:, out_i, :] = Gm
    gain_deep_eeg_3d[:, out_i, :] = Ge
    Gm_norm = Gm / (np.linalg.norm(Gm) + 1e-20)
    Ge_norm = Ge / (np.linalg.norm(Ge) + 1e-20)
    G_joint = np.vstack([Gm_norm, Ge_norm])
    _, _, vt = np.linalg.svd(G_joint, full_matrices=False)
    ori = vt[0]
    gain_deep_meg[:, out_i] = Gm @ ori
    gain_deep_eeg[:, out_i] = Ge @ ori
    deep_orientations[out_i] = ori


# %%
# ==================== 6. 拼接 mixed source space 和邻接矩阵 ====================

gain_mixed_meg = np.hstack([gain_surf_meg, gain_deep_meg])
gain_mixed_eeg = np.hstack([gain_surf_eeg, gain_deep_eeg])
gain_mixed_meg_3d = np.concatenate([gain_surf_meg_3d, gain_deep_meg_3d], axis=1)
gain_mixed_eeg_3d = np.concatenate([gain_surf_eeg_3d, gain_deep_eeg_3d], axis=1)
n_total = n_surf + n_deep
src_vertices = np.vstack([src_vertices_surf, deep_rr])

VertConn_surf = mne.spatial_src_adjacency(src_surf)
VertConn_surf = VertConn_surf.tocsr()

VertConn = sparse.lil_matrix((n_total, n_total), dtype=np.uint8)
VertConn[:n_surf, :n_surf] = VertConn_surf

for i in range(n_deep):
    d = np.linalg.norm(deep_rr - deep_rr[i], axis=1)
    nn = np.argsort(d)[1 : deep_neighbor_k + 1]
    for j in nn:
        VertConn[n_surf + i, n_surf + j] = 1
        VertConn[n_surf + j, n_surf + i] = 1

VertConn = VertConn.toarray().astype(np.uint8)

print("mixed MEG gain:", gain_mixed_meg.shape)
print("mixed EEG gain:", gain_mixed_eeg.shape)
print("mixed MEG free-orientation gain:", gain_mixed_meg_3d.shape)
print("mixed EEG free-orientation gain:", gain_mixed_eeg_3d.shape)
print("mixed VertConn:", VertConn.shape)


# %%
# ==================== 7. 构造真实深部源和皮层 patch ====================

times = np.arange(0, 1, 1.0 / sfreq)
active_from = int(times.size / 3)

deep_signal = source_amp * np.sin(2 * np.pi * 10 * times)
deep_signal[:active_from] = 0

surface_signal = surface_patch_amp * np.sin(2 * np.pi * 16 * times + np.pi / 5)
surface_signal[:active_from] = 0
surface_signal_2 = 0.85 * surface_patch_amp * np.sin(2 * np.pi * 13 * times - np.pi / 7)
surface_signal_2[:active_from] = 0

# 真实深部源取 ROI 候选点中最靠近 ROI 几何中心的点。
roi_center = deep_rr.mean(axis=0)
true_deep_local = int(np.argmin(np.linalg.norm(deep_rr - roi_center, axis=1)))
true_deep_idx = n_surf + true_deep_local

surface_center_idx = int(np.argmin(np.linalg.norm(src_vertices_surf - surface_center_target_m, axis=1)))
true_surface_indices = bfs_patch_indices(VertConn_surf, surface_center_idx, surface_patch_n_hops)
surface_center_2_idx = int(
    np.argmin(np.linalg.norm(src_vertices_surf - surface_center_target_2_m, axis=1))
)
true_surface_indices_2 = bfs_patch_indices(
    VertConn_surf,
    surface_center_2_idx,
    surface_patch_n_hops,
)

patch_pos = src_vertices_surf[true_surface_indices]
patch_dist = np.linalg.norm(patch_pos - src_vertices_surf[surface_center_idx], axis=1)
patch_weights = np.exp(-(patch_dist ** 2) / (2 * (0.010 ** 2)))
patch_weights = patch_weights / patch_weights.max()
patch_2_pos = src_vertices_surf[true_surface_indices_2]
patch_2_dist = np.linalg.norm(
    patch_2_pos - src_vertices_surf[surface_center_2_idx],
    axis=1,
)
patch_2_weights = np.exp(-(patch_2_dist ** 2) / (2 * (0.010 ** 2)))
patch_2_weights = patch_2_weights / patch_2_weights.max()

print("true deep index 0-based:", true_deep_idx)
print("true deep index 1-based:", true_deep_idx + 1)
print("true deep coordinate:", src_vertices[true_deep_idx])
print("surface patch center 1-based:", surface_center_idx + 1)
print("surface patch size:", len(true_surface_indices))
print("surface patch 2 center 1-based:", surface_center_2_idx + 1)
print("surface patch 2 size:", len(true_surface_indices_2))


# %%
# ==================== 8. 保存四个场景 ====================

scenario_specs = {
    "deep_only": {"deep": True, "patches": ()},
    "deep_plus_surface": {"deep": True, "patches": (1,)},
    "surface_only": {"deep": False, "patches": (1,)},
    "deep_plus_two_surface": {"deep": True, "patches": (1, 2)},
}

for scenario, spec in scenario_specs.items():
    s_true = np.zeros((n_total, len(times)))
    s_true_deep = np.zeros_like(s_true)
    s_true_surface = np.zeros_like(s_true)

    if spec["deep"]:
        s_true_deep[true_deep_idx, :] = deep_signal
        s_true += s_true_deep

    surface_index_parts = []
    surface_label_parts = []
    surface_centers_to_save = []
    if 1 in spec["patches"]:
        for idx, w in zip(true_surface_indices, patch_weights):
            s_true_surface[idx, :] = w * surface_signal
        surface_index_parts.append(true_surface_indices)
        surface_label_parts.append(np.ones(len(true_surface_indices), dtype=int))
        surface_centers_to_save.append(surface_center_idx)
    if 2 in spec["patches"]:
        for idx, w in zip(true_surface_indices_2, patch_2_weights):
            s_true_surface[idx, :] += w * surface_signal_2
        surface_index_parts.append(true_surface_indices_2)
        surface_label_parts.append(2 * np.ones(len(true_surface_indices_2), dtype=int))
        surface_centers_to_save.append(surface_center_2_idx)
    s_true += s_true_surface

    if surface_index_parts:
        surface_indices_to_save = np.concatenate(surface_index_parts)
        surface_patch_labels_to_save = np.concatenate(surface_label_parts)
        surface_centers_to_save = np.asarray(surface_centers_to_save, dtype=int)
    else:
        surface_indices_to_save = np.array([], dtype=int)
        surface_patch_labels_to_save = np.array([], dtype=int)
        surface_centers_to_save = np.array([], dtype=int)

    F_meg_clean = gain_mixed_meg @ s_true
    F_eeg_clean = gain_mixed_eeg @ s_true
    F_meg = awgn(F_meg_clean, snr_meg)
    F_eeg = awgn(F_eeg_clean, snr_eeg)

    save_dataset(
        out_dir=OUT_ROOT / scenario,
        F_meg=F_meg,
        F_eeg=F_eeg,
        gain_mixed_meg=gain_mixed_meg,
        gain_mixed_eeg=gain_mixed_eeg,
        gain_mixed_meg_3d=gain_mixed_meg_3d,
        gain_mixed_eeg_3d=gain_mixed_eeg_3d,
        VertConn=VertConn,
        src_vertices=src_vertices,
        times=times,
        sfreq=sfreq,
        s_true=s_true,
        s_true_deep=s_true_deep,
        s_true_surface=s_true_surface,
        true_deep_idx=true_deep_idx,
        true_surface_indices=surface_indices_to_save,
        true_surface_patch_labels=surface_patch_labels_to_save,
        true_surface_centers=surface_centers_to_save,
        has_deep_source=spec["deep"],
        n_surf=n_surf,
        n_deep=n_deep,
        deep_rr=deep_rr,
        deep_rr_mri=deep_rr_mri,
        deep_aseg_labels=labels[deep_sel],
        deep_orientations=deep_orientations,
    )
    print("saved scenario:", scenario, "->", OUT_ROOT / scenario)

print("done")
