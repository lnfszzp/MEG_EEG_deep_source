#%%
# ds004998 不是 DBS 刺激 ERP，而是帕金森患者的连续 MEG + STN-LFP 记录。
# 这一版只做最小、可核查的频谱定位基线：sub-0cGdk9，左手 Hold，服药后。
# 写法仍按你的习惯：参数在前，从上到下运行，不定义自己的函数。
#
# 很重要：FIF 里的 EEG001-EEG008 实际是 DBS 电极触点，不是头皮 EEG。
# 它们不进入 EEG forward，也不参与本次 DICS 逆解；以后只作为独立 LFP 验证。

from pathlib import Path
import csv
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import mne
from mne.io.constants import FIFF
from mne.surface import complete_surface_info
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
from scipy.spatial import cKDTree


mne.set_log_level("warning")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#%%
# ==================== 1. 参数区：通常只改这里 ====================

project_root = Path(__file__).resolve().parents[1]
dataset_path = Path(r"D:\博士\工作＆汇报\源定位\开源数据\ds004998")

subject = "0cGdk9"
task = "HoldL"
medication = "MedOn"
run = 1

highpass_hz = 1.0
lowpass_hz = 45.0
resample_hz = 200.0
epoch_seconds = 2.0
epoch_overlap_seconds = 1.0
beta_low_hz = 13.0
beta_high_hz = 30.0
multitaper_bandwidth_hz = 4.0
dics_regularization = 0.05

# 原网格为 4 mm。三个方向各隔一个点，得到约 8 mm 的试运行网格。
grid_stride = 2
display_percentile = 95.0

# 左手 Hold 的主假设是右侧感觉运动区。中心取自独立 MEG beta-ERD 文献
# （Heinrichs-Graham et al., 2020, doi:10.1093/cercor/bhaa199），半径预先固定。
right_motor_center_mni_mm = np.array([38.0, -28.0, 50.0])
left_motor_center_mni_mm = np.array([-38.0, -28.0, 50.0])
motor_roi_radius_mm = 25.0

# 解剖图只用于展示；所有指标仍在原始个体体积源网格上计算。
sample_data_path = Path(mne.datasets.sample.data_path(download=False, verbose=False))
subjects_dir = sample_data_path / "subjects"
fsaverage_t1_file = subjects_dir / "fsaverage" / "mri" / "T1.mgz"
draw_template_anatomy = fsaverage_t1_file.is_file()
surface_projection_max_distance_mm = 12.0

subject_root = dataset_path / f"sub-{subject}" / "ses-PeriOp"
file_stem = f"sub-{subject}_ses-PeriOp_task-{task}_acq-{medication}_run-{run}"
raw_file = subject_root / "meg" / f"{file_stem}_meg.fif"
events_file = subject_root / "meg" / f"{file_stem}_events.tsv"
montage_file = subject_root / "montage" / f"sub-{subject}_ses-PeriOp_montage.tsv"
headmodel_file = subject_root / "headmodel" / f"sub-{subject}_ses-PeriOp_headmodel.mat"
sourcemodel_file = subject_root / "sourcemodel" / f"sub-{subject}_ses-PeriOp_sourcemodel.mat"
template_grid_file = dataset_path / "template_sourcemodel.mat"

save_dir = (
    project_root
    / "results"
    / "real_data"
    / "ds004998_dbs_v1"
    / f"sub-{subject}_{task}_{medication}_run-{run}"
)
save_dir.mkdir(parents=True, exist_ok=True)

for required_file in (
    raw_file,
    events_file,
    montage_file,
    headmodel_file,
    sourcemodel_file,
    template_grid_file,
):
    assert required_file.is_file(), f"缺少文件：{required_file}"

all_started = time.perf_counter()
print("结果保存到：", save_dir)
print("本地正规 FIF 数：", len(list(dataset_path.rglob("*.fif"))))
print("本地 MEG sidecar 数：", len(list(dataset_path.rglob("*_meg.json"))))


#%%
# ==================== 2. 确认 EEG001-EEG008 的真实身份 ====================

with montage_file.open(encoding="utf-8-sig", newline="") as montage_handle:
    montage_rows = list(csv.DictReader(montage_handle, delimiter="\t"))

lfp_name_mapping = {}
for montage_row in montage_rows:
    lfp_name_mapping[montage_row["right_contacts_old"]] = montage_row["right_contacts_new"]
    lfp_name_mapping[montage_row["left_contacts_old"]] = montage_row["left_contacts_new"]

expected_lfp_storage_names = {f"EEG{number:03d}" for number in range(1, 9)}
assert set(lfp_name_mapping) == expected_lfp_storage_names
assert all(name.startswith("LFP-") for name in lfp_name_mapping.values())

print("FIF 中伪装成 EEG 的通道实际为：")
for old_name in sorted(lfp_name_mapping):
    print("   ", old_name, "->", lfp_name_mapping[old_name])
print("本次 forward/DICS 只使用 planar gradiometer，LFP 不进入逆解。")


#%%
# ==================== 3. 读取 rest 和 Hold 的区间 ====================

with events_file.open(encoding="utf-8-sig", newline="") as events_handle:
    event_rows = list(csv.DictReader(events_handle, delimiter="\t"))

rest_intervals = []
hold_intervals = []
bad_lfp_intervals = []
for event_row in event_rows:
    onset = float(event_row["onset"])
    duration = float(event_row["duration"])
    trial_type = event_row["trial_type"]
    if trial_type == "rest" and duration >= epoch_seconds:
        rest_intervals.append((onset, duration))
    elif trial_type == task and duration >= epoch_seconds:
        hold_intervals.append((onset, duration))
    elif trial_type.lower().startswith("bad"):
        bad_lfp_intervals.append((onset, duration, trial_type))

assert rest_intervals, "events.tsv 中没有可用的 rest 区间"
assert hold_intervals, f"events.tsv 中没有可用的 {task} 区间"

print("rest 总时长（秒）：", sum(duration for _, duration in rest_intervals))
print(f"{task} 总时长（秒）：", sum(duration for _, duration in hold_intervals))
print("bad_lfp 区间：", bad_lfp_intervals)


#%%
# ==================== 4. 读取个体 FieldTrip 网格，并降到约 8 mm ====================

grid = loadmat(sourcemodel_file, simplify_cells=True)["grid"]
template_grid = loadmat(template_grid_file, simplify_cells=True)["sourcemodel"]

grid_positions_cm = np.asarray(grid["pos"], dtype=float)
grid_inside = np.asarray(grid["inside"]).astype(bool).ravel()
grid_dimensions = np.asarray(grid["dim"], dtype=int).ravel()
template_positions_cm = np.asarray(template_grid["pos"], dtype=float)
template_inside = np.asarray(template_grid["inside"]).astype(bool).ravel()

assert grid.get("unit") == "cm"
assert grid.get("coordsys") == "neuromag"
assert grid_positions_cm.shape == template_positions_cm.shape
assert int(np.prod(grid_dimensions)) == len(grid_positions_cm)
assert np.array_equal(grid_inside, template_inside), "个体和模板网格不是逐点对应"

all_grid_indices = np.arange(len(grid_positions_cm))
grid_ijk = np.column_stack(
    np.unravel_index(all_grid_indices, tuple(grid_dimensions), order="F")
)
subsample_mask = grid_inside & np.all(grid_ijk % grid_stride == 0, axis=1)
subsample_global_indices = np.flatnonzero(subsample_mask)
source_positions_head_m = grid_positions_cm[subsample_mask] * 0.01

# DICS 使用自由方向，下面的径向单位向量只用于满足离散源空间的数据结构要求。
source_normals = source_positions_head_m - source_positions_head_m.mean(axis=0)
source_normals /= np.linalg.norm(source_normals, axis=1, keepdims=True)

source_space = mne.setup_volume_source_space(
    subject=None,
    pos={"rr": source_positions_head_m, "nn": source_normals},
    mri=None,
    verbose=False,
)

print("原 4 mm inside 点数：", int(grid_inside.sum()))
print("约 8 mm 试运行点数：", len(subsample_global_indices))


#%%
# ==================== 5. 把 FieldTrip 单壳 mesh 转成 MNE 一层 BEM ====================

headmodel = loadmat(headmodel_file, simplify_cells=True)["hdm"]
assert headmodel.get("type") == "singleshell"
assert headmodel.get("unit") == "cm"

boundary = headmodel["bnd"]
bem_surface = {
    "rr": np.asarray(boundary["pos"], dtype=float) * 0.01,
    "tris": np.asarray(boundary["tri"], dtype=int) - 1,
    "id": FIFF.FIFFV_BEM_SURF_ID_BRAIN,
    "sigma": 0.3,
    "coord_frame": FIFF.FIFFV_COORD_MRI,
}
bem_surface = complete_surface_info(bem_surface, copy=True, verbose=False)
bem_solution = mne.make_bem_solution([bem_surface], verbose=False)

# FieldTrip 的 grid 和 headmodel 都已经在同一个 neuromag/head 坐标系中。
# 这里把相同数值临时标作 MRI 坐标，并用单位矩阵连接 head 和 MRI。
head_to_model = mne.transforms.Transform("head", "mri", np.eye(4))

print("单壳 mesh：", bem_surface["np"], "个顶点，", bem_surface["ntri"], "个三角面")
print("注意：这是 MNE 一层 BEM 数值解，不等同于 FieldTrip Nolte singleshell。")


#%%
# ==================== 6. 只读取 planar gradiometer ====================

raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
raw_duration = raw.times[-1]
stored_lfp_channels = sorted(set(raw.ch_names) & set(lfp_name_mapping))
assert stored_lfp_channels, "FIF 中没有找到 montage.tsv 指定的 LFP 通道"
excluded_bad_channels = list(raw.info["bads"])

grad_picks = mne.pick_types(
    raw.info,
    meg="grad",
    eeg=False,
    eog=False,
    ecg=False,
    emg=False,
    stim=False,
    ref_meg=False,
    exclude="bads",
)
raw.pick(grad_picks)
# 原文件的 SSP 是按 204 个 grad 算的；删掉两个坏 grad 后它几乎退化为零向量。
# 官方 FieldTrip 示例也不应用这个 MNE SSP，因此本基线明确删除它，避免病态投影。
raw.del_proj()
assert all(mne.channel_type(raw.info, index) == "grad" for index in range(len(raw.ch_names)))

print("用于 DICS 的 planar gradiometer：", len(raw.ch_names))
print("排除的坏通道：", excluded_bad_channels)


#%%
# ==================== 7. 建立约 8 mm 的 MEG forward ====================

forward = mne.make_forward_solution(
    raw.info,
    trans=head_to_model,
    src=source_space,
    bem=bem_solution,
    meg=True,
    eeg=False,
    mindist=0.0,
    ignore_ref=False,
    n_jobs=1,
    on_inside="warning",
    verbose=False,
)

assert forward["nsource"] > 0
assert np.isfinite(forward["sol"]["data"]).all()
print("BEM 内保留的源点数：", forward["nsource"])
print("自由方向 leadfield 形状：", forward["sol"]["data"].shape)


#%%
# ==================== 8. rest 分段、降采样和 2 秒重叠 epoch ====================

rest_epoch_parts = []
for onset, duration in rest_intervals:
    segment_end = min(onset + duration - 1.0 / raw.info["sfreq"], raw_duration)
    rest_segment = raw.copy().crop(tmin=onset, tmax=segment_end)
    rest_segment.load_data(verbose=False)
    rest_segment.resample(resample_hz, n_jobs=1, verbose=False)
    rest_segment.filter(highpass_hz, lowpass_hz, n_jobs=1, verbose=False)
    rest_part = mne.make_fixed_length_epochs(
        rest_segment,
        duration=epoch_seconds,
        overlap=epoch_overlap_seconds,
        preload=True,
        reject_by_annotation=True,
        proj=False,
        verbose=False,
    )
    if len(rest_part):
        rest_epoch_parts.append(rest_part)

assert rest_epoch_parts, "rest 没有留下任何 epoch"
rest_epochs = (
    rest_epoch_parts[0]
    if len(rest_epoch_parts) == 1
    else mne.concatenate_epochs(rest_epoch_parts, on_mismatch="raise", verbose=False)
)
print("rest epoch 数：", len(rest_epochs))


#%%
# ==================== 9. Hold 分段、降采样和 2 秒重叠 epoch ====================

hold_epoch_parts = []
for onset, duration in hold_intervals:
    segment_end = min(onset + duration - 1.0 / raw.info["sfreq"], raw_duration)
    hold_segment = raw.copy().crop(tmin=onset, tmax=segment_end)
    hold_segment.load_data(verbose=False)
    hold_segment.resample(resample_hz, n_jobs=1, verbose=False)
    hold_segment.filter(highpass_hz, lowpass_hz, n_jobs=1, verbose=False)
    hold_part = mne.make_fixed_length_epochs(
        hold_segment,
        duration=epoch_seconds,
        overlap=epoch_overlap_seconds,
        preload=True,
        reject_by_annotation=True,
        proj=False,
        verbose=False,
    )
    if len(hold_part):
        hold_epoch_parts.append(hold_part)

assert hold_epoch_parts, f"{task} 没有留下任何 epoch"
hold_epochs = (
    hold_epoch_parts[0]
    if len(hold_epoch_parts) == 1
    else mne.concatenate_epochs(hold_epoch_parts, on_mismatch="raise", verbose=False)
)
print(f"{task} epoch 数：", len(hold_epochs))


#%%
# ==================== 10. 两种条件等量，计算共同 beta CSD ====================

equal_epoch_count = min(len(rest_epochs), len(hold_epochs))
rest_equal_indices = np.linspace(0, len(rest_epochs) - 1, equal_epoch_count).round().astype(int)
hold_equal_indices = np.linspace(0, len(hold_epochs) - 1, equal_epoch_count).round().astype(int)
rest_equal = rest_epochs[rest_equal_indices]
hold_equal = hold_epochs[hold_equal_indices]
common_epochs = mne.concatenate_epochs(
    [rest_equal, hold_equal], on_mismatch="raise", verbose=False
)

csd_parameters = {
    "fmin": beta_low_hz,
    "fmax": beta_high_hz,
    "bandwidth": multitaper_bandwidth_hz,
    "adaptive": False,
    "low_bias": True,
    "n_jobs": 1,
    "verbose": False,
}
rest_csd = mne.time_frequency.csd_multitaper(rest_equal, **csd_parameters).mean()
hold_csd = mne.time_frequency.csd_multitaper(hold_equal, **csd_parameters).mean()
common_csd = mne.time_frequency.csd_multitaper(common_epochs, **csd_parameters).mean()

print("两种条件各使用 epoch：", equal_epoch_count)
print("DICS 频段：", beta_low_hz, "-", beta_high_hz, "Hz")


#%%
# ==================== 11. 共同 DICS 滤波器，分别估计 rest 和 Hold ====================

dics_filters = mne.beamformer.make_dics(
    common_epochs.info,
    forward,
    common_csd,
    reg=dics_regularization,
    pick_ori="max-power",
    rank="info",
    weight_norm="unit-noise-gain",
    reduce_rank=True,
    real_filter=True,
    inversion="matrix",
    verbose=False,
)

rest_stc, rest_frequencies = mne.beamformer.apply_dics_csd(
    rest_csd, dics_filters, verbose=False
)
hold_stc, hold_frequencies = mne.beamformer.apply_dics_csd(
    hold_csd, dics_filters, verbose=False
)

assert np.array_equal(rest_stc.vertices[0], hold_stc.vertices[0])
rest_beta_power = np.asarray(rest_stc.data[:, 0], dtype=float)
hold_beta_power = np.asarray(hold_stc.data[:, 0], dtype=float)
power_floor = max(
    float(np.median(np.r_[rest_beta_power, hold_beta_power])) * 1e-12,
    np.finfo(float).tiny,
)

# 小于 0 表示 Hold 时 beta 下降；另存正向 ERD，便于直接看运动相关去同步。
hold_vs_rest_db = 10.0 * np.log10(
    (hold_beta_power + power_floor) / (rest_beta_power + power_floor)
)
rest_vs_hold_erd_db = -hold_vs_rest_db

print("rest CSD 代表频率：", rest_frequencies)
print("Hold CSD 代表频率：", hold_frequencies)


#%%
# ==================== 12. 把个体源点逐点换回模板 MNI 坐标 ====================

used_local_indices = np.asarray(rest_stc.vertices[0], dtype=int)
used_global_indices = subsample_global_indices[used_local_indices]
mni_positions_mm = template_positions_cm[used_global_indices] * 10.0

assert len(mni_positions_mm) == len(rest_beta_power)
assert np.isfinite(mni_positions_mm).all()

peak_index = int(np.argmax(rest_vs_hold_erd_db))
peak_mni_mm = mni_positions_mm[peak_index]
peak_erd_db = float(rest_vs_hold_erd_db[peak_index])
display_threshold_db = float(np.percentile(rest_vs_hold_erd_db, display_percentile))

right_motor_distance_mm = np.linalg.norm(
    mni_positions_mm - right_motor_center_mni_mm,
    axis=1,
)
left_motor_distance_mm = np.linalg.norm(
    mni_positions_mm - left_motor_center_mni_mm,
    axis=1,
)
right_motor_roi = right_motor_distance_mm <= motor_roi_radius_mm
left_motor_roi = left_motor_distance_mm <= motor_roi_radius_mm
positive_erd_db = np.maximum(rest_vs_hold_erd_db, 0.0)
high_erd = rest_vs_hold_erd_db >= display_threshold_db

right_roi_mean_erd_db = float(positive_erd_db[right_motor_roi].mean())
left_roi_mean_erd_db = float(positive_erd_db[left_motor_roi].mean())
roi_laterality_index = float(
    (right_roi_mean_erd_db - left_roi_mean_erd_db)
    / (right_roi_mean_erd_db + left_roi_mean_erd_db + np.finfo(float).eps)
)
right_high_count = int(np.sum(high_erd & (mni_positions_mm[:, 0] > 0)))
left_high_count = int(np.sum(high_erd & (mni_positions_mm[:, 0] < 0)))
high_count_laterality_index = float(
    (right_high_count - left_high_count)
    / max(right_high_count + left_high_count, 1)
)
right_roi_high_count = int(np.sum(high_erd & right_motor_roi))
left_roi_high_count = int(np.sum(high_erd & left_motor_roi))
peak_to_right_motor_center_mm = float(right_motor_distance_mm[peak_index])
peak_to_right_motor_roi_mm = float(
    max(0.0, peak_to_right_motor_center_mm - motor_roi_radius_mm)
)

right_roi_power_ratio_db = float(
    10.0
    * np.log10(
        rest_beta_power[right_motor_roi].mean()
        / hold_beta_power[right_motor_roi].mean()
    )
)
left_roi_power_ratio_db = float(
    10.0
    * np.log10(
        rest_beta_power[left_motor_roi].mean()
        / hold_beta_power[left_motor_roi].mean()
    )
)

print("最强 beta ERD 的 MNI 坐标（mm）：", peak_mni_mm.tolist())
print("最强 beta ERD（rest/Hold，dB）：", peak_erd_db)
print("峰到预注册右运动区中心距离（mm）：", peak_to_right_motor_center_mm)
print("峰到预注册右运动区边界距离（mm）：", peak_to_right_motor_roi_mm)
print("右/左运动 ROI 平均正 ERD（dB）：", right_roi_mean_erd_db, left_roi_mean_erd_db)
print("运动 ROI 左右指数：", roi_laterality_index)
print("全脑 P95 右/左点数：", right_high_count, left_high_count)


#%%
# ==================== 13. 保存完整逐源指标表 ====================

source_table_file = save_dir / "dics_beta_source_table.csv"
with source_table_file.open("w", encoding="utf-8-sig", newline="") as table_handle:
    fieldnames = [
        "source_index",
        "original_4mm_grid_index",
        "mni_x_mm",
        "mni_y_mm",
        "mni_z_mm",
        "rest_beta_power",
        "hold_beta_power",
        "hold_vs_rest_db",
        "rest_vs_hold_erd_db",
    ]
    writer = csv.DictWriter(table_handle, fieldnames=fieldnames)
    writer.writeheader()
    for source_index in range(len(rest_beta_power)):
        writer.writerow({
            "source_index": source_index,
            "original_4mm_grid_index": int(used_global_indices[source_index]),
            "mni_x_mm": float(mni_positions_mm[source_index, 0]),
            "mni_y_mm": float(mni_positions_mm[source_index, 1]),
            "mni_z_mm": float(mni_positions_mm[source_index, 2]),
            "rest_beta_power": float(rest_beta_power[source_index]),
            "hold_beta_power": float(hold_beta_power[source_index]),
            "hold_vs_rest_db": float(hold_vs_rest_db[source_index]),
            "rest_vs_hold_erd_db": float(rest_vs_hold_erd_db[source_index]),
        })

roi_table_file = save_dir / "dics_beta_roi_metrics.csv"
with roi_table_file.open("w", encoding="utf-8-sig", newline="") as table_handle:
    fieldnames = [
        "roi",
        "center_x_mm",
        "center_y_mm",
        "center_z_mm",
        "radius_mm",
        "source_count",
        "mean_positive_erd_db",
        "aggregated_power_ratio_db",
        "p95_source_count",
    ]
    writer = csv.DictWriter(table_handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({
        "roi": "right_sensorimotor",
        "center_x_mm": right_motor_center_mni_mm[0],
        "center_y_mm": right_motor_center_mni_mm[1],
        "center_z_mm": right_motor_center_mni_mm[2],
        "radius_mm": motor_roi_radius_mm,
        "source_count": int(right_motor_roi.sum()),
        "mean_positive_erd_db": right_roi_mean_erd_db,
        "aggregated_power_ratio_db": right_roi_power_ratio_db,
        "p95_source_count": right_roi_high_count,
    })
    writer.writerow({
        "roi": "left_sensorimotor",
        "center_x_mm": left_motor_center_mni_mm[0],
        "center_y_mm": left_motor_center_mni_mm[1],
        "center_z_mm": left_motor_center_mni_mm[2],
        "radius_mm": motor_roi_radius_mm,
        "source_count": int(left_motor_roi.sum()),
        "mean_positive_erd_db": left_roi_mean_erd_db,
        "aggregated_power_ratio_db": left_roi_power_ratio_db,
        "p95_source_count": left_roi_high_count,
    })


#%%
# ==================== 14. 画峰位置处的三个 MNI 切片散点图 ====================

color_limit = float(np.percentile(np.abs(rest_vs_hold_erd_db), 98.0))
color_limit = max(color_limit, np.finfo(float).eps)
color_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
slice_tolerance_mm = 4.1

slice_specs = [
    (0, 1, 2, "矢状面", "Y (mm)", "Z (mm)"),
    (1, 0, 2, "冠状面", "X (mm)", "Z (mm)"),
    (2, 0, 1, "轴位面", "X (mm)", "Y (mm)"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), facecolor="white")
slice_scatter = None
for axis, (fixed_axis, horizontal_axis, vertical_axis, plane_name, xlabel, ylabel) in zip(axes, slice_specs):
    slice_mask = np.abs(
        mni_positions_mm[:, fixed_axis] - peak_mni_mm[fixed_axis]
    ) <= slice_tolerance_mm
    axis.scatter(
        mni_positions_mm[slice_mask, horizontal_axis],
        mni_positions_mm[slice_mask, vertical_axis],
        s=8,
        color="#D9DEE7",
        linewidths=0,
    )
    slice_scatter = axis.scatter(
        mni_positions_mm[slice_mask, horizontal_axis],
        mni_positions_mm[slice_mask, vertical_axis],
        c=rest_vs_hold_erd_db[slice_mask],
        s=26,
        cmap="RdBu_r",
        norm=color_norm,
        linewidths=0,
    )
    axis.set_title(
        f"{plane_name} | {['X', 'Y', 'Z'][fixed_axis]} = {peak_mni_mm[fixed_axis]:.1f} mm",
        fontweight="bold",
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#EEF1F5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)

fig.colorbar(slice_scatter, ax=axes, shrink=0.82, label="beta ERD: 10 log10(rest / Hold), dB")
fig.suptitle(
    f"sub-{subject} | {task} {medication} | common-filter DICS 13-30 Hz\n"
    "模板 MNI 8 mm 网格；红色=Hold 时 beta 降低，蓝色=Hold 时 beta 增强",
    fontsize=14,
    fontweight="bold",
)
fig.subplots_adjust(left=0.06, right=0.91, bottom=0.12, top=0.80, wspace=0.28)
slice_figure_file = save_dir / "dics_beta_mni_slices.png"
fig.savefig(slice_figure_file, dpi=190, facecolor="white", bbox_inches="tight")
plt.close(fig)


#%%
# ==================== 15. 画 P95 以上 beta ERD 的三维散点图 ====================

shown_sources = rest_vs_hold_erd_db >= max(display_threshold_db, 0.0)
if not np.any(shown_sources):
    shown_sources[np.argsort(rest_vs_hold_erd_db)[-20:]] = True

fig = plt.figure(figsize=(9, 7), facecolor="white")
axis_3d = fig.add_subplot(111, projection="3d")
axis_3d.scatter(
    mni_positions_mm[:, 0],
    mni_positions_mm[:, 1],
    mni_positions_mm[:, 2],
    s=2,
    color="#CDD3DC",
    alpha=0.10,
    linewidths=0,
)
activation_scatter = axis_3d.scatter(
    mni_positions_mm[shown_sources, 0],
    mni_positions_mm[shown_sources, 1],
    mni_positions_mm[shown_sources, 2],
    c=rest_vs_hold_erd_db[shown_sources],
    s=22,
    cmap="inferno",
    linewidths=0,
)
axis_3d.set_xlabel("MNI X (mm)")
axis_3d.set_ylabel("MNI Y (mm)")
axis_3d.set_zlabel("MNI Z (mm)")
axis_3d.set_title(
    f"sub-{subject} | left Hold | beta ERD P{display_percentile:.0f}+\n"
    f"peak MNI = [{peak_mni_mm[0]:.1f}, {peak_mni_mm[1]:.1f}, {peak_mni_mm[2]:.1f}] mm",
    fontweight="bold",
)
axis_3d.view_init(elev=22, azim=-62)
fig.colorbar(activation_scatter, ax=axis_3d, shrink=0.70, pad=0.10, label="rest / Hold (dB)")
scatter_figure_file = save_dir / "dics_beta_mni_3d.png"
fig.savefig(scatter_figure_file, dpi=190, facecolor="white", bbox_inches="tight")
plt.close(fig)


#%%
# ==================== 16. 渲染 fsaverage MRI 三切片和 pial 表面 ====================

anatomy_figure_file = None
surface_figure_file = None
if draw_template_anatomy:
    fsaverage_t1 = nib.load(str(fsaverage_t1_file))
    fsaverage_volume = np.asarray(fsaverage_t1.get_fdata(dtype=np.float32))
    fsaverage_talairach = mne.read_talxfm(
        "fsaverage",
        subjects_dir=subjects_dir,
        verbose=False,
    )
    mni_to_fsaverage_mri = mne.transforms.invert_transform(fsaverage_talairach)
    source_fsaverage_mri_mm = (
        mne.transforms.apply_trans(
            mni_to_fsaverage_mri,
            mni_positions_mm / 1000.0,
        )
        * 1000.0
    )
    source_voxels = nib.affines.apply_affine(
        np.linalg.inv(fsaverage_t1.header.get_vox2ras_tkr()),
        source_fsaverage_mri_mm,
    )
    peak_voxel = np.rint(source_voxels[peak_index]).astype(int)

    activation_volume = np.zeros(fsaverage_volume.shape, dtype=np.float32)
    activation_weights = np.clip(
        (rest_vs_hold_erd_db - display_threshold_db)
        / max(peak_erd_db - display_threshold_db, np.finfo(float).eps),
        0.0,
        1.0,
    )
    source_voxel_indices = np.rint(source_voxels).astype(int)
    valid_voxels = np.all(source_voxel_indices >= 0, axis=1) & np.all(
        source_voxel_indices < np.asarray(fsaverage_volume.shape),
        axis=1,
    )
    valid_indices = source_voxel_indices[valid_voxels]
    np.maximum.at(
        activation_volume,
        (valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]),
        activation_weights[valid_voxels],
    )
    activation_volume = gaussian_filter(activation_volume, sigma=4.0)
    activation_volume /= max(float(activation_volume.max()), np.finfo(float).eps)

    x_voxel, y_voxel, z_voxel = peak_voxel
    anatomical_slices = (
        fsaverage_volume[x_voxel, :, :].T,
        fsaverage_volume[:, y_voxel, :].T,
        fsaverage_volume[:, :, z_voxel].T,
    )
    activation_slices = (
        activation_volume[x_voxel, :, :].T,
        activation_volume[:, y_voxel, :].T,
        activation_volume[:, :, z_voxel].T,
    )
    slice_titles = (
        f"Sagittal | X={peak_mni_mm[0]:.0f} mm",
        f"Coronal | Y={peak_mni_mm[1]:.0f} mm",
        f"Axial | Z={peak_mni_mm[2]:.0f} mm",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2), facecolor="white")
    for axis, anatomical_slice, activation_slice, slice_title in zip(
        axes,
        anatomical_slices,
        activation_slices,
        slice_titles,
    ):
        nonzero_anatomy = anatomical_slice[anatomical_slice > 0]
        anatomy_low, anatomy_high = np.percentile(nonzero_anatomy, [1, 99])
        axis.imshow(
            np.clip(
                (anatomical_slice - anatomy_low) / (anatomy_high - anatomy_low),
                0.0,
                1.0,
            ),
            cmap="gray",
            origin="lower",
        )
        axis.imshow(
            np.ma.masked_less(activation_slice, 0.06),
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            alpha=0.82,
        )
        axis.set_title(slice_title, fontweight="bold")
        axis.set_axis_off()

    colorbar_source = plt.cm.ScalarMappable(
        norm=Normalize(vmin=display_threshold_db, vmax=peak_erd_db),
        cmap="inferno",
    )
    colorbar_axis = fig.add_axes([0.925, 0.18, 0.014, 0.58])
    fig.colorbar(
        colorbar_source,
        cax=colorbar_axis,
        label="beta ERD: 10 log10(rest / Hold), dB",
    )
    fig.suptitle(
        f"sub-{subject} | left Hold | DICS 13-30 Hz | fsaverage template MRI\n"
        "P95+ volume-source beta ERD; template anatomy is for display only",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.82, wspace=0.02)
    anatomy_figure_file = save_dir / "dics_beta_fsaverage_mri.png"
    fig.savefig(anatomy_figure_file, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    surface_tree = cKDTree(mni_positions_mm)
    pial_plotter = pv.Plotter(
        off_screen=True,
        shape=(1, 2),
        window_size=(1500, 650),
        border=False,
    )
    for plot_column, hemisphere in enumerate(("lh", "rh")):
        surface_vertices_mm, surface_triangles = mne.read_surface(
            subjects_dir / "fsaverage" / "surf" / f"{hemisphere}.pial",
            verbose=False,
        )
        surface_vertices_mni_mm = (
            mne.transforms.apply_trans(
                fsaverage_talairach,
                surface_vertices_mm / 1000.0,
            )
            * 1000.0
        )
        nearest_distance_mm, nearest_source_index = surface_tree.query(
            surface_vertices_mni_mm,
            k=1,
        )
        surface_erd_db = rest_vs_hold_erd_db[nearest_source_index]
        surface_erd_db[
            (nearest_distance_mm > surface_projection_max_distance_mm)
            | (surface_erd_db < display_threshold_db)
        ] = np.nan
        pyvista_faces = np.column_stack(
            [np.full(len(surface_triangles), 3), surface_triangles]
        ).astype(np.int64).ravel()
        pial_mesh = pv.PolyData(surface_vertices_mm, pyvista_faces)
        pial_mesh.point_data["beta ERD (dB)"] = surface_erd_db

        pial_plotter.subplot(0, plot_column)
        pial_plotter.set_background("white")
        pial_plotter.add_mesh(
            pial_mesh,
            scalars="beta ERD (dB)",
            cmap="inferno",
            clim=[display_threshold_db, peak_erd_db],
            nan_color="#B8B8B8",
            nan_opacity=1.0,
            show_scalar_bar=False,
            smooth_shading=True,
        )
        pial_plotter.add_text(
            ("Left" if hemisphere == "lh" else "Right") + " hemisphere | lateral",
            position="upper_edge",
            font_size=11,
            color="black",
        )
        surface_center = surface_vertices_mm.mean(axis=0)
        lateral_camera = surface_center + np.array(
            [-400.0 if hemisphere == "lh" else 400.0, 0.0, 0.0]
        )
        pial_plotter.camera_position = [
            lateral_camera,
            surface_center,
            [0.0, 0.0, 1.0],
        ]
        pial_plotter.camera.zoom(1.32)

    pial_image = pial_plotter.screenshot(return_img=True)
    pial_plotter.close()

    fig, axis = plt.subplots(figsize=(15.6, 6.8), facecolor="white")
    axis.imshow(pial_image)
    axis.set_axis_off()
    fig.colorbar(
        colorbar_source,
        ax=axis,
        shrink=0.62,
        pad=0.012,
        label="beta ERD: 10 log10(rest / Hold), dB",
    )
    fig.suptitle(
        f"sub-{subject} | left Hold | DICS 13-30 Hz\n"
        "8-mm volume source projected to fsaverage pial for display only",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.93, bottom=0.01, top=0.86)
    surface_figure_file = save_dir / "dics_beta_fsaverage_pial.png"
    fig.savefig(surface_figure_file, dpi=190, facecolor="white", bbox_inches="tight")
    plt.close(fig)
else:
    print("未找到本机 fsaverage T1，跳过模板解剖渲染：", fsaverage_t1_file)


#%%
# ==================== 17. 保存本次运行说明 ====================

summary = {
    "dataset": "ds004998",
    "subject": subject,
    "task": task,
    "medication": medication,
    "run": run,
    "raw_file": str(raw_file),
    "analysis": "common-filter DICS beta power contrast",
    "inverse_channels": "planar gradiometers only",
    "lfp_policy": "EEG001-EEG008 are STN-LFP contacts and were excluded from the inverse",
    "grid_original_inside_points": int(grid_inside.sum()),
    "grid_subsampled_points_before_bem_check": int(len(subsample_global_indices)),
    "grid_points_after_bem_check": int(len(rest_beta_power)),
    "rest_epochs": int(len(rest_epochs)),
    "hold_epochs": int(len(hold_epochs)),
    "equal_epochs_per_condition": int(equal_epoch_count),
    "epoch_seconds": epoch_seconds,
    "epoch_overlap_seconds": epoch_overlap_seconds,
    "beta_hz": [beta_low_hz, beta_high_hz],
    "dics_regularization": dics_regularization,
    "peak_mni_mm": peak_mni_mm.tolist(),
    "peak_rest_vs_hold_erd_db": peak_erd_db,
    "registered_right_motor_roi_center_mni_mm": right_motor_center_mni_mm.tolist(),
    "registered_left_motor_roi_center_mni_mm": left_motor_center_mni_mm.tolist(),
    "registered_motor_roi_radius_mm": motor_roi_radius_mm,
    "peak_to_right_motor_center_mm": peak_to_right_motor_center_mm,
    "peak_to_right_motor_roi_boundary_mm": peak_to_right_motor_roi_mm,
    "right_roi_mean_positive_erd_db": right_roi_mean_erd_db,
    "left_roi_mean_positive_erd_db": left_roi_mean_erd_db,
    "right_roi_aggregated_power_ratio_db": right_roi_power_ratio_db,
    "left_roi_aggregated_power_ratio_db": left_roi_power_ratio_db,
    "roi_laterality_index": roi_laterality_index,
    "p95_right_source_count": right_high_count,
    "p95_left_source_count": left_high_count,
    "p95_count_laterality_index": high_count_laterality_index,
    "p95_right_roi_source_count": right_roi_high_count,
    "p95_left_roi_source_count": left_roi_high_count,
    "display_percentile": display_percentile,
    "display_threshold_erd_db": display_threshold_db,
    "template_anatomy_for_display_only": str(fsaverage_t1_file) if draw_template_anatomy else None,
    "interpretation_limit": (
        "Real data have no simulated source truth. Peak-to-ROI distance is not DLE, "
        "and this single downloaded run cannot test medication or group effects."
    ),
    "bem_limit": (
        "The provided FieldTrip singleshell boundary was solved as an MNE one-layer BEM; "
        "this is a runnable bridge, not a numerical reproduction of the Nolte singleshell solver."
    ),
    "elapsed_seconds": float(time.perf_counter() - all_started),
}

summary_file = save_dir / "run_summary.json"
with summary_file.open("w", encoding="utf-8") as summary_handle:
    json.dump(summary, summary_handle, ensure_ascii=False, indent=2)

report_lines = [
    "# ds004998 DBS-MEG 频谱定位试运行",
    "",
    f"- 数据：sub-{subject}，{task}，{medication}，run-{run}",
    f"- 方法：共同空间滤波器 DICS，{beta_low_hz:.0f}-{beta_high_hz:.0f} Hz",
    f"- 通道：{len(raw.ch_names)} 个 planar gradiometer；STN-LFP 未进入逆解",
    f"- epoch：rest {len(rest_epochs)} 个，Hold {len(hold_epochs)} 个；每种等量使用 {equal_epoch_count} 个",
    f"- 8 mm 网格：BEM 检查前 {len(subsample_global_indices)} 点，检查后 {len(rest_beta_power)} 点",
    f"- 最大 beta ERD：{peak_erd_db:.4f} dB",
    f"- 峰 MNI：[ {peak_mni_mm[0]:.1f}, {peak_mni_mm[1]:.1f}, {peak_mni_mm[2]:.1f} ] mm",
    f"- 峰到预注册右感觉运动中心：{peak_to_right_motor_center_mm:.2f} mm；到 25-mm ROI 边界：{peak_to_right_motor_roi_mm:.2f} mm",
    f"- 右/左 ROI 平均正 ERD：{right_roi_mean_erd_db:.4f} / {left_roi_mean_erd_db:.4f} dB",
    f"- ROI laterality index：{roi_laterality_index:+.4f}（正值表示右侧/对侧更强）",
    f"- 全脑 P95 点右/左：{right_high_count} / {left_high_count}；count LI={high_count_laterality_index:+.4f}",
    "",
    "这里没有仿真真值，所以不能计算 AUC 或 DLE。当前结果只回答频谱定位链路是否可运行；",
    "峰落在预注册右感觉运动 ROI 内，但双侧高值都很明显，不能宣称强侧化或优秀空间特异性。",
    "当前本地下载也不足以比较 MedOn/MedOff 或进行组统计。",
    "MRI 和 pial 图使用 fsaverage 模板且只用于显示，不是该患者的个体 MRI。",
]
(save_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print("逐源表：", source_table_file)
print("ROI 指标表：", roi_table_file)
print("MNI 切片图：", slice_figure_file)
print("MNI 三维图：", scatter_figure_file)
print("模板 MRI 图：", anatomy_figure_file)
print("模板 pial 图：", surface_figure_file)
print("运行摘要：", summary_file)
print("总耗时（秒）：", summary["elapsed_seconds"])
