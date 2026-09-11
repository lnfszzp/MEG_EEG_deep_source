#%%
# 这个脚本一次验证四类 ERP 任务：听觉、视觉、腕部感觉刺激、手指反应。
# MNE sample 中听觉和视觉各分左右；ds006035 中腕部刺激分 N20/P30。
# 写法按你的习惯：参数放在最前面，然后从上到下一块一块运行，不定义自己的函数。
# ROI 只在最后评价时读取，不参与逆解、峰时刻选择或参数选择。

from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.cov import compute_whitener
from mne.forward import _merge_fwds
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected
from pipelines import run_ds006035_somatomotor as real_pipeline


mne.set_log_level("warning")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#%%
# ==================== 1. 参数区：平时主要改这里 ====================

sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
sample_raw_file = sample_data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
sample_forward_file = sample_data_path / "MEG" / "sample" / "sample_audvis-meg-eeg-oct-6-fwd.fif"

ds006035_path = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
ds_subject = "sm09"
ds_run = 1

subjects_dir = sample_data_path / "subjects"
sample_subject = "sample"

filter_low = 1.0
sample_filter_high = 40.0
ds_filter_high = 100.0
lambda2 = 1.0 / 9.0
depth = 0.8

# 所有峰时刻都根据全传感器的白化后 GFP 选择，不读取脑区标签。
auditory_window = (0.080, 0.120)
visual_window = (0.080, 0.150)
wrist_n20_window = (0.018, 0.024)
wrist_p30_window = (0.028, 0.040)
finger_window = (0.020, 0.080)

method_names = ["OASTER ERP", "dSPM", "eLORETA"]
method_colors = {
    "OASTER ERP": "#0072B2",
    "dSPM": "#D55E00",
    "eLORETA": "#009E73",
}
method_markers = {"OASTER ERP": "s", "dSPM": "^", "eLORETA": "D"}
draw_brain = True

save_dir = project_root / "results" / "real_data" / "erp_four_paradigms_v2"
save_dir.mkdir(parents=True, exist_ok=True)

print("结果保存到：", save_dir)
assert sample_raw_file.is_file(), f"找不到 sample raw：{sample_raw_file}"
assert sample_forward_file.is_file(), f"找不到 sample forward：{sample_forward_file}"
assert (subjects_dir / sample_subject).is_dir(), f"找不到 sample 解剖：{subjects_dir}"


#%%
# ==================== 2. 读取 MNE sample，并保留 EEG 和 magnetometer ====================

all_started = time.perf_counter()

sample_raw = mne.io.read_raw_fif(sample_raw_file, preload=False, verbose=False)
sample_picks = mne.pick_types(
    sample_raw.info,
    meg="mag",
    eeg=True,
    eog=False,
    ecg=False,
    stim=True,
    ref_meg=False,
    exclude="bads",
)
sample_raw.pick(sample_picks)
sample_raw.load_data(verbose=False)
sample_raw.filter(filter_low, sample_filter_high, n_jobs=1, verbose=False)

sample_events = mne.find_events(sample_raw, stim_channel="STI 014", verbose=False)
sample_events = sample_events[np.isin(sample_events[:, 2], [1, 2, 3, 4])]

sample_event_id = {
    "auditory_left": 1,
    "auditory_right": 2,
    "visual_left": 3,
    "visual_right": 4,
}

print("sample 通道数：", len(sample_raw.ch_names))
print("sample 四种事件数：", {
    name: int(np.sum(sample_events[:, 2] == value))
    for name, value in sample_event_id.items()
})


#%%
# ==================== 3. sample 分段并进行叠加平均 ====================
# 下面四行 average() 就是四种刺激各自的 ERP/ERF 叠加平均。

sample_epochs = mne.Epochs(
    sample_raw,
    sample_events,
    event_id=sample_event_id,
    tmin=-0.20,
    tmax=0.50,
    baseline=(-0.20, 0.0),
    reject={"eeg": 150e-6, "mag": 5e-12},
    reject_tmin=-0.10,
    reject_tmax=0.30,
    preload=True,
    proj=False,
    event_repeated="drop",
    verbose=False,
)
sample_epochs.set_eeg_reference("average", projection=True, verbose=False)

sample_auditory_left = sample_epochs["auditory_left"].average()
sample_auditory_right = sample_epochs["auditory_right"].average()
sample_visual_left = sample_epochs["visual_left"].average()
sample_visual_right = sample_epochs["visual_right"].average()

sample_evoked_list = [
    sample_auditory_left,
    sample_auditory_right,
    sample_visual_left,
    sample_visual_right,
]
sample_task_names = [
    "听觉-左侧刺激",
    "听觉-右侧刺激",
    "视觉-左侧视野",
    "视觉-右侧视野",
]
sample_windows = [
    auditory_window,
    auditory_window,
    visual_window,
    visual_window,
]

print("sample 平均试次数：", {
    name: evoked.nave for name, evoked in zip(sample_task_names, sample_evoked_list)
})
assert min(evoked.nave for evoked in sample_evoked_list) >= 40


#%%
# ==================== 4. sample 基线噪声 ====================
# 噪声来自所有单试次的刺激前基线，不从平均后的 ERP 反推。

sample_noise_cov = mne.compute_covariance(
    sample_epochs,
    tmin=-0.20,
    tmax=-0.01,
    method="shrunk",
    rank="info",
    verbose=False,
)

sample_noise_trials = sample_epochs.get_data(
    tmin=-0.20,
    tmax=-0.01,
    copy=True,
    verbose=False,
)
sample_noise = sample_noise_trials.transpose(1, 0, 2).reshape(
    len(sample_epochs.ch_names), -1
)

print("sample 单试次基线样本数：", sample_noise.shape[1])


#%%
# ==================== 5. sample forward：EEG 和 MAG 使用同一皮层网格 ====================

sample_forward_all = mne.read_forward_solution(sample_forward_file, verbose=False)

sample_eeg_forward_free = mne.pick_types_forward(
    sample_forward_all,
    meg=False,
    eeg=True,
    ref_meg=False,
    exclude="bads",
)
sample_mag_forward_free = mne.pick_types_forward(
    sample_forward_all,
    meg="mag",
    eeg=False,
    ref_meg=False,
    exclude="bads",
)

sample_eeg_forward_free = mne.pick_channels_forward(
    sample_eeg_forward_free,
    include=[
        channel_name
        for channel_name in sample_eeg_forward_free["sol"]["row_names"]
        if channel_name in sample_epochs.ch_names
    ],
    ordered=True,
    copy=True,
)
sample_mag_forward_free = mne.pick_channels_forward(
    sample_mag_forward_free,
    include=[
        channel_name
        for channel_name in sample_mag_forward_free["sol"]["row_names"]
        if channel_name in sample_epochs.ch_names
    ],
    ordered=True,
    copy=True,
)

sample_eeg_forward_free = mne.convert_forward_solution(
    sample_eeg_forward_free,
    surf_ori=True,
    force_fixed=False,
    use_cps=True,
    verbose=False,
)
sample_mag_forward_free = mne.convert_forward_solution(
    sample_mag_forward_free,
    surf_ori=True,
    force_fixed=False,
    use_cps=True,
    verbose=False,
)

sample_eeg_forward_fixed = mne.convert_forward_solution(
    sample_eeg_forward_free,
    surf_ori=True,
    force_fixed=True,
    use_cps=True,
    verbose=False,
)
sample_mag_forward_fixed = mne.convert_forward_solution(
    sample_mag_forward_free,
    surf_ori=True,
    force_fixed=True,
    use_cps=True,
    verbose=False,
)

for hemi in range(2):
    assert np.array_equal(
        sample_eeg_forward_fixed["src"][hemi]["vertno"],
        sample_mag_forward_fixed["src"][hemi]["vertno"],
    )

print("sample 源点数：", sample_eeg_forward_fixed["nsource"])

sample_analysis_src = sample_eeg_forward_fixed["src"]
sample_source_xyz = np.vstack([
    hemi["rr"][hemi["vertno"]]
    for hemi in sample_analysis_src
])
sample_source_adjacency = mne.spatial_src_adjacency(
    sample_analysis_src,
    verbose=False,
).toarray()
sample_surface_kernels = protected.connected_euclidean_surface_kernels(
    sample_source_xyz,
    sample_source_adjacency,
    len(sample_source_xyz),
    scales_mm=oaster.SURFACE_SCALES_MM,
)
del sample_source_adjacency

print("sample OASTER 空间模板尺度：", [scale for scale, _ in sample_surface_kernels])


#%%
# ==================== 6. sample 对齐通道，并显式计算 EEG/MAG 白化 ====================

sample_reference_evoked = sample_evoked_list[0]

sample_eeg_reference, sample_eeg_gain, sample_eeg_noise = real_pipeline._aligned_modality(
    sample_reference_evoked.data,
    sample_noise,
    sample_reference_evoked.info,
    sample_eeg_forward_fixed,
    eeg=True,
)
sample_mag_reference, sample_mag_gain, sample_mag_noise = real_pipeline._aligned_modality(
    sample_reference_evoked.data,
    sample_noise,
    sample_reference_evoked.info,
    sample_mag_forward_fixed,
    eeg=False,
)

sample_eeg_channel_names = sample_eeg_forward_fixed["sol"]["row_names"]
sample_mag_channel_names = sample_mag_forward_fixed["sol"]["row_names"]
sample_eeg_order = np.asarray([
    sample_reference_evoked.ch_names.index(channel_name)
    for channel_name in sample_eeg_channel_names
])
sample_mag_order = np.asarray([
    sample_reference_evoked.ch_names.index(channel_name)
    for channel_name in sample_mag_channel_names
])
sample_eeg_info = mne.pick_info(sample_reference_evoked.info, sample_eeg_order, copy=True)
sample_mag_info = mne.pick_info(sample_reference_evoked.info, sample_mag_order, copy=True)
sample_eeg_cov = mne.pick_channels_cov(
    sample_noise_cov,
    include=sample_eeg_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)
sample_mag_cov = mne.pick_channels_cov(
    sample_noise_cov,
    include=sample_mag_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)

sample_eeg_whitener, sample_eeg_white_names, sample_eeg_rank = compute_whitener(
    sample_eeg_cov,
    sample_eeg_info,
    rank="info",
    pca=True,
    return_rank=True,
    verbose=False,
)
sample_mag_whitener, sample_mag_white_names, sample_mag_rank = compute_whitener(
    sample_mag_cov,
    sample_mag_info,
    rank="info",
    pca=True,
    return_rank=True,
    verbose=False,
)

assert sample_eeg_white_names == sample_eeg_channel_names
assert sample_mag_white_names == sample_mag_channel_names

sample_eeg_gain_white = sample_eeg_whitener @ sample_eeg_gain
sample_mag_gain_white = sample_mag_whitener @ sample_mag_gain
sample_joint_gain_white = np.vstack((sample_eeg_gain_white, sample_mag_gain_white))

print("sample 白化 EEG Gain：", sample_eeg_gain_white.shape)
print("sample 白化 MAG Gain：", sample_mag_gain_white.shape)
print("sample EEG/MAG 白化秩：", sample_eeg_rank, sample_mag_rank)


#%%
# ==================== 7. sample 多尺度 OASTER-ERP ====================
# 每个任务只把预先写在参数区的时间窗交给 OASTER；ROI 此时尚未读取。
# EEG/MAG 权重只根据各自 active 相对基线的传感器功率计算。

sample_records = []

for task_name, task_window, evoked in zip(
    sample_task_names,
    sample_windows,
    sample_evoked_list,
):
    sample_eeg_data, sample_eeg_gain_check, sample_eeg_noise_check = real_pipeline._aligned_modality(
        evoked.data,
        sample_noise,
        evoked.info,
        sample_eeg_forward_fixed,
        eeg=True,
    )
    sample_mag_data, sample_mag_gain_check, sample_mag_noise_check = real_pipeline._aligned_modality(
        evoked.data,
        sample_noise,
        evoked.info,
        sample_mag_forward_fixed,
        eeg=False,
    )

    assert np.allclose(sample_eeg_gain_check, sample_eeg_gain)
    assert np.allclose(sample_mag_gain_check, sample_mag_gain)

    sample_eeg_data_white = sample_eeg_whitener @ sample_eeg_data
    sample_mag_data_white = sample_mag_whitener @ sample_mag_data
    sample_joint_data_white = np.vstack((sample_eeg_data_white, sample_mag_data_white))

    sample_erp_baseline = (
        (evoked.times >= -0.20)
        & (evoked.times <= -0.01)
    )
    sample_erp_active = (
        (evoked.times >= task_window[0])
        & (evoked.times <= task_window[1])
    )
    sample_modality_weights = real_pipeline.modality_evidence_weights(
        (sample_eeg_data, sample_mag_data),
        sample_erp_baseline,
        sample_erp_active,
    )
    sample_channel_weights = np.r_[
        np.full(sample_eeg_data_white.shape[0], sample_modality_weights[0]),
        np.full(sample_mag_data_white.shape[0], sample_modality_weights[1]),
    ]

    oaster_source, oaster_information = oaster.reconstruct_evoked_oaster_from_whitened(
        sample_joint_data_white,
        sample_joint_gain_white,
        len(sample_source_xyz),
        sample_surface_kernels,
        baseline=sample_erp_baseline,
        active_windows=(sample_erp_active,),
        window_channel_weights=(sample_channel_weights,),
    )
    oaster_information["modality_weights"] = {
        "EEG": float(sample_modality_weights[0]),
        "MAG": float(sample_modality_weights[1]),
    }

    sample_records.append({
        "dataset": "MNE sample",
        "task": task_name,
        "window": task_window,
        "baseline": (-0.20, -0.01),
        "n_trials": evoked.nave,
        "times": evoked.times.copy(),
        "sensor_white": sample_joint_data_white,
        "OASTER ERP": oaster_source,
    })

    print(task_name, "OASTER：", oaster_source.shape, oaster_information)


#%%
# ==================== 8. sample 官方 dSPM 和 eLORETA ====================

sample_joint_forward = _merge_fwds(
    {
        "meg": sample_mag_forward_free.copy(),
        "eeg": sample_eeg_forward_free.copy(),
    },
    verbose=False,
)
sample_joint_channel_names = sample_joint_forward["sol"]["row_names"]
sample_inverse_evoked = sample_evoked_list[0].copy().reorder_channels(
    sample_joint_channel_names
)
sample_joint_forward["info"] = sample_inverse_evoked.info

sample_joint_cov = mne.pick_channels_cov(
    sample_noise_cov,
    include=sample_joint_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)

sample_inverse = mne.minimum_norm.make_inverse_operator(
    sample_inverse_evoked.info,
    sample_joint_forward,
    sample_joint_cov,
    loose=0.0,
    fixed=True,
    depth=depth,
    rank="info",
    use_cps=True,
    verbose=False,
)

for record, evoked in zip(sample_records, sample_evoked_list):
    sample_inverse_evoked = evoked.copy().reorder_channels(sample_joint_channel_names)

    sample_dspm = mne.minimum_norm.apply_inverse(
        sample_inverse_evoked,
        sample_inverse,
        lambda2=lambda2,
        method="dSPM",
        use_cps=True,
        verbose=False,
    )
    sample_eloreta = mne.minimum_norm.apply_inverse(
        sample_inverse_evoked,
        sample_inverse,
        lambda2=lambda2,
        method="eLORETA",
        use_cps=True,
        verbose=False,
    )

    record["dSPM"] = sample_dspm.data
    record["eLORETA"] = sample_eloreta.data

    assert record["OASTER ERP"].shape == record["dSPM"].shape
    assert record["dSPM"].shape == record["eLORETA"].shape

print("sample 三种方法已经完成")


#%%
# ==================== 9. 读取 ds006035 的腕部刺激和左指反应事件 ====================

ds_stem = f"sub-{ds_subject}_ses-meeg_task-somatomotor_run-{ds_run}"
ds_data_folder = ds006035_path / f"sub-{ds_subject}" / "ses-meeg" / "meg"
ds_raw_file = ds_data_folder / f"{ds_stem}_meg.fif"
ds_events_file = ds_data_folder / f"{ds_stem}_events.tsv"

assert ds_raw_file.is_file(), f"找不到 ds006035 raw：{ds_raw_file}"
assert ds_events_file.is_file(), f"找不到 ds006035 events：{ds_events_file}"

ds_raw = mne.io.read_raw_fif(ds_raw_file, preload=False, verbose=False)
ds_first_samp = int(ds_raw.first_samp)
ds_picks = mne.pick_types(
    ds_raw.info,
    meg="mag",
    eeg=True,
    eog=False,
    ecg=False,
    stim=False,
    ref_meg=False,
    exclude="bads",
)
ds_raw.pick(ds_picks)
ds_raw.load_data(verbose=False)

ds_line_frequency = float(ds_raw.info.get("line_freq") or 60.0)
if ds_line_frequency < ds_raw.info["sfreq"] / 2.0:
    ds_raw.notch_filter([ds_line_frequency], n_jobs=1, verbose=False)
ds_raw.filter(filter_low, ds_filter_high, n_jobs=1, verbose=False)

ds_event_table = pd.read_csv(ds_events_file, sep="\t")
ds_event_value = pd.to_numeric(ds_event_table["value"], errors="coerce")
ds_event_table = ds_event_table[ds_event_value.isin([16, 32])].copy()
ds_event_table["value"] = pd.to_numeric(ds_event_table["value"]).astype(int)
ds_event_table = ds_event_table.sort_values("sample")

ds_events = np.zeros((len(ds_event_table), 3), dtype=int)
ds_events[:, 0] = (
    ds_event_table["sample"].astype(float).astype(int).to_numpy() + ds_first_samp
)
ds_events[:, 2] = ds_event_table["value"].to_numpy()

print("右腕刺激 event32：", int(np.sum(ds_events[:, 2] == 32)))
print("左指反应 event16：", int(np.sum(ds_events[:, 2] == 16)))
assert np.all(np.diff(ds_events[:, 0]) > 0)


#%%
# ==================== 10. ds006035 坏道检查 ====================
# 坏道 probe 只用 event32，避免 Finger 前已有腕部刺激污染这一步的统计。

ds_epoch_parameter = dict(
    raw=ds_raw,
    events=ds_events,
    event_id={"right_wrist": 32},
    tmin=-0.40,
    tmax=0.15,
    baseline=(-0.40, -0.05),
    reject_tmin=-0.20,
    reject_tmax=0.10,
    preload=True,
    proj=False,
    event_repeated="drop",
    verbose=False,
)

ds_probe = mne.Epochs(**ds_epoch_parameter)
ds_eeg_probe = mne.pick_types(ds_probe.info, meg=False, eeg=True, exclude=[])
ds_baseline_window = (ds_probe.times >= -0.35) & (ds_probe.times <= -0.05)
ds_eeg_ptp = np.median(
    np.ptp(
        ds_probe.get_data(copy=False)[:, ds_eeg_probe][:, :, ds_baseline_window],
        axis=2,
    ),
    axis=0,
)
ds_ptp_center = float(np.median(ds_eeg_ptp))
ds_ptp_mad = float(np.median(np.abs(ds_eeg_ptp - ds_ptp_center)))
ds_ptp_cutoff = max(100e-6, ds_ptp_center + 6.0 * 1.4826 * ds_ptp_mad)
ds_auto_bad_channels = [
    ds_probe.ch_names[index]
    for index, value in zip(ds_eeg_probe, ds_eeg_ptp)
    if value > ds_ptp_cutoff
]

if ds_auto_bad_channels:
    ds_raw.drop_channels(ds_auto_bad_channels)

print("自动删除 EEG 坏道：", ds_auto_bad_channels)


#%%
# ==================== 11. 腕部刺激和 Finger 分开分段、分开计算基线 ====================
# Finger 发生前不久有腕部电刺激，不能把 -0.4~-0.05 秒当作 Finger 基线。
# 因此 Finger 单独取更早的 -0.55~-0.35 秒，避免把此前刺激反应混入噪声。

ds_wrist_epochs = mne.Epochs(
    ds_raw,
    ds_events,
    event_id={"right_wrist": 32},
    tmin=-0.40,
    tmax=0.15,
    baseline=(-0.40, -0.05),
    reject={"eeg": 200e-6, "mag": 6e-12},
    reject_tmin=-0.20,
    reject_tmax=0.10,
    preload=True,
    proj=False,
    event_repeated="drop",
    verbose=False,
)
ds_finger_epochs = mne.Epochs(
    ds_raw,
    ds_events,
    event_id={"left_finger": 16},
    tmin=-0.55,
    tmax=0.15,
    baseline=(-0.55, -0.35),
    reject={"eeg": 200e-6, "mag": 6e-12},
    reject_tmin=-0.55,
    reject_tmax=-0.35,
    preload=True,
    proj=False,
    event_repeated="drop",
    verbose=False,
)
ds_wrist_epochs.set_eeg_reference("average", projection=True, verbose=False)
ds_finger_epochs.set_eeg_reference("average", projection=True, verbose=False)

ds_wrist_evoked = ds_wrist_epochs.average()
ds_finger_evoked = ds_finger_epochs.average()

print("右腕叠加平均试次数：", ds_wrist_evoked.nave)
print("左指叠加平均试次数：", ds_finger_evoked.nave)
assert ds_wrist_evoked.nave >= 30
assert ds_finger_evoked.nave >= 30

ds_wrist_noise_cov = mne.compute_covariance(
    ds_wrist_epochs,
    tmin=-0.40,
    tmax=-0.05,
    method="shrunk",
    rank="info",
    verbose=False,
)
ds_finger_noise_cov = mne.compute_covariance(
    ds_finger_epochs,
    tmin=-0.55,
    tmax=-0.35,
    method="shrunk",
    rank="info",
    verbose=False,
)

ds_wrist_noise_trials = ds_wrist_epochs.get_data(
    tmin=-0.40,
    tmax=-0.05,
    copy=True,
    verbose=False,
)
ds_finger_noise_trials = ds_finger_epochs.get_data(
    tmin=-0.55,
    tmax=-0.35,
    copy=True,
    verbose=False,
)
ds_wrist_noise = ds_wrist_noise_trials.transpose(1, 0, 2).reshape(
    len(ds_wrist_epochs.ch_names), -1
)
ds_finger_noise = ds_finger_noise_trials.transpose(1, 0, 2).reshape(
    len(ds_finger_epochs.ch_names), -1
)

print("腕部基线样本数：", ds_wrist_noise.shape[1])
print("Finger 独立基线样本数：", ds_finger_noise.shape[1])


#%%
# ==================== 12. ds006035 模板配准和 forward ====================

ds_trans_file = (
    project_root
    / "results"
    / "real_data"
    / "ds006035"
    / f"sub-{ds_subject}_final"
    / f"sub-{ds_subject}_to_sample-trans.fif"
)
assert ds_trans_file.is_file(), f"找不到已经保存的配准：{ds_trans_file}"

ds_trans = mne.read_trans(ds_trans_file, verbose=False)
ds_source_space = mne.setup_source_space(
    sample_subject,
    spacing="ico4",
    subjects_dir=subjects_dir,
    add_dist=False,
    n_jobs=1,
    verbose=False,
)

ds_forwards = real_pipeline.make_fixed_forwards(
    ds_wrist_evoked.info,
    ds_trans,
    ds_source_space,
    subjects_dir,
)

ds_eeg_forward_fixed = ds_forwards[0]
ds_mag_forward_fixed = ds_forwards[1]
ds_eeg_forward_free = ds_forwards[2]
ds_mag_forward_free = ds_forwards[3]

print("ds006035 源点数：", ds_eeg_forward_fixed["nsource"])

ds_analysis_src = ds_eeg_forward_fixed["src"]
ds_source_xyz = np.vstack([
    hemi["rr"][hemi["vertno"]]
    for hemi in ds_analysis_src
])
ds_source_adjacency = mne.spatial_src_adjacency(
    ds_analysis_src,
    verbose=False,
).toarray()
ds_surface_kernels = protected.connected_euclidean_surface_kernels(
    ds_source_xyz,
    ds_source_adjacency,
    len(ds_source_xyz),
    scales_mm=oaster.SURFACE_SCALES_MM,
)
del ds_source_adjacency

print("ds006035 OASTER 空间模板尺度：", [scale for scale, _ in ds_surface_kernels])


#%%
# ==================== 13. ds006035 显式白化并运行多尺度 OASTER-ERP ====================

ds_records = []
ds_evoked_list = [ds_wrist_evoked, ds_finger_evoked]
ds_evoked_task_names = ["腕部刺激", "左指反应"]
ds_noise_list = [ds_wrist_noise, ds_finger_noise]
ds_noise_cov_list = [ds_wrist_noise_cov, ds_finger_noise_cov]

for task_name, evoked, trial_noise, noise_cov in zip(
    ds_evoked_task_names,
    ds_evoked_list,
    ds_noise_list,
    ds_noise_cov_list,
):
    ds_eeg_data, ds_eeg_gain_check, ds_eeg_noise_check = real_pipeline._aligned_modality(
        evoked.data,
        trial_noise,
        evoked.info,
        ds_eeg_forward_fixed,
        eeg=True,
    )
    ds_mag_data, ds_mag_gain_check, ds_mag_noise_check = real_pipeline._aligned_modality(
        evoked.data,
        trial_noise,
        evoked.info,
        ds_mag_forward_fixed,
        eeg=False,
    )

    ds_eeg_channel_names = ds_eeg_forward_fixed["sol"]["row_names"]
    ds_mag_channel_names = ds_mag_forward_fixed["sol"]["row_names"]
    ds_eeg_order = np.asarray([
        evoked.ch_names.index(channel_name)
        for channel_name in ds_eeg_channel_names
    ])
    ds_mag_order = np.asarray([
        evoked.ch_names.index(channel_name)
        for channel_name in ds_mag_channel_names
    ])
    ds_eeg_info = mne.pick_info(evoked.info, ds_eeg_order, copy=True)
    ds_mag_info = mne.pick_info(evoked.info, ds_mag_order, copy=True)

    ds_eeg_cov = mne.pick_channels_cov(
        noise_cov,
        include=ds_eeg_channel_names,
        exclude=[],
        ordered=True,
        copy=True,
        verbose=False,
    )
    ds_mag_cov = mne.pick_channels_cov(
        noise_cov,
        include=ds_mag_channel_names,
        exclude=[],
        ordered=True,
        copy=True,
        verbose=False,
    )
    ds_eeg_whitener, ds_eeg_white_names, ds_eeg_rank = compute_whitener(
        ds_eeg_cov,
        ds_eeg_info,
        rank="info",
        pca=True,
        return_rank=True,
        verbose=False,
    )
    ds_mag_whitener, ds_mag_white_names, ds_mag_rank = compute_whitener(
        ds_mag_cov,
        ds_mag_info,
        rank="info",
        pca=True,
        return_rank=True,
        verbose=False,
    )

    assert ds_eeg_white_names == ds_eeg_channel_names
    assert ds_mag_white_names == ds_mag_channel_names

    ds_eeg_data_white = ds_eeg_whitener @ ds_eeg_data
    ds_mag_data_white = ds_mag_whitener @ ds_mag_data
    ds_eeg_gain_white = ds_eeg_whitener @ ds_eeg_gain_check
    ds_mag_gain_white = ds_mag_whitener @ ds_mag_gain_check
    ds_joint_data_white = np.vstack((ds_eeg_data_white, ds_mag_data_white))
    ds_joint_gain_white = np.vstack((ds_eeg_gain_white, ds_mag_gain_white))

    if task_name == "腕部刺激":
        ds_erp_baseline = (
            (evoked.times >= -0.40)
            & (evoked.times <= -0.05)
        )
        ds_erp_windows = (
            (evoked.times >= wrist_n20_window[0])
            & (evoked.times <= wrist_n20_window[1]),
            (evoked.times >= wrist_p30_window[0])
            & (evoked.times <= wrist_p30_window[1]),
        )
        # N20/P30 各按一个预注册主导发生器验证，与正式 ds006035 链路一致。
        ds_max_templates = 1
    else:
        ds_erp_baseline = (
            (evoked.times >= -0.55)
            & (evoked.times <= -0.35)
        )
        ds_erp_windows = ((
            (evoked.times >= finger_window[0])
            & (evoked.times <= finger_window[1])
        ),)
        ds_max_templates = oaster.ERP_MAX_TEMPLATES

    ds_window_channel_weights = []
    ds_modality_weights = []
    for ds_erp_active in ds_erp_windows:
        current_modality_weights = real_pipeline.modality_evidence_weights(
            (ds_eeg_data, ds_mag_data),
            ds_erp_baseline,
            ds_erp_active,
        )
        ds_modality_weights.append({
            "EEG": float(current_modality_weights[0]),
            "MAG": float(current_modality_weights[1]),
        })
        ds_window_channel_weights.append(np.r_[
            np.full(ds_eeg_data_white.shape[0], current_modality_weights[0]),
            np.full(ds_mag_data_white.shape[0], current_modality_weights[1]),
        ])

    ds_oaster_source, ds_oaster_information = oaster.reconstruct_evoked_oaster_from_whitened(
        ds_joint_data_white,
        ds_joint_gain_white,
        len(ds_source_xyz),
        ds_surface_kernels,
        baseline=ds_erp_baseline,
        active_windows=ds_erp_windows,
        window_channel_weights=ds_window_channel_weights,
        max_templates=ds_max_templates,
    )
    ds_oaster_information["modality_weights"] = ds_modality_weights

    if task_name == "腕部刺激":
        ds_records.append({
            "dataset": "ds006035",
            "task": "右腕刺激-N20",
            "window": wrist_n20_window,
            "baseline": (-0.40, -0.05),
            "n_trials": evoked.nave,
            "times": evoked.times.copy(),
            "sensor_white": ds_joint_data_white,
            "OASTER ERP": ds_oaster_source,
        })
        ds_records.append({
            "dataset": "ds006035",
            "task": "右腕刺激-P30",
            "window": wrist_p30_window,
            "baseline": (-0.40, -0.05),
            "n_trials": evoked.nave,
            "times": evoked.times.copy(),
            "sensor_white": ds_joint_data_white,
            "OASTER ERP": ds_oaster_source,
        })
    else:
        ds_records.append({
            "dataset": "ds006035",
            "task": "左指反应-20到80ms",
            "window": finger_window,
            "baseline": (-0.55, -0.35),
            "n_trials": evoked.nave,
            "times": evoked.times.copy(),
            "sensor_white": ds_joint_data_white,
            "OASTER ERP": ds_oaster_source,
        })

    print(
        task_name,
        "OASTER：",
        ds_oaster_source.shape,
        "EEG/MAG 白化秩：",
        ds_eeg_rank,
        ds_mag_rank,
        ds_oaster_information,
    )


#%%
# ==================== 14. ds006035 官方 dSPM 和 eLORETA ====================

ds_joint_forward = _merge_fwds(
    {
        "meg": ds_mag_forward_free.copy(),
        "eeg": ds_eeg_forward_free.copy(),
    },
    verbose=False,
)
ds_joint_channel_names = ds_joint_forward["sol"]["row_names"]
ds_wrist_inverse_evoked = ds_wrist_evoked.copy().reorder_channels(ds_joint_channel_names)
ds_finger_inverse_evoked = ds_finger_evoked.copy().reorder_channels(ds_joint_channel_names)

ds_wrist_joint_forward = ds_joint_forward.copy()
ds_finger_joint_forward = ds_joint_forward.copy()
ds_wrist_joint_forward["info"] = ds_wrist_inverse_evoked.info
ds_finger_joint_forward["info"] = ds_finger_inverse_evoked.info

ds_wrist_joint_cov = mne.pick_channels_cov(
    ds_wrist_noise_cov,
    include=ds_joint_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)
ds_finger_joint_cov = mne.pick_channels_cov(
    ds_finger_noise_cov,
    include=ds_joint_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)

ds_wrist_inverse = mne.minimum_norm.make_inverse_operator(
    ds_wrist_inverse_evoked.info,
    ds_wrist_joint_forward,
    ds_wrist_joint_cov,
    loose=0.0,
    fixed=True,
    depth=depth,
    rank="info",
    use_cps=True,
    verbose=False,
)
ds_finger_inverse = mne.minimum_norm.make_inverse_operator(
    ds_finger_inverse_evoked.info,
    ds_finger_joint_forward,
    ds_finger_joint_cov,
    loose=0.0,
    fixed=True,
    depth=depth,
    rank="info",
    use_cps=True,
    verbose=False,
)

ds_wrist_dspm = mne.minimum_norm.apply_inverse(
    ds_wrist_inverse_evoked,
    ds_wrist_inverse,
    lambda2=lambda2,
    method="dSPM",
    use_cps=True,
    verbose=False,
)
ds_wrist_eloreta = mne.minimum_norm.apply_inverse(
    ds_wrist_inverse_evoked,
    ds_wrist_inverse,
    lambda2=lambda2,
    method="eLORETA",
    use_cps=True,
    verbose=False,
)

ds_finger_dspm = mne.minimum_norm.apply_inverse(
    ds_finger_inverse_evoked,
    ds_finger_inverse,
    lambda2=lambda2,
    method="dSPM",
    use_cps=True,
    verbose=False,
)
ds_finger_eloreta = mne.minimum_norm.apply_inverse(
    ds_finger_inverse_evoked,
    ds_finger_inverse,
    lambda2=lambda2,
    method="eLORETA",
    use_cps=True,
    verbose=False,
)

for record in ds_records:
    if record["task"].startswith("右腕刺激"):
        record["dSPM"] = ds_wrist_dspm.data
        record["eLORETA"] = ds_wrist_eloreta.data
    else:
        record["dSPM"] = ds_finger_dspm.data
        record["eLORETA"] = ds_finger_eloreta.data

    assert record["OASTER ERP"].shape == record["dSPM"].shape
    assert record["dSPM"].shape == record["eLORETA"].shape

print("ds006035 三种方法已经完成")


#%%
# ==================== 15. 所有逆解结束后，才读取 ROI 做评价 ====================
# 注意：从这一块开始才出现目标脑区。
# 前面的逆解和白化过程完全不知道下面这些 ROI。

all_records = sample_records + ds_records

sample_vertices = [
    np.asarray(space["vertno"], dtype=int)
    for space in sample_eeg_forward_fixed["src"]
]
sample_xyz = np.vstack([
    space["rr"][space["vertno"]]
    for space in sample_eeg_forward_fixed["src"]
])

ds_vertices = [
    np.asarray(space["vertno"], dtype=int)
    for space in ds_eeg_forward_fixed["src"]
]
ds_xyz = np.vstack([
    space["rr"][space["vertno"]]
    for space in ds_eeg_forward_fixed["src"]
])

sample_vertex_names = np.full(len(sample_xyz), "unknown", dtype=object)
ds_vertex_names = np.full(len(ds_xyz), "unknown", dtype=object)

sample_label_masks = {}
ds_label_masks = {}
sample_offsets = [0, len(sample_vertices[0])]
ds_offsets = [0, len(ds_vertices[0])]

anatomy_labels = mne.read_labels_from_annot(
    sample_subject,
    parc="aparc",
    subjects_dir=subjects_dir,
    verbose=False,
)

for label in anatomy_labels:
    hemi_index = 0 if label.hemi == "lh" else 1

    sample_label_index = (
        np.flatnonzero(np.isin(sample_vertices[hemi_index], label.vertices))
        + sample_offsets[hemi_index]
    )
    sample_label_mask = np.zeros(len(sample_xyz), dtype=bool)
    sample_label_mask[sample_label_index] = True
    sample_label_masks[label.name] = sample_label_mask
    sample_vertex_names[sample_label_index] = label.name

    ds_label_index = (
        np.flatnonzero(np.isin(ds_vertices[hemi_index], label.vertices))
        + ds_offsets[hemi_index]
    )
    ds_label_mask = np.zeros(len(ds_xyz), dtype=bool)
    ds_label_mask[ds_label_index] = True
    ds_label_masks[label.name] = ds_label_mask
    ds_vertex_names[ds_label_index] = label.name

target_roi_labels_by_task = {
    "听觉-左侧刺激": [
        "transversetemporal-lh",
        "transversetemporal-rh",
        "superiortemporal-lh",
        "superiortemporal-rh",
    ],
    "听觉-右侧刺激": [
        "transversetemporal-lh",
        "transversetemporal-rh",
        "superiortemporal-lh",
        "superiortemporal-rh",
    ],
    "视觉-左侧视野": [
        "pericalcarine-rh",
        "cuneus-rh",
        "lateraloccipital-rh",
        "lingual-rh",
    ],
    "视觉-右侧视野": [
        "pericalcarine-lh",
        "cuneus-lh",
        "lateraloccipital-lh",
        "lingual-lh",
    ],
    "右腕刺激-N20": ["postcentral-lh"],
    "右腕刺激-P30": ["postcentral-lh"],
    "左指反应-20到80ms": ["precentral-rh", "postcentral-rh"],
}

sample_roi_masks = {}
ds_roi_masks = {}

for task_name, roi_label_names in target_roi_labels_by_task.items():
    sample_roi_mask = np.zeros(len(sample_xyz), dtype=bool)
    ds_roi_mask = np.zeros(len(ds_xyz), dtype=bool)

    for roi_label_name in roi_label_names:
        sample_roi_mask |= sample_label_masks[roi_label_name]
        ds_roi_mask |= ds_label_masks[roi_label_name]

    sample_roi_masks[task_name] = sample_roi_mask
    ds_roi_masks[task_name] = ds_roi_mask

    assert sample_roi_mask.any(), f"sample 中复合 ROI 为空：{task_name}"
    assert ds_roi_mask.any(), f"ds006035 中复合 ROI 为空：{task_name}"


#%%
# ==================== 16. 用预注册时间窗的基线校正超额功率评价 ====================
# GFP 峰时刻只作为描述信息，不参与下面主空间图的计算。

metric_rows = []
brain_maps = {}

for record in all_records:
    times = record["times"]
    window_start, window_stop = record["window"]
    baseline_start, baseline_stop = record["baseline"]
    time_window = (times >= window_start) & (times <= window_stop)
    baseline_window = (times >= baseline_start) & (times <= baseline_stop)
    assert time_window.any(), f"时间窗为空：{record['task']}"
    assert baseline_window.any(), f"基线窗为空：{record['task']}"

    window_indices = np.flatnonzero(time_window)
    sensor_gfp = np.linalg.norm(record["sensor_white"][:, time_window], axis=0)
    peak_inside_window = int(np.argmax(sensor_gfp))
    peak_index = int(window_indices[peak_inside_window])
    peak_time_ms = float(times[peak_index] * 1000.0)

    if record["dataset"] == "MNE sample":
        xyz = sample_xyz
        vertex_names = sample_vertex_names
        roi_mask = sample_roi_masks[record["task"]]
        left_vertex_count = len(sample_vertices[0])
    else:
        xyz = ds_xyz
        vertex_names = ds_vertex_names
        roi_mask = ds_roi_masks[record["task"]]
        left_vertex_count = len(ds_vertices[0])

    roi_name = "+".join(target_roi_labels_by_task[record["task"]])
    roi_fraction = float(np.mean(roi_mask))

    for method_name in method_names:
        source = record[method_name]
        assert source.shape[1] == len(times)
        active_power = np.mean(source[:, time_window] ** 2, axis=1)
        baseline_power = np.mean(source[:, baseline_window] ** 2, axis=1)
        amplitude = np.sqrt(np.maximum(active_power - baseline_power, 0.0))
        brain_maps[(record["task"], method_name)] = amplitude.copy()
        power = amplitude**2
        total_power = float(np.sum(power))
        roi_power = float(np.sum(power[roi_mask]))
        roi_mass = roi_power / total_power if total_power > 0 else 0.0
        roi_enrichment = roi_mass / roi_fraction if total_power > 0 else 0.0

        source_peak = int(np.argmax(amplitude))
        peak_distance_mm = float(
            np.min(np.linalg.norm(xyz[roi_mask] - xyz[source_peak], axis=1))
            * 1000.0
        )

        metric_rows.append({
            "dataset": record["dataset"],
            "task": record["task"],
            "event_or_lock": (
                "event 1/2/3/4"
                if record["dataset"] == "MNE sample"
                else ("event 32" if record["task"].startswith("右腕刺激") else "event 16")
            ),
            "n_trials": record["n_trials"],
            "window_start_ms": window_start * 1000.0,
            "window_stop_ms": window_stop * 1000.0,
            "baseline_start_ms": baseline_start * 1000.0,
            "baseline_stop_ms": baseline_stop * 1000.0,
            "sensor_gfp_peak_ms": peak_time_ms,
            "source_map": "sqrt(max(mean(active^2)-mean(baseline^2),0))",
            "method": method_name,
            "target_roi": roi_name,
            "target_roi_vertices": int(np.sum(roi_mask)),
            "target_roi_mass_pct": roi_mass * 100.0,
            "target_roi_enrichment": roi_enrichment,
            "peak_distance_to_target_roi_mm": peak_distance_mm,
            "peak_label": str(vertex_names[source_peak]),
            "peak_hemi": "left" if source_peak < left_vertex_count else "right",
            "roi_used_before_evaluation": False,
        })

metric_table = pd.DataFrame(metric_rows)
metric_file = save_dir / "erp_four_paradigms_metrics.csv"
metric_table.to_csv(metric_file, index=False, encoding="utf-8-sig")

print(metric_table[[
    "task",
    "method",
    "sensor_gfp_peak_ms",
    "target_roi_enrichment",
    "peak_distance_to_target_roi_mm",
    "peak_label",
]].to_string(index=False))


#%%
# ==================== 17. 画一张统一比较图 ====================

task_order = list(target_roi_labels_by_task)
task_display = [
    "听觉\n左刺激",
    "听觉\n右刺激",
    "视觉\n左视野",
    "视觉\n右视野",
    "右腕\nN20",
    "右腕\nP30",
    "左指反应\n20–80 ms",
]
x = np.arange(len(task_order), dtype=float)
offsets = {"OASTER ERP": -0.22, "dSPM": 0.0, "eLORETA": 0.22}

figure, axes = plt.subplots(2, 1, figsize=(13.5, 9.0), sharex=True)

for method_name in method_names:
    method_table = metric_table[metric_table["method"] == method_name].set_index("task")
    method_table = method_table.loc[task_order]

    axes[0].scatter(
        x + offsets[method_name],
        method_table["target_roi_enrichment"],
        s=76,
        marker=method_markers[method_name],
        color=method_colors[method_name],
        edgecolor="white",
        linewidth=0.8,
        label=method_name,
        zorder=3,
    )
    axes[1].scatter(
        x + offsets[method_name],
        method_table["peak_distance_to_target_roi_mm"],
        s=76,
        marker=method_markers[method_name],
        color=method_colors[method_name],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

axes[0].axhline(1.0, color="#666666", linestyle="--", linewidth=1.2)
axes[0].text(
    len(task_order) - 0.55,
    1.05,
    "均匀源点零假设 = 1",
    color="#555555",
    fontsize=10,
    ha="right",
    va="bottom",
)
axes[0].set_ylabel("目标 ROI 富集")
axes[0].set_title("ERP 时间域源定位：预注册窗的基线校正超额功率")
axes[0].legend(frameon=False, ncol=3, loc="upper left")

axes[1].axhline(20.0, color="#666666", linestyle="--", linewidth=1.2)
axes[1].text(
    len(task_order) - 0.55,
    20.8,
    "距 ROI 20 mm",
    color="#555555",
    fontsize=10,
    ha="right",
    va="bottom",
)
axes[1].set_ylabel("峰距目标 ROI（mm）")
axes[1].set_xticks(x, task_display)

for axis in axes:
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

figure.tight_layout()
figure_file = save_dir / "erp_four_paradigms_comparison.png"
figure.savefig(figure_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)


#%%
# ==================== 18. 七个任务分别渲染解剖脑俯视图 ====================
# 每张图三行是三种方法；左列真正俯视，右列后上视以免枕叶被表面遮住。
# 不画 ROI、圆或星号；P95 只控制显示，不参与前面的数值指标。

task_slug_by_task = {
    "听觉-左侧刺激": "auditory_left",
    "听觉-右侧刺激": "auditory_right",
    "视觉-左侧视野": "visual_left",
    "视觉-右侧视野": "visual_right",
    "右腕刺激-N20": "right_wrist_n20",
    "右腕刺激-P30": "right_wrist_p30",
    "左指反应-20到80ms": "left_finger_response",
}
record_by_task = {record["task"]: record for record in all_records}

if draw_brain:
    for task_name in task_order:
        if record_by_task[task_name]["dataset"] == "MNE sample":
            brain_vertices = sample_vertices
        else:
            brain_vertices = ds_vertices

        brain_images = []

        for method_name in method_names:
            brain_map = brain_maps[(task_name, method_name)]
            brain_peak = max(
                float(brain_map.max(initial=0.0)),
                np.finfo(float).eps,
            )
            brain_normalized = brain_map / brain_peak
            brain_floor = float(np.percentile(brain_normalized, 95))
            if brain_floor >= 1.0:
                brain_floor = 0.0
            brain_display = np.where(
                brain_normalized >= brain_floor,
                brain_normalized,
                0.0,
            )

            brain_stc = mne.SourceEstimate(
                brain_display[:, None],
                vertices=brain_vertices,
                tmin=0.0,
                tstep=1.0,
                subject=sample_subject,
            )

            for view_name in ["dorsal", "parietal"]:
                brain = brain_stc.plot(
                    subject=sample_subject,
                    subjects_dir=subjects_dir,
                    surface="pial",
                    hemi="both",
                    views=view_name,
                    initial_time=0.0,
                    time_viewer=False,
                    show_traces=False,
                    time_label=None,
                    colormap="inferno",
                    clim={
                        "kind": "value",
                        "lims": [
                            brain_floor,
                            (brain_floor + 1.0) / 2.0,
                            1.0,
                        ],
                    },
                    smoothing_steps=5,
                    transparent=True,
                    background="white",
                    foreground="black",
                    cortex="classic",
                    colorbar=False,
                    size=(900, 650),
                    backend="pyvistaqt",
                    brain_kwargs={"show": False, "theme": "light"},
                )
                brain.plotter.camera.zoom(0.78)
                brain.plotter.render()

                brain_images.append((
                    method_name,
                    view_name,
                    brain.screenshot(mode="rgb", time_viewer=False),
                ))
                brain.close()

        brain_figure, brain_axes = plt.subplots(
            3,
            2,
            figsize=(16, 13),
            layout="constrained",
            facecolor="white",
        )

        for brain_axis, (method_name, view_name, brain_image) in zip(
            brain_axes.ravel(),
            brain_images,
        ):
            brain_axis.imshow(brain_image)
            view_chinese = "俯视" if view_name == "dorsal" else "后上视"
            brain_axis.set_title(f"{method_name} | {view_chinese}", fontsize=13)
            brain_axis.set_axis_off()

        brain_figure.suptitle(
            f"{task_name} | 解剖表面源定位（俯视）\n"
            "仅显示 ≥P95、各方法独立归一化",
            fontsize=16,
            fontweight="bold",
        )
        brain_file = save_dir / f"brain_{task_slug_by_task[task_name]}_dorsal_posterior.png"
        brain_figure.savefig(
            brain_file,
            dpi=170,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(brain_figure)
        print("脑图完成：", brain_file)


#%%
# ==================== 19. 完成检查 ====================

print("指标表：", metric_file)
print("比较图：", figure_file)
print("总耗时：", round(time.perf_counter() - all_started, 1), "秒")
assert len(metric_table) == 7 * len(method_names)
assert not metric_table["roi_used_before_evaluation"].any()
