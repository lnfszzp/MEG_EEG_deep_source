#%%
# 这个脚本只跑 ds006035 的一个受试者、一个 run。
# 写法按你原来的习惯：从上往下运行，每一步都保留变量，不用自己定义函数。
# 数据里有个体 T1，但还没有现成的 FreeSurfer/BEM/trans；这里先用 MNE sample 模板脑。
# 因而结果只能看方法是否定位到合理的感觉运动区，不能当作个体解剖精度结论。

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.forward import _merge_fwds
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected
from pipelines.run_ds006035_somatomotor import label_geometry


mne.set_log_level("warning")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


#%%
# ==================== 1. 参数区：平时主要改这里 ====================

dataset_path = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
subjects_dir = Path(r"C:\Users\zzp\mne_data\MNE-sample-data\subjects")

subject = "sm09"
run = 1
spacing = "ico4"

filter_low = 1.0
filter_high = 100.0
epoch_tmin = -0.4
epoch_tmax = 0.1
baseline = (-0.4, -0.05)

# OASTER 用前 200 个点作为基线，后面 31 个点作为早期 ERP。
target_times = np.r_[np.arange(-250, -50), np.arange(15, 46)] / 1000.0
noise_samples = 200
n20_window = (target_times >= 0.018) & (target_times <= 0.024)
p30_window = (target_times >= 0.028) & (target_times <= 0.040)

oaster_surface_scales_mm = (0.0, 4.0, 7.0)
oaster_ridge_fraction = 0.03
oaster_spectral_fraction = 0.05

# False 时只保存普通 matplotlib 图片，不启动三维脑窗口。
draw_brain = False

save_dir = project_root / "results" / "real_data" / "ds006035" / f"sub-{subject}_run-{run}_author_style"
save_dir.mkdir(parents=True, exist_ok=True)


#%%
# ==================== 2. 找到原始数据和事件文件 ====================

stem = f"sub-{subject}_ses-meeg_task-somatomotor_run-{run}"
data_folder = dataset_path / f"sub-{subject}" / "ses-meeg" / "meg"
raw_file = data_folder / f"{stem}_meg.fif"
events_file = data_folder / f"{stem}_events.tsv"

print("原始数据：", raw_file)
print("事件文件：", events_file)

assert raw_file.is_file(), f"找不到原始数据：{raw_file}"
assert events_file.is_file(), f"找不到事件文件：{events_file}"
assert (subjects_dir / "sample").is_dir(), f"找不到 sample 模板脑：{subjects_dir}"


#%%
# ==================== 3. 读取 EEG 和 MAG，并进行滤波 ====================
# 这里没有用 gradiometer，只保留 EEG 和 magnetometer，和当前真实数据验证保持一致。

raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
first_samp = int(raw.first_samp)
original_bad_channels = list(raw.info["bads"])

picks = mne.pick_types(
    raw.info,
    meg="mag",
    eeg=True,
    eog=False,
    ecg=False,
    stim=False,
    ref_meg=False,
    exclude="bads",
)

raw.pick(picks)
raw.load_data(verbose=False)

line_frequency = float(raw.info.get("line_freq") or 60.0)
if line_frequency < raw.info["sfreq"] / 2.0:
    raw.notch_filter([line_frequency], n_jobs=1, verbose=False)

raw.filter(filter_low, filter_high, n_jobs=1, verbose=False)

print("采样率：", raw.info["sfreq"])
print("保留通道数：", len(raw.ch_names))
print("原始坏道：", original_bad_channels)


#%%
# ==================== 4. 读取触觉刺激事件 ====================
# BIDS 的 sample 是相对采样点，MNE 需要加上 raw.first_samp。

event_table = pd.read_csv(events_file, sep="\t")

is_somatosensory = event_table["trial_type"].astype(str) == "somatosensory"
is_value_32 = pd.to_numeric(event_table["value"], errors="coerce") == 32
event_table = event_table[is_somatosensory | is_value_32].copy()

events = np.zeros((len(event_table), 3), dtype=int)
events[:, 0] = event_table["sample"].astype(float).astype(int).to_numpy() + first_samp
events[:, 2] = 1

print("触觉刺激事件数：", len(events))
print("first_samp：", first_samp)

assert len(events) > 0, "没有找到 somatosensory 事件"
assert np.all(np.diff(events[:, 0]) > 0), "事件采样点不是递增的"


#%%
# ==================== 5. 先检查 EEG 异常通道 ====================
# 用每个 EEG 通道在刺激前的中位峰峰值寻找明显坏道。
# sm09 的 EEG 033 会在这一步被自动发现，避免它让所有 epoch 都被删除。

epoch_parameter = dict(
    raw=raw,
    events=events,
    event_id={"somatosensory": 1},
    tmin=epoch_tmin,
    tmax=epoch_tmax,
    baseline=baseline,
    reject_tmin=-0.2,
    reject_tmax=0.08,
    preload=True,
    proj=False,
    event_repeated="drop",
    verbose=False,
)

probe_epochs = mne.Epochs(**epoch_parameter)
eeg_probe_picks = mne.pick_types(probe_epochs.info, meg=False, eeg=True, exclude=[])
reject_window = (probe_epochs.times >= -0.35) & (probe_epochs.times <= -0.05)

eeg_probe_data = probe_epochs.get_data(copy=False)[:, eeg_probe_picks][:, :, reject_window]
median_ptp = np.median(np.ptp(eeg_probe_data, axis=2), axis=0)
ptp_center = float(np.median(median_ptp))
ptp_mad = float(np.median(np.abs(median_ptp - ptp_center)))
ptp_cutoff = max(100e-6, ptp_center + 6.0 * 1.4826 * ptp_mad)

auto_bad_channels = []
for channel_index, channel_ptp in zip(eeg_probe_picks, median_ptp):
    if channel_ptp > ptp_cutoff:
        auto_bad_channels.append(probe_epochs.ch_names[channel_index])

if len(auto_bad_channels) > 0:
    raw.drop_channels(auto_bad_channels)

print("自动发现的 EEG 坏道：", auto_bad_channels)
print("EEG 峰峰值阈值：", ptp_cutoff * 1e6, "uV")


#%%
# ==================== 6. 分段并进行叠加平均 ====================
# 这一行 evoked = epochs.average() 就是 ERP/ERF 的叠加平均。
# OASTER、dSPM 和 eLORETA 后面使用的是同一份平均结果。

epoch_parameter["raw"] = raw
epoch_parameter["reject"] = {"eeg": 200e-6, "mag": 6e-12}

epochs = mne.Epochs(**epoch_parameter)
assert len(epochs) >= 15, f"可用 epoch 太少：{len(epochs)}"

epochs.set_eeg_reference("average", projection=True, verbose=False)
noise_cov = mne.compute_covariance(
    epochs,
    tmin=baseline[0],
    tmax=baseline[1],
    method="shrunk",
    rank="info",
    verbose=False,
)

evoked = epochs.average()

print("原始事件数：", len(events))
print("叠加平均使用的 epoch 数：", len(epochs))
print("ERP 数据形状：", evoked.data.shape)

assert evoked.nave == len(epochs)


#%%
# ==================== 7. 整理 OASTER 使用的基线和 ERP 时段 ====================
# 前 200 点：-250 到 -51 ms。
# 后 31 点：15 到 45 ms，包含 N20 和 P30/P35。

erp_data = np.vstack([
    np.interp(target_times, evoked.times, channel_data)
    for channel_data in evoked.data
])

noise_epochs = epochs.get_data(
    tmin=baseline[0],
    tmax=baseline[1],
    copy=True,
    verbose=False,
)
trial_noise = noise_epochs.transpose(1, 0, 2).reshape(len(evoked.ch_names), -1)

eeg_picks = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=[])
mag_picks = mne.pick_types(evoked.info, meg="mag", eeg=False, exclude=[])

eeg_baseline_rms = np.sqrt(np.mean(erp_data[eeg_picks, :noise_samples] ** 2))
eeg_n20_rms = np.sqrt(np.mean(erp_data[eeg_picks][:, n20_window] ** 2))
mag_baseline_rms = np.sqrt(np.mean(erp_data[mag_picks, :noise_samples] ** 2))
mag_n20_rms = np.sqrt(np.mean(erp_data[mag_picks][:, n20_window] ** 2))

eeg_n20_sensor_ratio = eeg_n20_rms / max(eeg_baseline_rms, np.finfo(float).eps)
mag_n20_sensor_ratio = mag_n20_rms / max(mag_baseline_rms, np.finfo(float).eps)

print("OASTER 输入形状：", erp_data.shape)
print("单试次基线噪声形状：", trial_noise.shape)
print(f"EEG N20/基线 RMS = {eeg_n20_sensor_ratio:.3f}")
print(f"MAG N20/基线 RMS = {mag_n20_sensor_ratio:.3f}")

assert erp_data.shape[1] == 231
assert int(n20_window.sum()) == 7
assert int(p30_window.sum()) == 13


#%%
# ==================== 8. 保存并画出叠加平均后的 ERP/ERF ====================
# EEG 和 MAG 单位不同，所以分成上下两张图。

show_time = (evoked.times >= -0.05) & (evoked.times <= 0.08)
eeg_gfp = np.sqrt(np.mean(evoked.data[eeg_picks] ** 2, axis=0)) * 1e6
mag_gfp = np.sqrt(np.mean(evoked.data[mag_picks] ** 2, axis=0)) * 1e15

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")

axes[0].plot(evoked.times[show_time] * 1000, eeg_gfp[show_time], color="#0072B2", linewidth=2)
axes[0].axvspan(18, 24, color="#56B4E9", alpha=0.20, label="N20: 18-24 ms")
axes[0].axvspan(28, 40, color="#E69F00", alpha=0.18, label="P30/P35: 28-40 ms")
axes[0].set_ylabel("EEG GFP (uV)")
axes[0].set_title(f"sub-{subject} run-{run} | {len(epochs)} 次叠加平均")
axes[0].legend(loc="upper right")

axes[1].plot(evoked.times[show_time] * 1000, mag_gfp[show_time], color="#009E73", linewidth=2)
axes[1].axvspan(18, 24, color="#56B4E9", alpha=0.20)
axes[1].axvspan(28, 40, color="#E69F00", alpha=0.18)
axes[1].set_xlabel("刺激后时间 (ms)")
axes[1].set_ylabel("MAG GFP (fT)")

for axis in axes:
    axis.axvline(0, color="#333333", linewidth=1)
    axis.grid(axis="y", color="#D9E1E8", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

fig.savefig(save_dir / "1-ERP叠加平均.png", dpi=180, bbox_inches="tight")
plt.close(fig)


#%%
# ==================== 9. 自动配准到 sample 模板脑 ====================
# 公开数据有个体 T1，但没有现成的 FreeSurfer/BEM/trans，所以这里只是模板脑初步验证。

coreg = mne.coreg.Coregistration(
    evoked.info,
    subject="sample",
    subjects_dir=subjects_dir,
    fiducials="estimated",
)
coreg.fit_fiducials(verbose=False)
coreg.fit_icp(n_iterations=20, nasion_weight=2.0, verbose=False)
coreg.omit_head_shape_points(distance=0.02)
coreg.fit_icp(n_iterations=30, nasion_weight=2.0, verbose=False)

trans = coreg.trans
coreg_distance = np.asarray(coreg.compute_dig_mri_distances()) * 1000

mne.write_trans(
    save_dir / f"sub-{subject}_run-{run}_to_sample-trans.fif",
    trans,
    overwrite=True,
)

print(f"配准距离均值 = {np.mean(coreg_distance):.2f} mm")
print(f"配准距离 P95 = {np.percentile(coreg_distance, 95):.2f} mm")


#%%
# ==================== 10. 建立皮层源空间和 EEG/MAG 正向模型 ====================
# EEG 使用三层 BEM，MAG 使用单层 BEM。
# 同时保留 fixed forward 给 OASTER，保留 free forward 给 MNE 官方逆解。

src = mne.setup_source_space(
    "sample",
    spacing=spacing,
    subjects_dir=subjects_dir,
    add_dist=False,
    n_jobs=1,
    verbose=False,
)

bem_folder = subjects_dir / "sample" / "bem"
eeg_bem = bem_folder / "sample-5120-5120-5120-bem-sol.fif"
mag_bem = bem_folder / "sample-5120-bem-sol.fif"

eeg_forward_picks = mne.pick_types(
    evoked.info, eeg=True, meg=False, ref_meg=False, exclude="bads"
)
eeg_forward_info = mne.pick_info(evoked.info, eeg_forward_picks, copy=True)

fwd_eeg = mne.make_forward_solution(
    eeg_forward_info,
    trans,
    src,
    eeg_bem,
    meg=False,
    eeg=True,
    mindist=5.0,
    n_jobs=1,
    on_inside="raise",
    verbose=False,
)
fwd_eeg_free = mne.convert_forward_solution(
    fwd_eeg, surf_ori=True, force_fixed=False, use_cps=True, verbose=False
)
fwd_eeg_fixed = mne.convert_forward_solution(
    fwd_eeg_free, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
)

mag_forward_picks = mne.pick_types(
    evoked.info, eeg=False, meg="mag", ref_meg=False, exclude="bads"
)
mag_forward_info = mne.pick_info(evoked.info, mag_forward_picks, copy=True)

fwd_mag = mne.make_forward_solution(
    mag_forward_info,
    trans,
    src,
    mag_bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=1,
    on_inside="raise",
    verbose=False,
)
fwd_mag_free = mne.convert_forward_solution(
    fwd_mag, surf_ori=True, force_fixed=False, use_cps=True, verbose=False
)
fwd_mag_fixed = mne.convert_forward_solution(
    fwd_mag_free, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
)

for hemi in range(2):
    assert np.array_equal(
        fwd_eeg_fixed["src"][hemi]["vertno"],
        fwd_mag_fixed["src"][hemi]["vertno"],
    ), "EEG 和 MAG 的皮层顶点不一致"

analysis_src = fwd_eeg_fixed["src"]
source_xyz = np.vstack([
    hemi["rr"][hemi["vertno"]]
    for hemi in analysis_src
])
n_sources = len(source_xyz)

print("EEG Gain：", fwd_eeg_fixed["sol"]["data"].shape)
print("MAG Gain：", fwd_mag_fixed["sol"]["data"].shape)
print("皮层源点数：", n_sources)


#%%
# ==================== 11. 对齐数据、通道和 Gain ====================
# EEG 还要显式做平均参考；已有投影也同时作用到数据、噪声和 Gain。

eeg_order = np.asarray([
    evoked.ch_names.index(channel_name)
    for channel_name in fwd_eeg_fixed["info"]["ch_names"]
])
eeg_data = erp_data[eeg_order].copy()
eeg_noise = trial_noise[eeg_order].copy()
eeg_gain = np.asarray(fwd_eeg_fixed["sol"]["data"], dtype=float).copy()
eeg_info = mne.pick_info(evoked.info, eeg_order, copy=True)

if len(eeg_info["projs"]) > 0:
    eeg_projector = mne.EvokedArray(
        np.eye(len(eeg_order)), eeg_info, tmin=0.0, verbose=False
    )
    eeg_projector.apply_proj(verbose=False)
    eeg_data = eeg_projector.data @ eeg_data
    eeg_noise = eeg_projector.data @ eeg_noise
    eeg_gain = eeg_projector.data @ eeg_gain

eeg_data = eeg_data - eeg_data.mean(axis=0, keepdims=True)
eeg_noise = eeg_noise - eeg_noise.mean(axis=0, keepdims=True)
eeg_gain = eeg_gain - eeg_gain.mean(axis=0, keepdims=True)

mag_order = np.asarray([
    evoked.ch_names.index(channel_name)
    for channel_name in fwd_mag_fixed["info"]["ch_names"]
])
mag_data = erp_data[mag_order].copy()
mag_noise = trial_noise[mag_order].copy()
mag_gain = np.asarray(fwd_mag_fixed["sol"]["data"], dtype=float).copy()
mag_info = mne.pick_info(evoked.info, mag_order, copy=True)

if len(mag_info["projs"]) > 0:
    mag_projector = mne.EvokedArray(
        np.eye(len(mag_order)), mag_info, tmin=0.0, verbose=False
    )
    mag_projector.apply_proj(verbose=False)
    mag_data = mag_projector.data @ mag_data
    mag_noise = mag_projector.data @ mag_noise
    mag_gain = mag_projector.data @ mag_gain

assert eeg_data.shape[0] == eeg_gain.shape[0]
assert mag_data.shape[0] == mag_gain.shape[0]
assert eeg_gain.shape[1] == mag_gain.shape[1] == n_sources

print("对齐后 EEG 数据/Gain：", eeg_data.shape, eeg_gain.shape)
print("对齐后 MAG 数据/Gain：", mag_data.shape, mag_gain.shape)


#%%
# ==================== 12. OASTER 第一步：EEG 和 MAG 分别白化 ====================
# 白化矩阵来自所有单试次的刺激前基线，不从 ERP 响应段估计噪声。

eeg_whitener = protected.whitening_matrix(eeg_noise, eeg_noise.shape[1])
mag_whitener = protected.whitening_matrix(mag_noise, mag_noise.shape[1])

eeg_white = eeg_whitener @ eeg_data
eeg_gain_white = eeg_whitener @ eeg_gain
mag_white = mag_whitener @ mag_data
mag_gain_white = mag_whitener @ mag_gain

joint_white = np.vstack((eeg_white, mag_white))
joint_gain_white = np.vstack((eeg_gain_white, mag_gain_white))

print("联合白化数据：", joint_white.shape)
print("联合白化 Gain：", joint_gain_white.shape)

assert np.isfinite(joint_white).all()
assert np.isfinite(joint_gain_white).all()


#%%
# ==================== 13. 建立 0、4、7 mm 的表层空间模板 ====================

source_adjacency = mne.spatial_src_adjacency(analysis_src, verbose=False).toarray()
surface_kernels = protected.connected_euclidean_surface_kernels(
    source_xyz,
    source_adjacency,
    n_sources,
    scales_mm=oaster_surface_scales_mm,
)

kernel_by_scale = dict(surface_kernels)

print("空间模板尺度：", list(kernel_by_scale))
assert 4.0 in kernel_by_scale


#%%
# ==================== 14. OASTER 第二步：提取时间基 ====================
# 当前还原版会先进行 active/baseline 频谱筛选，再对 15-45 ms 做 SVD。
# 只有超过基线噪声奇异值边界的时间成分会被保留。

time_basis = oaster._temporal_basis(
    joint_white,
    noise_samples=noise_samples,
)

print("OASTER 时间基形状：", time_basis.shape)
print("OASTER 时间秩：", time_basis.shape[0])


#%%
# ==================== 15. OASTER 第三步：EBIC 选择空间模板 ====================
# EBIC 会在 0、4、7 mm 模板中逐个加入候选源，直到继续加入不能降低 EBIC。
# 这份公开数据脚本目前只建了皮层表面源空间，没有深部候选，所以不运行 deep rescue。

oaster_primary, selected_templates = oaster._ebic_templates(
    joint_white,
    joint_gain_white,
    n_sources,
    surface_kernels,
    time_basis,
    ridge_fraction=oaster_ridge_fraction,
)

print("EBIC 选择的模板数：", selected_templates)
print("EBIC 主结果形状：", oaster_primary.shape)

assert oaster_primary.shape == (n_sources, len(target_times))


#%%
# ==================== 16. OASTER 第四步：计算 4 mm 频谱证据 ====================
# 这一步正是当前 OASTER 带有明显频谱定位倾向的地方。
# 它比较 active 和 baseline 的频谱功率，再投影回皮层源空间。

oaster_spectral = oaster._multiscale_spectral_evidence_source(
    joint_white,
    joint_gain_white,
    n_sources,
    kernel_by_scale[4.0],
    noise_samples=noise_samples,
)

print("频谱证据形状：", oaster_spectral.shape)


#%%
# ==================== 17. OASTER 第五步：主结果加 5% 频谱证据 ====================

oaster_joint = oaster._add_scaled_evidence(
    oaster_primary,
    oaster_spectral,
    oaster_spectral_fraction,
    noise_samples=noise_samples,
)

print("OASTER Joint 最终结果：", oaster_joint.shape)
assert np.isfinite(oaster_joint).all()


#%%
# ==================== 18. MNE 官方 dSPM 和 eLORETA ====================
# 两个对比方法直接使用同一份 epochs.average()，不是单试次定位以后再平均。

joint_forward = _merge_fwds(
    {"meg": fwd_mag_free.copy(), "eeg": fwd_eeg_free.copy()},
    verbose=False,
)

joint_channel_names = joint_forward["sol"]["row_names"]
evoked_ordered = evoked.copy().reorder_channels(joint_channel_names)
joint_forward["info"] = evoked_ordered.info

joint_noise_cov = mne.pick_channels_cov(
    noise_cov,
    include=joint_channel_names,
    exclude=[],
    ordered=True,
    copy=True,
    verbose=False,
)

inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked_ordered.info,
    joint_forward,
    joint_noise_cov,
    loose=0.0,
    fixed=True,
    depth=0.8,
    rank="info",
    use_cps=True,
    verbose=False,
)

dspm_stc = mne.minimum_norm.apply_inverse(
    evoked_ordered,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="dSPM",
    use_cps=True,
    verbose=False,
)

eloreta_stc = mne.minimum_norm.apply_inverse(
    evoked_ordered,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method="eLORETA",
    use_cps=True,
    verbose=False,
)

dspm_joint = np.vstack([
    np.interp(target_times, dspm_stc.times, source_data)
    for source_data in dspm_stc.data
])
eloreta_joint = np.vstack([
    np.interp(target_times, eloreta_stc.times, source_data)
    for source_data in eloreta_stc.data
])

print("dSPM 结果：", dspm_joint.shape)
print("eLORETA 结果：", eloreta_joint.shape)


#%%
# ==================== 19. 计算 N20 和 P30 的解剖合理性指标 ====================
# 真实数据没有源真值，所以这里不能计算 AUC、DLE 和 SD。
# S1 富集等于 1 是均匀全脑的面积零假设；大于 1 才表示左侧 S1 有富集。

geometry = label_geometry(analysis_src, subjects_dir)

method_source = {
    "OASTER Joint": oaster_joint,
    "dSPM Joint": dspm_joint,
    "eLORETA Joint": eloreta_joint,
}

method_color = {
    "OASTER Joint": "#0072B2",
    "dSPM Joint": "#CC79A7",
    "eLORETA Joint": "#D55E00",
}

result_rows = []
n20_maps = {}
p30_maps = {}

for method, source in method_source.items():
    baseline_power = np.mean(source[:, :noise_samples] ** 2, axis=1)

    n20_power = np.mean(source[:, n20_window] ** 2, axis=1)
    n20_map = np.sqrt(np.maximum(n20_power - baseline_power, 0.0))

    p30_power = np.mean(source[:, p30_window] ** 2, axis=1)
    p30_map = np.sqrt(np.maximum(p30_power - baseline_power, 0.0))

    n20_maps[method] = n20_map
    p30_maps[method] = p30_map

    n20_map_power = n20_map ** 2
    p30_map_power = p30_map ** 2

    n20_total = float(n20_map_power.sum())
    p30_total = float(p30_map_power.sum())

    n20_left_mass = float(n20_map_power[geometry["left"]].sum()) / n20_total if n20_total > 0 else 0.0
    p30_left_mass = float(p30_map_power[geometry["left"]].sum()) / p30_total if p30_total > 0 else 0.0

    n20_enrichment = n20_left_mass / float(np.mean(geometry["left"]))
    p30_enrichment = p30_left_mass / float(np.mean(geometry["left"]))

    n20_peak = int(np.argmax(n20_map))
    p30_peak = int(np.argmax(p30_map))

    n20_distance = np.min(
        np.linalg.norm(geometry["xyz"][geometry["left"]] - geometry["xyz"][n20_peak], axis=1)
    ) * 1000
    p30_distance = np.min(
        np.linalg.norm(geometry["xyz"][geometry["left"]] - geometry["xyz"][p30_peak], axis=1)
    ) * 1000

    result_rows.append({
        "subject": subject,
        "run": run,
        "method": method,
        "epochs": len(epochs),
        "n20_left_s1_enrichment": n20_enrichment,
        "n20_peak_euclidean_distance_to_left_s1_mm": n20_distance,
        "n20_peak_label": geometry["names"][n20_peak],
        "p30_left_s1_enrichment": p30_enrichment,
        "p30_peak_euclidean_distance_to_left_s1_mm": p30_distance,
        "p30_peak_label": geometry["names"][p30_peak],
    })

result_table = pd.DataFrame(result_rows)
result_table.to_csv(save_dir / "2-方法指标.csv", index=False, encoding="utf-8-sig")

print("\n定位指标：")
print(result_table.to_string(index=False))


#%%
# ==================== 20. 保存所有方法的源结果 ====================

np.savez_compressed(
    save_dir / "3-源定位结果.npz",
    target_times=target_times,
    vertices_lh=geometry["vertices"][0],
    vertices_rh=geometry["vertices"][1],
    oaster_joint=oaster_joint,
    dspm_joint=dspm_joint,
    eloreta_joint=eloreta_joint,
    oaster_n20=n20_maps["OASTER Joint"],
    dspm_n20=n20_maps["dSPM Joint"],
    eloreta_n20=n20_maps["eLORETA Joint"],
    oaster_p30=p30_maps["OASTER Joint"],
    dspm_p30=p30_maps["dSPM Joint"],
    eloreta_p30=p30_maps["eLORETA Joint"],
)


#%%
# ==================== 21. 画三个方法的指标对比 ====================

method_names = result_table["method"].tolist()
x = np.arange(len(method_names))
colors = [method_color[name] for name in method_names]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")

plot_items = [
    ("n20_left_s1_enrichment", "N20 左S1富集", "富集倍数"),
    ("p30_left_s1_enrichment", "P30/P35 左S1富集", "富集倍数"),
    ("n20_peak_euclidean_distance_to_left_s1_mm", "N20 峰到左S1距离", "距离 (mm)"),
    ("p30_peak_euclidean_distance_to_left_s1_mm", "P30/P35 峰到左S1距离", "距离 (mm)"),
]

for axis, (column, title, ylabel) in zip(axes.ravel(), plot_items):
    values = result_table[column].to_numpy(dtype=float)
    bars = axis.bar(x, values, color=colors, width=0.62)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x, method_names, rotation=18, ha="right")
    axis.grid(axis="y", color="#D9E1E8", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    if "enrichment" in column:
        axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1, label="面积零假设=1")
        axis.legend(loc="upper right")

fig.suptitle(f"sub-{subject} run-{run} | ERP叠加平均后的源定位", fontsize=15, fontweight="bold")
fig.savefig(save_dir / "4-方法指标对比.png", dpi=180, bbox_inches="tight")
plt.close(fig)


#%%
# ==================== 22. 可选：把 N20 结果渲染到脑表面 ====================
# 参数区 draw_brain=False 时不会运行这一块，也不会打开三维窗口。

if draw_brain:
    brain_images = []

    for method in method_names:
        brain_map = n20_maps[method]
        brain_peak = max(float(brain_map.max(initial=0.0)), np.finfo(float).eps)
        brain_floor = float(np.percentile(brain_map, 95))
        if brain_floor >= brain_peak:
            brain_floor = 0.0

        display_map = np.where(brain_map >= brain_floor, brain_map, 0.0)
        stc = mne.SourceEstimate(
            display_map[:, None],
            vertices=geometry["vertices"],
            tmin=0.0,
            tstep=1.0,
            subject="sample",
        )

        brain = stc.plot(
            subject="sample",
            subjects_dir=subjects_dir,
            surface="pial",
            hemi="split",
            views="lateral",
            initial_time=0.0,
            time_viewer=False,
            show_traces=False,
            colormap="inferno",
            clim={"kind": "value", "lims": [brain_floor, (brain_floor + brain_peak) / 2, brain_peak]},
            smoothing_steps=5,
            transparent=True,
            background="white",
            foreground="black",
            cortex="classic",
            colorbar=False,
            size=(1000, 520),
            backend="pyvistaqt",
            brain_kwargs={"show": False, "theme": "light"},
        )

        brain_images.append((method, brain.screenshot(mode="rgb", time_viewer=False)))
        brain.close()

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), layout="constrained")

    for axis, (method, brain_image) in zip(axes, brain_images):
        axis.imshow(brain_image)
        axis.set_title(f"{method} | N20 | 只显示 P95 以上")
        axis.set_axis_off()

    fig.savefig(save_dir / "5-N20脑空间结果.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


#%%
# ==================== 23. 完成 ====================

print("\n运行完成，结果保存在：")
print(save_dir)
print("这份脚本没有自定义 def，可以在 PyCharm 里按 #%% 一块一块运行。")
