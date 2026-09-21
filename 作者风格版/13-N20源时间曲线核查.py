# %%
# sm09 开发数据：查看现有逆解的左 S1 源时程，而不只看 18–24 ms 的功率图。
# OASTER-v3 当前只在 N20/P30 两个选定窗口重建；窗口外零值不是生理学静默。

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# %%
# ==================== 1. 固定数据和输入 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
dataset = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
subjects_dir = Path(r"C:\Users\zzp\mne_data\MNE-sample-data\subjects")
output = project_root / "results" / "real_data" / "ds006035_n20_diagnostic" / "source_timecourse_sm09_run1"
output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project_root))

from benchmark import methods as comparators
from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected
from pipelines import run_ds006035_somatomotor as real

mne.set_log_level("WARNING")
raw_path, events_path = real._paths(dataset, "sm09", 1)
sensor, noise, evoked, noise_cov, qc = real.preprocess_run(raw_path, events_path)
trans, coreg = real.coregister(evoked.info, subjects_dir)
src = mne.setup_source_space(
    "sample", spacing="ico4", subjects_dir=subjects_dir,
    add_dist=False, n_jobs=1, verbose=False,
)
eeg_fwd, mag_fwd, _eeg_free, _mag_free = real.make_fixed_forwards(
    evoked.info, trans, src, subjects_dir,
)
geometry = real.label_geometry(eeg_fwd["src"], subjects_dir)
eeg, mag, ge, gm, eeg_noise, mag_noise = real.aligned_data_and_gain(
    sensor, noise, evoked.info, (eeg_fwd, mag_fwd),
)
eeg_white, eeg_gain = real.whiten_from_trials(eeg, ge, eeg_noise)
mag_white, mag_gain = real.whiten_from_trials(mag, gm, mag_noise)
data = np.vstack((eeg_white, mag_white))
gain = np.vstack((eeg_gain, mag_gain))
baseline = real.TARGET_TIMES < -0.05
data -= data[:, baseline].mean(axis=1, keepdims=True)
assert gain.shape[1] == len(geometry["left"]) and geometry["left"].any()


# %%
# ==================== 2. 原样重算两种方法 ====================

adjacency = mne.spatial_src_adjacency(eeg_fwd["src"], verbose=False).toarray()
kernels = protected.connected_euclidean_surface_kernels(
    geometry["xyz"], adjacency, gain.shape[1],
)
channel_weights = []
for window in (real.N20, real.P30):
    eeg_weight, mag_weight = real.modality_evidence_weights(
        (data[:len(eeg_white)], data[len(eeg_white):]), baseline, window,
    )
    channel_weights.append(np.r_[
        np.full(len(eeg_white), eeg_weight),
        np.full(len(mag_white), mag_weight),
    ])

oaster_source, oaster_diagnostics = oaster.reconstruct_evoked_oaster_v3_from_whitened(
    data, gain, gain.shape[1], kernels,
    baseline=baseline, active_windows=(real.N20, real.P30),
    window_channel_weights=channel_weights, require_one=False,
)
dspm_source = comparators.minimum_norm_family(data, gain)["dSPM"]
time_ms = real.TARGET_TIMES * 1000.0
poststim = time_ms >= 15.0
unfilled = poststim & ~real.N20 & ~real.P30
assert np.all(oaster_source[:, unfilled] == 0.0)
assert np.any(np.abs(dspm_source[:, unfilled]) > 0.0)


# %%
# ==================== 3. 左 S1 时程：每种方法各按全皮层峰值归一 ====================

fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, facecolor="white")
roi_curves = {}
for ax, name, source, color in zip(
    axes,
    ("OASTER-ERP-v3", "dSPM"),
    (oaster_source, dspm_source),
    ("#0072B2", "#D55E00"),
    strict=True,
):
    denominator = max(
        float(np.max(np.abs(source[:, poststim]))), np.finfo(float).eps,
    )
    left_s1_rms = np.sqrt(np.mean(source[geometry["left"]][:, poststim] ** 2, axis=0))
    global_peak = np.max(np.abs(source[:, poststim]), axis=0)
    roi_curves[name] = 100.0 * left_s1_rms / denominator
    ax.plot(time_ms[poststim], 100.0 * global_peak / denominator,
            color="#555555", lw=1.4, label="全皮层逐时刻最大绝对值")
    ax.plot(time_ms[poststim], 100.0 * left_s1_rms / denominator,
            color=color, lw=2.4, label="左 S1 (postcentral) ROI RMS")
    ax.axvspan(18, 24, color="#56B4E9", alpha=0.16, label="N20 18–24 ms")
    ax.axvspan(28, 40, color="#E69F00", alpha=0.13, label="P30 28–40 ms")
    ax.set_ylim(-2, 104)
    ax.set_ylabel("占本方法全皮层峰值 (%)")
    ax.set_title(name, loc="left", fontsize=12)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=8, ncol=2, loc="upper right")

axes[-1].set_xlabel("电刺激后时间 (ms)")
axes[-1].set_xlim(15, 45)
fig.suptitle(
    "ds006035 sm09/run-1｜右腕刺激｜左 S1 源时程（固定 ico4 模板）\n"
    "OASTER 段外零值为实现限制；RMS 不表示 N20 波形极性",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
image_path = output / "n20_left_s1_source_timecourse_oaster_vs_dspm.png"
fig.savefig(image_path, dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
for name, color in (("OASTER-ERP-v3", "#0072B2"), ("dSPM", "#D55E00")):
    ax.plot(time_ms[poststim], roi_curves[name], lw=2.3, color=color, label=name)
ax.axvspan(18, 24, color="#56B4E9", alpha=0.16, label="N20 18–24 ms")
ax.set(xlim=(15, 27), ylim=(0, 12), xlabel="电刺激后时间 (ms)",
       ylabel="左 S1 ROI RMS / 本方法全皮层峰值 (%)",
       title="sm09/run-1｜N20 窗左 S1 源活动局部放大")
ax.grid(alpha=0.18)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(output / "n20_left_s1_roi_zoom.png", dpi=180)
plt.close(fig)

oaster_metrics, _, _ = real.source_metrics(oaster_source, geometry)
dspm_metrics, _, _ = real.source_metrics(dspm_source, geometry)
report = (
    "# sm09/run-1 左 S1 源时程诊断\n\n"
    "同一 EEG+MEG evoked、白化前向模型、sample ico4 皮层网格。"
    "图中两方法分别按各自全皮层、全时间绝对峰值归一，不能跨方法比较物理幅值。\n\n"
    "OASTER-v3 只在 18–24 和 28–40 ms 填充源估计；"
    "15–17、25–27、41–45 ms 的零值由代码强制产生，不是神经静默。"
    "ROI RMS 只显示幅度包络，不代表 N20 的有符号极性。\n\n"
    f"sm09 N20 左 S1 富集：OASTER {oaster_metrics['n20_left_s1_enrichment']:.3f}，"
    f"dSPM {dspm_metrics['n20_left_s1_enrichment']:.3f}。"
    "因此此图不能证明 OASTER 成功重建 N20；必须结合传感器 ERP 和独立重复验证。\n"
)
(output / "REPORT.md").write_text(report, encoding="utf-8")
print("图：", image_path)
print("说明：", output / "REPORT.md")
