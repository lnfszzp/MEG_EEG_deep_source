#%%
# ds004998 HoldL：在同一批 rest/Hold beta epoch、同一 MEG forward 上比较源功率变化。
# 先复用 4-DBS频谱定位.py 的 DICS/LCMV 和预处理，另存 v3；不改写已有 v2。
# MNE、dSPM、sLORETA、eLORETA 使用同一个 pooled inverse operator。
# Dipole、RAP-MUSIC 和旧频谱 OASTER 目前不是这个 beta 功率对比的直接估计器，
# 在表中保留“需适配”，不填伪造数值。

from pathlib import Path
import csv
import json
import os
import runpy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.ndimage import gaussian_filter


#%%
# ==================== 1. 参数区 ====================

base_script = Path(__file__).with_name("4-DBS频谱定位.py")
output_root = "ds004998_dbs_v3_multimethod"
lambda2 = 1.0 / 9.0
depth = 0.8
figure_percentile = 95.0
started = time.perf_counter()

previous_output_root = os.environ.get("OASTER_DBS_SAVE_ROOT")
os.environ["OASTER_DBS_SAVE_ROOT"] = output_root
try:
    previous = runpy.run_path(str(base_script))
finally:
    if previous_output_root is None:
        os.environ.pop("OASTER_DBS_SAVE_ROOT", None)
    else:
        os.environ["OASTER_DBS_SAVE_ROOT"] = previous_output_root

save_dir = previous["save_dir"]
info = previous["common_beta_epochs"].info
forward = previous["forward"]
rest_cov = previous["rest_beta_cov"]
hold_cov = previous["hold_beta_cov"]
pooled_cov = previous["common_beta_cov"]
source_vertices = np.asarray(previous["used_local_indices"], dtype=int)
positions_mm = previous["mni_positions_mm"]
right_roi = previous["right_motor_roi"]
left_roi = previous["left_motor_roi"]
right_distance_mm = previous["right_motor_distance_mm"]
assert len(source_vertices) == len(positions_mm) == forward["nsource"]


#%%
# ==================== 2. 共同最小范数逆算子与四种标准化 ====================

# 这里没有独立空室噪声，使用两条件等量合并后的 beta 协方差作固定白化参考。
# 这不是独立噪声测量；但条件间绝不重建逆算子，避免滤波器变化伪装成 ERD。
inverse = mne.minimum_norm.make_inverse_operator(
    info,
    forward,
    pooled_cov,
    loose=1.0,
    fixed=False,
    depth=depth,
    rank="info",
    verbose=False,
)

erd_by_method = {
    "DICS": np.asarray(previous["rest_vs_hold_erd_db"], dtype=float),
    "LCMV": np.asarray(previous["lcmv_erd_db"], dtype=float),
}
power_by_method = {
    "DICS": (
        np.asarray(previous["rest_beta_power"], dtype=float),
        np.asarray(previous["hold_beta_power"], dtype=float),
    ),
    "LCMV": (
        np.asarray(previous["lcmv_rest_power"], dtype=float),
        np.asarray(previous["lcmv_hold_power"], dtype=float),
    ),
}
for method in ("MNE", "dSPM", "sLORETA", "eLORETA"):
    rest_stc = mne.minimum_norm.apply_inverse_cov(
        rest_cov, info, inverse, lambda2=lambda2, method=method, verbose=False
    )
    hold_stc = mne.minimum_norm.apply_inverse_cov(
        hold_cov, info, inverse, lambda2=lambda2, method=method, verbose=False
    )
    assert np.array_equal(rest_stc.vertices[0], source_vertices)
    assert np.array_equal(hold_stc.vertices[0], source_vertices)
    rest_power = np.asarray(rest_stc.data[:, 0], dtype=float)
    hold_power = np.asarray(hold_stc.data[:, 0], dtype=float)
    assert len(rest_power) == len(positions_mm)
    assert np.isfinite(rest_power).all() and np.isfinite(hold_power).all()
    assert np.all(rest_power >= 0) and np.all(hold_power >= 0)
    floor = max(
        float(np.median(np.r_[rest_power, hold_power])) * 1e-12,
        np.finfo(float).tiny,
    )
    erd_by_method[method] = 10.0 * np.log10(
        (rest_power + floor) / (hold_power + floor)
    )
    power_by_method[method] = (rest_power, hold_power)
    print(method, "beta-ERD 完成")

normalization_checks = {}
for method in ("dSPM", "sLORETA"):
    maximum_difference = float(np.max(np.abs(erd_by_method[method] - erd_by_method["MNE"])))
    normalization_checks[method] = {
        "max_abs_erd_difference_db_vs_mne": maximum_difference,
        "same_erd_as_mne_tolerance_1e-6_db": bool(maximum_difference < 1e-6),
    }
    print(method, "与 MNE ERD 最大差值 (dB)：", maximum_difference)


#%%
# ==================== 3. 同一解剖假设下的指标表 ====================

comparison_rows = []
estimator_by_method = {
    "DICS": "pooled_filter_multitaper_CSD_13-30Hz",
    "LCMV": "pooled_filter_bandpass_covariance_13-30Hz",
    "MNE": "pooled_inverse_bandpass_covariance_13-30Hz",
    "dSPM": "pooled_inverse_bandpass_covariance_13-30Hz",
    "sLORETA": "pooled_inverse_bandpass_covariance_13-30Hz",
    "eLORETA": "pooled_inverse_bandpass_covariance_13-30Hz",
}
for method, erd_db in erd_by_method.items():
    peak_index = int(np.argmax(erd_db))
    high = (erd_db > 0.0) & (
        erd_db >= max(float(np.percentile(erd_db, figure_percentile)), 0.0)
    )
    positive = np.maximum(erd_db, 0.0)
    right_mean = float(positive[right_roi].mean())
    left_mean = float(positive[left_roi].mean())
    right_count = int(np.sum(high & (positions_mm[:, 0] > 0)))
    left_count = int(np.sum(high & (positions_mm[:, 0] < 0)))
    comparison_rows.append({
        "method": method,
        "status": "computed",
        "estimator": estimator_by_method[method],
        "peak_mni_x_mm": float(positions_mm[peak_index, 0]),
        "peak_mni_y_mm": float(positions_mm[peak_index, 1]),
        "peak_mni_z_mm": float(positions_mm[peak_index, 2]),
        "peak_erd_db": float(erd_db[peak_index]),
        "peak_to_right_roi_center_mm": float(right_distance_mm[peak_index]),
        "peak_to_right_roi_boundary_mm": float(max(0.0, right_distance_mm[peak_index] - previous["motor_roi_radius_mm"])),
        "right_roi_mean_positive_erd_db": right_mean,
        "left_roi_mean_positive_erd_db": left_mean,
        "roi_laterality_index": float((right_mean - left_mean) / (right_mean + left_mean + np.finfo(float).eps)),
        "p95_right_count": right_count,
        "p95_left_count": left_count,
        "p95_count_laterality_index": float((right_count - left_count) / max(right_count + left_count, 1)),
        "display_p95_threshold_db": float(np.percentile(erd_db, figure_percentile)),
        "limitation": "No source truth; ROI distance is not DLE.",
    })
for method, reason in (
    ("Dipole fitting (grid)", "Existing fit estimates evoked time courses, not pooled beta power contrast."),
    ("RAP-MUSIC", "Existing fit estimates evoked time courses, not pooled beta power contrast."),
    ("OASTER spectral", "Existing active-over-baseline filter targets power increase and needs surface kernels; Hold beta-ERD is a decrease on a volume grid."),
):
    comparison_rows.append({
        "method": method,
        "status": "requires_adaptation",
        "estimator": "",
        "limitation": reason,
    })

comparison_file = save_dir / "beta_multimethod_comparison.csv"
fieldnames = list(comparison_rows[0])
with comparison_file.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_rows)

source_file = save_dir / "beta_multimethod_source_maps.csv"
with source_file.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["source_index", "mni_x_mm", "mni_y_mm", "mni_z_mm"]
        + [f"{method}_erd_db" for method in erd_by_method]
    )
    for index in range(len(positions_mm)):
        writer.writerow(
            [index, *positions_mm[index].tolist()]
            + [float(values[index]) for values in erd_by_method.values()]
        )


#%%
# ==================== 4. 完全相同模板 MRI 切面上的六方法图 ====================

brain_figure_file = None
if previous["draw_template_anatomy"]:
    fsaverage_volume = previous["fsaverage_volume"]
    valid_indices = previous["valid_indices"]
    valid_voxels = previous["valid_voxels"]
    center_voxel = previous["center_voxel"]
    x_voxel, y_voxel, z_voxel = center_voxel
    anatomy_slices = (
        fsaverage_volume[x_voxel, :, :].T,
        fsaverage_volume[:, y_voxel, :].T,
        fsaverage_volume[:, :, z_voxel].T,
    )
    fig, axes = plt.subplots(len(erd_by_method), 3, figsize=(15.0, 22.0), facecolor="white")
    for row_index, (method, erd_db) in enumerate(erd_by_method.items()):
        threshold = max(float(np.percentile(erd_db, figure_percentile)), 0.0)
        peak = float(np.max(erd_db))
        weights = np.clip((erd_db - threshold) / max(peak - threshold, np.finfo(float).eps), 0.0, 1.0)
        activation_volume = np.zeros(fsaverage_volume.shape, dtype=np.float32)
        np.maximum.at(
            activation_volume,
            (valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]),
            weights[valid_voxels],
        )
        activation_volume = gaussian_filter(activation_volume, sigma=4.0)
        activation_volume /= max(float(activation_volume.max()), np.finfo(float).eps)
        activation_slices = (
            activation_volume[x_voxel, :, :].T,
            activation_volume[:, y_voxel, :].T,
            activation_volume[:, :, z_voxel].T,
        )
        for axis, anatomy, activation, plane in zip(
            axes[row_index], anatomy_slices, activation_slices, ("Sagittal", "Coronal", "Axial")
        ):
            nonzero = anatomy[anatomy > 0]
            low, high = np.percentile(nonzero, [1, 99])
            axis.imshow(np.clip((anatomy - low) / (high - low), 0.0, 1.0), cmap="gray", origin="lower")
            axis.imshow(np.ma.masked_less(activation, 0.06), cmap="inferno", origin="lower", vmin=0, vmax=1, alpha=0.82)
            axis.set_title(f"{method} | {plane}", fontsize=11, fontweight="bold")
            axis.set_axis_off()
    fig.suptitle(
        "ds004998 HoldL | 13–30 Hz beta ERD: 10 log10(rest / Hold)\n"
        "Identical fsaverage MRI slices; each method independently displays its positive P95+",
        fontsize=14,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, hspace=0.18, wspace=0.02)
    brain_figure_file = save_dir / "beta_multimethod_same_slices_mri.png"
    fig.savefig(brain_figure_file, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


#%%
# ==================== 5. 校验与解读边界 ====================

with comparison_file.open(encoding="utf-8-sig", newline="") as handle:
    saved_rows = list(csv.DictReader(handle))
assert len(saved_rows) == 9
assert sum(row["status"] == "computed" for row in saved_rows) == 6
assert all(len(values) == len(positions_mm) for values in erd_by_method.values())
assert len({row["method"] for row in saved_rows}) == len(saved_rows)

metadata = {
    "dataset": "ds004998",
    "subject": previous["subject"],
    "task": previous["task"],
    "medication": previous["medication"],
    "run": previous["run"],
    "methods_computed": list(erd_by_method),
    "methods_requiring_adaptation": ["Dipole fitting (grid)", "RAP-MUSIC", "OASTER spectral"],
    "channels": len(info["ch_names"]),
    "equal_epochs_per_condition": previous["equal_epoch_count"],
    "source_points": len(positions_mm),
    "beta_hz": [previous["beta_low_hz"], previous["beta_high_hz"]],
    "inverse_noise_reference": "pooled rest/Hold beta covariance, not independent empty-room noise",
    "inverse_lambda2": lambda2,
    "inverse_depth": depth,
    "normalization_checks": normalization_checks,
    "interpretation_limit": "One run, no source truth: peak-to-ROI is not DLE; method-specific P95 does not compare absolute power or extent; fsaverage anatomy only for display.",
    "elapsed_seconds": float(time.perf_counter() - started),
}
metadata_file = save_dir / "beta_multimethod_metadata.json"
metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

report = [
    "# ds004998 HoldL：六种可比 beta-ERD 方法",
    "",
    "DICS、LCMV、MNE、dSPM、sLORETA、eLORETA 使用同一条记录、等量 rest/Hold epoch、"
    "202 个 grad、同一体积源网格与 13–30 Hz，比较 `10 log10(rest power / Hold power)`。",
    "所有逆算子在两条件间固定。MNE 家族使用 pooled beta 协方差作白化参考，"
    "它不是独立空室噪声，故此结果只能用于探索性方法比较。",
    "",
    f"- [指标及方法状态]({comparison_file.name})",
    f"- [逐源 ERD 图数据]({source_file.name})",
    f"- [相同 MRI 切面图]({brain_figure_file.name})" if brain_figure_file else "- 本机缺少 fsaverage T1，未渲染 MRI 图。",
    "",
    "固定方向时 MNE/dSPM/sLORETA 的逐源常数缩放会在功率比中抵消；"
    "本次自由方向逆解是否数值相同，以 metadata 的逐点最大差值为准，不能把相同图重复当独立证据。",
    "Dipole fitting、RAP-MUSIC 与旧频谱 OASTER 尚无同目标 beta 功率对比实现，"
    "表中保留未运行状态，不能以 ERP/波形定位替代。",
    "",
    "单个患者的一条真实记录没有源真值，不计算 AUC/DLE；峰到右感觉运动 ROI 的距离只是解剖合理性检查。"
    "各方法独立 P95 显示阈值仅用于看空间形状，不能通过颜色亮度或面积判断胜负。"
    " MRI 是 fsaverage 模板而非患者个体解剖；STN-LFP 未进入 MEG 逆解，不能据此定位 DBS 电极。",
]
(save_dir / "MULTIMETHOD_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

print("六方法对比表：", comparison_file)
print("逐源表：", source_file)
print("MRI 脑图：", brain_figure_file)
print("归一化等价性：", normalization_checks)
print("总耗时（秒）：", metadata["elapsed_seconds"])
