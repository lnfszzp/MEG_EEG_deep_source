#%%
# ds006035 左指反应锁定 beta-ERD：DICS、LCMV 和最小范数家族。
# 复用脚本 5 的严格配对、刺激伪迹处理、逐 run forward 和 DICS，
# 所有方法只比较同一批 response epoch、15–30 Hz、-1.50~-1.00 s 基线
# 与 -0.40~0.20 s 活动窗。原脚本结果另存，不覆盖历史输出。
# 这是一个受试者 sm09 的探索性解剖合理性比较，不是定位真值验证。

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


#%%
# ==================== 1. 参数区 ====================

base_script = Path(__file__).with_name("5-左指运动双链定位.py")
output_root = "sub-sm09_finger_beta_multimethod"
beta_low_hz = 15.0
beta_high_hz = 30.0
sample_rate_hz = 200.0
lambda2 = 1.0 / 9.0
depth = 0.8
started = time.perf_counter()

old_output_root = os.environ.get("OASTER_FINGER_SAVE_ROOT")
os.environ["OASTER_FINGER_SAVE_ROOT"] = output_root
try:
    previous = runpy.run_path(str(base_script))
finally:
    if old_output_root is None:
        os.environ.pop("OASTER_FINGER_SAVE_ROOT", None)
    else:
        os.environ["OASTER_FINGER_SAVE_ROOT"] = old_output_root

save_dir = previous["save_dir"]
run_inputs = previous["run_inputs"]
source_xyz = previous["source_xyz"]
source_vertices = previous["source_vertices"]
roi_masks = previous["roi_masks"]
source_labels = previous["source_label_names"]
n_sources = len(source_xyz)
assert len(run_inputs) == 3
assert previous["response_baseline"] == (-1.50, -1.00)
assert previous["dics_erd_window"] == (-0.40, 0.20)
assert np.array_equal(previous["dics_frequencies"]["beta_15_30Hz"], np.arange(15.0, 31.0, 2.0))


#%%
# ==================== 2. 每个 run 固定一个滤波器/逆算子 ====================
# DICS 原样使用脚本 5 的单试次 Morlet CSD。LCMV 和最小范数用同一
# 15–30 Hz 带通信号的协方差；三者是不同估计器，不能直接比绝对幅值。

run_maps = {}
for run_input in run_inputs:
    run = run_input["run"]
    epochs = run_input["response_epochs"].copy().apply_proj(verbose=False)
    epochs.filter(beta_low_hz, beta_high_hz, n_jobs=1, verbose=False)
    epochs.resample(sample_rate_hz, npad="auto", n_jobs=1, verbose=False)
    forward = run_input["joint_forward"].copy()
    epochs.reorder_channels(forward["sol"]["row_names"])
    forward["info"] = epochs.info

    common_cov = mne.compute_covariance(
        epochs, tmin=-1.50, tmax=1.00, method="empirical", rank="info", verbose=False
    )
    baseline_cov = mne.compute_covariance(
        epochs, tmin=-1.50, tmax=-1.00, method="empirical", rank="info", verbose=False
    )
    movement_cov = mne.compute_covariance(
        epochs, tmin=-0.40, tmax=0.20, method="empirical", rank="info", verbose=False
    )

    filters = mne.beamformer.make_lcmv(
        epochs.info, forward, common_cov, reg=0.05, noise_cov=baseline_cov,
        pick_ori="max-power", rank="info", weight_norm="unit-noise-gain",
        reduce_rank=True, depth=depth, inversion="matrix", verbose=False,
    )
    baseline_stc = mne.beamformer.apply_lcmv_cov(baseline_cov, filters, verbose=False)
    movement_stc = mne.beamformer.apply_lcmv_cov(movement_cov, filters, verbose=False)
    assert all(np.array_equal(v, expected) for v, expected in zip(baseline_stc.vertices, source_vertices))
    baseline_power = np.asarray(baseline_stc.data[:, 0], dtype=float)
    movement_power = np.asarray(movement_stc.data[:, 0], dtype=float)
    floor = max(float(np.median(np.r_[baseline_power, movement_power])) * 1e-12, np.finfo(float).tiny)
    run_maps[(run, "LCMV")] = 10.0 * np.log10((baseline_power + floor) / (movement_power + floor))

    # 没有独立空室噪声：基线协方差只作为两个条件共用的逆算子参考，
    # 它不构成独立信噪比或临床精度证明。
    inverse = mne.minimum_norm.make_inverse_operator(
        epochs.info, forward, baseline_cov, loose=0.0, fixed=True,
        depth=depth, rank="info", verbose=False,
    )
    for method in ("MNE", "dSPM", "sLORETA", "eLORETA"):
        baseline_stc = mne.minimum_norm.apply_inverse_cov(
            baseline_cov, epochs.info, inverse, lambda2=lambda2,
            method=method, verbose=False,
        )
        movement_stc = mne.minimum_norm.apply_inverse_cov(
            movement_cov, epochs.info, inverse, lambda2=lambda2,
            method=method, verbose=False,
        )
        assert all(np.array_equal(v, expected) for v, expected in zip(baseline_stc.vertices, source_vertices))
        baseline_power = np.asarray(baseline_stc.data[:, 0], dtype=float)
        movement_power = np.asarray(movement_stc.data[:, 0], dtype=float)
        floor = max(float(np.median(np.r_[baseline_power, movement_power])) * 1e-12, np.finfo(float).tiny)
        run_maps[(run, method)] = 10.0 * np.log10(
            (baseline_power + floor) / (movement_power + floor)
        )

    run_maps[(run, "DICS")] = previous["dics_run_maps"][(run, "beta_15_30Hz", "ERD")]
    assert all(np.isfinite(run_maps[(run, method)]).all() for method in ("DICS", "LCMV", "MNE", "dSPM", "sLORETA", "eLORETA"))
    assert all(run_maps[(run, method)].shape == (n_sources,) for method in ("DICS", "LCMV", "MNE", "dSPM", "sLORETA", "eLORETA"))
    print(f"run-{run}：六种 beta-ERD 图完成，试次数 {len(epochs)}")


#%%
# ==================== 3. 同一网格、同一 ROI 和同一评分公式 ====================

methods = ("DICS", "LCMV", "MNE", "dSPM", "sLORETA", "eLORETA")
weights = np.asarray([len(item["response_epochs"]) for item in run_inputs], dtype=float)
pooled_maps = {
    method: np.average(
        np.asarray([run_maps[(item["run"], method)] for item in run_inputs]),
        axis=0, weights=weights,
    )
    for method in methods
}

rows = []
for run, n_trials, maps in [
    (item["run"], len(item["response_epochs"]), {method: run_maps[(item["run"], method)] for method in methods})
    for item in run_inputs
] + [("pooled_1_2_3", int(weights.sum()), pooled_maps)]:
    for method, erd_db in maps.items():
        positive = np.maximum(erd_db, 0.0)
        peak = int(np.argmax(positive))
        mass = positive**2
        total = float(mass.sum())
        for roi_name, roi_mask in roi_masks.items():
            roi_fraction = float(roi_mask.mean())
            rows.append({
                "subject": "sm09", "run": run, "method": method,
                "n_trials": n_trials, "band_hz": "15-30",
                "baseline_ms": "-1500:-1000", "movement_ms": "-400:200",
                "roi": roi_name,
                "roi_mass_pct": 100.0 * float(mass[roi_mask].sum()) / total if total > 0 else 0.0,
                "roi_enrichment": float(mass[roi_mask].sum()) / total / roi_fraction if total > 0 else 0.0,
                "peak_distance_to_roi_mm": float(np.min(np.linalg.norm(source_xyz[roi_mask] - source_xyz[peak], axis=1)) * 1000.0),
                "peak_positive_erd_db": float(positive[peak]),
                "peak_label": str(source_labels[peak]),
                "peak_hemi": "left" if peak < len(source_vertices[0]) else "right",
                "ground_truth_available": False,
            })

for method, reason in (
    ("Dipole fitting (grid)", "Existing fit estimates evoked time courses, not beta power contrast."),
    ("RAP-MUSIC", "Existing fit estimates evoked time courses, not beta power contrast."),
    ("OASTER spectral", "Current active-over-baseline filter selects power increase; beta-ERD is power decrease."),
):
    rows.append({"subject": "sm09", "run": "pooled_1_2_3", "method": method,
                 "n_trials": int(weights.sum()), "band_hz": "15-30",
                 "baseline_ms": "-1500:-1000", "movement_ms": "-400:200",
                 "roi": "requires_adaptation", "limitation": reason,
                 "ground_truth_available": False})

comparison_file = save_dir / "beta_multimethod_comparison.csv"
with comparison_file.open("w", encoding="utf-8-sig", newline="") as handle:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

np.savez_compressed(
    save_dir / "beta_multimethod_source_maps.npz",
    vertices_lh=source_vertices[0], vertices_rh=source_vertices[1],
    source_xyz=source_xyz, **{f"pooled__{method}": source for method, source in pooled_maps.items()},
    **{f"run_{run}__{method}": source for (run, method), source in run_maps.items()},
)

normalization_check = {
    method: float(np.max(np.abs(pooled_maps[method] - pooled_maps["MNE"])))
    for method in ("dSPM", "sLORETA")
}
(save_dir / "beta_normalization_check.json").write_text(
    json.dumps(normalization_check, ensure_ascii=False, indent=2), encoding="utf-8"
)

#%%
# ==================== 4. 指标图：只比较 ROI 分布，不比较方法间绝对幅值 ====================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
colors = ("#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#E69F00")
for index, method in enumerate(methods):
    result = next(row for row in rows if row["run"] == "pooled_1_2_3" and row["method"] == method and row["roi"] == "right_precentral")
    axes[0].scatter(result["roi_enrichment"], index, s=70, color=colors[index])
    axes[1].scatter(result["peak_distance_to_roi_mm"], index, s=70, color=colors[index])
axes[0].axvline(1.0, color="0.45", linestyle="--", linewidth=1)
for axis in axes:
    axis.set_yticks(range(len(methods)), methods)
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
axes[0].set_xlabel("Right precentral enrichment; 1 = uniform-grid null")
axes[1].set_xlabel("Peak-to-right-precentral boundary (mm)")
fig.suptitle("ds006035 sm09 finger: matched 15–30 Hz beta-ERD, 3 runs")
figure_file = save_dir / "beta_multimethod_roi_comparison.png"
fig.savefig(figure_file, dpi=180)
plt.close(fig)

print("指标表：", comparison_file)
print("逐源图：", save_dir / "beta_multimethod_source_maps.npz")
print("指标图：", figure_file)
print("总耗时（秒）：", round(time.perf_counter() - started, 1))
