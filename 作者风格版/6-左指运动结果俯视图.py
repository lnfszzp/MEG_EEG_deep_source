#%%
# 读取 5-左指运动双链定位.py 已保存的源图，直接渲染完整 pial 俯视图。
# 不重新做预处理和逆解；没有五角星、圆圈或 ROI 叠加。

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
subjects_dir = sample_data_path / "subjects"
subject = "sample"
result_dir = (
    project_root
    / "results"
    / "real_data"
    / "ds006035"
    / "sub-sm09_finger_dual_chain"
)

time_file = result_dir / "time_domain_run_source_maps.npz"
dics_file = result_dir / "dics_pooled_source_maps.npz"
assert time_file.is_file(), f"找不到逐 run 时域源图，请先重跑脚本 5：{time_file}"
assert dics_file.is_file(), f"找不到频域源图：{dics_file}"
assert (subjects_dir / subject).is_dir(), f"找不到 sample 解剖：{subjects_dir}"

display_percentile = 95.0
camera_zoom = 0.72
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
mne.set_log_level("warning")


#%%
time_run_maps = np.load(time_file)
dics_maps = np.load(dics_file)
time_metrics = pd.read_csv(result_dir / "time_domain_method_comparison.csv")
vertices = [time_run_maps["vertices_lh"], time_run_maps["vertices_rh"]]
assert np.array_equal(vertices[0], dics_maps["vertices_lh"])
assert np.array_equal(vertices[1], dics_maps["vertices_rh"])

methods = ("OASTER-ERP", "dSPM", "eLORETA")
components = (
    ("MF_response_-80_-20ms", "MF：-80~-20 ms"),
    ("MEFI_response_20_60ms", "MEFI：20~60 ms"),
    ("MEFII_response_120_180ms", "MEFII：120~180 ms"),
)
single_run_metrics = time_metrics[
    time_metrics["aggregation"].astype(str) == "single_run"
].copy()
single_run_metrics["run"] = single_run_metrics["run"].astype(int)
oaster_detection = single_run_metrics[
    single_run_metrics["method"] == "OASTER-ERP"
].drop_duplicates(["component", "run"])[
    ["component", "run", "localization_available"]
]
oaster_detection["localization_available"] = (
    oaster_detection["localization_available"].astype(str).str.lower().eq("true")
)
oaster_detection = oaster_detection.groupby("component", as_index=False).agg(
    successful_runs=("localization_available", "sum"),
    total_runs=("run", "nunique"),
)
oaster_detection["success_rate_pct"] = (
    100.0 * oaster_detection["successful_runs"] / oaster_detection["total_runs"]
)
oaster_detection.to_csv(
    result_dir / "oaster_detection_rate.csv",
    index=False,
    encoding="utf-8-sig",
)
oaster_detection_by_component = oaster_detection.set_index("component")

# 每一行先求三种方法共同可定位的 run，再用相同试次数权重聚合。
matched_maps = {}
matched_rows = []
matched_summary = {}
source_xyz = time_run_maps["source_xyz"]
roi_masks = {
    "right_precentral": time_run_maps["right_precentral_mask"].astype(bool),
    "right_postcentral": time_run_maps["right_postcentral_mask"].astype(bool),
}
for component, component_title in components:
    component_metrics = single_run_metrics[single_run_metrics["component"] == component]
    candidate_runs = sorted(component_metrics["run"].unique())
    common_runs = []
    for run in candidate_runs:
        all_methods_available = True
        for method in methods:
            available = component_metrics[
                (component_metrics["run"] == run)
                & (component_metrics["method"] == method)
            ]["localization_available"]
            available = available.astype(str).str.lower().eq("true")
            all_methods_available &= bool(len(available)) and bool(available.all())
        if all_methods_available:
            common_runs.append(int(run))
    assert common_runs, f"{component} 没有三种方法共同可比较的 run"

    run_weights = []
    for run in common_runs:
        trial_counts = component_metrics[
            component_metrics["run"] == run
        ]["n_trials"].unique()
        assert len(trial_counts) == 1, f"{component} run-{run} 三法试次数不一致"
        run_weights.append(trial_counts[0])
    run_weights = np.asarray(run_weights, dtype=float)
    matched_summary[component] = {
        "title": component_title,
        "runs": common_runs,
        "n_trials": int(run_weights.sum()),
    }
    for method in methods:
        run_maps = np.asarray([
            time_run_maps[f"run_{run}__{component}__{method}"]
            for run in common_runs
        ])
        source_map = np.sqrt(np.average(run_maps**2, axis=0, weights=run_weights))
        matched_maps[(component, method)] = source_map
        peak_index = int(np.argmax(source_map))
        source_power = source_map**2
        total_power = float(source_power.sum())
        for roi, roi_mask in roi_masks.items():
            roi_power = float(source_power[roi_mask].sum())
            matched_rows.append({
                "component": component,
                "method": method,
                "matched_runs": ";".join(str(run) for run in common_runs),
                "n_trials": int(run_weights.sum()),
                "comparison_scope": "conditional_on_common_successful_runs",
                "oaster_successful_runs": int(
                    oaster_detection_by_component.loc[component, "successful_runs"]
                ),
                "oaster_total_runs": int(
                    oaster_detection_by_component.loc[component, "total_runs"]
                ),
                "oaster_success_rate_pct": float(
                    oaster_detection_by_component.loc[component, "success_rate_pct"]
                ),
                "roi": roi,
                "roi_mass_pct": 100.0 * roi_power / total_power,
                "roi_enrichment": (roi_power / total_power) / float(roi_mask.mean()),
                "peak_distance_to_roi_mm": float(
                    np.min(np.linalg.norm(
                        source_xyz[roi_mask] - source_xyz[peak_index], axis=1
                    )) * 1000.0
                ),
            })

matched_metrics = pd.DataFrame(matched_rows)
matched_metrics.to_csv(
    result_dir / "time_domain_matched_method_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)
matched_npz = {
    f"{component}__{method}": source_map
    for (component, method), source_map in matched_maps.items()
}
np.savez_compressed(
    result_dir / "time_domain_matched_source_maps.npz",
    vertices_lh=vertices[0],
    vertices_rh=vertices[1],
    **matched_npz,
)

time_brain_jobs = []
for component, component_title in components:
    summary = matched_summary[component]
    runs_text = ",".join(str(run) for run in summary["runs"])
    for method in methods:
        time_brain_jobs.append((
            f"matched_{component}_{method}",
            f"{component_title}\n{method}（runs {runs_text}；n={summary['n_trials']}）",
            matched_maps[(component, method)],
        ))
dics_brain_jobs = [
    (
        "DICS_beta_ERD",
        "DICS：beta-ERD（-400~200 ms）",
        np.maximum(dics_maps["beta_15_30Hz__ERD"], 0.0),
    ),
    (
        "DICS_beta_PMBR",
        "DICS：beta-PMBR（500~1000 ms）",
        np.maximum(dics_maps["beta_15_30Hz__PMBR"], 0.0),
    ),
]
brain_jobs = time_brain_jobs + dics_brain_jobs


#%%
brain_image_files = []
for file_stem, title, source_map in brain_jobs:
    peak = max(float(np.max(source_map)), np.finfo(float).eps)
    normalized = source_map / peak
    floor = float(np.percentile(normalized, display_percentile))
    if floor >= 1.0:
        floor = 0.0
    display_map = np.where(normalized >= floor, normalized, 0.0)

    stc = mne.SourceEstimate(
        display_map[:, None],
        vertices=vertices,
        tmin=0.0,
        tstep=1.0,
        subject=subject,
    )
    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        surface="pial",
        hemi="both",
        views="dorsal",
        initial_time=0.0,
        time_viewer=False,
        show_traces=False,
        time_label=None,
        colormap="inferno",
        clim={"kind": "value", "lims": [floor, (floor + 1.0) / 2.0, 1.0]},
        smoothing_steps=5,
        transparent=True,
        background="white",
        foreground="black",
        cortex="classic",
        colorbar=True,
        size=(900, 720),
        backend="pyvistaqt",
        brain_kwargs={"show": False, "theme": "light"},
    )
    brain.plotter.camera.zoom(camera_zoom)
    brain.plotter.render()
    image_file = result_dir / f"brain_{file_stem}_pial_dorsal.png"
    brain.save_image(image_file)
    brain.close()
    brain_image_files.append((image_file, title))
    print("已保存：", image_file)


#%%
figure, axes = plt.subplots(3, 3, figsize=(15.5, 14.0), layout="constrained")
for axis, (image_file, title) in zip(axes.ravel(), brain_image_files[:9]):
    axis.imshow(plt.imread(image_file))
    axis.set_title(title, fontsize=13)
    axis.axis("off")

figure.suptitle(
    "ds006035 sub-sm09：同一时间窗、同一 run 与试次的时域比较",
    fontsize=17,
)
figure.text(
    0.5,
    0.006,
    "每行三法使用相同 response-lock 窗与匹配 runs；逐图峰值归一化并使用 P95，仅定性比较峰位和空间模式。"
    " OASTER 全 run 成功率：MF 2/3，MEFI 3/3，MEFII 2/3。",
    ha="center",
    fontsize=10,
    color="#444444",
)
time_montage_file = result_dir / "time_domain_same_window_pial_dorsal_montage.png"
figure.savefig(time_montage_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), layout="constrained")
for axis, (image_file, title) in zip(axes.ravel(), brain_image_files[9:]):
    axis.imshow(plt.imread(image_file))
    axis.set_title(title, fontsize=13)
    axis.axis("off")
figure.suptitle("beta 频域定位（补充证据，不与 ERP 方法作优劣比较）", fontsize=16)
dics_montage_file = result_dir / "dics_beta_pial_dorsal_montage.png"
figure.savefig(dics_montage_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)

assert len(brain_image_files) == 11
assert time_montage_file.is_file()
assert dics_montage_file.is_file()
print("同窗、同 run、同试次时域比较：", time_montage_file)
print("DICS 频域图：", dics_montage_file)
print("OASTER 定位成功率：", result_dir / "oaster_detection_rate.csv")


#%%
# 只画三个 response-lock 主终点；旧的 Stim-M1 行保留在原始 CSV 中供审计。
primary_components = [component for component, _ in components]
primary_labels = ["MF\n-80~-20", "MEFI\n20~60", "MEFII\n120~180"]
method_colors = {
    "OASTER-ERP": "#0072B2",
    "dSPM": "#D55E00",
    "eLORETA": "#009E73",
}
matched_metrics.to_csv(
    result_dir / "time_domain_primary_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, layout="constrained")
figure.get_layout_engine().set(rect=(0.0, 0.055, 1.0, 0.94))
x = np.arange(len(primary_components), dtype=float)
for column, (roi, roi_title) in enumerate((
    ("right_precentral", "右侧 precentral（M1）"),
    ("right_postcentral", "右侧 postcentral（S1）"),
)):
    for method, color in method_colors.items():
        selected = matched_metrics[
            (matched_metrics["roi"] == roi) & (matched_metrics["method"] == method)
        ].set_index("component").loc[primary_components]
        axes[0, column].plot(
            x, selected["roi_enrichment"], marker="o", linewidth=2.0,
            color=color, label=method,
        )
        axes[1, column].plot(
            x, selected["peak_distance_to_roi_mm"], marker="o", linewidth=2.0,
            color=color,
        )
    axes[0, column].axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axes[0, column].set_title(roi_title)
    axes[0, column].set_ylabel("ROI 富集")
    axes[1, column].set_ylabel("峰距 ROI（mm）")
    axes[1, column].set_xticks(x, primary_labels)

for axis in axes.ravel():
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
figure.suptitle(
    "sub-sm09 左指运动：同窗、同 run、同试次的 response-lock 主终点",
    fontsize=15,
)
figure.text(
    0.5,
    0.005,
    "共同成功 run 上的条件性定位指标；OASTER 全 run 成功率：MF 2/3，MEFI 3/3，MEFII 2/3。",
    ha="center",
    fontsize=10,
    color="#444444",
)
primary_figure_file = result_dir / "time_domain_primary_method_comparison.png"
figure.savefig(primary_figure_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("response-lock 主终点比较：", primary_figure_file)
