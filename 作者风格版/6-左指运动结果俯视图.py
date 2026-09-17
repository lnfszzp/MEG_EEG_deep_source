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

time_file = result_dir / "time_domain_pooled_source_maps.npz"
dics_file = result_dir / "dics_pooled_source_maps.npz"
assert time_file.is_file(), f"找不到时域源图：{time_file}"
assert dics_file.is_file(), f"找不到频域源图：{dics_file}"
assert (subjects_dir / subject).is_dir(), f"找不到 sample 解剖：{subjects_dir}"

display_percentile = 95.0
camera_zoom = 0.72
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
mne.set_log_level("warning")


#%%
time_maps = np.load(time_file)
dics_maps = np.load(dics_file)
vertices = [time_maps["vertices_lh"], time_maps["vertices_rh"]]
assert np.array_equal(vertices[0], dics_maps["vertices_lh"])
assert np.array_equal(vertices[1], dics_maps["vertices_rh"])

brain_jobs = [
    (
        "01_OASTER_MF",
        "OASTER-ERP：运动前 MF（-80~-20 ms）",
        time_maps["MF_response_-80_-20ms__OASTER-ERP"],
    ),
    (
        "02_eLORETA_MF",
        "eLORETA：运动前 MF（-80~-20 ms）",
        time_maps["MF_response_-80_-20ms__eLORETA"],
    ),
    (
        "03_dSPM_MEFI",
        "dSPM：运动后 MEFI（20~60 ms）",
        time_maps["MEFI_response_20_60ms__dSPM"],
    ),
    (
        "04_eLORETA_MEFI",
        "eLORETA：运动后 MEFI（20~60 ms）",
        time_maps["MEFI_response_20_60ms__eLORETA"],
    ),
    (
        "05_DICS_beta_ERD",
        "DICS：beta-ERD（-400~200 ms）",
        np.maximum(dics_maps["beta_15_30Hz__ERD"], 0.0),
    ),
    (
        "06_DICS_beta_PMBR",
        "DICS：beta-PMBR（500~1000 ms）",
        np.maximum(dics_maps["beta_15_30Hz__PMBR"], 0.0),
    ),
]


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
figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.8), layout="constrained")
for axis, (image_file, title) in zip(axes.ravel(), brain_image_files):
    axis.imshow(plt.imread(image_file))
    axis.set_title(title, fontsize=13)
    axis.axis("off")

figure.suptitle(
    "ds006035 sub-sm09 左指运动源定位：完整 pial 俯视图",
    fontsize=17,
)
figure.text(
    0.5,
    0.006,
    "三 run 独立 forward/inverse 后聚合；显示阈值为各图 P95，仅用于展示；当前使用 sample 模板解剖。",
    ha="center",
    fontsize=10,
    color="#444444",
)
montage_file = result_dir / "finger_localization_pial_dorsal_montage.png"
figure.savefig(montage_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)

assert len(brain_image_files) == 6
assert montage_file.is_file()
print("俯视脑图总览：", montage_file)


#%%
# 只画三个 response-lock 主终点；旧的 Stim-M1 行保留在原始 CSV 中供审计。
primary_components = [
    "MF_response_-80_-20ms",
    "MEFI_response_20_60ms",
    "MEFII_response_120_180ms",
]
primary_labels = ["MF\n-80~-20", "MEFI\n20~60", "MEFII\n120~180"]
method_colors = {
    "OASTER-ERP": "#0072B2",
    "dSPM": "#D55E00",
    "eLORETA": "#009E73",
}
time_metrics = pd.read_csv(result_dir / "time_domain_method_comparison.csv")
primary_metrics = time_metrics[
    time_metrics["component"].isin(primary_components)
].copy()
primary_metrics.to_csv(
    result_dir / "time_domain_primary_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)
pooled = primary_metrics[
    primary_metrics["run"].astype(str) == "pooled_1_2_3"
]

figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), sharex=True, layout="constrained")
x = np.arange(len(primary_components), dtype=float)
for column, (roi, roi_title) in enumerate((
    ("right_precentral", "右侧 precentral（M1）"),
    ("right_postcentral", "右侧 postcentral（S1）"),
)):
    for method, color in method_colors.items():
        selected = pooled[
            (pooled["roi"] == roi) & (pooled["method"] == method)
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
    "sub-sm09 三 run 左指运动：response-lock 主终点",
    fontsize=15,
)
primary_figure_file = result_dir / "time_domain_primary_method_comparison.png"
figure.savefig(primary_figure_file, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(figure)
print("response-lock 主终点比较：", primary_figure_file)
