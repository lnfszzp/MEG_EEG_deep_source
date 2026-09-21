# %%
# ds006035 右腕电刺激 N20 可见性核查：同一 run 的传感器波形、头皮拓扑和两种源图。
# 只读已保存的八方法源图；不重新拟合逆解，也不把显示阈值用于定量指标。

from pathlib import Path
import csv
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np


# %%
# ==================== 1. 固定输入与检查 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
dataset = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
subjects_dir = Path(r"C:\Users\zzp\mne_data\MNE-sample-data\subjects")
comparison_root = project_root / "results" / "real_data" / "ds006035_erp_eight_methods"
output_root = project_root / "results" / "real_data" / "ds006035_n20_diagnostic"
cases = (("sm09", 1, "development / S1 miss"),
         ("sm04", 1, "prior validation cohort / S1 miss"),
         ("sm12", 1, "prior validation cohort / S1 hit"))
methods = (("OASTER-ERP-v3", "n20_oaster_erp_v3"), ("dSPM", "n20_dspm"))
display_floor = 0.05  # 固定 5% 峰值；两法只比较峰位/形状，不比较物理振幅。

assert dataset.is_dir() and (subjects_dir / "sample").is_dir()
assert 0 < display_floor < 1
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from pipelines import run_ds006035_somatomotor as real

mne.set_log_level("WARNING")
output_root.mkdir(parents=True, exist_ok=True)
report_rows = []


# %%
# ==================== 2. 叠加平均后的 EEG/MEG 有符号波形和头皮拓扑 ====================

for subject, run, role in cases:
    case = f"sub-{subject}_run-{run}"
    prior = comparison_root / case
    assert (prior / "source_maps.npz").is_file() and (prior / "method_comparison.csv").is_file(), prior
    raw_path, events_path = real._paths(dataset, subject, run)
    _sensor_data, _noise, evoked, _noise_cov, qc = real.preprocess_run(raw_path, events_path)
    with (prior / "method_comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    with np.load(prior / "source_maps.npz") as archive:
        vertices = [archive["vertices_lh"].copy(), archive["vertices_rh"].copy()]
        source_maps = {method: archive[key].copy() for method, key in methods}
    assert qc["epochs"] == int(rows["OASTER-ERP-v3"]["epochs"])
    assert all(np.isfinite(values).all() and values.max(initial=0) > 0 for values in source_maps.values())

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), facecolor="white")
    sensor_lines = {}
    for row, (modality, unit, scale, kind) in enumerate((
        ("EEG", "µV", 1e6, "eeg"), ("MEG magnetometers", "fT", 1e15, "mag")
    )):
        picks = mne.pick_types(evoked.info, meg="mag" if kind == "mag" else False,
                               eeg=kind == "eeg", exclude="bads")
        assert len(picks) > 0
        data = evoked.data[picks] * scale
        times = evoked.times * 1000.0
        n20 = (times >= 18) & (times <= 24)
        # 描述性选道：每个 run 独立选 18–24 ms 平均电位/磁场绝对值最大的通道。
        # 不参与源定位、ROI 选择或统计推断；同时画全部通道以暴露选择偏差。
        chosen = int(np.argmax(np.abs(data[:, n20].mean(axis=1))))
        line = data[chosen]
        baseline = (times >= -100) & (times <= -50)
        sensor_lines[kind] = {
            "channels": len(picks), "selected": evoked.ch_names[picks[chosen]],
            "n20_window_mean": float(line[n20].mean()),
            "baseline_rms": float(np.sqrt(np.mean(line[baseline] ** 2))),
            "unit": unit,
        }
        ax = axes[row, 0]
        show = (times >= -50) & (times <= 80)
        ax.plot(times[show], data[:, show].T, color="#9AA5B1", alpha=0.20, lw=0.7)
        ax.plot(times[show], line[show], color="#165E8A" if kind == "eeg" else "#B65043", lw=2.1)
        ax.axhline(0, color="#64748B", lw=0.8)
        ax.axvspan(0, 8, color="#CFD8DF", alpha=0.45, label="stimulus interpolation")
        ax.axvspan(18, 24, color="#2676A7", alpha=0.15, label="N20 18–24 ms")
        ax.axvspan(28, 40, color="#DB8751", alpha=0.13, label="P30 28–40 ms")
        ax.axvline(20, color="#2676A7", ls="--", lw=1)
        ax.axvline(35, color="#DB8751", ls="--", lw=1)
        ax.set_xlim(-50, 80)
        ax.set_xlabel("Time after right-wrist stimulus (ms)")
        ax.set_ylabel(unit)
        ax.set_title(f"{modality} evoked, {len(picks)} sensors\nHighlighted: {evoked.ch_names[picks[chosen]]} (post-hoc N20 maximum)")
        ax.grid(color="#E3E8ED", lw=0.6)
        for column, moment in enumerate((20.0, 35.0), start=1):
            topo = np.asarray([np.interp(moment, times, channel) for channel in data])
            limit = max(float(np.max(np.abs(topo))), np.finfo(float).eps)
            topomap = mne.viz.plot_topomap(
                topo, evoked.copy().pick([evoked.ch_names[index] for index in picks]).info,
                axes=axes[row, column], show=False, cmap="RdBu_r", vlim=(-limit, limit),
                contours=0, extrapolate="head", sensors=True,
            )[0]
            axes[row, column].set_title(f"{modality} at {moment:.0f} ms (signed {unit})")
            fig.colorbar(topomap, ax=axes[row, column], fraction=0.045, pad=0.04)
    fig.suptitle(
        f"ds006035 {case} ({role}) | right-wrist stimulus | {qc['epochs']} averaged trials\n"
        "N20/P30 are prespecified windows; selected traces are descriptive, not source-localization input",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    sensor_path = output_root / f"{case}_signed_evoked_topomaps.png"
    fig.savefig(sensor_path, dpi=170)
    plt.close(fig)


# %%
# ==================== 3. 左侧外侧 + 双侧俯视源图，统一显示尺度 ====================

    brain_images = {}
    for method, _key in methods:
        amplitude = source_maps[method]
        normalized = amplitude / float(np.max(amplitude))
        displayed = np.where(normalized >= display_floor, normalized, 0.0)
        stc = mne.SourceEstimate(displayed[:, None], vertices, tmin=0.0, tstep=1.0, subject="sample")
        for surface, view, hemi in (("pial", "lateral", "lh"),
                                    ("inflated", "lateral", "lh"),
                                    ("pial", "dorsal", "both")):
            brain = None
            try:
                brain = stc.plot(
                    surface=surface, hemi=hemi, views=view,
                    colormap="inferno", time_label=None, smoothing_steps="nearest",
                    transparent=True, subjects_dir=subjects_dir, size=(850, 600),
                    clim={"kind": "value", "lims": [display_floor, 0.5, 1.0]},
                    background="white", foreground="black", cortex="classic",
                    initial_time=0.0, time_viewer=False, show_traces=False,
                    colorbar=False, backend="pyvistaqt",
                    brain_kwargs={"show": False, "theme": "light"},
                )
                screenshot = brain.screenshot(mode="rgb", time_viewer=False)
                # Crop only white screenshot margins: same camera and map are unchanged.
                extent = np.any(screenshot < 245, axis=2)
                yy, xx = np.where(extent)
                assert len(xx) > 0 and len(yy) > 0
                brain_images[(method, surface, view)] = screenshot[
                    max(0, yy.min() - 12):min(screenshot.shape[0], yy.max() + 13),
                    max(0, xx.min() - 12):min(screenshot.shape[1], xx.max() + 13),
                ]
            finally:
                if brain is not None:
                    brain.close()
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.8), facecolor="white")
    for row, (method, _key) in enumerate(methods):
        metric = rows[method]
        for column, (surface, view) in enumerate((("pial", "lateral"),
                                                   ("inflated", "lateral"),
                                                   ("pial", "dorsal"))):
            ax = axes[row, column]
            ax.imshow(brain_images[(method, surface, view)])
            title = f"{method} | {surface} | {'left lateral' if view == 'lateral' else 'bilateral dorsal'}"
            if column == 0:
                title += (f"\nPeak {metric['n20_peak_label']}, "
                          f"{float(metric['n20_peak_euclidean_distance_to_left_s1_mm']):.1f} mm from left S1, "
                          f"enrichment {float(metric['n20_left_s1_enrichment']):.2f}")
            ax.set_title(title, fontsize=11)
            ax.axis("off")
    fig.suptitle(
        f"{case} ({role}) | unsigned N20 18–24 ms excess-power vs baseline | identical evoked, forward, grid, window\n"
        "Each method peak-normalized to 1; fixed 5% display floor and [0.05, 0.5, 1] color scale; "
        "NOT comparable absolute amplitudes",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.91), h_pad=3.0)
    source_path = output_root / f"{case}_n20_pial_inflated_dorsal.png"
    fig.savefig(source_path, dpi=170)
    plt.close(fig)
    report_rows.append({
        "case": case, "role": role, "epochs": qc["epochs"], "events": qc["events"],
        "sensor_lines": sensor_lines,
        "source_metrics": {method: {
            "peak_label": rows[method]["n20_peak_label"],
            "peak_to_left_s1_mm": float(rows[method]["n20_peak_euclidean_distance_to_left_s1_mm"]),
            "left_s1_enrichment": float(rows[method]["n20_left_s1_enrichment"]),
        } for method, _key in methods},
        "sensor_figure": sensor_path.name, "source_figure": source_path.name,
    })
    print(f"Saved {case}: {qc['epochs']} epochs; {sensor_path.name}; {source_path.name}", flush=True)


# %%
# ==================== 4. 仅记录诊断事实，不把传感器挑选当作验证 ====================

assert len(report_rows) == len(cases)
metadata = {
    "dataset": str(dataset), "source_maps": str(comparison_root), "cases": report_rows,
    "event": "right wrist somatosensory electrical stimulus",
    "source_window_ms": [18, 24], "topomap_times_ms": [20, 35],
    "trace_rule": "post-hoc largest absolute 18-24 ms mean among all good EEG or MAG channels; descriptive only",
    "display": "per-method peak normalization; shared 5% floor and [0.05,0.5,1] colormap; not used for metrics",
    "limitations": [
        "No measured source ground truth: neither S1 enrichment nor peak distance is an accuracy score.",
        "sm09/run-1 is development; sm04/run-1 and sm12/run-1 are a previously examined validation cohort, not a fresh independent test.",
        "MNE sample template cortex, not participant MRI; coordinate error can dominate small peak distances.",
        "A visible evoked deflection at 20 ms does not prove that an inverse method localized its generator.",
    ],
}
(output_root / "diagnostic_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
lines = [
    "# ds006035 N20 可见性核查", "",
    "三个 run 使用原八方法实验的同一已保存源图；波形来自相同预处理的单试次叠加平均。",
    "N20 源图是 18–24 ms 相对基线的无符号超额功率幅度 sqrt(max(mean(active²)−mean(baseline²),0))，不是 20 ms 的有符号源电流波形。头皮拓扑分别是 20 与 35 ms。两法源图各自峰值归一化，",
    "统一 5% 显示下限及相同归一化色标；不能据此比较绝对源电流。", "",
]
for item in report_rows:
    lines += [
        f"## {item['case']} ({item['role']})", "",
        f"事件 {item['events']}，保留并平均 {item['epochs']} 试次。",
        f"EEG 描述性通道 {item['sensor_lines']['eeg']['selected']}，"
        f"N20 平均 {item['sensor_lines']['eeg']['n20_window_mean']:.3f} µV；"
        f"MAG 描述性通道 {item['sensor_lines']['mag']['selected']}，"
        f"N20 平均 {item['sensor_lines']['mag']['n20_window_mean']:.2f} fT。", "",
        f"![有符号叠加平均与拓扑]({item['sensor_figure']})", "",
        f"![N20 源图]({item['source_figure']})", "",
    ]
    for method, metrics in item["source_metrics"].items():
        lines.append(
            f"- {method}：峰 {metrics['peak_label']}；距左 S1 最近点 "
            f"{metrics['peak_to_left_s1_mm']:.1f} mm；左 S1 富集 {metrics['left_s1_enrichment']:.2f}。"
        )
    lines.append("")
lines += [
    "传感器高亮通道是看过本 run 的 18–24 ms 后选出的描述性通道；灰线是同类全部通道，",
    "不可把高亮峰值当独立显著性或作为逆解输入。刺激伪迹 −2 至 +8 ms 已插值。", "",
    "sm09 的 OASTER 峰是单点稀疏的缘上回峰：pial 外侧及俯视角未显示，inflated 外侧图仅见一个很小的亮点；",
    "它并未落入左 S1。sm04 的 dSPM 全局峰在右侧上顶叶，不能把 dSPM 作为真值。", "",
    "真实数据无已知源真值；模板皮层不是个体 MRI；显示出来的 N20 波形和脑图只能证明",
    "信号成分可见，不能证明定位准确或 OASTER 优于 dSPM。",
]
(output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("Report:", output_root / "REPORT.md")
