# %%
# ds006035 右腕电刺激：OASTER-ERP-v3 与七种仿真对比方法，同一 ERP、同一 forward、同一皮层网格。
# 本文件不定义函数。sm09/run-1 是开发检查；其余四人/run-1 是固定的外部检查。
# 真实数据无源真值；这里比较左 S1 解剖合理性、侧化、跨 run 稳定性，不计算 AUC/DLE。

from pathlib import Path
import csv
import json
import sys
import time
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.stats import spearmanr


# %%
# ==================== 1. 参数区 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
dataset = Path(r"D:\博士\工作＆汇报\源定位\开源数据\v1.0.0")
subjects_dir = Path(r"C:\Users\zzp\mne_data\MNE-sample-data\subjects")
output_root = project_root / "results" / "real_data" / "ds006035_erp_eight_methods"
subjects = ("sm04", "sm06", "sm07", "sm12")  # sm09 仅作开发图，不参与锁定验证汇总
runs = (1, 2, 3)
spacing = "ico4"
render_brains = False  # sm09/run-1 代表脑图已保存
skip_completed = True
analysis_revision = "common-lcmv-covariance-v3"

methods = (
    "OASTER-ERP-v3", "MNE", "dSPM", "sLORETA", "eLORETA",
    "LCMV", "Dipole fitting (grid)", "RAP-MUSIC",
)
windows = ("n20", "p30")

assert dataset.is_dir(), dataset
assert (subjects_dir / "sample").is_dir(), subjects_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmark import methods as comparators
from candidates import oaster_rebuilt as oaster
import protected_multilayer as protected
from pipelines import run_ds006035_somatomotor as real

assert methods[1:] == (
    "MNE", "dSPM", "sLORETA", "eLORETA", "LCMV",
    "Dipole fitting (grid)", "RAP-MUSIC",
)
assert len(methods) == 8
assert np.sum(real.N20) > 0 and np.sum(real.P30) > 0
assert not np.any(real.N20 & real.P30)
assert np.array_equal(np.flatnonzero(real.N20), np.arange(203, 210))
assert np.array_equal(np.flatnonzero(real.P30), np.arange(213, 226))
lcmv_covariance_indices = np.flatnonzero(real.TARGET_TIMES >= 0.015)
assert np.array_equal(lcmv_covariance_indices, np.arange(200, 231))


# %%
# ==================== 2. 同输入八方法逐 run 定位 ====================

mne.set_log_level("WARNING")
output_root.mkdir(parents=True, exist_ok=True)

for subject in subjects:
    for run in runs:
        output = output_root / f"sub-{subject}_run-{run}"
        table_path = output / "method_comparison.csv"
        maps_path = output / "source_maps.npz"
        metadata_path = output / "metadata.json"
        previous_revision = (
            json.loads(metadata_path.read_text(encoding="utf-8")).get("analysis_revision")
            if metadata_path.is_file() else None
        )
        if skip_completed and table_path.is_file() and maps_path.is_file() and previous_revision == analysis_revision:
            print("已完成，跳过：", output, flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        raw_path, events_path = real._paths(dataset, subject, run)
        assert raw_path.is_file() and events_path.is_file(), (raw_path, events_path)
        print("预处理：", subject, "run", run, flush=True)
        sensor_data, trial_noise, evoked, noise_cov, qc = real.preprocess_run(
            raw_path, events_path
        )
        trans, coreg = real.coregister(evoked.info, subjects_dir)
        src = mne.setup_source_space(
            "sample", spacing=spacing, subjects_dir=subjects_dir,
            add_dist=False, n_jobs=1, verbose=False,
        )
        eeg_fwd, mag_fwd, _eeg_free, _mag_free = real.make_fixed_forwards(
            evoked.info, trans, src, subjects_dir
        )
        geometry = real.label_geometry(eeg_fwd["src"], subjects_dir)
        eeg, mag, ge, gm, eeg_noise, mag_noise = real.aligned_data_and_gain(
            sensor_data, trial_noise, evoked.info, (eeg_fwd, mag_fwd)
        )
        eeg_white, eeg_gain = real.whiten_from_trials(eeg, ge, eeg_noise)
        mag_white, mag_gain = real.whiten_from_trials(mag, gm, mag_noise)
        data = np.vstack((eeg_white, mag_white))
        gain = np.vstack((eeg_gain, mag_gain))
        baseline = real.TARGET_TIMES < -0.05
        data -= data[:, baseline].mean(axis=1, keepdims=True)
        n_sources = gain.shape[1]
        assert n_sources == eeg_fwd["nsource"] == mag_fwd["nsource"]
        assert n_sources == len(geometry["xyz"])  # 当前只有皮层源，没有深部体积源
        assert np.isfinite(data).all() and np.isfinite(gain).all()
        assert data.shape[1] == len(real.TARGET_TIMES)

        adjacency = mne.spatial_src_adjacency(eeg_fwd["src"], verbose=False).toarray()
        kernels = protected.connected_euclidean_surface_kernels(
            geometry["xyz"], adjacency, n_sources
        )
        channel_weights = []
        for window in (real.N20, real.P30):
            weights = real.modality_evidence_weights(
                (data[:len(eeg_white)], data[len(eeg_white):]), baseline, window
            )
            channel_weights.append(np.r_[
                np.full(len(eeg_white), weights[0]),
                np.full(len(mag_white), weights[1]),
            ])

        print("八方法逆解：", subject, "run", run, flush=True)
        oaster_estimate, oaster_diagnostics = oaster.reconstruct_evoked_oaster_v3_from_whitened(
            data, gain, n_sources, kernels, baseline=baseline,
            active_windows=(real.N20, real.P30),
            window_channel_weights=channel_weights,
            require_one=False,
        )
        family = comparators.minimum_norm_family(data, gain)
        assert set(family) == set(methods[1:5])
        estimates = {"OASTER-ERP-v3": {window: oaster_estimate for window in windows}}
        estimates.update({method: {window: value for window in windows} for method, value in family.items()})
        lcmv_estimate = comparators.lcmv(data, gain, lcmv_covariance_indices)
        estimates["LCMV"] = {window: lcmv_estimate for window in windows}
        for method, solver in (
            ("Dipole fitting (grid)", comparators.dipole_fit),
            ("RAP-MUSIC", comparators.rap_music),
        ):
            estimates[method] = {
                window: solver(data, gain, np.flatnonzero(active))
                for window, active in zip(windows, (real.N20, real.P30), strict=True)
            }
        assert tuple(estimates) == methods

        maps = {}
        rows = []
        for method in methods:
            row = {
                "subject": subject, "run": run, "method": method,
                "epochs": qc["epochs"], "sources": n_sources,
                "source_space": "sample ico4 cortex only",
                "eeg_channels_whitened": data[:len(eeg_white)].shape[0],
                "mag_channels_whitened": data[len(eeg_white):].shape[0],
                "coreg_mean_mm": coreg["mean_mm"],
            }
            for window, active in zip(windows, (real.N20, real.P30), strict=True):
                estimate = estimates[method][window]
                assert estimate.shape == (n_sources, len(real.TARGET_TIMES)), method
                assert np.isfinite(estimate).all(), method
                metric, n20_map, p30_map = real.source_metrics(estimate, geometry)
                row.update({key: value for key, value in metric.items() if key.startswith(window)})
                maps[f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}"] = (
                    n20_map if window == "n20" else p30_map
                )
            rows.append(row)
            print(
                f"  {method}: N20 S1={row['n20_left_s1_enrichment']:.2f}, "
                f"P30 S1={row['p30_left_s1_enrichment']:.2f}", flush=True,
            )

        with table_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            maps_path, methods=np.asarray(methods),
            vertices_lh=geometry["vertices"][0],
            vertices_rh=geometry["vertices"][1], **maps,
        )
        metadata_path.write_text(
            json.dumps({
                "analysis_revision": analysis_revision,
                "dataset": str(dataset), "subject": subject, "run": run,
                "event": "somatosensory/right wrist stimulation (32)",
                "windows_ms": {"n20": [18, 24], "p30": [28, 40]},
                "baseline_ms": [-250, -51], "methods": methods,
                "source_space": "MNE sample ico4 cortical surface, no deep volume source",
                "comparators": "benchmark.methods (same implementation as ERP simulation)",
                "active_indices": {
                    window: np.flatnonzero(active).tolist()
                    for window, active in zip(windows, (real.N20, real.P30), strict=True)
                },
                "lcmv_covariance_window_ms": [15, 45],
                "shared_whitened_data_shape": data.shape,
                "shared_whitened_gain_shape": gain.shape,
                "oaster_modality_weights": [weights.tolist() for weights in (
                    real.modality_evidence_weights(
                        (data[:len(eeg_white)], data[len(eeg_white):]), baseline, active
                    ) for active in (real.N20, real.P30)
                )],
                "oaster_diagnostics": oaster_diagnostics,
                "qc": qc, "coregistration": coreg,
                "elapsed_seconds": time.perf_counter() - started,
                "caveat": "No source truth, no AUC/DLE. Template anatomy is exploratory.",
            }, ensure_ascii=False, indent=2, default=lambda value: value.item()),
            encoding="utf-8",
        )
        assert len(rows) == len(methods) and len(maps) == 2 * len(methods)
        print("数值结果已保存：", output, flush=True)


# %%
# ==================== 3. 八方法俯视皮层图（显示阈值不参与指标计算） ====================

        if not render_brains:
            continue
        try:
            for window in windows:
                images = []
                for method in methods:
                    key = f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}"
                    amplitude = maps[key]
                    peak = max(float(amplitude.max(initial=0.0)), np.finfo(float).eps)
                    normalized = amplitude / peak
                    floor = max(0.10, float(np.percentile(normalized, 95)))
                    displayed = np.where(normalized >= floor, normalized, 0.0)
                    stc = mne.SourceEstimate(
                        displayed[:, None], geometry["vertices"],
                        tmin=0.0, tstep=1.0, subject="sample",
                    )
                    brain = None
                    try:
                        brain = stc.plot(
                            surface="pial", hemi="both", views="dorsal",
                            colormap="inferno", time_label=None,
                            smoothing_steps=5, transparent=True,
                            subjects_dir=subjects_dir, size=(700, 500),
                            clim={"kind": "value", "lims": [floor, 0.5 * (floor + 1.0), 1.0]},
                            background="white", foreground="black", cortex="classic",
                            initial_time=0.0, time_viewer=False, show_traces=False,
                            colorbar=False, backend="pyvistaqt",
                            brain_kwargs={"show": False, "theme": "light"},
                        )
                        images.append(brain.screenshot(mode="rgb", time_viewer=False))
                    finally:
                        if brain is not None:
                            brain.close()
                fig, axes = plt.subplots(4, 2, figsize=(14, 16), facecolor="white")
                for ax, method, image in zip(axes.ravel(), methods, images, strict=True):
                    ax.imshow(image)
                    ax.set_title(method, fontsize=13)
                    ax.axis("off")
                fig.suptitle(
                    f"ds006035 sub-{subject} run-{run} | {window.upper()} | "
                    "8 methods, same evoked/forward/window\n"
                    "Pial dorsal; each map peak-normalized; display floor=max(10%, P95)",
                    fontsize=15, y=0.995,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.95))
                fig.savefig(output / f"{window}_eight_methods_pial_dorsal.png", dpi=160)
                plt.close(fig)
        except Exception as error:
            (output / "brain_render_error.txt").write_text(repr(error), encoding="utf-8")
            print("脑图渲染失败，数值结果仍已保存：", repr(error), flush=True)

# %%
# ==================== 4. 按受试者汇总；每人 3 个 run 先取中位数 ====================

validation_subjects = ("sm04", "sm06", "sm07", "sm12")
subject_rows = []
missing_runs = []
for subject in validation_subjects:
    by_run = []
    by_run_maps = []
    reference_vertices = None
    for run in runs:
        output = output_root / f"sub-{subject}_run-{run}"
        table_path = output / "method_comparison.csv"
        metadata_path = output / "metadata.json"
        if not table_path.is_file() or not metadata_path.is_file():
            missing_runs.append(f"sub-{subject} run-{run}: missing")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("analysis_revision") != analysis_revision:
            missing_runs.append(f"sub-{subject} run-{run}: wrong revision")
            continue
        with table_path.open(encoding="utf-8-sig", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        assert [row["method"] for row in run_rows] == list(methods), output
        with np.load(output / "source_maps.npz") as archive:
            vertices = (archive["vertices_lh"], archive["vertices_rh"])
            if reference_vertices is None:
                reference_vertices = vertices
            else:
                assert all(np.array_equal(a, b) for a, b in zip(reference_vertices, vertices, strict=True))
            by_run_maps.append({
                f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}":
                archive[f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}"].copy()
                for window in windows for method in methods
            })
        by_run.append({row["method"]: row for row in run_rows})
    if len(by_run) != len(runs):
        continue
    for method in methods:
        row = {"subject": subject, "method": method, "runs": len(runs)}
        for window in windows:
            map_key = f"{window}_{method.lower().replace(' ', '_').replace('-', '_')}"
            source_maps = [item[map_key] for item in by_run_maps]
            correlations = [
                float(spearmanr(first, second).statistic)
                for first, second in combinations(source_maps, 2)
                if np.any(first) and np.any(second)
            ]
            row[f"{window}_run_map_spearman_median"] = (
                float(np.median(correlations)) if correlations else float("nan")
            )
            row[f"{window}_zero_map_runs"] = sum(not np.any(values) for values in source_maps)
            row[f"{window}_peak_left_s1_run_majority"] = int(sum(
                run_rows[method][f"{window}_peak_label"] == "postcentral-lh"
                and np.any(values)
                for run_rows, values in zip(by_run, source_maps, strict=True)
            ) >= 2)
            for field in (
                "left_s1_enrichment", "left_s1_mass_pct",
                "postcentral_laterality", "peak_euclidean_distance_to_left_s1_mm",
            ):
                key = f"{window}_{field}"
                row[f"{key}_run_median"] = float(np.median([
                    float(run_rows[method][key]) for run_rows in by_run
                ]))
        subject_rows.append(row)

if subject_rows:
    summary_path = output_root / "heldout_subject_method_metrics.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(subject_rows[0]))
        writer.writeheader()
        writer.writerows(subject_rows)
    complete_subjects = sorted({row["subject"] for row in subject_rows})
    comparison_rows = []
    for method in methods:
        selected = [row for row in subject_rows if row["method"] == method]
        row = {"method": method, "subjects": len(selected), "runs_per_subject": len(runs)}
        for window in windows:
            enrichment = np.asarray([
                item[f"{window}_left_s1_enrichment_run_median"] for item in selected
            ])
            distance = np.asarray([
                item[f"{window}_peak_euclidean_distance_to_left_s1_mm_run_median"]
                for item in selected
            ])
            row[f"{window}_left_s1_enrichment_subject_median"] = float(np.median(enrichment))
            row[f"{window}_subjects_enrichment_above_uniform"] = int(np.sum(enrichment > 1))
            row[f"{window}_peak_distance_to_roi_subject_median_mm"] = float(np.median(distance))
            row[f"{window}_subjects_peak_in_left_s1_run_majority"] = sum(
                item[f"{window}_peak_left_s1_run_majority"] for item in selected
            )
            row[f"{window}_zero_map_runs"] = sum(
                item[f"{window}_zero_map_runs"] for item in selected
            )
            stability = [
                item[f"{window}_run_map_spearman_median"] for item in selected
                if np.isfinite(item[f"{window}_run_map_spearman_median"])
            ]
            row[f"{window}_run_map_spearman_subject_median"] = (
                float(np.median(stability)) if stability else float("nan")
            )
        comparison_rows.append(row)
    comparison_path = output_root / "heldout_method_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    report_lines = [
        "# ds006035 右腕刺激：八方法 ERP 同输入比较",
        "",
        f"完成的锁定验证受试者：{', '.join(complete_subjects)}；每人 {len(runs)} 个 run 先取中位数，统计单位为受试者。",
        f"分析版本：`{analysis_revision}`；N20=18–24 ms，P30=28–40 ms。",
        "",
        "| 方法 | N20 左S1富集：受试者中位数 | N20 >均匀基线人数 | P30 左S1富集：受试者中位数 | P30 >均匀基线人数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        if row["method"] == "LCMV" and row["n20_zero_map_runs"] == len(runs) * row["subjects"]:
            report_lines.append("| LCMV | 未检出 | — | 未检出 | — |")
            continue
        report_lines.append(
            f"| {row['method']} | {row['n20_left_s1_enrichment_subject_median']:.2f} | "
            f"{row['n20_subjects_enrichment_above_uniform']}/{row['subjects']} | "
            f"{row['p30_left_s1_enrichment_subject_median']:.2f} | "
            f"{row['p30_subjects_enrichment_above_uniform']}/{row['subjects']} |"
        )
    report_lines.extend((
        "",
        "N20 额外检查（峰距离仅指峰到模板左 postcentral ROI 的最近距离，绝非真值 DLE）：",
        "",
        "| 方法 | N20 峰进入左S1：受试者 run 多数 | N20 峰到 ROI 距离中位数(mm) | N20 跨 run 地图 Spearman 中位数 | N20 全零 run |",
        "|---|---:|---:|---:|---:|",
    ))
    for row in comparison_rows:
        if row["method"] == "LCMV" and row["n20_zero_map_runs"] == len(runs) * row["subjects"]:
            report_lines.append(
                f"| LCMV | — | — | — | {row['n20_zero_map_runs']}/{len(runs)*row['subjects']} |"
            )
            continue
        stability = row["n20_run_map_spearman_subject_median"]
        report_lines.append(
            f"| {row['method']} | {row['n20_subjects_peak_in_left_s1_run_majority']}/{row['subjects']} | "
            f"{row['n20_peak_distance_to_roi_subject_median_mm']:.1f} | "
            f"{stability:.2f}" if np.isfinite(stability) else
            f"| {row['method']} | {row['n20_subjects_peak_in_left_s1_run_majority']}/{row['subjects']} | "
            f"{row['n20_peak_distance_to_roi_subject_median_mm']:.1f} | —"
        )
        report_lines[-1] += f" | {row['n20_zero_map_runs']}/{len(runs)*row['subjects']} |"
    report_lines.extend((
        "",
        "同一原始 run 的 EEG/MAG 通道、evoked、sample 模板 ico4 皮层前向模型、白化数据与源网格供八方法共用。"
        "七个对比方法来自 `benchmark/methods.py` 的固定方向 Python 网格实现，不是 MNE 官方逆解的全套实现。"
        "LCMV 使用同一 15–45 ms 协方差滤波器，Dipole/RAP 各使用目标窗的整数时间索引。",
        "",
        "真实数据没有源位置真值；左 S1 富集只检验解剖合理性，不是 AUC。"
        "峰到 ROI 距离不是 DLE；sample 刚性模板也不是个体 MRI。"
        "当前网格只有皮层，没有深部体积源，因此不验证 OASTER-v3 的深层定位。",
        "",
        "LCMV 在这个短窗平均 ERP 的基线扣除 excess-power 指标下得到全零图；这应记为未检出，"
        "不能以零富集把它排名为最差算法，也不对参数做事后搜索。"
        "峰值归一化和 P95/10% 仅用于脑图显示，不参与上述指标。",
        "",
        "OASTER-v3 及稀疏偶极方法的富集值接近 36–37 时，主要反映支持集中于仅占网格约 2.7% 的左 S1 ROI；"
        "它不是几十倍的定位精度，须和命中人数、峰位置、跨 run 稳定性一起看。",
        "锁定验证仅 4 名受试者，不足以建立任一方法的显著优势；本表是外部数据探索性对照。",
        "",
        "sm09 是开发检查，不进入锁定验证统计。代表脑图："
        "[`N20`](sub-sm09_run-1/n20_eight_methods_pial_dorsal.png)、"
        "[`P30`](sub-sm09_run-1/p30_eight_methods_pial_dorsal.png)。",
        "逐 run 原始指标与源图在各 `sub-*_run-*` 目录。",
    ))
    if missing_runs:
        report_lines.extend(("", "未纳入的 run：" + "；".join(missing_runs)))
    (output_root / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
    for ax, window in zip(axes, windows, strict=True):
        values = np.asarray([
            [next(item[f"{window}_left_s1_enrichment_run_median"]
                  for item in subject_rows if item["subject"] == subject and item["method"] == method)
             for subject in complete_subjects]
            for method in methods
        ], dtype=float)
        display = np.ma.masked_invalid(np.log1p(values))
        display[methods.index("LCMV")] = np.ma.masked
        color_map = plt.get_cmap("viridis").copy()
        color_map.set_bad("#e5e7eb")
        image = ax.imshow(display, cmap=color_map, vmin=0, vmax=np.log1p(37), aspect="auto")
        ax.set_xticks(range(len(complete_subjects)), complete_subjects)
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(f"{window.upper()} | left S1 enrichment | n={len(complete_subjects)}")
        for row_index, method in enumerate(methods):
            for col_index in range(len(complete_subjects)):
                label = "ND" if method == "LCMV" else f"{values[row_index, col_index]:.1f}"
                color = "white" if values[row_index, col_index] > 9 and method != "LCMV" else "black"
                ax.text(col_index, row_index, label, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=axes, shrink=0.76, label="log(1 + left S1 enrichment); ND = no positive excess-power map")
    fig.suptitle("ds006035 held-out subjects | 3-run median per subject | same ERP and source grid", fontsize=13)
    fig.savefig(output_root / "heldout_eight_method_enrichment_heatmaps.png", dpi=170)
    plt.close(fig)

print("完成。结果目录：", output_root, flush=True)
