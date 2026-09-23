"""# %% 同一 trial-covariance 链路下，比较当前代码的 20 与 40 试次定位。"""

# %% 路径、病例和固定算法；只读 development，不读取校准/验证。
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_predictive import fit_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

base = root / "results/erp_whole_head/adaptive_v6"
output = base / "development_diagnosis/combined40_trial_covariance_convex_m10_m10"
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")
output.mkdir(parents=True)

cases = []
for number in ("01", "03"):
    manifest = base / f"protocol/development_component_balanced_m10_m10_case{number}.json"
    cases.extend(json.loads(manifest.read_text(encoding="utf-8")))
assert len(cases) == 2 and [case["case_number"] for case in cases] == [1, 3]
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
solver_settings = {"solver_kind": "admm", "mrf_strength": .8, "outer_iterations": 1}
seed_root = 2026092206
tracked = [root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_balanced.py",
    "candidates/graph_reweight_solver.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py",
    "benchmark/erp_protocol.py", "benchmark/protocol.py", "benchmark/metrics.py")]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in tracked}

# %% 每个病例用完全相同的观测、白化器和当前代码分别拟合 train20 与 combined40。
rows, details = [], []
started = time.perf_counter()
for case in cases:
    tick = time.perf_counter()
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    datasets = {"train20": observation["training"],
                "combined40": (observation["training"] + observation["confirmation"]) / 2}
    estimates, convergence, fitting = {}, {}, {}
    for trials, data in datasets.items():
        null, full, fit = fit_predictive_models(
            data, observation["gain"], shared["n_surf"], adjacency=shared["adjacency"],
            baseline=observation["baseline"], active=observation["active_windows"][0],
            channel_weights=observation["channel_weights"], solver_settings=solver_settings)
        estimates[trials] = {"surface-only": null, "full-ungated": full}
        fitting[trials] = fit
        convergence[trials] = {name: bool(fit[name + "_model"]["windows"][0]["solver"]["converged"])
                               for name in ("null", "full")}
        for family, estimate in estimates[trials].items():
            values = metrics.evaluate_estimate(
                estimate, observation["truth"], shared["vertices"], observation["groups"],
                shared["n_surf"], observation["active"], shared["auc_cortex"],
                baseline=observation["baseline"])
            rows.append({"case_id": case["case_id"], "case_number": case["case_number"],
                "scenario": case["scenario"], "eeg_snr_db": case["eeg_snr_db"],
                "meg_snr_db": case["meg_snr_db"], "method": trials + "-" + family, **values})
    np.savez_compressed(output / (case["case_id"] + ".npz"),
        truth=observation["truth"].astype(np.float32),
        train_null=estimates["train20"]["surface-only"].astype(np.float32),
        train_full=estimates["train20"]["full-ungated"].astype(np.float32),
        combined_null=estimates["combined40"]["surface-only"].astype(np.float32),
        combined_full=estimates["combined40"]["full-ungated"].astype(np.float32),
        vertices=shared["vertices"], times=shared["times"],
        active=observation["active"], baseline=observation["baseline"])
    detail = {"case_id": case["case_id"], "complete": True,
        "comparison": "current-code train20 versus combined40 using the same trial-covariance whitener",
        "combined_formula": "(training + confirmation) / 2",
        "preparation": "prepare_trial_covariance_case",
        "truth_used_by_inverse": False, "truth_used_only_for_development_metrics": True,
        "deployable_presence_decision_evaluated": False,
        "solver_settings": solver_settings, "convergence": convergence,
        "elapsed_seconds": time.perf_counter() - tick, "fitting": fitting}
    details.append(detail)
    (output / (case["case_id"] + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
    print(case["case_id"], convergence, f"{detail['elapsed_seconds']:.1f}s", flush=True)

# %% 只作算法能力诊断：case01 看 H0、case03 看 H1；这不是可部署门控。
records = []
for case, family in (("01", "surface-only"), ("03", "full-ungated")):
    case_id = f"erp-v6-development-{case}-eeg-10-meg-10"
    for trials in ("train20", "combined40"):
        row = next(row for row in rows if row["case_id"] == case_id
                   and row["method"] == trials + "-" + family)
        records.append({"case": case, "trials": trials, "oracle_family": family,
            "local_An_cal_AUC": float(row["auc"]), "An_auc": float(row["auc_tie_corrected"]),
            "surface_An_auc": float(row["surface_auc_tie_corrected"]),
            "deep_An_auc": float(row["deep_auc_tie_corrected"]),
            "surface_SD_mm": float(row["surface_sd_mm"]),
            "surface_DLE_mm": float(row["surface_dle_mm"]),
            "deep_DLE_mm": float(row["deep_dle_mm"]), "deep_detected": int(row["deep_detected"]),
            "deep_false_positive": int(row["deep_false_positive"])})

with (output / "paired_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

combined_rows = [row for row in records if row["trials"] == "combined40"]
case01_raw_full = next(row for row in rows if row["case_number"] == 1
                       and row["method"] == "combined40-full-ungated")
summary = {"complete": True, "phase": "development", "covariance": "trial",
    "snr_eeg_meg_db": [-10, -10], "solver_settings": solver_settings,
    "comparison_is_same_current_code_and_whitener": True,
    "oracle_family_selection_for_diagnostic": True,
    "deployable_presence_decision_evaluated": False,
    "local_AUC_target": .9,
    "exploratory_local_target_met": all(row["local_An_cal_AUC"] >= .9 for row in combined_rows),
    "case01_local_AUC_delta": combined_rows[0]["local_An_cal_AUC"] - records[0]["local_An_cal_AUC"],
    "case03_local_AUC_delta": combined_rows[1]["local_An_cal_AUC"] - records[2]["local_An_cal_AUC"],
    "case01_full_raw_deep_false_positive": int(case01_raw_full["deep_false_positive"]),
    "case03_deep_detected": combined_rows[1]["deep_detected"],
    "all_converged": all(all(all(block.values()) for block in detail["convergence"].values())
                         for detail in details),
    "accepted": False, "wall_seconds": time.perf_counter() - started,
    "code_sha256": code_hashes}
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 配对图。
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
x = np.arange(2)
for color, trials in (("#9C9C9C", "train20"), ("#007C83", "combined40")):
    pair = [row for row in records if row["trials"] == trials]
    axes[0].plot(x, [row["local_An_cal_AUC"] for row in pair], "o-", color=color,
                 linewidth=2.4, label=trials)
    axes[1].plot(x, [row["An_auc"] for row in pair], "o-", color=color,
                 linewidth=2.4, label=trials)
    axes[2].plot(x, [row["surface_DLE_mm"] for row in pair], "o-", color=color,
                 linewidth=2.4, label=trials)
axes[0].axhline(.9, color="#C44E52", linestyle=":", linewidth=1.5)
axes[0].set(title="Local An_cal_AUC", ylabel="AUC", ylim=(.45, 1.02))
axes[1].set(title="Global An_auc", ylabel="AUC", ylim=(.85, 1.02))
axes[2].set(title="Surface DLE", ylabel="mm")
for axis in axes:
    axis.set(xticks=x, xticklabels=["S+S (oracle H0)", "S+D (oracle H1)"])
axes[0].legend(frameon=False)
figure.suptitle("Trial-covariance current-code 20 vs 40 trial localization", fontweight="bold")
figure.savefig(output / "comparison.png", dpi=220, facecolor="white")
plt.close(figure)

# %% 报告：oracle 诊断不写成验收通过。
lines = ["# 同链路 20 与 40 试次定位诊断", "",
    "两次拟合使用当前同一代码、同一 trial-covariance 白化器。case01 的 H0 与 case03 的 H1 是按开发真值选取，只用于判断定位上限，不是可部署 presence 门控。", "",
    "| case | 试次数 | oracle family | 局部 AUC | An_auc | 表层 SD/DLE mm | 深层 DLE mm | 深层检出 |",
    "|---:|---|---|---:|---:|---:|---:|---:|"]
for row in records:
    deep = "—" if not np.isfinite(row["deep_DLE_mm"]) else f"{row['deep_DLE_mm']:.2f}"
    lines.append(f"| {row['case']} | {row['trials']} | {row['oracle_family']} | "
        f"{row['local_An_cal_AUC']:.3f} | {row['An_auc']:.3f} | "
        f"{row['surface_SD_mm']:.2f}/{row['surface_DLE_mm']:.2f} | {deep} | "
        f"{row['deep_detected']} |")
lines += ["", f"局部 0.90 探索目标：{'达到' if summary['exploratory_local_target_met'] else '未达到'}。",
    f"case01 的未门控 H1 原始深源误报：{summary['case01_full_raw_deep_false_positive']}。",
    "没有冻结 presence 决策，因此本实验固定 accepted=false。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in tracked}
print(json.dumps(summary, ensure_ascii=False, indent=2))
