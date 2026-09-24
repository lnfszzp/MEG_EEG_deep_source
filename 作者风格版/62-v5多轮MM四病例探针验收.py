"""# %% 按冻结条件验收 v5 多轮 MM 四病例探针；不调阈值。"""

# %% 1. 输入、代码和求解设置必须与预声明探针一致。
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--candidate", choices=("floor", "alias"), required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if output.exists():
    raise FileExistsError(f"不覆盖旧结果：{output}")

completion = json.loads((source / "completion.json").read_text(encoding="utf-8"))
metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
expected_settings = {
    "solver_kind": "admm", "mrf_strength": .8, "outer_iterations": 20,
    "max_inner_retries": 20, "epsilon_fraction": .05, "tolerance": .001,
    "outer_tolerance": .01, "surface_reweight_floor": .5,
    "deep_reweight_floor": .5, "edge_weight_floor": .5,
}
if args.candidate == "alias":
    expected_settings["deep_alias_penalty"] = True
if metadata["phase"] != "development" or metadata["covariance"] != "trial" or \
        metadata.get("score_kind") != "excess" or metadata["solver_settings"] != expected_settings:
    raise ValueError("探针必须使用预声明的 trial/excess/多轮 MM 设置")
if not completion["complete"] or completion["case_count"] != 4:
    raise ValueError("四病例探针尚未完整运行")
for relative, expected in metadata["code_sha256"].items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"探针代码已变化：{path}")

manifest = Path(metadata["manifest"])
if hashlib.sha256(manifest.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
    raise ValueError("探针 manifest 与上游记录不一致")
cases = json.loads(manifest.read_text(encoding="utf-8"))
if [case["case_number"] for case in cases] != [0, 2, 4, 10]:
    raise ValueError("不是冻结的 00/02/04/10 探针")


# %% 2. 验证八个拟合都是真正的多轮固定点，并读取预先选定的 excess 分数。
rows, fit_rows = [], []
for case in cases:
    name = case["case_id"]
    payload = json.loads((source / (name + ".json")).read_text(encoding="utf-8"))
    score = float(payload["evidence"]["excess_fraction_score"])
    is_surface = case["scenario"] == "surface_only"
    rows.append({"case_id": name, "case_number": case["case_number"],
                 "scenario": case["scenario"], "truth_group": "H0" if is_surface else "H1",
                 "excess_score": score})
    for family in ("null", "full"):
        solver = payload["fitting"][family + "_model"]["windows"][0]["solver"]
        history = solver["history"]
        objectives = np.array([item["log_objective"] for item in history], float)
        monotone = bool(np.all(np.diff(objectives) <= 1e-8 * np.maximum(1, np.abs(objectives[:-1]))))
        fit_rows.append({
            "case_id": name, "case_number": case["case_number"], "family": family,
            "outer_steps": len(history), "converged": int(solver["converged"]),
            "final_inner_converged": int(solver["final_inner_converged"]),
            "outer_converged": int(solver["outer_converged"]),
            "stationarity_gap": float(solver["final_stationarity_gap_relative"]),
            "minimum_amplitude_weight": float(solver["amplitude_weight_range"][0]),
            "minimum_edge_weight": float(solver["edge_weight_range"][0]),
            "objective_monotone": int(monotone),
        })

h0_scores = [row["excess_score"] for row in rows if row["truth_group"] == "H0"]
h1_scores = [row["excess_score"] for row in rows if row["truth_group"] == "H1"]
separation_margin = float(min(h1_scores) - max(h0_scores))
all_fits_valid = all(
    row["converged"] and row["final_inner_converged"] and row["outer_converged"]
    and row["outer_steps"] > 1 and row["stationarity_gap"] <= .001 + 1e-12
    and row["minimum_amplitude_weight"] >= .5 - 1e-12
    and row["minimum_edge_weight"] >= .5 - 1e-12 and row["objective_monotone"]
    for row in fit_rows)
probe_passed = bool(all_fits_valid and separation_margin > 0)


# %% 3. 事后定位指标只做诊断，不进入预声明主条件。
with (source / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    raw_metrics = list(csv.DictReader(stream))
for row in rows:
    method = "v6-surface-only" if row["truth_group"] == "H0" else "v6-full-ungated"
    selected = next(item for item in raw_metrics
                    if item["case_id"] == row["case_id"] and item["method"] == method)
    row.update(posthoc_method=method, posthoc_AUC=float(selected["auc"]),
               posthoc_surface_DLE_mm=float(selected["surface_dle_mm"]),
               posthoc_deep_peak_distance_mm=float(selected["deep_peak_distance_mm"]))

summary = {
    "complete": True, "development_only": True, "formal_acceptance_allowed": False,
    "candidate": args.candidate,
    "predeclared_rule": "max(T00,T10) < min(T02,T04)",
    "all_eight_fits_valid": all_fits_valid,
    "max_h0_score": float(max(h0_scores)), "min_h1_score": float(min(h1_scores)),
    "separation_margin": separation_margin, "probe_passed": probe_passed,
    "old_v4_threshold_used_for_acceptance": False,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
output.mkdir(parents=True)
(output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
archive._atomic_csv(output / "case_scores.csv", rows, rows[0].keys())
archive._atomic_csv(output / "fit_diagnostics.csv", fit_rows, fit_rows[0].keys())


# %% 4. 图和表直接显示分离边界与真正的外层 stationarity gap。
figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
colors = ["#D55E00" if row["truth_group"] == "H0" else "#007C83" for row in rows]
markers = ["s" if row["truth_group"] == "H0" else "o" for row in rows]
for index, (row, color, marker) in enumerate(zip(rows, colors, markers)):
    axes[0].scatter(index, row["excess_score"], color=color, marker=marker, s=65)
axes[0].axhline(max(h0_scores), color="#D55E00", linestyle=":", linewidth=1.4)
axes[0].axhline(min(h1_scores), color="#007C83", linestyle=":", linewidth=1.4)
axes[0].set(title=f"Predeclared score separation (margin={separation_margin:.4f})",
            ylabel="Independent excess score", xticks=np.arange(4),
            xticklabels=[f"case {row['case_number']:02d}\n{row['truth_group']}" for row in rows])
fit_x = np.arange(len(fit_rows))
axes[1].scatter(fit_x, [row["stationarity_gap"] for row in fit_rows],
                color=["#4C78A8" if row["family"] == "null" else "#6A51A3" for row in fit_rows],
                s=48)
axes[1].axhline(.001, color="#C44E52", linestyle=":", linewidth=1.4)
axes[1].set_yscale("log")
axes[1].set(title="Final adaptive stationarity gaps", ylabel="Relative gap",
            xticks=fit_x, xticklabels=[f"{row['case_number']:02d}-{row['family']}" for row in fit_rows])
axes[1].tick_params(axis="x", labelrotation=45, labelsize=8)
for axis in axes:
    axis.grid(axis="y", color="#DDDDDD", linewidth=.7)
figure.suptitle(f"v5 {args.candidate} bounded-weight multi-round MM probe", fontweight="bold")
figure.savefig(output / "probe.png", dpi=220, facecolor="white")
plt.close(figure)

lines = [f"# v5 多轮 MM 四病例探针结果（{args.candidate}）", "",
    f"预声明分离条件：`max(H0) < min(H1)`；margin = {separation_margin:.6f}。",
    f"八个拟合全部达到多轮固定点：{all_fits_valid}；探针通过：{probe_passed}。", "",
    "| 病例 | 真值组 | excess | 事后方法 | AUC | 表层DLE mm | 深峰距离 mm |",
    "|---:|---|---:|---|---:|---:|---:|"]
for row in rows:
    surface_dle = "—" if not np.isfinite(row["posthoc_surface_DLE_mm"]) else f"{row['posthoc_surface_DLE_mm']:.2f}"
    deep_distance = ("—" if not np.isfinite(row["posthoc_deep_peak_distance_mm"])
                     else f"{row['posthoc_deep_peak_distance_mm']:.2f}")
    lines.append(f"| {row['case_number']:02d} | {row['truth_group']} | {row['excess_score']:.6f} | "
                 f"{row['posthoc_method']} | {row['posthoc_AUC']:.3f} | {surface_dle} | {deep_distance} |")
lines += ["", "这是有标签开发探针，不是正式校准或 validation。", ""]
(output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
