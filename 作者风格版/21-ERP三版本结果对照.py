"""Pair the existing 20 pilot cases across v3/v4/v5; descriptive results only."""

# %% 只读取已有结果；四个源配置的重复SNR不作为独立样本做CI或p值。
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
import run_strict_oaster as archive
from run_erp_whole_head_matrix import _shared_fingerprint
import plot_erp_whole_head_results as style

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / "results/erp_whole_head/adaptive_v5/paired_v3_v4_v5")
args = parser.parse_args()
inputs = {
    "v3": root / "results/erp_whole_head/development_full_v3/v3_locked",
    "v4": root / "results/erp_whole_head/adaptive_v4/pilot_five_snr_all_methods",
    "v5": root / "results/erp_whole_head/adaptive_v5/pilot_five_snr_all_methods",
}
metadata = {version: json.loads((path / "metadata.json").read_text(encoding="utf-8")) for version, path in inputs.items()}
reference = metadata["v5"]
for version, meta in metadata.items():
    complete = json.loads((inputs[version] / "completion.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete" and complete["error_count"] == 0
    assert meta["oaster_algorithm_version"] == version
    for key in ("manifest_sha256", "erp_seed_root", "checkpoint_shared_fingerprint", "checkpoint_environment",
                "oaster_modality_weighting", "protocol", "observation_rule", "window_rule", "snr_level"):
        assert meta[key] == reference[key], f"{version}: incompatible {key}"
    assert meta["oaster_modality_weighting"] == "evidence" and meta["erp_seed_root"] == 20260921
    assert meta["provenance"]["code_sha256"]["erp_protocol"] == reference["provenance"]["code_sha256"]["erp_protocol"]

# v3旧指标版本只差深检阈值；核对Git源码，避免混入AUC/距离定义变化。
old_commit = metadata["v3"]["provenance"]["git_commit_at_merge"]
old_metric = subprocess.run(["git", "show", f"{old_commit}:benchmark/metrics.py"], cwd=root,
                            capture_output=True, text=True, encoding="utf-8", check=True).stdout
normalized_old_metric = old_metric.replace("DEEP_DETECTION_THRESHOLD = 0.125", "DEEP_DETECTION_THRESHOLD = 0.14").replace(
    "    0.1,\n    DEEP_DETECTION_THRESHOLD,", "    0.1,\n    0.125,\n    DEEP_DETECTION_THRESHOLD,")
assert normalized_old_metric == Path(metrics.__file__).read_text(encoding="utf-8")
assert not subprocess.run(["git", "diff", old_commit, "--", "metrics/user_metrics"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()
assert hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest() == reference["provenance"]["code_sha256"]["metrics"]
manifest_path = Path(reference["manifest"])
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == reference["manifest_sha256"]
manifest = {case["case_id"]: case for case in json.loads(manifest_path.read_text(encoding="utf-8"))}
shared = protocol.load_shared(Path(reference["data_root"]), Path(reference["sample_path"]))
assert _shared_fingerprint(shared) == reference["checkpoint_shared_fingerprint"]
penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000)
threshold = metrics.DEEP_DETECTION_THRESHOLD
radius_mm = float(reference["deep_detection_rule"]["maximum_peak_distance_mm"])
assert threshold == reference["deep_detection_rule"]["relative_amplitude_threshold"]

# %% case_id、场景、SNR及真实源配置全部严格匹配，保留60行原始指标。
tables = {}
input_hashes = {}
for version, path in inputs.items():
    with (path / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == f"OASTER-ERP-{version}"]
    assert rows and len({row["case_id"] for row in rows}) == len(rows)
    tables[version] = {row["case_id"]: row for row in rows}
    input_hashes[version] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                             for name in ("rows.csv", "metadata.json", "completion.json")}
case_ids = sorted(tables["v5"], key=lambda key: int(tables["v5"][key]["case_number"]))
assert len(case_ids) == 20 and set(tables["v4"]) == set(case_ids)
assert set(case_ids) <= tables["v3"].keys()
assert len({tables["v5"][key]["configuration_id"] for key in case_ids}) == 4
assert len({(tables["v5"][key]["eeg_snr_db"], tables["v5"][key]["meg_snr_db"]) for key in case_ids}) == 5
diagnostics = {}
for version in ("v4", "v5"):
    pattern = f"parts_*_run_{metadata[version]['checkpoint_fingerprint'][:12]}/diagnostics/*.json"
    diagnostics[version] = {}
    for path in inputs[version].glob(pattern):
        item = json.loads(path.read_text(encoding="utf-8"))
        assert item["case_id"] not in diagnostics[version]
        windows = item["diagnostics"]["windows"]
        diagnostics[version][item["case_id"]] = int(all(window.get("solver") is not None and window["solver"].get("converged", False) for window in windows))
    assert set(diagnostics[version]) == set(case_ids)

paired = []
pair_fields = ("case_number", "configuration_number", "configuration_id", "pair_index", "scenario",
               "eeg_snr_db", "meg_snr_db", "surface_centers", "deep_index", "deep_surface_ratio", "correlation")
for case_id in case_ids:
    expected = manifest[case_id]
    for version in inputs:
        row = tables[version][case_id]
        assert row["status"] == "ok" and row["manifest_sha256"] == reference["manifest_sha256"]
        for key in pair_fields:
            assert row[key] == tables["v5"][case_id][key], f"{case_id}: mismatched {key}"
            actual = json.loads(row[key]) if key == "surface_centers" else row[key]
            expected_value = expected[key]
            assert actual == expected_value or str(actual) == ("" if expected_value is None else str(expected_value)), f"manifest mismatch: {case_id}/{key}"
        record = {"version": version, **row,
                  "original_deep_detected": row["deep_detected"], "original_deep_false_positive": row["deep_false_positive"],
                  "original_deep_threshold": metadata[version]["deep_detection_rule"]["relative_amplitude_threshold"],
                  "comparison_deep_threshold": threshold,
                  "solver_converged": diagnostics.get(version, {}).get(case_id, "")}
        has_deep = bool(int(row["has_deep_true"]))
        positive = float(row["deep_score"]) >= threshold
        distance = float(row["deep_peak_distance_mm"])
        record["deep_detected"] = int(has_deep and positive and np.isfinite(distance) and distance <= radius_mm + 1e-6)
        record["deep_false_positive"] = int(not has_deep and positive)
        for layer in ("surface", "deep"):
            for name in ("sd_mm", "dle_mm"):
                field = f"{layer}_{name}"
                value = float(row[field])
                present = bool(int(row["has_surface_true" if layer == "surface" else "has_deep_true"]))
                record[field + "_penalized"] = (np.nan if not present else penalty_mm
                    if not np.isfinite(value) or (layer == "deep" and not record["deep_detected"]) else value)
        paired.append(record)
assert len(paired) == 60
summary = []
for version in inputs:
    rows = [row for row in paired if row["version"] == version]
    counts = {"true_deep_cases": sum(int(row["has_deep_true"]) for row in rows),
              "true_surface_only_cases": sum(not int(row["has_deep_true"]) for row in rows),
              "deep_TP": sum(row["deep_detected"] for row in rows),
              "deep_FP": sum(row["deep_false_positive"] for row in rows)}
    summary.append({"version": version, "configuration_count": 4, "snr_pair_count": 5,
                    "solver_converged_cases": sum(diagnostics[version].values()) if version in diagnostics else "not_applicable",
                    "deep_decision_changes_after_standardization": sum(int(row["original_deep_detected"]) != row["deep_detected"] or int(row["original_deep_false_positive"]) != row["deep_false_positive"] for row in rows),
                    **counts, **archive._aggregate(rows, penalty_mm)})

# %% 均值及计数，不计算CI/p值；源层不存在时保持NaN。
args.output.mkdir(parents=True, exist_ok=True)
for filename, rows in (("matched_cases.csv", paired), ("version_summary.csv", summary)):
    with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
style._style()
plt = style.plt
colors = ("#0072B2", "#D55E00", "#009E73")
figure, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
panels = (
    (("auc_tie_corrected", "Overall", "^"), ("surface_auc_tie_corrected", "Surface", "o"), ("deep_auc_tie_corrected", "Deep", "s")),
    (("surface_sd_mm_penalized", "SD", "o"), ("surface_dle_mm_penalized", "DLE", "s")),
    (("deep_sd_mm_penalized", "SD", "o"), ("deep_dle_mm_penalized", "DLE", "s")),
    (("deep_TP", "TP", "o"), ("deep_FP", "FP", "X")),
)
for axis, panel, title in zip(axes, panels, ("An_auc", "Surface distances", "Deep miss-penalized distances", "Deep TP / FP counts")):
    for offset, (field, label, marker) in zip(np.linspace(-.18, .18, len(panel)), panel):
        values = [row[field] for row in summary]
        axis.scatter(values, np.arange(3) + offset, c=colors, marker=marker, s=75, edgecolors="white", linewidths=.6, zorder=3)
        axis.scatter([], [], c="#555555", marker=marker, s=45, label=label)
        if field.startswith("deep_T") or field.startswith("deep_F"):
            for index, value in enumerate(values):
                axis.annotate(str(value), (value, index + offset), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10)
    axis.set_title(title, fontsize=11, pad=14)
    axis.set_yticks(range(3), ("v3 fixed templates", "v4 adaptive graph", "v5 balanced graph"))
    axis.set_ylim(2.5, -.5)
    axis.grid(axis="x", alpha=.2)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False, ncol=len(panel))
axes[0].set(xlim=(.55, 1.02), xlabel="Higher is better")
axes[1].set(xlim=(0, None), xlabel="mm; lower is better")
axes[2].set(xlim=(0, penalty_mm * 1.03), xlabel=f"Miss penalty = {penalty_mm:.2f} mm")
axes[3].set(xlim=(-.5, max(row["true_deep_cases"] for row in summary) + 1.5),
            xlabel=f"TP / {summary[0]['true_deep_cases']} true-deep; FP / {summary[0]['true_surface_only_cases']} surface-only")
figure.suptitle("Same 20 cases · 4 source configurations · 5 SNR pairs\nDescriptive version comparison; no CI or p-values", fontsize=14)
figure.tight_layout(rect=(0, .02, 1, .97))
figure.savefig(args.output / "three_version_comparison.png", dpi=200, bbox_inches="tight")
plt.close(figure)

# %% 保存可核查来源，并指出这不是仅改一个因素的等价消融。
audit = {"inputs": {key: str(path) for key, path in inputs.items()}, "input_hashes": input_hashes,
    "matched_case_count": len(case_ids), "matched_row_count": len(paired), "manifest_sha256": reference["manifest_sha256"],
    "seed_root": reference["erp_seed_root"], "modality_weighting": reference["oaster_modality_weighting"],
    "shared_fingerprint": reference["checkpoint_shared_fingerprint"], "n_surf": shared["n_surf"], "n_deep": shared["n_deep"],
    "missing_distance_penalty_mm": penalty_mm, "comparison_deep_threshold": threshold, "deep_radius_mm": radius_mm,
    "metric_labels": {"auc": "An_cal_AUC parcel/group AUC", "auc_tie_corrected": "An_auc tie-aware whole-grid AUC"},
    "metric_compatibility": "Git audit confirms v3 metric source differs only in deep threshold and candidate list; user metric implementations unchanged",
    "algorithm_parameters": {key: meta["oaster_kwargs"] for key, meta in metadata.items()},
    "scope": "4 configurations repeated across 5 SNR pairs; no independent-subject CI or p-values",
    "not_an_equivalent_ablation": "v3 uses fixed multiscale templates and added temporal evidence; v4/v5 use graph reweighting. v5 also changes temporal basis, layer calibration and numerical stopping criteria.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(args.output / "metadata.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
report = ["# ERP三版本同病例描述对照", "", "同20例、4个源配置、5组SNR；共60行严格配对，不计算置信区间或p值。",
    "", "| 版本 | An_auc | 表层An_auc | 深层An_auc | 深源TP | 纯浅FP | 数值收敛 |", "|---|---:|---:|---:|---:|---:|---:|"]
for row in summary:
    report.append(f"| {row['version']} | {row['auc_tie_corrected']:.4f} | {row['surface_auc_tie_corrected']:.4f} | {row['deep_auc_tie_corrected']:.4f} | {row['deep_TP']}/{row['true_deep_cases']} | {row['deep_FP']}/{row['true_surface_only_cases']} | {row['solver_converged_cases']} |")
report += ["", f"所有深检统一采用幅度比≥{threshold:g}且峰距≤{radius_mm:g}mm。v3原阈值0.125已从保存的deep_score重算，原标记另列保留。漏检距离罚值来自实际7,514点源空间bbox对角线：{penalty_mm:.6f}mm。",
    "", "CSV同时保存原始及罚分SD/DLE；深层漏检即使存在某个近邻峰也计漏检罚分，未过滤未收敛病例。AUC与对应层的均值仅在该层真值存在时计算。",
    "", "auc字段是An_cal_AUC；auc_tie_corrected以及surface/deep_auc_tie_corrected才是An_auc。图使用后者。",
    "", "三版manifest、源位置/比例/相关性、20260921随机根种子、evidence模态权重、前向/噪声共享指纹及环境均匹配。v3指标源码与当前仅深检阈值和候选列表不同，已读取Git历史核查。",
    "", "这是算法版本演进对照，不是单因素等价消融：v3含预制多尺度模板和附加时间证据；v4/v5使用源及边权重更新；v5还改变了时间基、分层校准与数值停止检查。不能把差异全部归因于某个空间正则项。",
    "", "相对v4，v5提高深源TP的同时也提高了纯浅FP；与v3的TP相同而FP更多，必须同时阅读灵敏度与特异度。四个空间配置尚不足以证明全头或真实数据优势。",
    "", "文件：matched_cases.csv（60行）、version_summary.csv（3行）、three_version_comparison.png、metadata.json。"]
(args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
print(json.dumps([{key: row[key] for key in ("version", "auc_tie_corrected", "surface_auc_tie_corrected", "deep_auc_tie_corrected", "deep_TP", "deep_FP", "solver_converged_cases")} for row in summary], indent=2))
print(args.output)
