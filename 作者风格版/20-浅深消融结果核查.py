# %% 开发消融的配对结果与求解诊断；只读已完成实验，不启动仿真，不计算 p 值或 CI。
from pathlib import Path
import argparse
import csv
import json
import os
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import run_erp_whole_head_matrix as simulation
import plot_erp_whole_head_results as style

parser = argparse.ArgumentParser(description="Paired development ablations; descriptive results only")
parser.add_argument("--input", type=Path, default=root / "results/erp_whole_head/adaptive_v5")
parser.add_argument("--output", type=Path)
args = parser.parse_args()
output = args.output or args.input / "ablation_summary"

# 直接复用 runner 的几何与计算规则，绝不猜测或重新选择漏检惩罚距离。
shared = simulation.protocol.load_shared(simulation.DEFAULT_DATA_ROOT, simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
penalty_mm = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000.0)
skipped, summaries, paired_rows, audits = [], [], [], []
reference = None


# %% 按 metadata 的完整参数匹配行，再按 case_id 配对。未完成、错版或坏数据列出原因。
for directory in sorted(args.input.glob("ablation_*")):
    if not directory.is_dir() or directory.resolve() == output.resolve():
        continue
    if not (directory / "completion.json").is_file():
        skipped.append({"experiment": directory.name, "reason": "no completion.json; unfinished or aborted"})
        continue
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert completion["code_fingerprint"] == metadata["code_fingerprint"], "code fingerprint mismatch"
        assert metadata["shared_fingerprint"] == shared_fingerprint, "geometry/observation inputs changed"
        assert completion["error_count"] == 0 and all(row["status"] == "ok" for row in rows), "completed experiment contains failed cases"
        assert completion["cases"] == len(rows) == len(metadata["settings"]) * len(metadata["case_ids"]), "incomplete rows"
        pairing = (metadata["manifest_sha256"], metadata["seed_root"], tuple(sorted(metadata["case_ids"])))
        assert reference is None or pairing == reference, "different observations/case IDs; not paired with this summary"
        local_groups = []
        for parameters in metadata["settings"]:
            tag = f"{parameters['solver_kind']}_{parameters['calibration']}_{parameters['temporal_mode']}"
            group = [row for row in rows if row["setting"] == tag]
            by_case = {row["case_id"]: row for row in group}
            assert len(by_case) == len(group) == len(metadata["case_ids"]), "missing or duplicated setting/case"
            assert set(by_case) == set(metadata["case_ids"]), "setting uses different case IDs"
            assert {row["scenario"] for row in group} == set(simulation.SCENARIOS), "four scenarios required"
            records = []
            for case_id in sorted(by_case):
                row = by_case[case_id]
                assert row["manifest_sha256"] == metadata["manifest_sha256"], "row manifest mismatch"
                path = directory / tag / "diagnostics" / f"case_{int(row['case_number']):05d}.json"
                diagnostic = json.loads(path.read_text(encoding="utf-8"))
                assert diagnostic["case_id"] == case_id and diagnostic["parameters"] == parameters, "diagnostic parameter/case mismatch"
                assert diagnostic["method"] == row["method"], "diagnostic method mismatch"
                assert diagnostic["manifest_sha256"] == metadata["manifest_sha256"], "diagnostic manifest mismatch"
                records.append((row, diagnostic["diagnostics"]))
            local_groups.append((parameters, tag, records))
        reference = pairing
    except (OSError, ValueError, KeyError, AssertionError) as error:
        skipped.append({"experiment": directory.name, "reason": str(error) or type(error).__name__})
        continue

    for parameters, tag, records in local_groups:
        setting_id = f"S{len(summaries) + 1:02d}"
        label = f"{setting_id} {parameters['solver_kind'].upper()} / {parameters['calibration']} / {parameters['temporal_mode']} / tau={parameters['mrf_strength']:g}"
        aggregate = simulation.archive._aggregate([row for row, _ in records], penalty_mm)
        positives = [row for row, _ in records if int(row["has_deep_true"]) == 1]
        negatives = [row for row, _ in records if row["scenario"] == "surface_only"]
        summary = {
            "setting_id": setting_id, "label": label, "experiment": directory.name, "setting": tag,
            "parameters_json": json.dumps(parameters, sort_keys=True), "code_fingerprint": metadata["code_fingerprint"],
            "case_count": len(records), "configuration_count": len({row["configuration_id"] for row, _ in records}),
            **{name: aggregate[name] for name in ("auc_tie_corrected", "surface_auc_tie_corrected", "deep_auc_tie_corrected",
                "surface_sd_mm_penalized", "surface_dle_mm_penalized", "deep_sd_mm_penalized", "deep_dle_mm_penalized")},
            "deep_TP": sum(int(row["deep_detected"]) for row in positives), "deep_positive_count": len(positives),
            "deep_FP": sum(int(row["deep_false_positive"]) for row in negatives), "surface_only_count": len(negatives),
        }
        setting_audits = []
        for row, diagnostic in records:
            windows = [window["solver"] for window in diagnostic["windows"] if window.get("solver") is not None]
            changes, violations = [], 0
            for window in windows:
                previous = window.get("initial_log_objective")
                for step in window.get("history", []):
                    current = float(step["log_objective"])
                    if previous is not None:
                        delta = current - previous
                        changes.append(delta)
                        slack = float(step.get("objective_descent_slack", 128 * np.finfo(float).eps * max(1., abs(previous), abs(current))))
                        violations += delta > slack
                    previous = current
            audit = {
                "setting_id": setting_id, "case_id": row["case_id"], "scenario": row["scenario"],
                "solver_windows": len(windows), "converged": bool(windows) and all(window["converged"] for window in windows),
                "inner_converged": bool(windows) and all(window.get("inner_converged", False) for window in windows),
                "outer_converged": bool(windows) and all(window.get("outer_converged", False) for window in windows),
                "objective_steps_checked": len(changes), "objective_monotonicity_violations": int(violations),
                "objective_delta_min": min(changes) if changes else np.nan,
                "objective_delta_max": max(changes) if changes else np.nan,
            }
            setting_audits.append(audit)
            paired_rows.append({"setting_id": setting_id, "experiment": directory.name,
                                "parameters_json": summary["parameters_json"], **row})
        audits.extend(setting_audits)
        for field in ("converged", "inner_converged", "outer_converged", "solver_windows", "objective_steps_checked", "objective_monotonicity_violations"):
            summary[field + "_count"] = sum(int(audit[field]) for audit in setting_audits)
        finite_min = [audit["objective_delta_min"] for audit in setting_audits if np.isfinite(audit["objective_delta_min"])]
        finite_max = [audit["objective_delta_max"] for audit in setting_audits if np.isfinite(audit["objective_delta_max"])]
        summary["objective_delta_min"] = min(finite_min) if finite_min else np.nan
        summary["objective_delta_max"] = max(finite_max) if finite_max else np.nan
        summaries.append(summary)


# %% 输出完整表、逐病例配对表、数值审计与跳过清单；重新运行可更新新增的完成实验。
output.mkdir(parents=True, exist_ok=True)
notes = {
    "purpose": "paired development ablation; not independent validation; no p-values or confidence intervals",
    "included_settings": len(summaries), "paired_case_ids": [] if reference is None else list(reference[2]),
    "skipped": skipped, "miss_penalty_mm": penalty_mm,
    "penalty_rule": "norm(ptp(shared.vertices, axis=0)) * 1000; archive._aggregate applies missing/deep-nondetection penalties",
    "objective_audit": "Accepted log-objective deltas only: IRLS starts from recorded initial objective; ADMM starts at first recorded MM objective. Increases exceeding recorded IRLS slack or 128*eps*max(1,abs(adjacent values)) count as violations. Different solver objectives are never compared against each other.",
}
(output / "summary.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
if not summaries:
    raise SystemExit("No comparable completed ablations. See summary.json skipped list.")
for filename, table in (("setting_comparison.csv", summaries), ("paired_cases.csv", paired_rows), ("solver_audit_by_case.csv", audits)):
    simulation.archive._atomic_csv(output / filename, table, table[0].keys())
assert all(summary["case_count"] == len(reference[2]) for summary in summaries)
assert len({(row["setting_id"], row["case_id"]) for row in paired_rows}) == len(paired_rows)


# %% 清晰的描述性 2×2 比较：点是病例，菱形是均值；没有虚假的独立样本误差棒。
style._style()
plt = style.plt
figure, axes = plt.subplots(2, 2, figsize=(16, max(8.5, 0.68 * len(summaries))), sharey=True)
y = np.arange(len(summaries))
colors = ["#4477AA" if "IRLS" in row["label"] else "#AA3377" for row in summaries]
for axis, metric, title in ((axes[0, 0], "deep_auc_tie_corrected", "Deep An_auc"),
                            (axes[0, 1], "surface_auc_tie_corrected", "Surface An_auc")):
    for index, summary in enumerate(summaries):
        values = [float(row[metric]) for row in paired_rows if row["setting_id"] == summary["setting_id"] and np.isfinite(float(row[metric]))]
        axis.scatter(values, index + np.linspace(-0.12, 0.12, len(values)), color=colors[index], alpha=.35, s=24)
    axis.scatter([row[metric] for row in summaries], y, c=colors, marker="D", s=45)
    axis.set_title(title + " (faint: cases; diamond: mean)")
    axis.set_xlim(-.03, 1.04)
for offset, count, total, marker, label in ((-.22, "deep_TP", "deep_positive_count", "o", "TP / deep-present"),
                                           (.22, "deep_FP", "surface_only_count", "s", "FP / surface-only")):
    values = [row[count] / row[total] for row in summaries]
    axes[1, 0].scatter(values, y + offset, c=colors, marker=marker, s=45)
    axes[1, 0].scatter([], [], color="#555555", marker=marker, label=label)
    for index, row in enumerate(summaries):
        axes[1, 0].text(values[index] + .035, index + offset, f"{row[count]}/{row[total]}", va="center", fontsize=8)
axes[1, 0].set_title("Deep detection and false positives")
axes[1, 0].set_xlim(-.03, 1.2)
axes[1, 0].legend(loc="lower center", bbox_to_anchor=(.5, -.2), ncol=2, frameon=False, fontsize=8)
values = [row["converged_count"] / row["case_count"] for row in summaries]
axes[1, 1].scatter(values, y, c=colors, marker="D", s=45)
for index, row in enumerate(summaries):
    axes[1, 1].text(values[index] + .035, index, f"{row['converged_count']}/{row['case_count']}", va="center", fontsize=8)
axes[1, 1].set_title("Reported convergence / all paired cases")
axes[1, 1].set_xlim(-.03, 1.2)
for axis in axes.flat:
    axis.set_yticks(y, [row["label"] for row in summaries])
    axis.set_ylim(len(summaries) - .5, -.5)
    axis.grid(axis="x", alpha=.2)
    axis.spines[["top", "right"]].set_visible(False)
figure.suptitle(f"Development ablation | {summaries[0]['configuration_count']} paired configurations | no validation claim", fontsize=15)
figure.text(.5, .025, "Descriptive only. IRLS and ADMM optimize different objectives; convergence flags refer to their own stopping criteria.", ha="center", fontsize=9)
figure.subplots_adjust(left=.28, right=.97, top=.90, bottom=.12, hspace=.35, wspace=.18)
figure.savefig(output / "development_ablation_comparison.png", dpi=180, facecolor="white")
plt.close(figure)
print("Included settings:", len(summaries), "; skipped experiments:", len(skipped), "; miss penalty mm:", penalty_mm)
print("Results:", output)
