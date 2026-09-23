"""# %% 仅用开发病例 case01 的独立 confirmation 盲选三点正则化路径。"""

# %% 环境、路径和固定设置
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original

save_dir = Path(__file__).resolve().parent
manifest_path = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
reference_dir = root / "results/erp_whole_head/adaptive_v6/dev_trial_admm_mrf080_m10_m10"
case_id = "erp-v6-development-01-eeg-10-meg-10"
seed_root = 2026092206
multipliers = (1.0, 0.5, 0.25)
solver_fixed = {"solver_kind": "admm", "mrf_strength": 0.8}

cases = json.loads(manifest_path.read_text(encoding="utf-8"))
selected_cases = [case for case in cases if case["case_id"] == case_id]
assert len(selected_cases) == 1
case = selected_cases[0]
assert case["scenario"] == "surface_only" and case["deep_index"] is None
assert case["eeg_snr_db"] == -10 and case["meg_snr_db"] == -10

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
baseline = observation["baseline"]
active = observation["active_windows"][0]
weights = observation["channel_weights"]

# %% multiplier=1 复用同一输入、同一求解设置的已完成开发结果，避免重复计算。
reference_json = json.loads((reference_dir / f"{case_id}.json").read_text(encoding="utf-8"))
reference_rows = []
with (reference_dir / "rows.csv").open(newline="", encoding="utf-8-sig") as stream:
    for row in csv.DictReader(stream):
        if row["case_id"] == case_id and row["method"] in {"v6-surface-only", "v6-full-ungated"}:
            reference_rows.append(row)
assert len(reference_rows) == 2
reference_by_model = {row["method"]: row for row in reference_rows}
assert reference_json["fitting"]["solver_kind"] == "admm"
assert reference_json["fitting"]["structural_settings"]["mrf_strength"] == 0.8
assert reference_json["fitting"]["structural_settings"]["noise_multiplier"] == 1.0

candidate_rows = []
started = time.perf_counter()
for multiplier in multipliers:
    print(f"lambda multiplier {multiplier:g} starting", flush=True)
    if multiplier == 1.0:
        fitting = reference_json["fitting"]
        evidence = reference_json["evidence"]
        model_metrics = {}
        for model_name in ("v6-surface-only", "v6-full-ungated"):
            source = reference_by_model[model_name]
            model_metrics[model_name] = {
                key: (None if source[key].lower() == "nan" else float(source[key]))
                for key in ("auc", "auc_tie_corrected", "surface_dle_mm", "deep_dle_mm",
                            "deep_score", "deep_false_positive", "active_count",
                            "surface_active_count", "deep_active_count")
            }
    else:
        null, full, fitting = fit_predictive_models(
            observation["training"], observation["gain"], shared["n_surf"],
            adjacency=shared["adjacency"], baseline=baseline, active=active,
            channel_weights=weights,
            solver_settings={**solver_fixed, "noise_multiplier": multiplier})
        _, evidence = score_predictive_models(
            observation["confirmation"], observation["gain"], null, full,
            baseline=baseline, active=active, channel_weights=weights)
        model_metrics = {}
        for model_name, estimate in (("v6-surface-only", null), ("v6-full-ungated", full)):
            values = metrics.evaluate_estimate(
                estimate, observation["truth"], shared["vertices"], observation["groups"],
                shared["n_surf"], observation["active"], shared["auc_cortex"], baseline=baseline)
            model_metrics[model_name] = {
                key: (None if isinstance(values[key], float) and not np.isfinite(values[key])
                      else float(values[key]))
                for key in ("auc", "auc_tie_corrected", "surface_dle_mm", "deep_dle_mm",
                            "deep_score", "deep_false_positive", "active_count",
                            "surface_active_count", "deep_active_count")
            }

    for family, model_name, loss_name in (
            ("H0", "v6-surface-only", "null_loss"),
            ("H1", "v6-full-ungated", "full_loss")):
        solver = fitting[("null" if family == "H0" else "full") + "_model"]["windows"][0]["solver"]
        candidate_rows.append({
            "lambda_multiplier": multiplier,
            "family": family,
            "confirmation_loss": float(evidence[loss_name]),
            "converged": bool(solver["converged"]),
            "outer_converged": bool(solver["outer_converged"]),
            "inner_converged": bool(solver["inner_converged"]),
            "final_stationarity_gap_relative": float(solver["final_stationarity_gap_relative"]),
            **model_metrics[model_name],
        })
    (save_dir / "partial_results.json").write_text(json.dumps({
        "complete_multipliers": [value for value in multipliers
                                 if any(row["lambda_multiplier"] == value for row in candidate_rows)],
        "candidates": candidate_rows,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)

# %% 只看 confirmation loss 盲选；不收敛候选没有资格获选。
selected = {}
for family in ("H0", "H1"):
    eligible = [row for row in candidate_rows if row["family"] == family and row["converged"]]
    selected[family] = min(eligible, key=lambda row: row["confirmation_loss"]) if eligible else None
selected_family = min((family for family in ("H0", "H1") if selected[family] is not None),
                      key=lambda family: selected[family]["confirmation_loss"])
winner = selected[selected_family]

with (save_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=candidate_rows[0].keys())
    writer.writeheader()
    writer.writerows(candidate_rows)

summary = {
    "scope": "development case01 only; calibration and validation were not read",
    "case_id": case_id,
    "manifest": str(manifest_path),
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "seed_root": seed_root,
    "covariance": "trial-baseline contrast covariance from training only",
    "fixed_settings": solver_fixed,
    "candidate_lambda_multipliers": list(multipliers),
    "selection_rule": "within each family, minimum independent-confirmation prediction loss among converged candidates; then minimum loss between selected H0/H1; no truth used",
    "selected": selected,
    "selected_family": selected_family,
    "selected_result": winner,
    "deep_false_positive": bool(winner["deep_false_positive"]),
    "all_candidates": candidate_rows,
    "wall_seconds": time.perf_counter() - started,
    "reference_multiplier_1": str(reference_dir),
}
(save_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

lines = [
    "# case01 三点正则化路径开发核查",
    "",
    "仅使用 `erp-v6-development-01-eeg-10-meg-10`。候选为当前数据驱动 lambda 的 "
    "`1 / 0.5 / 0.25` 倍，MRF 固定为 0.8。拟合只看 train-20；候选选择只看独立 "
    "confirmation-20 的预测损失。没有读取 calibration 或 validation。",
    "",
    "| lambda倍数 | 模型 | 收敛 | confirmation loss | An_cal AUC | An_auc | 表层DLE(mm) | 深源分数 | 深源误报 |",
    "|---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|",
]
for row in candidate_rows:
    dle = "NA" if row["surface_dle_mm"] is None else f"{row['surface_dle_mm']:.3f}"
    lines.append(f"| {row['lambda_multiplier']:.2f} | {row['family']} | {row['converged']} | "
                 f"{row['confirmation_loss']:.6f} | {row['auc']:.6f} | "
                 f"{row['auc_tie_corrected']:.6f} | {dle} | {row['deep_score']:.6f} | "
                 f"{bool(row['deep_false_positive'])} |")
lines += [
    "",
    f"盲选结果：H0 选择 lambda×{selected['H0']['lambda_multiplier'] if selected['H0'] else '无'}；"
    f"H1 选择 lambda×{selected['H1']['lambda_multiplier'] if selected['H1'] else '无'}；"
    f"两模型按 confirmation loss 最终选择 **{selected_family}**。",
    "",
    f"最终 An_cal AUC={winner['auc']:.6f}，tie-corrected An_auc={winner['auc_tie_corrected']:.6f}，"
    f"表层 DLE={'NA' if winner['surface_dle_mm'] is None else f'{winner['surface_dle_mm']:.3f} mm'}，"
    f"深源误报={bool(winner['deep_false_positive'])}。",
    "",
    "这是一例开发诊断，不能据此冻结正则化参数或宣称泛化改善。",
]
(save_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(save_dir / "partial_results.json").unlink()
print(json.dumps({"selected_family": selected_family, "selected_result": winner}, ensure_ascii=False, indent=2), flush=True)
