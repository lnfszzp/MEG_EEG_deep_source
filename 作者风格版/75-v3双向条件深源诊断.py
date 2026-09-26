"""旧验证 case13/15 的已揭盲开发诊断：双向预测分数与条件深源独有证据。"""

# %% 1. 固定输入。旧 21 例已经揭盲，只能诊断，不能重新称为盲验证。
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_partial_confirmation import score_partial_deep_presence
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original

source = root / "results/erp_whole_head/adaptive_v6/formal_hierarchical_v2_validation"
manifest = root / "results/erp_whole_head/adaptive_v6/protocol/hierarchical_v1_validation_manifest.json"
consumed_marker = root / "results/erp_whole_head/adaptive_v6/protocol/hierarchical_v1_validation_manifest_consumed.json"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/v3_bidirectional_conditional_13_15"
if output.exists():
    raise FileExistsError(f"不覆盖旧诊断：{output}")
output.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))

source_metadata_path = source / "metadata.json"
source_completion_path = source / "completion.json"
source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
source_completion = json.loads(source_completion_path.read_text(encoding="utf-8"))
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
consumed_marker_sha256 = hashlib.sha256(consumed_marker.read_bytes()).hexdigest()
if source_metadata.get("stage") != "validation" or source_metadata.get("covariance") != "trial" or \
        source_metadata.get("manifest_sha256") != manifest_sha256 or \
        Path(source_metadata["manifest"]).resolve() != manifest.resolve() or \
        source_metadata.get("consumed_marker_sha256") != consumed_marker_sha256:
    raise ValueError("旧验证 metadata、manifest 或 consumed marker 绑定不一致")
if source_completion != {
        "stage": "validation", "complete": True, "passed": False,
        "case_count": 21, "failure_count": 0,
        "wall_seconds": source_completion.get("wall_seconds")}:
    raise ValueError("这里只诊断完整运行但算法验收失败的旧 21 例")

seed_roots = source_metadata.get("seed_roots")
expected_seed_roots = {"fit": 2026100101, "check": 2026100102}
expected_core_settings = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 80, "max_inner_retries": 20,
    "epsilon_fraction": .05, "tolerance": .001,
    "outer_tolerance": .01, "surface_reweight_floor": .5,
    "deep_reweight_floor": .5, "edge_weight_floor": .5,
}
if seed_roots != expected_seed_roots or source_metadata.get("solver_settings") != expected_core_settings:
    raise ValueError("旧病例的 seed 或 v2 outer80 bounded-MM 设置不一致")

solver_settings = {
    **expected_core_settings,
    "max_iter": 2000, "rho": 1., "adaptive_rho": True,
    "edge_fraction": .5, "noise_multiplier": 1., "calibration": "layer",
    "temporal_mode": "smooth", "ridge_fraction": 0.,
    "edge_penalty_mode": "group", "source_penalty_mode": "group",
    "deep_alias_penalty": False,
}
code_relatives = [
    "benchmark/erp_trial_covariance.py",
    "benchmark/metrics.py",
    "benchmark/protocol.py",
    "candidates/oaster_balanced.py",
    "candidates/oaster_partial_confirmation.py",
    "candidates/oaster_predictive.py",
    "run_erp_whole_head_matrix.py",
]
code_sha256 = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in code_relatives
}
for relative in set(code_relatives) - {"candidates/oaster_partial_confirmation.py"}:
    expected = source_metadata["code_sha256"].get(relative)
    if expected is None or code_sha256[relative] != expected:
        raise ValueError(f"v2 冻结依赖已变化：{relative}")
script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

all_cases = json.loads(manifest.read_text(encoding="utf-8"))
if len(all_cases) != 21 or [case["case_id"] for case in all_cases] != source_metadata["case_ids"]:
    raise ValueError("旧验证 manifest 的 21 例顺序与 metadata 不一致")
cases = [case for case in all_cases if int(case["case_number"]) in {13, 15}]
if [int(case["case_number"]) for case in cases] != [13, 15] or \
        [case["scenario"] for case in cases] != ["deep_plus_two_surface", "surface_only"]:
    raise ValueError("没有唯一找到旧验证 case13/15")
if any(case.get("replica_seed_roots") != expected_seed_roots for case in cases):
    raise ValueError("case13/15 的复制种子与冻结 seed 不一致")

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
shared_fingerprint = original._shared_fingerprint(shared)
if shared_fingerprint != source_metadata["shared_fingerprint"]:
    raise ValueError("共享 forward 指纹已变化")
n_surf = int(shared["n_surf"])
adjacency = shared["adjacency"]


# %% 2. 每一方向只在一个 20-trial ERP 拟合，在另一个独立 20-trial ERP 打分。
rows = []
source_case_sha256 = {}
started = time.perf_counter()
for case in cases:
    tick = time.perf_counter()
    name = case["case_id"]
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_roots["fit"])
    baseline = np.asarray(observation["baseline"], bool)
    active = np.asarray(observation["active_windows"][0], bool)
    gain = np.asarray(observation["gain"], float)
    primary_weights = np.ones(gain.shape[0], float)
    modality_sizes = tuple(int(value) for value in observation["metadata"]["retained_channels"])
    if len(modality_sizes) != 2 or sum(modality_sizes) != gain.shape[0] or \
            observation["metadata"].get("replica_seed_roots") != expected_seed_roots:
        raise RuntimeError(f"{name}: EEG/MEG 通道块或复制种子不一致")

    direction_details = []
    for direction, fit_key, check_key in (
            ("A_to_B", "training", "confirmation"),
            ("B_to_A", "confirmation", "training")):
        null, full, fitting = fit_predictive_models(
            observation[fit_key], gain, n_surf,
            adjacency=adjacency, baseline=baseline, active=active,
            channel_weights=primary_weights, solver_settings=solver_settings)
        solver_checks = {}
        for family in ("null", "full"):
            solver = fitting[family + "_model"]["windows"][0]["solver"]
            strict = bool(
                solver["converged"] and solver["outer_converged"]
                and solver["final_inner_converged"]
                and solver["final_stationarity_gap_relative"]
                <= solver_settings["tolerance"] + 1e-12)
            solver_checks[family] = {
                "strict_converged": strict,
                "converged": bool(solver["converged"]),
                "outer_converged": bool(solver["outer_converged"]),
                "final_inner_converged": bool(solver["final_inner_converged"]),
                "final_stationarity_gap_relative": float(
                    solver["final_stationarity_gap_relative"]),
            }
        if not all(item["strict_converged"] for item in solver_checks.values()):
            raise RuntimeError(f"{name} {direction}: H0/H1 未全部严格收敛")

        _, predictive = score_predictive_models(
            observation[check_key], gain, null, full,
            baseline=baseline, active=active, channel_weights=primary_weights,
            modality_sizes=modality_sizes)
        g = np.asarray(predictive["modality_noise_scores"], float)
        q = np.asarray(predictive["modality_response_excess_ratios"], float)
        if g.shape != (2,) or q.shape != (2,) or not np.isfinite(g).all() or \
                not np.isfinite(q).all() or np.any(q < 0):
            raise RuntimeError(f"{name} {direction}: EEG/MEG 的 g 或 q 非法")
        h = g / np.sqrt(1. + q)

        partial_score, partial = score_partial_deep_presence(
            observation[check_key], gain, null, full, n_surf,
            baseline=baseline, active=active, channel_weights=primary_weights)
        if not np.isfinite(partial_score):
            raise RuntimeError(f"{name} {direction}: conditional partial score 非有限")
        fraction = partial["conditional_gain_fraction"]
        if fraction is not None and (not np.isfinite(fraction) or not 0 <= fraction <= 1 + 1e-12):
            raise RuntimeError(f"{name} {direction}: conditional gain fraction 非法")
        direction_details.append({
            "direction": direction, "fit_half": fit_key, "held_out_half": check_key,
            "modality_order": ["EEG", "MEG"],
            "g": g.tolist(), "q": q.tolist(), "h": h.tolist(),
            "equal_modality_mean_h": float(h.mean()),
            "solver_checks": solver_checks,
            "partial_score": float(partial_score),
            "partial_evidence": partial,
        })

    h_matrix = np.asarray([item["h"] for item in direction_details], float)
    fold_scores = h_matrix.mean(axis=1)
    s_cf = float(h_matrix.mean())
    partial_scores = np.asarray([item["partial_score"] for item in direction_details], float)
    selected_deep = [item["partial_evidence"]["selected_deep_index"]
                     for item in direction_details]
    conditional_fractions = [item["partial_evidence"]["conditional_gain_fraction"]
                             for item in direction_details]
    selected_deep_consistent = bool(
        selected_deep[0] is not None and selected_deep[0] == selected_deep[1])
    partial_sign_consistent = bool(np.sign(partial_scores[0]) == np.sign(partial_scores[1]))
    predictive_sign_consistent = bool(np.sign(fold_scores[0]) == np.sign(fold_scores[1]))
    all_four_fits_converged = bool(all(
        item["solver_checks"][family]["strict_converged"]
        for item in direction_details for family in ("null", "full")))
    if len(direction_details) != 2 or not all_four_fits_converged:
        raise RuntimeError(f"{name}: 双向四个 H0/H1 拟合没有全部严格收敛")

    # 旧结果和真值只在新分数完全固定后读取，用于已揭盲的失败解释，不参与拟合或判决。
    old_case_path = source / (name + ".json")
    old_case_sha256 = hashlib.sha256(old_case_path.read_bytes()).hexdigest()
    source_case_sha256[name] = old_case_sha256
    old_case = json.loads(old_case_path.read_text(encoding="utf-8"))
    if old_case.get("case_id") != name or old_case.get("complete") is not True:
        raise ValueError(f"{name}: 旧验证病例记录不完整")

    row = {
        "case_id": name, "case_number": int(case["case_number"]),
        "scenario_posthoc": case["scenario"],
        "eeg_snr_db_posthoc": float(case["eeg_snr_db"]),
        "meg_snr_db_posthoc": float(case["meg_snr_db"]),
        "known_deep_index_posthoc": "" if case.get("deep_index") is None else int(case["deep_index"]),
        "prior_v2_gate_score_posthoc": float(old_case["gate_score"]),
        "prior_v2_deep_decision_posthoc": int(old_case["deep_present_decision"]),
        "prior_v2_selected_deep_posthoc": "" if old_case.get("selected_deep_index") is None
            else int(old_case["selected_deep_index"]),
        "symmetric_crossfit_score": s_cf,
        "a_to_b_equal_modality_mean_h": float(fold_scores[0]),
        "b_to_a_equal_modality_mean_h": float(fold_scores[1]),
        "a_to_b_eeg_g": float(direction_details[0]["g"][0]),
        "a_to_b_meg_g": float(direction_details[0]["g"][1]),
        "b_to_a_eeg_g": float(direction_details[1]["g"][0]),
        "b_to_a_meg_g": float(direction_details[1]["g"][1]),
        "a_to_b_eeg_q": float(direction_details[0]["q"][0]),
        "a_to_b_meg_q": float(direction_details[0]["q"][1]),
        "b_to_a_eeg_q": float(direction_details[1]["q"][0]),
        "b_to_a_meg_q": float(direction_details[1]["q"][1]),
        "a_to_b_eeg_h": float(direction_details[0]["h"][0]),
        "a_to_b_meg_h": float(direction_details[0]["h"][1]),
        "b_to_a_eeg_h": float(direction_details[1]["h"][0]),
        "b_to_a_meg_h": float(direction_details[1]["h"][1]),
        "predictive_direction_sign_consistent": int(predictive_sign_consistent),
        "partial_a_to_b": float(partial_scores[0]),
        "partial_b_to_a": float(partial_scores[1]),
        "partial_symmetric_mean": float(partial_scores.mean()),
        "partial_direction_sign_consistent": int(partial_sign_consistent),
        "conditional_gain_fraction_a_to_b": "" if conditional_fractions[0] is None
            else float(conditional_fractions[0]),
        "conditional_gain_fraction_b_to_a": "" if conditional_fractions[1] is None
            else float(conditional_fractions[1]),
        "selected_deep_a_to_b": "" if selected_deep[0] is None else int(selected_deep[0]),
        "selected_deep_b_to_a": "" if selected_deep[1] is None else int(selected_deep[1]),
        "selected_deep_direction_consistent": int(selected_deep_consistent),
        "all_four_fits_strictly_converged": int(all_four_fits_converged),
        "elapsed_seconds": float(time.perf_counter() - tick),
    }
    rows.append(row)
    detail = {
        "case_id": name,
        "diagnostic_status": "old_validation_revealed_diagnosis_only",
        "truth_used_by_fit_or_score": False,
        "truth_and_old_result_read_only_after_new_scores_frozen": True,
        "decision_threshold": None, "family_decision_made": False,
        "score_formula": "mean over directions A->B,B->A and modalities EEG,MEG of g/sqrt(1+q)",
        "direction_details": direction_details,
        "symmetric_crossfit_score": s_cf,
        "partial_symmetric_mean": float(partial_scores.mean()),
        "selected_deep_direction_consistent": selected_deep_consistent,
        "partial_direction_sign_consistent": partial_sign_consistent,
        "predictive_direction_sign_consistent": predictive_sign_consistent,
        "all_four_fits_strictly_converged": all_four_fits_converged,
        "posthoc_context": {
            "scenario": case["scenario"],
            "eeg_snr_db": float(case["eeg_snr_db"]),
            "meg_snr_db": float(case["meg_snr_db"]),
            "known_deep_index": case.get("deep_index"),
            "prior_v2_gate_score": float(old_case["gate_score"]),
            "prior_v2_deep_decision": bool(old_case["deep_present_decision"]),
            "prior_v2_selected_deep": old_case.get("selected_deep_index"),
            "source_case_json_sha256": old_case_sha256,
        },
    }
    (staging / (name + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(f"{name}: S_cf={s_cf:.6g}, partial={partial_scores.tolist()}, "
          f"deep={selected_deep}, converged=4/4", flush=True)


# %% 3. 只报告排序和方向稳定性；没有阈值、判决、调参或验证结论。
with (staging / "rows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_number = {row["case_number"]: row for row in rows}
summary = {
    "complete": True,
    "phase": "development_diagnosis",
    "scope": "revealed old hierarchical-v1 validation cases 13 and 15 only",
    "old_21_cases_are_diagnostic_only": True,
    "blind_validation_claim": False,
    "threshold_tuned_or_applied": False,
    "decision_threshold": None,
    "family_decision_made": False,
    "case_count": 2,
    "all_eight_direction_family_fits_strictly_converged": bool(all(
        row["all_four_fits_strictly_converged"] for row in rows)),
    "case13_symmetric_crossfit_score": by_number[13]["symmetric_crossfit_score"],
    "case15_symmetric_crossfit_score": by_number[15]["symmetric_crossfit_score"],
    "crossfit_orders_case13_above_case15": bool(
        by_number[13]["symmetric_crossfit_score"] > by_number[15]["symmetric_crossfit_score"]),
    "case13_partial_symmetric_mean": by_number[13]["partial_symmetric_mean"],
    "case15_partial_symmetric_mean": by_number[15]["partial_symmetric_mean"],
    "partial_orders_case13_above_case15": bool(
        by_number[13]["partial_symmetric_mean"] > by_number[15]["partial_symmetric_mean"]),
    "interpretation_limit": (
        "Ordering in two revealed failures is diagnostic only. It cannot select a threshold, "
        "estimate error rates, or replace fresh development, calibration and one-time validation."),
    "wall_seconds": float(time.perf_counter() - started),
}
(staging / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")

table_lines = []
for row in rows:
    fractions = "/".join(str(value) for value in (
        row["conditional_gain_fraction_a_to_b"],
        row["conditional_gain_fraction_b_to_a"]))
    deep_pair = f"{row['selected_deep_a_to_b']}/{row['selected_deep_b_to_a']}"
    table_lines.append(
        f"| {row['case_number']} | {row['scenario_posthoc']} | "
        f"{row['symmetric_crossfit_score']:.6g} | "
        f"{row['a_to_b_equal_modality_mean_h']:.6g}/{row['b_to_a_equal_modality_mean_h']:.6g} | "
        f"{row['partial_a_to_b']:.6g}/{row['partial_b_to_a']:.6g} | "
        f"{fractions} | {deep_pair} | {row['selected_deep_direction_consistent']} |")
report = [
    "# v3 双向条件深源诊断：旧 case13/15",
    "",
    "这是旧 21 例正式验证揭盲后的失败诊断，不是新的盲验证。脚本只处理 case13/15，"
    "不读取或应用门限，不作家族判决，也不据这两个标签调参。",
    "",
    "## 固定计算",
    "",
    "A、B 分别是独立的 20-trial ERP。A→B 与 B→A 分别用 v2 outer80、"
    "全白化通道权重恒为 1 的同一 H0/H1 联合模型；四个模型均须严格收敛。"
    "每个方向和模态先算 `h=g/sqrt(1+q)`，最终 `S_cf` 是 EEG、MEG、"
    "A→B、B→A 四项的等权平均。没有固定 EEG 偏重、min veto 或 alias/VIF penalty。",
    "",
    "同一批训练折的 H0/H1 解再交给 `score_partial_deep_presence`；它只检查训练所选深点"
    "在训练 H0 皮层 nuisance 子空间之外的确认集增量。partial 分数仍未校准，不能解释成概率。",
    "",
    "## 结果",
    "",
    "| case | 已揭盲情形 | S_cf | A→B/B→A h均值 | partial A→B/B→A | conditional fraction | 深点 A→B/B→A | 深点一致 |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
    *table_lines,
    "",
    f"- `S_cf(case13) > S_cf(case15)`：`{summary['crossfit_orders_case13_above_case15']}`。",
    f"- `partial_mean(case13) > partial_mean(case15)`：`{summary['partial_orders_case13_above_case15']}`。",
    "- 两例排序只能回答‘对称复用/条件独有证据是否值得进入新的开发面板’，不能提供阈值或误报率。",
    "- 下一步若保留候选，必须使用新病例开发，随后冻结算法并建立全新的 H0 校准与一次性验证。",
]
(staging / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

result_metadata = {
    "schema_version": 1,
    "phase": "development_diagnosis",
    "protocol": "erp-v6-v3-bidirectional-conditional-revealed-13-15",
    "output": str(output),
    "source": str(source),
    "source_metadata_sha256": hashlib.sha256(source_metadata_path.read_bytes()).hexdigest(),
    "source_completion_sha256": hashlib.sha256(source_completion_path.read_bytes()).hexdigest(),
    "source_case_json_sha256": source_case_sha256,
    "manifest": str(manifest), "manifest_sha256": manifest_sha256,
    "consumed_marker": str(consumed_marker),
    "consumed_marker_sha256": consumed_marker_sha256,
    "seed_roots": seed_roots,
    "case_ids": [case["case_id"] for case in cases],
    "shared_fingerprint": shared_fingerprint,
    "solver_settings": solver_settings,
    "primary_channel_weights": "all whitened EEG/MEG channels exactly one",
    "score_formula": "mean_{direction,modality} g/sqrt(1+q)",
    "partial_score_role": "uncalibrated conditional unique-deep diagnostic only",
    "deep_alias_penalty": False,
    "threshold": None,
    "truth_access": "case selection and context only; never used by fitting, scoring or a decision",
    "old_21_cases_are_diagnostic_only": True,
    "blind_validation_claim": False,
    "code_sha256": code_sha256,
    "script_sha256": script_sha256,
}
(staging / "metadata.json").write_text(
    json.dumps(result_metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
(staging / "completion.json").write_text(
    json.dumps({
        "complete": True, "phase": "development_diagnosis", "case_count": 2,
        "all_eight_direction_family_fits_strictly_converged":
            summary["all_eight_direction_family_fits_strictly_converged"],
        "threshold_tuned_or_applied": False, "blind_validation_claim": False,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")

for relative, expected in code_sha256.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"运行中代码发生变化，结果不发布：{relative}")
if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha256:
    raise RuntimeError("运行中诊断脚本发生变化，结果不发布")
os.replace(staging, output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
