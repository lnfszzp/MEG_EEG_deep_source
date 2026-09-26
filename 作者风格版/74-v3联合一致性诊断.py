"""# %% v3 联合图一致性诊断：旧21例只作已揭盲排障，不作验证或调阈值。"""

# %% 1. 固定旧决定、输入哈希和作者风格。
import argparse
import ast
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
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import reconstruct_evoked_oaster_v5_from_whitened
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path,
    default=root / "results/erp_whole_head/adaptive_v6/development_diagnosis/"
                   "hierarchical_v3_joint_consistency")
parser.add_argument("--preflight", action="store_true",
                    help="只核验输入、哈希和作者风格，不创建结果目录")
args = parser.parse_args()
output = args.output if args.output.is_absolute() else root / args.output
output = output.resolve()
case_numbers = (6, 13, 14, 15, 17)
localization_target_cases = (6, 14, 17)
known_family_error_cases = (13, 15)

script_path = Path(__file__).resolve()
script_text = script_path.read_text(encoding="utf-8")
script_tree = ast.parse(script_text)
if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
       for node in ast.walk(script_tree)):
    raise RuntimeError("作者风格自检失败：本脚本不应定义 def/class")
if script_text.count("# %%") < 5:
    raise RuntimeError("作者风格自检失败：缺少顺序式 # %% 分段")
if output.exists() and not args.preflight:
    raise FileExistsError(f"不覆盖旧结果：{output}")

validation_dir = root / "results/erp_whole_head/adaptive_v6/formal_hierarchical_v2_validation"
source_metadata_path = validation_dir / "metadata.json"
source_rows_path = validation_dir / "rows.csv"
source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
manifest = Path(source_metadata["manifest"])
manifest = manifest if manifest.is_absolute() else root / manifest
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
if source_metadata.get("stage") != "validation" or \
        manifest_sha256 != source_metadata.get("manifest_sha256"):
    raise ValueError("旧 v2 validation 的 metadata/manifest 绑定已变化")
seed_roots = source_metadata.get("seed_roots")
if not isinstance(seed_roots, dict) or set(seed_roots) != {"fit", "check"} or \
        seed_roots["check"] != seed_roots["fit"] + 1:
    raise ValueError("旧面板没有完整绑定 fit/check seed roots")

expected_solver_settings = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 80, "max_inner_retries": 20,
    "epsilon_fraction": .05, "tolerance": .001,
    "outer_tolerance": .01, "surface_reweight_floor": .5,
    "deep_reweight_floor": .5, "edge_weight_floor": .5,
}
if source_metadata.get("solver_settings") != expected_solver_settings:
    raise ValueError("旧 v2 validation 的数值求解设置已变化")

all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for number in case_numbers for case in all_cases
         if case.get("case_number") == number]
if [case.get("case_number") for case in cases] != list(case_numbers) or \
        len({case.get("case_id") for case in cases}) != len(case_numbers):
    raise ValueError("诊断病例必须且只能是旧21例中的 06/13/14/15/17")
if any(case.get("replica_seed_roots") != seed_roots or
       case.get("seed", [None])[0] != seed_roots["fit"] for case in cases):
    raise ValueError("诊断病例没有使用 metadata 冻结的 seed roots")

with source_rows_path.open("r", encoding="utf-8-sig", newline="") as stream:
    old_rows_all = list(csv.DictReader(stream))
old_rows = {int(row["case_number"]): row for row in old_rows_all
            if int(row["case_number"]) in case_numbers}
if set(old_rows) != set(case_numbers):
    raise ValueError("旧 v2 rows.csv 缺少目标失败病例")
for case_number, row in old_rows.items():
    gate_score = float(row["gate_score"])
    gate_threshold = float(row["gate_threshold"])
    if int(row["deep_present_decision"]) != int(gate_score > gate_threshold):
        raise ValueError(f"case {case_number:02d} 的旧冻结决定不可复算")

alias_rejections = {
    "results/erp_whole_head/adaptive_v6/dev_localization_v5_alias_mm_probe/EARLY_STOPPED.md":
        "dd470165819a7517c9e72e2dac7ee446d67f523230bdfbf392f8b83de8be2fb0",
    "results/erp_whole_head/adaptive_v6/dev_localization_v5_alias_mm_rho_fixed_probe/EARLY_STOPPED.md":
        "f209ec7d4e479f644e032cd0bf2da9f8a64a95ea9cbc6b81d594b9223614d308",
}
for relative, expected in alias_rejections.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise ValueError(f"已淘汰 alias 探针记录发生变化：{relative}")

code_files = (
    "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py",
    "benchmark/metrics.py",
    "benchmark/protocol.py",
    "candidates/graph_reweight_solver.py",
    "candidates/oaster_balanced.py",
    "candidates/oaster_rebuilt.py",
    "run_erp_whole_head_matrix.py",
)
code_sha256 = {}
for relative in code_files:
    path = root / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    code_sha256[relative] = digest
    old_digest = source_metadata.get("code_sha256", {}).get(relative)
    if old_digest is not None and digest != old_digest:
        raise ValueError(f"旧 v2 后核心代码已变化，不能作同链路诊断：{relative}")
code_sha256[str(script_path.relative_to(root)).replace("\\", "/")] = \
    hashlib.sha256(script_path.read_bytes()).hexdigest()

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
if original._shared_fingerprint(shared) != source_metadata.get("shared_fingerprint"):
    raise ValueError("共享 forward 与旧 v2 validation 不一致")
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"], float)
adjacency = shared["adjacency"]

if args.preflight:
    print(json.dumps({
        "preflight": "PASS", "development_diagnosis_only": True,
        "formal_or_blind_validation": False, "threshold_tuning": False,
        "inverse_calls_planned": len(cases), "case_numbers": list(case_numbers),
        "manifest_sha256": manifest_sha256, "seed_roots": seed_roots,
        "shared_fingerprint": source_metadata["shared_fingerprint"],
        "frozen_decisions": {str(number): int(old_rows[number]["deep_present_decision"])
                             for number in case_numbers},
        "alias_penalty_status": "archived_rejected_not_rerun",
        "script_has_def_or_class": False, "output_exists": output.exists(),
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


# %% 2. 新目录只在全部完成后发布；每例严格只有一次所选联合图反演。
output.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
started = time.perf_counter()
joint_settings = {
    **expected_solver_settings,
    "max_iter": 2000, "rho": 1., "adaptive_rho": True,
    "noise_multiplier": 1., "edge_fraction": .5, "calibration": "layer",
    "temporal_mode": "smooth", "ridge_fraction": 0.,
    "edge_penalty_mode": "group", "source_penalty_mode": "group",
    "deep_alias_penalty": False,
}
metadata = {
    "schema_version": 1,
    "protocol": "erp-v6-hierarchical-v3-joint-map-consistency-development-diagnosis",
    "development_diagnosis_only": True,
    "formal_acceptance_allowed": False,
    "blind_validation_claim_allowed": False,
    "old_21_cases_already_unblinded": True,
    "threshold_fitting_performed": False,
    "gate_recomputed": False,
    "family_decision_source": "copied verbatim from frozen v2 rows.csv before observation reconstruction",
    "localization_change_only": "replace v2 MRF-innovation/sequential residual final map by the same adaptive graph objective for H0 or H1",
    "map_semantics": "combined40 selected-family joint map; H0 surface-only, H1 surface+deep",
    "channel_weights": "all whitened channels exactly one, matching the v2 primary H0/H1 objective",
    "localization_target_cases": list(localization_target_cases),
    "known_family_error_cases_not_fixed_here": list(known_family_error_cases),
    "inverse_call_budget": len(cases),
    "alias_penalty_status": "archived_rejected_not_rerun",
    "alias_rejection_records_sha256": alias_rejections,
    "manifest": str(manifest), "manifest_sha256": manifest_sha256,
    "source_metadata": str(source_metadata_path),
    "source_metadata_sha256": hashlib.sha256(source_metadata_path.read_bytes()).hexdigest(),
    "source_rows": str(source_rows_path),
    "source_rows_sha256": hashlib.sha256(source_rows_path.read_bytes()).hexdigest(),
    "seed_roots": seed_roots,
    "shared_fingerprint": source_metadata["shared_fingerprint"],
    "code_sha256": code_sha256,
    "solver_settings": joint_settings,
    "publication_mode": "atomic directory promotion from a same-volume staging directory",
}
rows = []
details = []
inverse_call_count = 0


# %% 3. 先复制旧盲决定，再合并40试次；真值只在唯一候选图固定后读取。
for case in cases:
    tick = time.perf_counter()
    case_number = int(case["case_number"])
    name = case["case_id"]
    old = old_rows[case_number]
    deep_present = bool(int(old["deep_present_decision"]))
    frozen_family = "H1" if deep_present else "H0"

    observation = prepare_trial_covariance_case(
        shared, case, seed_root=seed_roots["fit"])
    baseline = np.asarray(observation["baseline"], bool)
    active_window = np.asarray(observation["active_windows"][0], bool)
    active_samples = np.asarray(observation["active"], int)
    gain = np.asarray(observation["gain"], float)
    combined = (observation["training"] + observation["confirmation"]) / 2
    primary_weights = np.ones(gain.shape[0])

    branch_gain = gain if deep_present else gain[:, :n_surf]
    branch_adjacency = adjacency if deep_present else adjacency[:n_surf, :n_surf]
    branch_estimate, fitting = reconstruct_evoked_oaster_v5_from_whitened(
        combined, branch_gain, n_surf, kernels=(), adjacency=branch_adjacency,
        baseline=baseline, active_windows=(active_window,),
        window_channel_weights=(primary_weights,), require_one=False,
        **joint_settings)
    inverse_call_count += 1
    solver = fitting["windows"][0]["solver"]
    solver_ok = bool(
        solver["converged"] and solver["outer_converged"]
        and solver["final_inner_converged"]
        and solver["final_stationarity_gap_relative"]
        <= expected_solver_settings["tolerance"] + 1e-12)
    if not solver_ok:
        raise RuntimeError(f"{name}: selected {frozen_family} 联合图没有到达固定点")
    selected = np.zeros((gain.shape[1], combined.shape[1]))
    selected[:branch_estimate.shape[0]] = branch_estimate
    if selected.shape != (gain.shape[1], combined.shape[1]) or \
            not np.isfinite(selected).all():
        raise RuntimeError(f"{name}: 联合图的形状或有限性错误")
    if not deep_present and np.any(selected[n_surf:] != 0):
        raise RuntimeError(f"{name}: H0 联合图意外包含深源")

    # 唯一图到这里已经冻结；下面才读取真值并计算事后指标。
    truth = np.asarray(observation["truth"], float)
    values = metrics.evaluate_estimate(
        selected, truth, vertices, observation["groups"], n_surf,
        active_samples, shared["auc_cortex"], baseline=baseline)
    surface_groups = [np.asarray(group, int) for group in observation["groups"]
                      if np.asarray(group).size and np.all(np.asarray(group) < n_surf)]
    surface_component_dle = []
    surface_patch_hits = []
    if surface_groups:
        surface_energy = np.sum(selected[:n_surf, active_samples] ** 2, axis=1)
        truth_indices = np.concatenate(surface_groups)
        truth_owner = np.concatenate([
            np.full(len(group), index, int)
            for index, group in enumerate(surface_groups)])
        owner = truth_owner[cKDTree(vertices[truth_indices]).query(vertices[:n_surf])[1]]
        support = surface_energy > .1 * surface_energy.max(initial=0.)
        for index, group in enumerate(surface_groups):
            region = np.flatnonzero(owner == index)
            supported_region = region[support[region]]
            if supported_region.size:
                peak = int(supported_region[np.argmax(surface_energy[supported_region])])
                center = vertices[group].mean(axis=0)
                surface_component_dle.append(float(
                    np.linalg.norm(vertices[peak] - center) * 1000))
            else:
                surface_component_dle.append(None)
            surface_patch_hits.append(int(np.count_nonzero(support[group])))
    finite_components = [value for value in surface_component_dle if value is not None]
    component_max = (np.nan if len(finite_components) != len(surface_groups)
                     else max(finite_components, default=np.nan))

    old_global_auc = float(old["auc_tie_corrected"])
    old_surface_auc = float(old["surface_auc_tie_corrected"])
    old_surface_dle = float(old["surface_dle_mm"])
    old_component_dle = float(old["surface_component_dle_max_mm"])
    old_surface_sd = float(old["surface_sd_mm"])
    old_deep_distance = float(old["deep_peak_distance_mm"])
    row = {
        "case_number": case_number, "case_id": name,
        "scenario_posthoc": case["scenario"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "frozen_gate_score": float(old["gate_score"]),
        "frozen_gate_threshold": float(old["gate_threshold"]),
        "frozen_deep_present_decision": int(deep_present),
        "selected_family": frozen_family,
        "localization_target": int(case_number in localization_target_cases),
        "known_family_error_not_fixed": int(case_number in known_family_error_cases),
        "solver_converged": int(solver_ok),
        "truth_used_only_for_posthoc_metrics": 1,
        "old_global_An_auc": old_global_auc,
        "new_global_An_auc": values["auc_tie_corrected"],
        "global_An_auc_change": values["auc_tie_corrected"] - old_global_auc,
        "old_surface_An_auc": old_surface_auc,
        "new_surface_An_auc": values["surface_auc_tie_corrected"],
        "surface_An_auc_change": (
            values["surface_auc_tie_corrected"] - old_surface_auc
            if np.isfinite(values["surface_auc_tie_corrected"]) and np.isfinite(old_surface_auc)
            else np.nan),
        "old_surface_DLE_mm": old_surface_dle,
        "new_surface_DLE_mm": values["surface_dle_mm"],
        "old_surface_component_DLE_mm": old_component_dle,
        "new_surface_component_DLE_mm": component_max,
        "old_surface_SD_mm": old_surface_sd,
        "new_surface_SD_mm": values["surface_sd_mm"],
        "old_deep_peak_distance_mm": old_deep_distance,
        "new_deep_peak_distance_mm": values["deep_peak_distance_mm"],
        "old_deep_false_positive": int(old["deep_false_positive"]),
        "new_deep_false_positive": values["deep_false_positive"],
        "old_surface_patch_hit_count": int(old["surface_patch_hit_count"]),
        "new_surface_patch_hit_count": int(sum(value > 0 for value in surface_patch_hits)),
        "surface_patch_count": len(surface_patch_hits),
        "new_surface_active_count": values["surface_active_count"],
        "new_deep_active_count": values["deep_active_count"],
    }
    rows.append(row)

    map_path = staging / f"case_{case_number:02d}_combined40_joint.npz"
    np.savez_compressed(
        map_path, truth=truth.astype(np.float32), selected=selected.astype(np.float32),
        vertices=vertices.astype(np.float32), times=np.asarray(shared["times"]),
        active=active_samples, baseline=baseline)
    clean_metrics = {
        key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value)
              else value.item() if isinstance(value, np.generic) else value)
        for key, value in values.items()}
    details.append({
        "case_number": case_number, "case_id": name,
        "frozen_decision_loaded_before_observation": True,
        "truth_used_by_decision_or_localization": False,
        "truth_used_only_for_posthoc_metrics": True,
        "frozen_gate_score": float(old["gate_score"]),
        "frozen_gate_threshold": float(old["gate_threshold"]),
        "frozen_deep_present_decision": deep_present,
        "selected_family": frozen_family,
        "combined_trial_count": 40,
        "joint_objective": fitting["mode"],
        "deep_alias_penalty": False,
        "solver_converged": solver_ok,
        "solver_outer_iterations": len(solver["history"]),
        "solver_final_stationarity_gap_relative": solver[
            "final_stationarity_gap_relative"],
        "posthoc_metrics": clean_metrics,
        "surface_component_dle_mm": surface_component_dle,
        "surface_patch_support_hits": surface_patch_hits,
        "map_file": map_path.name,
        "map_file_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - tick,
    })
    print(name, frozen_family,
          f"old/new AUC={old_global_auc:.3f}/{values['auc_tie_corrected']:.3f}",
          f"old/new surface DLE={old_surface_dle}/{values['surface_dle_mm']}", flush=True)


# %% 4. 只汇总定位改动；13/15 的 family 错误明确留给下一轮门控开发。
if inverse_call_count != len(case_numbers) or len(rows) != len(case_numbers):
    raise RuntimeError("反演次数必须严格等于5，每例只能一次")
row_by_case = {row["case_number"]: row for row in rows}
diagnostic_checks = {
    "case06_surface_recovered": bool(
        np.isfinite(row_by_case[6]["new_surface_DLE_mm"])
        and row_by_case[6]["new_surface_patch_hit_count"]
        == row_by_case[6]["surface_patch_count"]),
    "case14_surface_DLE_improved": bool(
        np.isfinite(row_by_case[14]["new_surface_DLE_mm"])
        and row_by_case[14]["new_surface_DLE_mm"]
        < row_by_case[14]["old_surface_DLE_mm"]),
    "case17_all_surface_patches_recovered": bool(
        row_by_case[17]["new_surface_patch_hit_count"]
        == row_by_case[17]["surface_patch_count"]
        and np.isfinite(row_by_case[17]["new_surface_component_DLE_mm"])),
    "case13_family_error_intentionally_unchanged": bool(
        row_by_case[13]["frozen_deep_present_decision"] == 0),
    "case15_family_error_intentionally_unchanged": bool(
        row_by_case[15]["frozen_deep_present_decision"] == 1),
}
summary = {
    "complete": True, "development_diagnosis_only": True,
    "formal_or_blind_validation": False, "threshold_tuning_performed": False,
    "gate_recomputed": False, "case_count": len(rows),
    "inverse_call_count": inverse_call_count,
    "all_selected_joint_solvers_converged": bool(
        all(row["solver_converged"] for row in rows)),
    "localization_target_cases": list(localization_target_cases),
    "known_family_error_cases_not_fixed_here": list(known_family_error_cases),
    "diagnostic_checks": diagnostic_checks,
    "localization_target_checks_passed": bool(all(
        diagnostic_checks[name] for name in (
            "case06_surface_recovered", "case14_surface_DLE_improved",
            "case17_all_surface_patches_recovered"))),
    "wall_seconds": time.perf_counter() - started,
    "next_required_step": "develop a separate conditional-residual family gate on new data; then fresh calibration and untouched validation",
}
if not summary["all_selected_joint_solvers_converged"]:
    raise RuntimeError("存在未收敛的 selected joint map")


# %% 5. 临时目录写完并自检后一次性发布，绝不覆盖已有结果。
(staging / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
(staging / "details.json").write_text(
    json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
(staging / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
with (staging / "rows.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

report_lines = [
    "# v3 联合图一致性诊断（旧21例已揭盲排障）", "",
    "没有重跑或调整门控：直接复制 v2 已冻结的 H0/H1 决定，再将40试次合并后只运行一次同一 adaptive-graph 目标。H0 是纯表层联合图，H1 是表层+深层联合图。", "",
    "|病例|诊断范围|SNR EEG/MEG|冻结family|旧/新 An_auc|旧/新 surface AUC|旧/新 surface DLE|旧/新 component DLE|旧/新 SD|旧/新 deep mm|旧/新 deep FP|旧/新 patch hit|",
    "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    scope = "定位" if row["localization_target"] else "family错误保留"
    pairs = []
    for old_key, new_key in (
            ("old_surface_An_auc", "new_surface_An_auc"),
            ("old_surface_DLE_mm", "new_surface_DLE_mm"),
            ("old_surface_component_DLE_mm", "new_surface_component_DLE_mm"),
            ("old_surface_SD_mm", "new_surface_SD_mm"),
            ("old_deep_peak_distance_mm", "new_deep_peak_distance_mm")):
        old_value, new_value = row[old_key], row[new_key]
        pairs.append(
            ("—" if not np.isfinite(old_value) else f"{old_value:.3f}") + "/" +
            ("—" if not np.isfinite(new_value) else f"{new_value:.3f}"))
    report_lines.append(
        f"|{row['case_number']:02d}|{scope}|{row['eeg_snr_db']:+d}/{row['meg_snr_db']:+d}|"
        f"{row['selected_family']}|{row['old_global_An_auc']:.3f}/{row['new_global_An_auc']:.3f}|"
        f"{pairs[0]}|{pairs[1]}|{pairs[2]}|{pairs[3]}|{pairs[4]}|"
        f"{row['old_deep_false_positive']}/{row['new_deep_false_positive']}|"
        f"{row['old_surface_patch_hit_count']}/{row['new_surface_patch_hit_count']} of {row['surface_patch_count']}|")
report_lines += [
    "", "## 诊断检查", "",
    *[f"- {name}: `{value}`" for name, value in diagnostic_checks.items()],
    "", "## 边界", "",
    "- 旧21例已经揭盲；本结果只能回答‘统一联合图是否修复最终定位图’，不能作为性能验证。",
    "- case 13/15 的旧 family 决定原样保留，因此这里不尝试用定位器掩盖门控错误。",
    "- VIF alias 惩罚已有两次提前终止记录：一次数值不收敛，一次把低SNR真深源压为零；本轮没有重复运行。",
    "- 下一步必须另用新开发数据研究 conditional-residual gate，再重新独立校准和一次性验证。", "",
]
(staging / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
(staging / "completion.json").write_text(
    json.dumps({
        "complete": True, "development_diagnosis_only": True,
        "case_count": len(rows), "inverse_call_count": inverse_call_count,
        "all_selected_joint_solvers_converged": summary[
            "all_selected_joint_solvers_converged"],
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

reloaded_summary = json.loads((staging / "summary.json").read_text(encoding="utf-8"))
with (staging / "rows.csv").open("r", encoding="utf-8-sig", newline="") as stream:
    reloaded_rows = list(csv.DictReader(stream))
if reloaded_summary.get("complete") is not True or len(reloaded_rows) != 5 or \
        any(not (staging / detail["map_file"]).is_file() for detail in details):
    raise RuntimeError("发布前自检失败")
if output.exists():
    raise FileExistsError(f"运行期间目标目录被创建，停止发布：{output}")
os.replace(staging, output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
