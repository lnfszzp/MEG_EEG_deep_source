"""# %% v3 图扩散强度诊断：旧21例已揭盲，只比较 MRF 0.8/0.5/0 的机制差异。"""

# %% 1. 只绑定旧结果、病例、家族决定和唯一改动参数。
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
                   "v3_mrf_strength_05_00")
parser.add_argument("--preflight", action="store_true",
                    help="只核验输入、哈希、6次调用预算和作者风格，不创建结果目录")
args = parser.parse_args()
output = args.output if args.output.is_absolute() else root / args.output
output = output.resolve()
case_numbers = (6, 14, 17)
new_strengths = (.5, 0.)
comparison_strengths = (.8, .5, 0.)
expected_family = {6: "H1", 14: "H0", 17: "H0"}

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

source_dir = root / (
    "results/erp_whole_head/adaptive_v6/development_diagnosis/"
    "hierarchical_v3_joint_consistency")
source_files = (
    "metadata.json", "rows.csv", "summary.json", "details.json",
    "completion.json", "REPORT.md", "case_06_combined40_joint.npz",
    "case_14_combined40_joint.npz", "case_17_combined40_joint.npz",
)
source_result_sha256 = {}
for name in source_files:
    path = source_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少 v3 MRF=0.8 基线文件：{path}")
    source_result_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()

source_metadata_path = source_dir / "metadata.json"
source_rows_path = source_dir / "rows.csv"
source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
source_completion = json.loads(
    (source_dir / "completion.json").read_text(encoding="utf-8"))
if source_summary.get("complete") is not True or \
        source_completion.get("complete") is not True or \
        source_summary.get("all_selected_joint_solvers_converged") is not True:
    raise ValueError("MRF=0.8 来源结果不完整或存在未收敛图")
if source_metadata.get("old_21_cases_already_unblinded") is not True or \
        source_metadata.get("formal_acceptance_allowed") is not False:
    raise ValueError("来源结果不再是已揭盲的开发诊断")

expected_source_settings = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 80, "max_inner_retries": 20,
    "epsilon_fraction": .05, "tolerance": .001,
    "outer_tolerance": .01, "surface_reweight_floor": .5,
    "deep_reweight_floor": .5, "edge_weight_floor": .5,
    "max_iter": 2000, "rho": 1., "adaptive_rho": True,
    "noise_multiplier": 1., "edge_fraction": .5, "calibration": "layer",
    "temporal_mode": "smooth", "ridge_fraction": 0.,
    "edge_penalty_mode": "group", "source_penalty_mode": "group",
    "deep_alias_penalty": False,
}
if source_metadata.get("solver_settings") != expected_source_settings:
    raise ValueError("MRF=0.8 来源求解设置已变化")

manifest = Path(source_metadata["manifest"])
manifest = manifest if manifest.is_absolute() else root / manifest
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
if manifest_sha256 != source_metadata.get("manifest_sha256"):
    raise ValueError("旧 validation manifest 与 v3 来源绑定不一致")
seed_roots = source_metadata.get("seed_roots")
if not isinstance(seed_roots, dict) or set(seed_roots) != {"fit", "check"} or \
        seed_roots["check"] != seed_roots["fit"] + 1:
    raise ValueError("来源结果缺少冻结 fit/check seed roots")

all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for number in case_numbers for case in all_cases
         if case.get("case_number") == number]
if [case.get("case_number") for case in cases] != list(case_numbers) or \
        len({case.get("case_id") for case in cases}) != len(case_numbers):
    raise ValueError("诊断病例必须且只能是旧21例中的 06/14/17")
if any(case.get("replica_seed_roots") != seed_roots or
       case.get("seed", [None])[0] != seed_roots["fit"] for case in cases):
    raise ValueError("诊断病例没有使用来源结果冻结的 seed roots")

with source_rows_path.open("r", encoding="utf-8-sig", newline="") as stream:
    source_rows_all = list(csv.DictReader(stream))
source_rows = {int(row["case_number"]): row for row in source_rows_all
               if int(row["case_number"]) in case_numbers}
if set(source_rows) != set(case_numbers):
    raise ValueError("MRF=0.8 rows.csv 缺少 06/14/17")
for number in case_numbers:
    row = source_rows[number]
    family = row["selected_family"]
    decision = int(row["frozen_deep_present_decision"])
    if family != expected_family[number] or decision != int(family == "H1"):
        raise ValueError(f"case {number:02d} 的冻结 family 决定已变化")

core_files = (
    "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py",
    "benchmark/metrics.py",
    "benchmark/protocol.py",
    "candidates/graph_reweight_solver.py",
    "candidates/oaster_balanced.py",
    "candidates/oaster_rebuilt.py",
    "run_erp_whole_head_matrix.py",
)
core_code_sha256 = {}
for relative in core_files:
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    core_code_sha256[relative] = digest
    if digest != source_metadata.get("code_sha256", {}).get(relative):
        raise ValueError(f"MRF=0.8 以后核心代码已变化：{relative}")
core_code_bundle_sha256 = hashlib.sha256("\n".join(
    f"{relative}:{core_code_sha256[relative]}" for relative in core_files
).encode("utf-8")).hexdigest()

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
if original._shared_fingerprint(shared) != source_metadata.get("shared_fingerprint"):
    raise ValueError("共享 forward 与 MRF=0.8 来源结果不一致")
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"], float)
adjacency = shared["adjacency"]

if args.preflight:
    print(json.dumps({
        "preflight": "PASS", "development_diagnosis_only": True,
        "old_21_cases_already_unblinded": True,
        "formal_or_blind_validation": False,
        "final_parameter_selection_allowed": False,
        "case_numbers": list(case_numbers),
        "frozen_family": {str(key): value for key, value in expected_family.items()},
        "baseline_strength_loaded_not_rerun": .8,
        "new_strengths": list(new_strengths), "inverse_calls_planned": 6,
        "only_changed_setting": "mrf_strength",
        "manifest_sha256": manifest_sha256, "seed_roots": seed_roots,
        "source_result_sha256": source_result_sha256,
        "core_code_bundle_sha256": core_code_bundle_sha256,
        "script_has_def_or_class": False, "output_exists": output.exists(),
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


# %% 2. 目标目录只在6张图全部完成后发布；0.8直接读已有结果，不重复反演。
output.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
started = time.perf_counter()
metadata = {
    "schema_version": 1,
    "protocol": "erp-v6-v3-mrf-strength-05-00-development-diagnosis",
    "development_diagnosis_only": True,
    "formal_acceptance_allowed": False,
    "blind_validation_claim_allowed": False,
    "final_parameter_selection_allowed": False,
    "old_21_cases_already_unblinded": True,
    "diagnostic_question": "does reducing graph diffusion repair cases 06/14 without destroying case 17",
    "case_numbers": list(case_numbers),
    "family_decision_source": "copied from hierarchical_v3_joint_consistency before reconstructing observations",
    "frozen_family": {str(key): value for key, value in expected_family.items()},
    "map_semantics": "combined40 selected-family map; H0 surface-only, H1 surface+deep",
    "channel_weights": "all whitened channels exactly one",
    "baseline_strength_loaded_not_rerun": .8,
    "new_strengths": list(new_strengths),
    "inverse_call_budget": 6,
    "only_changed_setting": "mrf_strength",
    "no_multiplier_added": True,
    "no_other_parameter_scan": True,
    "source_result": str(source_dir),
    "source_result_sha256": source_result_sha256,
    "manifest": str(manifest), "manifest_sha256": manifest_sha256,
    "seed_roots": seed_roots,
    "shared_fingerprint": source_metadata["shared_fingerprint"],
    "core_code_sha256": core_code_sha256,
    "core_code_bundle_sha256": core_code_bundle_sha256,
    "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    "baseline_solver_settings": expected_source_settings,
    "solver_settings_by_new_strength": {
        str(strength): {**expected_source_settings, "mrf_strength": strength}
        for strength in new_strengths
    },
    "publication_mode": "atomic directory promotion from same-volume staging",
}
new_rows = []
details = []
inverse_call_count = 0


# %% 3. 每例复用固定 family，0.5 与 0 各反演一次；图固定后才读取真值。
for case in cases:
    case_number = int(case["case_number"])
    name = case["case_id"]
    frozen_family = expected_family[case_number]
    deep_present = frozen_family == "H1"

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

    for strength in new_strengths:
        tick = time.perf_counter()
        settings = {**expected_source_settings, "mrf_strength": strength}
        changed = {key for key in settings
                   if settings[key] != expected_source_settings[key]}
        if changed != {"mrf_strength"}:
            raise RuntimeError("除 mrf_strength 外不允许改变任何求解参数")
        branch_estimate, fitting = reconstruct_evoked_oaster_v5_from_whitened(
            combined, branch_gain, n_surf, kernels=(), adjacency=branch_adjacency,
            baseline=baseline, active_windows=(active_window,),
            window_channel_weights=(primary_weights,), require_one=False,
            **settings)
        inverse_call_count += 1
        solver = fitting["windows"][0]["solver"]
        solver_ok = bool(
            solver["converged"] and solver["outer_converged"]
            and solver["final_inner_converged"]
            and solver["final_stationarity_gap_relative"]
            <= expected_source_settings["tolerance"] + 1e-12)
        if not solver_ok:
            raise RuntimeError(
                f"{name} MRF={strength:g}: 联合图没有到达固定点")
        selected = np.zeros((gain.shape[1], combined.shape[1]))
        selected[:branch_estimate.shape[0]] = branch_estimate
        if selected.shape != (gain.shape[1], combined.shape[1]) or \
                not np.isfinite(selected).all():
            raise RuntimeError(f"{name} MRF={strength:g}: 图形状或有限性错误")
        if not deep_present and np.any(selected[n_surf:] != 0):
            raise RuntimeError(f"{name} MRF={strength:g}: H0 图意外包含深源")

        # 到这里图已经完全固定；以下真值只参与事后指标，未参与 family 或反演。
        truth = np.asarray(observation["truth"], float)
        values = metrics.evaluate_estimate(
            selected, truth, vertices, observation["groups"], n_surf,
            active_samples, shared["auc_cortex"], baseline=baseline)
        surface_groups = [np.asarray(group, int) for group in observation["groups"]
                          if np.asarray(group).size and
                          np.all(np.asarray(group) < n_surf)]
        surface_component_dle = []
        surface_patch_hits = []
        if surface_groups:
            surface_energy = np.sum(selected[:n_surf, active_samples] ** 2, axis=1)
            truth_indices = np.concatenate(surface_groups)
            truth_owner = np.concatenate([
                np.full(len(group), index, int)
                for index, group in enumerate(surface_groups)])
            owner = truth_owner[
                cKDTree(vertices[truth_indices]).query(vertices[:n_surf])[1]]
            support = surface_energy > .1 * surface_energy.max(initial=0.)
            for index, group in enumerate(surface_groups):
                region = np.flatnonzero(owner == index)
                supported_region = region[support[region]]
                if supported_region.size:
                    peak = int(supported_region[
                        np.argmax(surface_energy[supported_region])])
                    center = vertices[group].mean(axis=0)
                    surface_component_dle.append(float(
                        np.linalg.norm(vertices[peak] - center) * 1000))
                else:
                    surface_component_dle.append(None)
                surface_patch_hits.append(int(np.count_nonzero(support[group])))
        finite_components = [value for value in surface_component_dle
                             if value is not None]
        component_max = (
            np.nan if len(finite_components) != len(surface_groups)
            else max(finite_components, default=np.nan))
        row = {
            "case_number": case_number, "case_id": name,
            "scenario_posthoc": case["scenario"],
            "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
            "selected_family": frozen_family, "mrf_strength": strength,
            "result_source": "new_single_run",
            "solver_converged": int(solver_ok),
            "truth_used_only_after_map_fixed": 1,
            "global_An_auc": values["auc_tie_corrected"],
            "surface_An_auc": values["surface_auc_tie_corrected"],
            "surface_DLE_mm": values["surface_dle_mm"],
            "surface_component_DLE_mm": component_max,
            "surface_SD_mm": values["surface_sd_mm"],
            "surface_patch_hit_count": int(
                sum(value > 0 for value in surface_patch_hits)),
            "surface_patch_count": len(surface_patch_hits),
            "deep_false_positive": values["deep_false_positive"],
            "deep_peak_distance_mm": values["deep_peak_distance_mm"],
            "surface_active_count": values["surface_active_count"],
            "deep_active_count": values["deep_active_count"],
        }
        new_rows.append(row)
        label = "05" if strength == .5 else "00"
        map_path = staging / f"case_{case_number:02d}_mrf_{label}.npz"
        np.savez_compressed(
            map_path, truth=truth.astype(np.float32),
            selected=selected.astype(np.float32),
            vertices=vertices.astype(np.float32),
            times=np.asarray(shared["times"]), active=active_samples,
            baseline=baseline)
        clean_metrics = {
            key: (None if isinstance(value, (float, np.floating))
                  and not np.isfinite(value)
                  else value.item() if isinstance(value, np.generic) else value)
            for key, value in values.items()}
        details.append({
            "case_number": case_number, "case_id": name,
            "selected_family": frozen_family, "mrf_strength": strength,
            "frozen_decision_loaded_before_observation": True,
            "truth_used_by_decision_or_localization": False,
            "truth_used_only_after_map_fixed": True,
            "combined_trial_count": 40,
            "all_whitened_channel_weights_one": True,
            "only_setting_changed_from_baseline": "mrf_strength",
            "joint_objective": fitting["mode"],
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
        print(name, frozen_family, f"MRF={strength:g}",
              f"AUC={values['auc_tie_corrected']:.3f}",
              f"surface DLE={values['surface_dle_mm']}", flush=True)


# %% 4. 拼接未重跑的0.8基线，逐例并排呈现；不据此选最终参数。
if inverse_call_count != 6 or len(new_rows) != 6 or len(details) != 6:
    raise RuntimeError("反演次数必须严格为6：3例乘2个新强度")
if not all(row["solver_converged"] for row in new_rows):
    raise RuntimeError("存在未收敛的新图")

baseline_rows = []
for case in cases:
    number = int(case["case_number"])
    source = source_rows[number]
    baseline_rows.append({
        "case_number": number, "case_id": case["case_id"],
        "scenario_posthoc": case["scenario"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "selected_family": expected_family[number], "mrf_strength": .8,
        "result_source": "existing_hierarchical_v3_joint_consistency",
        "solver_converged": int(source["solver_converged"]),
        "truth_used_only_after_map_fixed": 1,
        "global_An_auc": float(source["new_global_An_auc"]),
        "surface_An_auc": float(source["new_surface_An_auc"]),
        "surface_DLE_mm": float(source["new_surface_DLE_mm"]),
        "surface_component_DLE_mm": float(
            source["new_surface_component_DLE_mm"]),
        "surface_SD_mm": float(source["new_surface_SD_mm"]),
        "surface_patch_hit_count": int(source["new_surface_patch_hit_count"]),
        "surface_patch_count": int(source["surface_patch_count"]),
        "deep_false_positive": int(source["new_deep_false_positive"]),
        "deep_peak_distance_mm": float(source["new_deep_peak_distance_mm"]),
        "surface_active_count": int(source["new_surface_active_count"]),
        "deep_active_count": int(source["new_deep_active_count"]),
    })

all_rows_unordered = baseline_rows + new_rows
comparison_rows = []
for number in case_numbers:
    for strength in comparison_strengths:
        matches = [row for row in all_rows_unordered
                   if row["case_number"] == number and
                   row["mrf_strength"] == strength]
        if len(matches) != 1:
            raise RuntimeError(
                f"case {number:02d} MRF={strength:g} 比较行数量错误")
        comparison_rows.append(matches[0])

comparison = {}
for number in case_numbers:
    comparison[str(number)] = {}
    for row in comparison_rows:
        if row["case_number"] != number:
            continue
        clean_row = {
            key: (None if isinstance(value, (float, np.floating))
                  and not np.isfinite(value)
                  else value.item() if isinstance(value, np.generic) else value)
            for key, value in row.items()
            if key in (
                "global_An_auc", "surface_An_auc", "surface_DLE_mm",
                "surface_component_DLE_mm", "surface_SD_mm",
                "surface_patch_hit_count", "surface_patch_count",
                "deep_false_positive", "deep_peak_distance_mm")}
        comparison[str(number)][str(row["mrf_strength"])] = clean_row

summary = {
    "complete": True, "development_diagnosis_only": True,
    "formal_or_blind_validation": False,
    "old_21_cases_already_unblinded": True,
    "final_parameter_selection_allowed": False,
    "parameter_selected": None,
    "case_count": 3, "new_inverse_call_count": inverse_call_count,
    "baseline_inverse_call_count": 0,
    "comparison_row_count": len(comparison_rows),
    "all_new_joint_solvers_converged": True,
    "only_changed_setting": "mrf_strength",
    "comparison": comparison,
    "interpretation_boundary": (
        "mechanism diagnosis on revealed cases only; these rows cannot choose "
        "the final MRF strength"),
    "wall_seconds": time.perf_counter() - started,
}


# %% 5. 临时目录写完、自检完整后一次性发布。
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
    writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
    writer.writeheader()
    writer.writerows(comparison_rows)

report_lines = [
    "# v3 图扩散强度诊断（旧21例已揭盲）", "",
    "固定已有 family 决定、combined40、全1白化通道权重和 outer80；只把 MRF 强度从 0.8 改为 0.5 或 0。0.8 行直接读取已有联合一致性结果，没有重复反演。", "",
    "|病例|family|MRF|来源|An_auc|surface AUC|surface DLE mm|component DLE mm|SD mm|patch hit|deep FP|",
    "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in comparison_rows:
    formatted = {}
    for key in ("global_An_auc", "surface_An_auc", "surface_DLE_mm",
                "surface_component_DLE_mm", "surface_SD_mm"):
        value = row[key]
        formatted[key] = "—" if not np.isfinite(value) else f"{value:.3f}"
    source_label = "已有基线" if row["mrf_strength"] == .8 else "本轮单次"
    report_lines.append(
        f"|{row['case_number']:02d}|{row['selected_family']}|"
        f"{row['mrf_strength']:g}|{source_label}|{formatted['global_An_auc']}|"
        f"{formatted['surface_An_auc']}|{formatted['surface_DLE_mm']}|"
        f"{formatted['surface_component_DLE_mm']}|{formatted['surface_SD_mm']}|"
        f"{row['surface_patch_hit_count']}/{row['surface_patch_count']}|"
        f"{row['deep_false_positive']}|")
report_lines += [
    "", "## 使用边界", "",
    "- 这3例来自已经揭盲的旧21例，只能作图扩散机制诊断，不是盲验证。",
    "- 结果不得用于选择最终 MRF 强度；最终参数必须在新的开发/校准面板确定，再由未见数据验证。",
    "- 本轮没有加入 multiplier，也没有扫描任何其他参数；新增反演严格为6次。",
    "- 真值只在每张候选图固定后用于计算 An_auc、surface AUC、DLE、component DLE、SD、patch hit 和 deep FP。", "",
]
(staging / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
(staging / "completion.json").write_text(
    json.dumps({
        "complete": True, "development_diagnosis_only": True,
        "new_inverse_call_count": inverse_call_count,
        "comparison_row_count": len(comparison_rows),
        "all_new_joint_solvers_converged": True,
        "final_parameter_selection_allowed": False,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

reloaded_summary = json.loads((staging / "summary.json").read_text(encoding="utf-8"))
with (staging / "rows.csv").open("r", encoding="utf-8-sig", newline="") as stream:
    reloaded_rows = list(csv.DictReader(stream))
map_files = [path for path in staging.glob("case_*_mrf_*.npz")]
if reloaded_summary.get("complete") is not True or len(reloaded_rows) != 9 or \
        len(map_files) != 6 or inverse_call_count != 6:
    raise RuntimeError("发布前自检失败")
if output.exists():
    raise FileExistsError(f"运行期间目标目录被创建，停止发布：{output}")
os.replace(staging, output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
