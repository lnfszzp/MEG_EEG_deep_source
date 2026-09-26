"""# %% v3 图扩散盲选择诊断：旧21例已揭盲，只检验 held-out 选参机制。"""

# %% 1. 固定输入、病例、family 和唯一候选参数；不读取旧定位指标。
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

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import (
    _smooth_temporal_basis, reconstruct_evoked_oaster_v5_from_whitened)
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path,
    default=root / "results/erp_whole_head/adaptive_v6/development_diagnosis/"
                   "v3_mrf_heldout_selector")
parser.add_argument("--preflight", action="store_true",
                    help="只核验输入、哈希、6次调用预算和作者风格，不创建结果目录")
args = parser.parse_args()
output = args.output if args.output.is_absolute() else root / args.output
output = output.resolve()
case_numbers = (6, 14, 17)
strengths = (.5, .8)
frozen_family = {6: "H1", 14: "H0", 17: "H0"}

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
    "v3_mrf_strength_05_00")
source_names = (
    "metadata.json", "details.json", "rows.csv", "summary.json",
    "completion.json", "REPORT.md", "case_06_mrf_05.npz",
    "case_14_mrf_05.npz", "case_17_mrf_05.npz",
)
source_sha256 = {}
for name in source_names:
    path = source_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少 76 的来源文件：{path}")
    source_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest()

source_metadata = json.loads(
    (source_dir / "metadata.json").read_text(encoding="utf-8"))
source_completion = json.loads(
    (source_dir / "completion.json").read_text(encoding="utf-8"))
if source_metadata.get("protocol") != \
        "erp-v6-v3-mrf-strength-05-00-development-diagnosis" or \
        source_metadata.get("old_21_cases_already_unblinded") is not True or \
        source_metadata.get("formal_acceptance_allowed") is not False or \
        source_metadata.get("frozen_family") != {
            str(key): value for key, value in frozen_family.items()}:
    raise ValueError("76 的开发诊断身份或冻结 family 已变化")
if source_completion.get("complete") is not True or \
        source_completion.get("new_inverse_call_count") != 6 or \
        source_completion.get("all_new_joint_solvers_converged") is not True:
    raise ValueError("76 的 MRF 强度诊断不完整")

expected_settings = {
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
if source_metadata.get("baseline_solver_settings") != expected_settings:
    raise ValueError("76 的 outer80/all-one 求解设置已变化")

source_script = root / "作者风格版/76-v3图扩散强度诊断.py"
if hashlib.sha256(source_script.read_bytes()).hexdigest() != \
        source_metadata.get("script_sha256"):
    raise ValueError("76 的脚本与运行记录不一致")

manifest = Path(source_metadata["manifest"])
manifest = manifest if manifest.is_absolute() else root / manifest
manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
seed_roots = source_metadata.get("seed_roots")
if manifest_sha256 != source_metadata.get("manifest_sha256") or \
        seed_roots != {"fit": 2026100101, "check": 2026100102}:
    raise ValueError("旧 manifest 或复制种子与 76 不一致")

all_cases = json.loads(manifest.read_text(encoding="utf-8"))
cases = [case for number in case_numbers for case in all_cases
         if case.get("case_number") == number]
if [case.get("case_number") for case in cases] != list(case_numbers) or \
        len({case.get("case_id") for case in cases}) != len(case_numbers):
    raise ValueError("诊断病例必须且只能是旧21例中的 06/14/17")
if any(case.get("replica_seed_roots") != seed_roots or
       case.get("seed", [None])[0] != seed_roots["fit"] for case in cases):
    raise ValueError("病例没有绑定 76 使用的冻结复制种子")

core_files = tuple(source_metadata.get("core_code_sha256", {}))
core_code_sha256 = {}
for relative in core_files:
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    core_code_sha256[relative] = digest
    if digest != source_metadata["core_code_sha256"][relative]:
        raise ValueError(f"76 以后核心代码已变化：{relative}")
core_code_bundle_sha256 = hashlib.sha256("\n".join(
    f"{relative}:{core_code_sha256[relative]}" for relative in core_files
).encode("utf-8")).hexdigest()
if core_code_bundle_sha256 != source_metadata.get("core_code_bundle_sha256"):
    raise ValueError("核心代码 bundle 哈希与 76 不一致")

shared = protocol.load_shared(
    root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
shared_fingerprint = original._shared_fingerprint(shared)
if shared_fingerprint != source_metadata.get("shared_fingerprint"):
    raise ValueError("共享 forward 与 76 不一致")
n_surf = int(shared["n_surf"])
adjacency = shared["adjacency"]


# %% 2. preflight 额外核验76保存的0.5图和其指向的0.8图；不创建目录。
if args.preflight:
    source_details = json.loads(
        (source_dir / "details.json").read_text(encoding="utf-8"))
    details_05 = {int(item["case_number"]): item for item in source_details
                  if float(item["mrf_strength"]) == .5}
    upstream_dir = Path(source_metadata["source_result"])
    upstream_dir = upstream_dir if upstream_dir.is_absolute() else root / upstream_dir
    checked_maps = {}
    for number in case_numbers:
        item = details_05.get(number)
        if item is None:
            raise ValueError(f"76 details 缺少 case {number:02d} 的 MRF=0.5 图")
        map_05 = source_dir / item["map_file"]
        map_08_name = f"case_{number:02d}_combined40_joint.npz"
        map_08 = upstream_dir / map_08_name
        expected_05 = item["map_file_sha256"]
        expected_08 = source_metadata["source_result_sha256"].get(map_08_name)
        for strength, path, expected in (
                (.5, map_05, expected_05), (.8, map_08, expected_08)):
            if not path.is_file() or expected is None or \
                    hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"case {number:02d} MRF={strength:g} combined40 图哈希不一致")
            checked_maps[f"case_{number:02d}_mrf_{strength:g}"] = expected
    print(json.dumps({
        "preflight": "PASS", "development_diagnosis_only": True,
        "old_21_cases_already_unblinded": True,
        "formal_or_blind_validation": False,
        "case_numbers": list(case_numbers),
        "frozen_family": {str(key): value for key, value in frozen_family.items()},
        "fit_half": "training20", "held_out_half": "confirmation20",
        "selection_rule": "minimum all-whitened-channel raw squared loss; tie -> MRF 0.8",
        "strengths": list(strengths), "inverse_calls_planned": 6,
        "manifest_sha256": manifest_sha256, "seed_roots": seed_roots,
        "source_sha256": source_sha256, "checked_combined40_maps": checked_maps,
        "core_code_bundle_sha256": core_code_bundle_sha256,
        "script_has_def_or_class": False, "output_exists": output.exists(),
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


# %% 3. 每例、每个MRF只用training20拟合；confirmation20只算平滑DCT预测损失。
output.parent.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
started = time.perf_counter()
candidate_rows = []
candidate_details = []
selected_strengths = {}
inverse_call_count = 0

for case in cases:
    case_number = int(case["case_number"])
    name = case["case_id"]
    family = frozen_family[case_number]
    observation = prepare_trial_covariance_case(
        shared, case, seed_root=seed_roots["fit"])
    baseline = np.asarray(observation["baseline"], bool)
    active = np.asarray(observation["active_windows"][0], bool)
    training = np.asarray(observation["training"], float)
    confirmation = np.asarray(observation["confirmation"], float)
    gain = np.asarray(observation["gain"], float)
    modality_sizes = tuple(
        int(value) for value in observation["metadata"]["retained_channels"])
    if observation["metadata"].get("n_trials_per_half") != 20 or \
            observation["metadata"].get("replica_seed_roots") != seed_roots or \
            len(modality_sizes) != 2 or sum(modality_sizes) != gain.shape[0]:
        raise RuntimeError(f"{name}: 不是冻结的 EEG/MEG training20-confirmation20 输入")

    branch_gain = gain if family == "H1" else gain[:, :n_surf]
    branch_adjacency = adjacency if family == "H1" else adjacency[:n_surf, :n_surf]
    weights = np.ones(gain.shape[0], float)
    centered = confirmation - confirmation[:, baseline].mean(axis=1, keepdims=True)
    basis, basis_info = _smooth_temporal_basis(confirmation, baseline, active)
    response = centered @ basis.T
    mean_correction = float(
        np.sum(basis[:, active].sum(axis=1) ** 2) / baseline.sum())
    noise_factor = len(basis) + mean_correction
    case_candidates = []

    for strength in strengths:
        tick = time.perf_counter()
        settings = {**expected_settings, "mrf_strength": strength}
        changed = {key for key in settings
                   if settings[key] != expected_settings[key]}
        if changed != (set() if strength == .8 else {"mrf_strength"}):
            raise RuntimeError("除 mrf_strength 外不允许改变 76 的任何求解参数")
        branch_estimate, fitting = reconstruct_evoked_oaster_v5_from_whitened(
            training, branch_gain, n_surf, kernels=(),
            adjacency=branch_adjacency, baseline=baseline,
            active_windows=(active,), window_channel_weights=(weights,),
            require_one=False, **settings)
        inverse_call_count += 1
        solver = fitting["windows"][0]["solver"]
        solver_ok = bool(
            solver["converged"] and solver["outer_converged"]
            and solver["final_inner_converged"]
            and solver["final_stationarity_gap_relative"]
            <= expected_settings["tolerance"] + 1e-12)
        if not solver_ok:
            raise RuntimeError(
                f"{name} MRF={strength:g}: training20 联合图没有严格收敛")

        estimate = np.zeros((gain.shape[1], training.shape[1]))
        estimate[:branch_estimate.shape[0]] = branch_estimate
        if not np.isfinite(estimate).all() or \
                (family == "H0" and np.any(estimate[n_surf:] != 0)):
            raise RuntimeError(f"{name} MRF={strength:g}: 候选图非法")
        prediction = gain @ (estimate @ basis.T)
        total_loss = float(np.sum((response - prediction) ** 2))
        baseline_variance = float(
            np.sum(centered[:, baseline] ** 2) / (baseline.sum() - 1))
        total_noise = baseline_variance * noise_factor
        if not np.isfinite([total_loss, total_noise]).all() or total_noise <= 0:
            raise RuntimeError(f"{name} MRF={strength:g}: 总 held-out 损失非法")

        block_losses = []
        block_normalized_losses = []
        start = 0
        for size in modality_sizes:
            block = slice(start, start + size)
            block_loss = float(np.sum(
                (response[block] - prediction[block]) ** 2))
            block_variance = float(
                np.sum(centered[block, baseline] ** 2) / (baseline.sum() - 1))
            block_noise = block_variance * noise_factor
            if not np.isfinite([block_loss, block_noise]).all() or block_noise <= 0:
                raise RuntimeError(
                    f"{name} MRF={strength:g}: 模态 held-out 损失非法")
            block_losses.append(block_loss)
            block_normalized_losses.append(block_loss / block_noise)
            start += size
        if not np.isclose(sum(block_losses), total_loss, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"{name} MRF={strength:g}: 分块损失不等于总损失")

        row = {
            "case_number": case_number, "case_id": name,
            "selected_family": family, "mrf_strength": strength,
            "fit_half": "training20", "held_out_half": "confirmation20",
            "solver_converged": int(solver_ok),
            "total_heldout_squared_loss": total_loss,
            "total_noise_normalized_loss": total_loss / total_noise,
            "eeg_heldout_squared_loss": block_losses[0],
            "meg_heldout_squared_loss": block_losses[1],
            "eeg_noise_normalized_loss": block_normalized_losses[0],
            "meg_noise_normalized_loss": block_normalized_losses[1],
            "response_energy": float(np.sum(response ** 2)),
            "prediction_energy": float(np.sum(prediction ** 2)),
        }
        candidate_rows.append(row)
        case_candidates.append(row)
        candidate_details.append({
            **row, "truth_or_old_metric_used_for_selection": False,
            "all_whitened_channel_weights_one": True,
            "temporal_basis": basis_info,
            "modality_order": ["EEG", "MEG"],
            "modality_sizes": list(modality_sizes),
            "solver_outer_iterations": len(solver["history"]),
            "solver_final_stationarity_gap_relative": float(
                solver["final_stationarity_gap_relative"]),
            "elapsed_seconds": time.perf_counter() - tick,
        })
        print(name, family, f"MRF={strength:g}",
              f"held-out loss={total_loss:.9g}", flush=True)

    loss_05 = next(row["total_heldout_squared_loss"] for row in case_candidates
                   if row["mrf_strength"] == .5)
    loss_08 = next(row["total_heldout_squared_loss"] for row in case_candidates
                   if row["mrf_strength"] == .8)
    selected_strengths[case_number] = .8 if loss_08 <= loss_05 else .5

if inverse_call_count != 6 or len(candidate_rows) != 6 or \
        len(selected_strengths) != 3 or \
        not all(row["solver_converged"] for row in candidate_rows):
    raise RuntimeError("必须完成且只完成 3例×2强度=6 次严格收敛反演")


# %% 4. 三例盲选全部固定后，才核验并引用76的combined40图，再读取真值指标。
source_details = json.loads(
    (source_dir / "details.json").read_text(encoding="utf-8"))
with (source_dir / "rows.csv").open(
        "r", encoding="utf-8-sig", newline="") as stream:
    source_rows = list(csv.DictReader(stream))
details_05 = {int(item["case_number"]): item for item in source_details
              if float(item["mrf_strength"]) == .5}
upstream_dir = Path(source_metadata["source_result"])
upstream_dir = upstream_dir if upstream_dir.is_absolute() else root / upstream_dir
map_references = {}
selected_rows = []
posthoc_expected_strength = {6: .5, 14: .5, 17: .8}

for case in cases:
    number = int(case["case_number"])
    selected_strength = selected_strengths[number]
    all_case_references = {}
    for strength in strengths:
        if strength == .5:
            item = details_05.get(number)
            if item is None:
                raise ValueError(f"76 details 缺少 case {number:02d} 的 MRF=0.5 图")
            map_path = source_dir / item["map_file"]
            expected_hash = item["map_file_sha256"]
        else:
            map_name = f"case_{number:02d}_combined40_joint.npz"
            map_path = upstream_dir / map_name
            expected_hash = source_metadata["source_result_sha256"].get(map_name)
        actual_hash = (hashlib.sha256(map_path.read_bytes()).hexdigest()
                       if map_path.is_file() else None)
        if expected_hash is None or actual_hash != expected_hash:
            raise ValueError(
                f"case {number:02d} MRF={strength:g} combined40 图哈希不一致")
        all_case_references[str(strength)] = {
            "path": str(map_path), "sha256": actual_hash,
            "copied_to_output": False,
        }
    map_references[str(number)] = all_case_references

    selected_map = all_case_references[str(selected_strength)]
    with np.load(selected_map["path"], allow_pickle=False) as archive:
        if set(archive.files) != {
                "truth", "selected", "vertices", "times", "active", "baseline"} or \
                archive["selected"].shape != archive["truth"].shape or \
                not np.isfinite(archive["selected"]).all():
            raise ValueError(f"case {number:02d} 所选 combined40 图内容非法")

    matches = [row for row in source_rows
               if int(row["case_number"]) == number
               and float(row["mrf_strength"]) == selected_strength]
    if len(matches) != 1 or matches[0]["selected_family"] != frozen_family[number]:
        raise ValueError(f"76 rows 缺少 case {number:02d} 所选图的唯一指标行")
    source_row = matches[0]
    metrics = {}
    for key in (
            "global_An_auc", "surface_An_auc", "surface_DLE_mm",
            "surface_component_DLE_mm", "surface_SD_mm",
            "deep_peak_distance_mm"):
        value = float(source_row[key])
        metrics[key] = None if not np.isfinite(value) else value
    for key in (
            "surface_patch_hit_count", "surface_patch_count",
            "deep_false_positive", "surface_active_count", "deep_active_count"):
        metrics[key] = int(source_row[key])

    candidates = [row for row in candidate_rows if row["case_number"] == number]
    loss_05 = next(row["total_heldout_squared_loss"] for row in candidates
                   if row["mrf_strength"] == .5)
    loss_08 = next(row["total_heldout_squared_loss"] for row in candidates
                   if row["mrf_strength"] == .8)
    selected_rows.append({
        "case_number": number, "case_id": case["case_id"],
        "selected_family": frozen_family[number],
        "mrf_05_total_heldout_squared_loss": loss_05,
        "mrf_08_total_heldout_squared_loss": loss_08,
        "selected_mrf_strength": selected_strength,
        "tie_selected_08": int(loss_05 == loss_08),
        "posthoc_expected_mrf_strength": posthoc_expected_strength[number],
        "posthoc_expected_match": int(
            selected_strength == posthoc_expected_strength[number]),
        "selected_map_path": selected_map["path"],
        "selected_map_sha256": selected_map["sha256"],
        **metrics,
    })

all_posthoc_expected = all(row["posthoc_expected_match"] for row in selected_rows)


# %% 5. 输出损失、盲选结果、引用图和使用边界；完成后原子发布。
metadata = {
    "schema_version": 1,
    "protocol": "erp-v6-v3-mrf-heldout-selector-development-diagnosis",
    "development_diagnosis_only": True,
    "formal_acceptance_allowed": False,
    "blind_validation_claim_allowed": False,
    "final_parameter_selection_allowed": False,
    "old_21_cases_already_unblinded": True,
    "case_numbers": list(case_numbers),
    "frozen_family": {str(key): value for key, value in frozen_family.items()},
    "fit_half": "training20", "held_out_half": "confirmation20",
    "candidate_strengths": list(strengths),
    "inverse_call_budget": 6,
    "selection_rule": (
        "minimum confirmation20 all-whitened-channel raw squared loss in the "
        "fixed smooth-DCT subspace; exact tie selects MRF 0.8"),
    "modality_loss_role": "recorded only; EEG/MEG normalized losses do not select MRF",
    "truth_and_old_localization_metric_role": (
        "read only after all three MRF selections were fixed; posthoc reporting only"),
    "combined40_map_role": (
        "no refit after selection; directly reference hash-verified maps saved by 76 "
        "or its hash-bound MRF=0.8 source"),
    "solver_settings_by_strength": {
        str(strength): {**expected_settings, "mrf_strength": strength}
        for strength in strengths
    },
    "manifest": str(manifest), "manifest_sha256": manifest_sha256,
    "seed_roots": seed_roots, "shared_fingerprint": shared_fingerprint,
    "source_result": str(source_dir), "source_sha256": source_sha256,
    "combined40_map_references": map_references,
    "core_code_sha256": core_code_sha256,
    "core_code_bundle_sha256": core_code_bundle_sha256,
    "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    "publication_mode": "atomic directory promotion from same-volume staging",
}
summary = {
    "complete": True, "development_diagnosis_only": True,
    "formal_or_blind_validation": False,
    "old_21_cases_already_unblinded": True,
    "final_parameter_selection_allowed": False,
    "case_count": 3, "inverse_call_count": inverse_call_count,
    "all_training_solvers_converged": True,
    "selected_strengths": {
        str(number): strength for number, strength in selected_strengths.items()},
    "posthoc_expected_strengths": {
        str(number): strength
        for number, strength in posthoc_expected_strength.items()},
    "posthoc_expected_match_count": int(sum(
        row["posthoc_expected_match"] for row in selected_rows)),
    "all_posthoc_expected_strengths_selected": bool(all_posthoc_expected),
    "interpretation_boundary": (
        "selection mechanism diagnosis on revealed cases only; this cannot freeze "
        "MRF strength or support a blind-validation claim"),
    "wall_seconds": time.perf_counter() - started,
}

(staging / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
(staging / "candidate_details.json").write_text(
    json.dumps(candidate_details, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
(staging / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")
with (staging / "candidate_losses.csv").open(
        "w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
    writer.writeheader()
    writer.writerows(candidate_rows)
with (staging / "rows.csv").open(
        "w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
    writer.writeheader()
    writer.writerows(selected_rows)

report_lines = [
    "# v3 图扩散 held-out 盲选择诊断（旧21例已揭盲）", "",
    "固定 family 后，MRF 0.5/0.8 各自只拟合 training20；confirmation20 在同一平滑 DCT 子空间计算全白化通道平方损失。损失较小者胜，精确平局预设选 0.8。", "",
    "|病例|family|loss 0.5|loss 0.8|盲选MRF|事后期望|匹配|An_auc|surface AUC|DLE mm|component DLE mm|SD mm|patch hit|",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in selected_rows:
    shown = {}
    for key in (
            "global_An_auc", "surface_An_auc", "surface_DLE_mm",
            "surface_component_DLE_mm", "surface_SD_mm"):
        value = row[key]
        shown[key] = "—" if value is None else f"{value:.3f}"
    report_lines.append(
        f"|{row['case_number']:02d}|{row['selected_family']}|"
        f"{row['mrf_05_total_heldout_squared_loss']:.6g}|"
        f"{row['mrf_08_total_heldout_squared_loss']:.6g}|"
        f"{row['selected_mrf_strength']:g}|"
        f"{row['posthoc_expected_mrf_strength']:g}|"
        f"{'是' if row['posthoc_expected_match'] else '否'}|"
        f"{shown['global_An_auc']}|{shown['surface_An_auc']}|"
        f"{shown['surface_DLE_mm']}|{shown['surface_component_DLE_mm']}|"
        f"{shown['surface_SD_mm']}|"
        f"{row['surface_patch_hit_count']}/{row['surface_patch_count']}|")
report_lines += [
    "", "## 使用边界", "",
    "- 06/14/17 来自已经揭盲的旧21例，只能检验 held-out 选参机制，不是验证集。",
    "- 主选择只看 confirmation20 的全白化通道总平方损失；EEG/MEG 归一损失仅记录。",
    "- case06/14 期望0.5、case17期望0.8及定位指标都在三例选择固定后才作事后对照。",
    "- 所选定位图没有再次拟合或复制，直接引用76保存并经 SHA-256 核验的 combined40 图。", "",
]
(staging / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
(staging / "completion.json").write_text(
    json.dumps({
        "complete": True, "development_diagnosis_only": True,
        "inverse_call_count": inverse_call_count,
        "candidate_row_count": len(candidate_rows),
        "selected_row_count": len(selected_rows),
        "all_training_solvers_converged": True,
        "all_posthoc_expected_strengths_selected": bool(all_posthoc_expected),
        "final_parameter_selection_allowed": False,
        "wall_seconds": summary["wall_seconds"],
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

reloaded_summary = json.loads(
    (staging / "summary.json").read_text(encoding="utf-8"))
with (staging / "candidate_losses.csv").open(
        "r", encoding="utf-8-sig", newline="") as stream:
    reloaded_candidates = list(csv.DictReader(stream))
with (staging / "rows.csv").open(
        "r", encoding="utf-8-sig", newline="") as stream:
    reloaded_selected = list(csv.DictReader(stream))
if reloaded_summary.get("complete") is not True or \
        len(reloaded_candidates) != 6 or len(reloaded_selected) != 3 or \
        inverse_call_count != 6 or set(map_references) != {"6", "14", "17"}:
    raise RuntimeError("发布前自检失败")
if output.exists():
    raise FileExistsError(f"运行期间目标目录被创建，停止发布：{output}")
os.replace(staging, output)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
