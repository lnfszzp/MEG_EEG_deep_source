"""# %% v3 全新开发运行：双向条件门控、held-out MRF 选择和同 family 联合重拟合。"""

# %% 1. 固定输入、作者风格和不可越过的开发边界。
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

if not __debug__:
    raise RuntimeError("v3 开发运行禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import (
    _smooth_temporal_basis, reconstruct_evoked_oaster_v5_from_whitened)
from candidates.oaster_partial_confirmation import score_partial_deep_presence
from candidates.oaster_predictive import fit_predictive_models
from refined_grid import load_refined_forward, refined_sd_dle
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output", type=Path,
    default=root / "results/erp_whole_head/adaptive_v6/development_diagnosis/"
                   "hierarchical_v3_fresh_development")
parser.add_argument("--preflight", action="store_true",
                    help="只核验冻结面板、off-grid forward、调用预算和作者风格")
parser.add_argument(
    "--evaluate-frozen", action="store_true",
    help="只续算已完整冻结的39张final map；沿用原盲运行身份并另记评估脚本哈希")
args = parser.parse_args()
output = args.output if args.output.is_absolute() else root / args.output
output = output.resolve()

script_path = Path(__file__).resolve()
script_text = script_path.read_text(encoding="utf-8")
script_tree = ast.parse(script_text)
if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
       for node in ast.walk(script_tree)):
    raise RuntimeError("作者风格自检失败：79 号脚本不应定义 def/class")
if script_text.count("# %%") < 8:
    raise RuntimeError("作者风格自检失败：79 号脚本缺少顺序式 # %% 分段")

adaptive_root = root / "results/erp_whole_head/adaptive_v6"
protocol_dir = adaptive_root / "protocol"
manifest_path = protocol_dir / "hierarchical_v3_fresh_development_manifest.json"
audit_path = protocol_dir / "hierarchical_v3_fresh_development_audit.json"
generator_path = root / "作者风格版/78-v3全新开发面板.py"
for path in (manifest_path, audit_path, generator_path):
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少78冻结输入，79拒绝猜测 schema 或病例：{path}")

manifest_payload = manifest_path.read_bytes()
audit_payload = audit_path.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
audit_sha256 = hashlib.sha256(audit_payload).hexdigest()
for path, digest in ((manifest_path, manifest_sha256), (audit_path, audit_sha256)):
    sidecar = Path(str(path) + ".sha256")
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected:
        raise ValueError(f"78 冻结文件旁车不一致：{path}")

cases = json.loads(manifest_payload)
audit = json.loads(audit_payload)
required_case_fields = {
    "case_id", "case_number", "configuration_id", "configuration_number",
    "configuration_kind", "scenario", "scenario_code", "pair_index",
    "eeg_snr_db", "meg_snr_db", "replica_seed_roots", "seed", "check_seed",
    "h0_role", "alias_tier", "surface_centers", "deep_index", "deep_local",
    "truth_grid", "truth_forward_sha256", "truth_forward_array_bundle_sha256",
    "truth_surface_patch_indices_refined", "truth_surface_patch_weights",
    "evaluation_surface_projection", "truth_deep_fine_index",
    "evaluation_deep_index", "truth_surface_patch_nearest_inverse_distance_mm",
    "inverse_forward_fingerprint",
    "channel_order_sha256",
}
if not isinstance(cases, list) or len(cases) != 39 or \
        [case.get("case_number") for case in cases] != list(range(39)) or \
        len({case.get("case_id") for case in cases}) != 39 or \
        any(not required_case_fields.issubset(case) for case in cases):
    raise ValueError("78 manifest 不是39例连续编号的已知 schema")
if audit.get("protocol") != "erp-v3-fresh-development-manifest" or \
        audit.get("schema_version") != 1 or audit.get("case_count") != 39 or \
        audit.get("manifest_sha256") != manifest_sha256 or \
        audit.get("formal_performance_claim_allowed") is not False:
    raise ValueError("78 audit 身份、manifest 哈希或开发边界不一致")

seed_roots = {"fit": 2026100301, "check": 2026100302}
snr_pairs = {(-10, -10), (-10, 20), (20, -10)}
if any(case["replica_seed_roots"] != seed_roots or
       case["seed"] != [seed_roots["fit"], case["pair_index"],
                        case["configuration_number"]] or
       case["check_seed"] != [seed_roots["check"], case["pair_index"],
                              case["configuration_number"]]
       for case in cases):
    raise ValueError("39例没有逐例绑定78冻结的 fit/check seed")
if {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cases} != snr_pairs:
    raise ValueError("78 manifest 的三个 EEG/MEG SNR 组合已变化")
for case in cases:
    patches = case["truth_surface_patch_indices_refined"]
    weights = case["truth_surface_patch_weights"]
    projection = case["evaluation_surface_projection"]
    mappings = projection.get("inverse_indices_by_patch", [])
    distances = case["truth_surface_patch_nearest_inverse_distance_mm"]
    if len(patches) != len(case["surface_centers"]) or \
            not (len(patches) == len(weights) == len(mappings) == len(distances)) or \
            projection.get("rule") != "nearest coarse surface vertex in HEAD coordinates" or \
            any(not (len(a) == len(b) == len(c) == len(d))
                    for a, b, c, d in zip(patches, weights, mappings, distances)):
        raise ValueError(f"{case['case_id']}: refined patch 与 coarse 映射不是逐点一一对应")


# %% 2. 锁定 coarse inverse、refined truth forward、图结构和算法设置。
truth_audit = audit.get("truth_forward", {})
refined_path = root / str(truth_audit.get("path", ""))
if not refined_path.is_file() or \
        hashlib.sha256(refined_path.read_bytes()).hexdigest() != \
        truth_audit.get("file_sha256"):
    raise FileNotFoundError("78 指向的 refined forward 缓存缺失或哈希变化")
refined = load_refined_forward(refined_path)
expected_refined = {
    "gain_eeg", "gain_meg", "vertices", "surface_xyz_head", "surface_xyz_mri",
    "deep_xyz_head", "deep_xyz_mri", "surface_edges", "n_surf", "n_deep",
    "mri_head_t", "eeg_ch_names", "meg_ch_names",
}
if not expected_refined.issubset(refined):
    raise ValueError(f"refined forward 缺少字段：{sorted(expected_refined - refined.keys())}")

shared = protocol.load_shared(
    root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
shared_fingerprint = original._shared_fingerprint(shared)
if shared_fingerprint != audit["inverse_forward"]["fingerprint"] or \
        any(case["inverse_forward_fingerprint"] != shared_fingerprint for case in cases):
    raise ValueError("coarse inverse forward 与78不一致")
n_surf = int(shared["n_surf"])
n_deep = int(shared["n_deep"])
n_sources = n_surf + n_deep
n_truth_surf = int(np.asarray(refined["n_surf"]).ravel()[0])
n_truth_deep = int(np.asarray(refined["n_deep"]).ravel()[0])
n_truth = n_truth_surf + n_truth_deep
surface_edges = np.asarray(refined["surface_edges"], int)
truth_adjacency = sparse.coo_matrix(
    (np.ones(surface_edges.shape[0] * 2, dtype=np.uint8),
     (np.r_[surface_edges[:, 0], surface_edges[:, 1]],
      np.r_[surface_edges[:, 1], surface_edges[:, 0]])),
    shape=(n_truth, n_truth)).tocsr()
truth_shared = dict(shared)
truth_shared.update(
    gain_eeg=np.asarray(refined["gain_eeg"], float),
    gain_meg=np.asarray(refined["gain_meg"], float),
    vertices=np.asarray(refined["vertices"], float),
    adjacency=truth_adjacency, n_surf=n_truth_surf, n_deep=n_truth_deep)
if truth_shared["gain_eeg"].shape != (shared["gain_eeg"].shape[0], n_truth) or \
        truth_shared["gain_meg"].shape != (shared["gain_meg"].shape[0], n_truth) or \
        truth_shared["vertices"].shape != (n_truth, 3):
    raise ValueError("refined truth forward 的通道或source轴与78不一致")

for case in cases:
    for center, expected_indices, expected_weights in zip(
            case["surface_centers"], case["truth_surface_patch_indices_refined"],
            case["truth_surface_patch_weights"]):
        indices, weights = protocol._surface_patch(truth_shared, int(center))
        if not np.array_equal(indices, np.asarray(expected_indices, int)) or \
                not np.allclose(weights, np.asarray(expected_weights, float),
                                rtol=0., atol=1e-12):
            raise ValueError(f"{case['case_id']}: 78冻结的 refined patch 已变化")

base_settings = {
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
strengths = (.5, .8)
gate_statistic_rule = "min(partial_A_to_B, partial_B_to_A)"
gate_decision_rule = "H1 iff statistic > max(6 gate_calibration H0 statistics)"

core_files = (
    "benchmark/erp_protocol.py", "benchmark/erp_replicates.py",
    "benchmark/erp_trial_covariance.py", "benchmark/metrics.py",
    "benchmark/protocol.py", "candidates/graph_reweight_solver.py",
    "candidates/oaster_balanced.py", "candidates/oaster_partial_confirmation.py",
    "candidates/oaster_predictive.py", "refined_grid.py",
    "run_erp_whole_head_matrix.py", "作者风格版/78-v3全新开发面板.py",
)
code_sha256 = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in core_files
}
code_bundle_sha256 = hashlib.sha256("\n".join(
    f"{name}:{digest}" for name, digest in code_sha256.items()
).encode("utf-8")).hexdigest()
script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
identity_path = output / "run_identity.json"
blind_script_sha256 = script_sha256
if args.evaluate_frozen:
    final_root = output / "final_cases"
    final_directories = ([] if not final_root.is_dir() else
                         list(final_root.glob("case_*")))
    if len(final_directories) != 39 or not identity_path.is_file() or \
            any(not (path / "decision.json").is_file() or
                    not (path / "final_map.npz").is_file()
                    for path in final_directories):
        raise ValueError("--evaluate-frozen 只允许续算完整冻结的39张final map")
    frozen_identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if frozen_identity.get("code_bundle_sha256") != code_bundle_sha256:
        raise ValueError("--evaluate-frozen 的核心代码bundle与原盲运行不一致")
    blind_script_sha256 = str(frozen_identity.get("script_sha256", ""))
    if len(blind_script_sha256) != 64:
        raise ValueError("原盲运行缺少合法script哈希")


# %% 3. preflight 只读核验；不创建结果或checkpoint。
if args.preflight:
    role_counts = Counter(case["h0_role"] for case in cases)
    scenario_counts = Counter(case["scenario"] for case in cases)
    if role_counts != {None: 27, "gate_calibration": 6, "gate_audit": 6} or \
            scenario_counts != {
                "surface_only": 12, "deep_only": 9,
                "deep_plus_surface": 9, "deep_plus_two_surface": 9}:
        raise ValueError("78 的 H0角色或四种情形计数已变化")
    print(json.dumps({
        "preflight": "PASS", "development_only": True,
        "formal_or_blind_validation": False, "case_count": len(cases),
        "gate_score_cases": 39, "gate_calibration_h0": 6,
        "gate_audit_h0_not_used_for_threshold": 6,
        "h1_not_used_for_threshold": 27,
        "gate_statistic": gate_statistic_rule,
        "gate_decision": gate_decision_rule,
        "mrf_strengths": list(strengths),
        "inverse_calls_planned": 39 * 6,
        "inverse_calls_per_case": {
            "bidirectional_gate_H0_H1": 4,
            "heldout_mrf_new_fit": 1,
            "heldout_mrf_08_reused_from_A_to_B_gate_fit": 1,
            "combined40_final": 1},
        "manifest_sha256": manifest_sha256, "audit_sha256": audit_sha256,
        "refined_forward_sha256": truth_audit["file_sha256"],
        "shared_fingerprint": shared_fingerprint,
        "code_bundle_sha256": code_bundle_sha256,
        "script_has_def_or_class": False,
        "checkpoint_resume": True, "output_exists": output.exists(),
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


# %% 4. 建立或核验断点身份；Stage 1 对39例只保存双向conditional分数。
identity = {
    "schema_version": 1,
    "protocol": "erp-v3-fresh-development-two-stage-runner",
    "development_only": True, "formal_or_blind_validation": False,
    "manifest": str(manifest_path), "manifest_sha256": manifest_sha256,
    "audit": str(audit_path), "audit_sha256": audit_sha256,
    "refined_forward": str(refined_path),
    "refined_forward_sha256": truth_audit["file_sha256"],
    "shared_fingerprint": shared_fingerprint, "seed_roots": seed_roots,
    "gate_statistic_rule": gate_statistic_rule,
    "gate_decision_rule": gate_decision_rule,
    "solver_settings": base_settings, "mrf_strengths": list(strengths),
    "code_sha256": code_sha256, "code_bundle_sha256": code_bundle_sha256,
    "script_sha256": blind_script_sha256,
    "truth_access_rule": (
        "The simulator receives each complete case, but Stage1 decision code retains "
        "only sensor observations and an identity; no scenario/SNR/source field enters "
        "a score. Only 6 gate_calibration H0 roles set the threshold. Audit-H0 and H1 "
        "labels/positions are read for metrics only after each final-map hash is fixed."),
}
identity_payload = (json.dumps(identity, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
identity_sha256 = hashlib.sha256(identity_payload).hexdigest()
if output.exists():
    if not identity_path.is_file() or identity_path.read_bytes() != identity_payload:
        raise ValueError("已有输出不是同一79脚本/manifest/forward/settings的可续跑checkpoint")
else:
    output.mkdir(parents=True)
    with identity_path.open("xb") as stream:
        stream.write(identity_payload)
(output / "gate_scores").mkdir(exist_ok=True)
(output / "final_cases").mkdir(exist_ok=True)
(output / "posthoc").mkdir(exist_ok=True)

started = time.perf_counter()
gate_score_sha256 = {}
for original_case in cases:
    number = int(original_case["case_number"])
    name = str(original_case["case_id"])
    score_path = output / "gate_scores" / f"case_{number:02d}.json"
    if score_path.is_file():
        score_payload = score_path.read_bytes()
        score_record = json.loads(score_payload)
        if score_record.get("complete") is not True or \
                score_record.get("case_number") != number or \
                score_record.get("case_id") != name or \
                score_record.get("identity_sha256") != identity_sha256 or \
                score_record.get("truth_or_label_used") is not False or \
                len(score_record.get("directions", [])) != 2 or \
                set(score_record.get("directions", [{}])[0].get(
                    "mrf_08_candidates", {})) != {"H0", "H1"} or \
                not np.isfinite(score_record.get("gate_statistic", np.nan)):
            raise ValueError(f"{name}: gate score checkpoint 非法，拒绝静默重算")
        gate_score_sha256[number] = hashlib.sha256(score_payload).hexdigest()
        print(name, "gate score checkpoint 已核验", flush=True)
        continue

    tick = time.perf_counter()
    observation = prepare_trial_covariance_case(
        shared, original_case, seed_root=seed_roots["fit"],
        truth_shared=truth_shared)
    metadata = observation["metadata"]
    if metadata.get("truth_grid") != original_case["truth_grid"] or \
            metadata.get("truth_source_count") != n_truth or \
            metadata.get("inverse_source_count") != n_sources or \
            metadata.get("replica_seed_roots") != seed_roots or \
            metadata.get("n_trials_per_half") != 20:
        raise RuntimeError(f"{name}: off-grid 20+20观测身份不一致")

    # 仿真器必须知道truth；从这里起只留下传感器数据和coarse inverse，决策变量不含标签。
    blind = {
        "training": np.asarray(observation["training"], float),
        "confirmation": np.asarray(observation["confirmation"], float),
        "gain": np.asarray(observation["gain"], float),
        "baseline": np.asarray(observation["baseline"], bool),
        "active": np.asarray(observation["active_windows"][0], bool),
        "modality_sizes": tuple(map(int, metadata["retained_channels"])),
    }
    del observation, metadata
    case = {"case_id": name, "case_number": number}
    if len(blind["modality_sizes"]) != 2 or \
            sum(blind["modality_sizes"]) != blind["gain"].shape[0] or \
            blind["gain"].shape[1] != n_sources:
        raise RuntimeError(f"{name}: EEG/MEG白化块或coarse gain非法")

    weights = np.ones(blind["gain"].shape[0], float)
    directions = []
    for direction, fit_key, check_key in (
            ("A_to_B", "training", "confirmation"),
            ("B_to_A", "confirmation", "training")):
        null, full, fitting = fit_predictive_models(
            blind[fit_key], blind["gain"], n_surf,
            adjacency=shared["adjacency"], baseline=blind["baseline"],
            active=blind["active"], channel_weights=weights,
            solver_settings=base_settings)
        solver_checks = {}
        for family in ("null", "full"):
            solver = fitting[family + "_model"]["windows"][0]["solver"]
            strict = bool(
                solver["converged"] and solver["outer_converged"]
                and solver["final_inner_converged"]
                and solver["final_stationarity_gap_relative"]
                <= base_settings["tolerance"] + 1e-12)
            solver_checks[family] = {
                "strict_converged": strict,
                "converged": bool(solver["converged"]),
                "outer_converged": bool(solver["outer_converged"]),
                "final_inner_converged": bool(solver["final_inner_converged"]),
                "outer_iterations": len(solver["history"]),
                "final_stationarity_gap_relative": float(
                    solver["final_stationarity_gap_relative"]),
            }
        if not all(item["strict_converged"] for item in solver_checks.values()):
            raise RuntimeError(f"{name} {direction}: conditional gate H0/H1未严格收敛")
        partial_score, partial = score_partial_deep_presence(
            blind[check_key], blind["gain"], null, full, n_surf,
            baseline=blind["baseline"], active=blind["active"],
            channel_weights=weights)
        fraction = partial["conditional_gain_fraction"]
        if not np.isfinite(partial_score) or \
                (fraction is not None and
                 (not np.isfinite(fraction) or not 0 <= fraction <= 1 + 1e-12)):
            raise RuntimeError(f"{name} {direction}: conditional score非法")
        heldout_candidates = {}
        heldout_basis = None
        if direction == "A_to_B":
            centered = blind[check_key] - blind[check_key][
                :, blind["baseline"]].mean(axis=1, keepdims=True)
            basis, heldout_basis = _smooth_temporal_basis(
                blind[check_key], blind["baseline"], blind["active"])
            response = centered @ basis.T
            mean_correction = float(
                np.sum(basis[:, blind["active"]].sum(axis=1) ** 2)
                / blind["baseline"].sum())
            noise_factor = len(basis) + mean_correction
            total_noise = float(np.sum(
                centered[:, blind["baseline"]] ** 2)
                / (blind["baseline"].sum() - 1) * noise_factor)
            for family, estimate, solver_family in (
                    ("H0", null, "null"), ("H1", full, "full")):
                prediction = blind["gain"] @ (estimate @ basis.T)
                total_loss = float(np.sum((response - prediction) ** 2))
                block_losses = []
                block_normalized_losses = []
                start = 0
                for size in blind["modality_sizes"]:
                    block = slice(start, start + size)
                    loss = float(np.sum(
                        (response[block] - prediction[block]) ** 2))
                    expected_noise = float(np.sum(
                        centered[block, blind["baseline"]] ** 2)
                        / (blind["baseline"].sum() - 1) * noise_factor)
                    if expected_noise <= 0 or \
                            not np.isfinite([loss, expected_noise]).all():
                        raise RuntimeError(
                            f"{name} {direction} {family}: MRF=0.8 held-out损失非法")
                    block_losses.append(loss)
                    block_normalized_losses.append(loss / expected_noise)
                    start += size
                if total_noise <= 0 or \
                        not np.isfinite([total_loss, total_noise]).all() or \
                        not np.isclose(sum(block_losses), total_loss,
                                       rtol=1e-12, atol=1e-12):
                    raise RuntimeError(
                        f"{name} {direction} {family}: MRF=0.8总损失非法")
                heldout_candidates[family] = {
                    "mrf_strength": .8,
                    "total_heldout_squared_loss": total_loss,
                    "total_noise_normalized_loss": total_loss / total_noise,
                    "eeg_heldout_squared_loss": block_losses[0],
                    "meg_heldout_squared_loss": block_losses[1],
                    "eeg_noise_normalized_loss": block_normalized_losses[0],
                    "meg_noise_normalized_loss": block_normalized_losses[1],
                    "solver": solver_checks[solver_family],
                    "reused_from_gate_fit": True,
                }
        directions.append({
            "direction": direction, "fit_half": fit_key,
            "held_out_half": check_key, "partial_score": float(partial_score),
            "selected_deep_index": partial["selected_deep_index"],
            "conditional_gain_fraction": fraction,
            "candidate_available": bool(partial["candidate_available"]),
            "identifiable": bool(partial["identifiable"]),
            "mrf_08_candidates": heldout_candidates,
            "mrf_08_temporal_basis": heldout_basis,
            "solver_checks": solver_checks,
        })
    statistic = float(min(item["partial_score"] for item in directions))
    score_record = {
        "complete": True, "stage": "gate_score_sealed",
        "case_number": number, "case_id": name,
        "identity_sha256": identity_sha256,
        "fit_check": "bidirectional independent 20-trial half means",
        "gate_statistic_rule": gate_statistic_rule,
        "gate_statistic": statistic, "directions": directions,
        "all_four_fits_strictly_converged": True,
        "truth_or_label_used": False,
        "scenario_snr_source_fields_used_by_score": False,
        "case_id_saved_as_identity_and_human_readable_only": True,
        "elapsed_seconds": float(time.perf_counter() - tick),
    }
    score_payload = (json.dumps(score_record, ensure_ascii=False, indent=2,
                                sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = score_path.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(score_payload)
    os.replace(temporary, score_path)
    gate_score_sha256[number] = hashlib.sha256(score_payload).hexdigest()
    print(name, f"sealed gate statistic={statistic:.6g}", flush=True)

if len(gate_score_sha256) != 39:
    raise RuntimeError("Stage1必须先完整冻结39例gate score")


# %% 5. 只解封6个gate_calibration H0角色；最大统计量冻结为严格大于阈值。
calibration_cases = [case for case in cases if case["h0_role"] == "gate_calibration"]
if len(calibration_cases) != 6 or \
        any(case["scenario"] != "surface_only" or case["deep_index"] is not None
            for case in calibration_cases):
    raise ValueError("gate calibration 必须且只能是78预声明的6个H0")
calibration_scores = []
for case in calibration_cases:
    number = int(case["case_number"])
    record = json.loads(
        (output / "gate_scores" / f"case_{number:02d}.json").read_text(
            encoding="utf-8"))
    calibration_scores.append({
        "case_number": number, "case_id": case["case_id"],
        "gate_statistic": float(record["gate_statistic"]),
        "gate_score_sha256": gate_score_sha256[number],
    })
gate_threshold = float(max(item["gate_statistic"] for item in calibration_scores))
threshold_record = {
    "complete": True, "stage": "gate_threshold_frozen",
    "development_only": True, "formal_or_blind_validation": False,
    "identity_sha256": identity_sha256,
    "gate_statistic_rule": gate_statistic_rule,
    "threshold_rule": "maximum statistic among 6 predeclared gate_calibration H0 cases",
    "decision_rule": "H1 iff statistic > threshold (strict)",
    "threshold": gate_threshold, "calibration_case_count": 6,
    "calibration_cases": calibration_scores,
    "audit_h0_labels_used": False, "h1_labels_used": False,
    "all_gate_score_sha256": {
        str(number): gate_score_sha256[number] for number in range(39)},
}
threshold_payload = (json.dumps(
    threshold_record, ensure_ascii=False, indent=2, sort_keys=True,
    allow_nan=False) + "\n").encode("utf-8")
threshold_path = output / "gate_threshold.json"
if threshold_path.is_file():
    if threshold_path.read_bytes() != threshold_payload:
        raise ValueError("已冻结gate threshold与当前39个score或6个calibration角色不一致")
else:
    temporary = threshold_path.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(threshold_payload)
    os.replace(temporary, threshold_path)
threshold_sha256 = hashlib.sha256(threshold_payload).hexdigest()
print(f"gate threshold已冻结：{gate_threshold:.9g}（仅6个calibration H0）", flush=True)


# %% 6. family固定后盲选MRF 0.5/0.8，再用combined40在同family联合重拟合并先落图哈希。
for original_case in cases:
    number = int(original_case["case_number"])
    name = str(original_case["case_id"])
    case_dir = output / "final_cases" / f"case_{number:02d}"
    decision_path = case_dir / "decision.json"
    map_path = case_dir / "final_map.npz"
    if case_dir.exists():
        if not decision_path.is_file() or not map_path.is_file():
            raise ValueError(f"{name}: final checkpoint目录不完整，拒绝猜测或覆盖")
        final_record = json.loads(decision_path.read_text(encoding="utf-8"))
        if final_record.get("complete") is not True or \
                final_record.get("identity_sha256") != identity_sha256 or \
                final_record.get("threshold_sha256") != threshold_sha256 or \
                final_record.get("case_id") != name or \
                hashlib.sha256(map_path.read_bytes()).hexdigest() != \
                final_record.get("final_map_sha256"):
            raise ValueError(f"{name}: final checkpoint身份或map哈希错误")
        print(name, "final checkpoint 已核验", flush=True)
        continue

    tick = time.perf_counter()
    gate_record = json.loads(
        (output / "gate_scores" / f"case_{number:02d}.json").read_text(
            encoding="utf-8"))
    statistic = float(gate_record["gate_statistic"])
    deep_present = bool(statistic > gate_threshold)
    family = "H1" if deep_present else "H0"

    observation = prepare_trial_covariance_case(
        shared, original_case, seed_root=seed_roots["fit"],
        truth_shared=truth_shared)
    metadata = observation["metadata"]
    if metadata.get("truth_grid") != original_case["truth_grid"] or \
            metadata.get("truth_source_count") != n_truth or \
            metadata.get("inverse_source_count") != n_sources:
        raise RuntimeError(f"{name}: Stage2 off-grid观测身份不一致")
    blind = {
        "training": np.asarray(observation["training"], float),
        "confirmation": np.asarray(observation["confirmation"], float),
        "gain": np.asarray(observation["gain"], float),
        "baseline": np.asarray(observation["baseline"], bool),
        "active": np.asarray(observation["active_windows"][0], bool),
        "active_samples": np.asarray(observation["active"], int),
        "modality_sizes": tuple(map(int, metadata["retained_channels"])),
    }
    del observation, metadata
    case = {"case_id": name, "case_number": number}
    branch_gain = blind["gain"] if deep_present else blind["gain"][:, :n_surf]
    branch_adjacency = (shared["adjacency"] if deep_present else
                        shared["adjacency"][:n_surf, :n_surf])
    weights = np.ones(blind["gain"].shape[0], float)

    centered = blind["confirmation"] - blind["confirmation"][:, blind["baseline"]].mean(
        axis=1, keepdims=True)
    basis, basis_info = _smooth_temporal_basis(
        blind["confirmation"], blind["baseline"], blind["active"])
    response = centered @ basis.T
    mean_correction = float(
        np.sum(basis[:, blind["active"]].sum(axis=1) ** 2)
        / blind["baseline"].sum())
    noise_factor = len(basis) + mean_correction
    stored_08 = dict(
        gate_record["directions"][0]["mrf_08_candidates"][family])
    if stored_08.get("mrf_strength") != .8 or \
            stored_08.get("reused_from_gate_fit") is not True or \
            not np.isfinite(stored_08.get(
                "total_heldout_squared_loss", np.nan)):
        raise RuntimeError(f"{name}: Stage1 MRF=0.8 held-out候选非法")
    candidates = [stored_08]
    for strength in (.5,):
        settings = {**base_settings, "mrf_strength": strength}
        branch_estimate, fitting = reconstruct_evoked_oaster_v5_from_whitened(
            blind["training"], branch_gain, n_surf, kernels=(),
            adjacency=branch_adjacency, baseline=blind["baseline"],
            active_windows=(blind["active"],),
            window_channel_weights=(weights,), require_one=False, **settings)
        solver = fitting["windows"][0]["solver"]
        strict = bool(
            solver["converged"] and solver["outer_converged"]
            and solver["final_inner_converged"]
            and solver["final_stationarity_gap_relative"]
            <= base_settings["tolerance"] + 1e-12)
        if not strict:
            raise RuntimeError(f"{name} MRF={strength:g}: training20未严格收敛")
        estimate = np.zeros((n_sources, blind["training"].shape[1]))
        estimate[:branch_estimate.shape[0]] = branch_estimate
        prediction = blind["gain"] @ (estimate @ basis.T)
        total_loss = float(np.sum((response - prediction) ** 2))
        baseline_variance = float(np.sum(
            centered[:, blind["baseline"]] ** 2) / (blind["baseline"].sum() - 1))
        total_noise = baseline_variance * noise_factor
        block_losses = []
        block_normalized_losses = []
        start = 0
        for size in blind["modality_sizes"]:
            block = slice(start, start + size)
            loss = float(np.sum((response[block] - prediction[block]) ** 2))
            variance = float(np.sum(
                centered[block, blind["baseline"]] ** 2)
                / (blind["baseline"].sum() - 1))
            expected_noise = variance * noise_factor
            if expected_noise <= 0 or not np.isfinite([loss, expected_noise]).all():
                raise RuntimeError(f"{name} MRF={strength:g}: held-out模态损失非法")
            block_losses.append(loss)
            block_normalized_losses.append(loss / expected_noise)
            start += size
        if total_noise <= 0 or not np.isfinite([total_loss, total_noise]).all() or \
                not np.isclose(sum(block_losses), total_loss, rtol=1e-12, atol=1e-12):
            raise RuntimeError(f"{name} MRF={strength:g}: held-out总损失非法")
        candidates.append({
            "mrf_strength": strength, "total_heldout_squared_loss": total_loss,
            "total_noise_normalized_loss": total_loss / total_noise,
            "eeg_heldout_squared_loss": block_losses[0],
            "meg_heldout_squared_loss": block_losses[1],
            "eeg_noise_normalized_loss": block_normalized_losses[0],
            "meg_noise_normalized_loss": block_normalized_losses[1],
            "solver": {
                "strict_converged": True, "converged": bool(solver["converged"]),
                "outer_converged": bool(solver["outer_converged"]),
                "final_inner_converged": bool(solver["final_inner_converged"]),
                "outer_iterations": len(solver["history"]),
                "final_stationarity_gap_relative": float(
                    solver["final_stationarity_gap_relative"]),
            },
            "reused_from_gate_fit": False,
        })
    loss_05 = next(item["total_heldout_squared_loss"] for item in candidates
                   if item["mrf_strength"] == .5)
    loss_08 = next(item["total_heldout_squared_loss"] for item in candidates
                   if item["mrf_strength"] == .8)
    selected_strength = .8 if loss_08 <= loss_05 else .5

    combined = (blind["training"] + blind["confirmation"]) / 2.
    final_settings = {**base_settings, "mrf_strength": selected_strength}
    branch_final, fitting = reconstruct_evoked_oaster_v5_from_whitened(
        combined, branch_gain, n_surf, kernels=(), adjacency=branch_adjacency,
        baseline=blind["baseline"], active_windows=(blind["active"],),
        window_channel_weights=(weights,), require_one=False, **final_settings)
    solver = fitting["windows"][0]["solver"]
    strict = bool(
        solver["converged"] and solver["outer_converged"]
        and solver["final_inner_converged"]
        and solver["final_stationarity_gap_relative"]
        <= base_settings["tolerance"] + 1e-12)
    if not strict:
        raise RuntimeError(f"{name}: combined40 {family} MRF={selected_strength:g}未严格收敛")
    selected = np.zeros((n_sources, combined.shape[1]))
    selected[:branch_final.shape[0]] = branch_final
    if not np.isfinite(selected).all() or \
            (family == "H0" and np.any(selected[n_surf:] != 0)):
        raise RuntimeError(f"{name}: final map非法")

    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f"case_{number:02d}.tmp-", dir=output / "final_cases"))
    temporary_map = temporary_dir / "final_map.npz"
    np.savez_compressed(
        temporary_map, selected=selected.astype(np.float32),
        active=blind["active_samples"], baseline=blind["baseline"],
        times=np.asarray(shared["times"], float))
    map_sha256 = hashlib.sha256(temporary_map.read_bytes()).hexdigest()
    final_record = {
        "complete": True, "stage": "final_map_frozen_before_truth_unseal",
        "case_number": number, "case_id": name,
        "identity_sha256": identity_sha256,
        "gate_score_sha256": gate_score_sha256[number],
        "threshold_sha256": threshold_sha256,
        "gate_statistic": statistic, "gate_threshold": gate_threshold,
        "strict_greater_than_threshold": True,
        "selected_family": family, "deep_present_decision": deep_present,
        "mrf_selection_rule": (
            "minimum confirmation20 all-whitened-channel raw squared loss; exact tie -> 0.8"),
        "mrf_candidates": candidates, "selected_mrf_strength": selected_strength,
        "temporal_basis": basis_info,
        "final_solver": {
            "strict_converged": True, "converged": bool(solver["converged"]),
            "outer_converged": bool(solver["outer_converged"]),
            "final_inner_converged": bool(solver["final_inner_converged"]),
            "outer_iterations": len(solver["history"]),
            "final_stationarity_gap_relative": float(
                solver["final_stationarity_gap_relative"]),
        },
        "final_map_file": "final_map.npz", "final_map_sha256": map_sha256,
        "truth_scenario_snr_source_positions_used_by_decision": False,
        "elapsed_seconds": float(time.perf_counter() - tick),
    }
    (temporary_dir / "decision.json").write_text(
        json.dumps(final_record, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary_dir, case_dir)
    print(name, family, f"MRF={selected_strength:g}",
          f"map={map_sha256[:12]}", flush=True)


# %% 7. 每例final map与决定固定后，才解封fine truth/角色/SNR并计算投影AUC和真实坐标SD/DLE。
coarse_surface_head = np.asarray(shared["vertices"], float)[:n_surf]
coarse_deep_head = np.asarray(shared["deep_rr"], float)
fine_vertices_head = np.asarray(refined["vertices"], float)
coarse_vertices_head = np.asarray(shared["vertices"], float)
auc_cortex_head = dict(shared["auc_cortex"])
auc_cortex_head["Vertices"] = coarse_vertices_head
if coarse_surface_head.shape != (n_surf, 3) or \
        coarse_deep_head.shape != (n_deep, 3) or \
        fine_vertices_head.shape != (n_truth, 3) or \
        coarse_vertices_head.shape != (n_sources, 3) or \
        not np.allclose(coarse_vertices_head[n_surf:], coarse_deep_head,
                        rtol=0., atol=1e-12):
    raise ValueError("真实HEAD坐标轴不完整")

for case in cases:
    number = int(case["case_number"])
    name = str(case["case_id"])
    posthoc_path = output / "posthoc" / f"case_{number:02d}.json"
    case_dir = output / "final_cases" / f"case_{number:02d}"
    decision_path = case_dir / "decision.json"
    map_path = case_dir / "final_map.npz"
    decision_sha256 = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if posthoc_path.is_file():
        posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
        if posthoc.get("complete") is not True or \
                posthoc.get("decision_sha256") != decision_sha256 or \
                posthoc.get("final_map_sha256") != decision["final_map_sha256"]:
            raise ValueError(f"{name}: posthoc checkpoint身份错误")
        print(name, "posthoc checkpoint 已核验", flush=True)
        continue

    if hashlib.sha256(map_path.read_bytes()).hexdigest() != decision["final_map_sha256"]:
        raise ValueError(f"{name}: truth解封前final map哈希变化")
    with np.load(map_path, allow_pickle=False) as archive:
        if set(archive.files) != {"selected", "active", "baseline", "times"}:
            raise ValueError(f"{name}: final map字段非法")
        selected = np.asarray(archive["selected"], float)
        active_samples = np.asarray(archive["active"], int)
        baseline = np.asarray(archive["baseline"], bool)
    if selected.shape != (n_sources, len(shared["times"])) or \
            not np.isfinite(selected).all():
        raise ValueError(f"{name}: final map形状或有限性错误")

    # 这里才重新生成并读取fine truth；decision/map/checkpoint已不可变。
    observation = prepare_trial_covariance_case(
        shared, case, seed_root=seed_roots["fit"], truth_shared=truth_shared)
    truth = np.asarray(observation["truth"], float)
    groups = [np.asarray(group, int) for group in observation["groups"]]
    if truth.shape != (n_truth, selected.shape[1]) or \
            not np.array_equal(np.asarray(observation["active"], int), active_samples):
        raise RuntimeError(f"{name}: 解封truth与冻结map时间轴不一致")
    expected_groups = [np.asarray(group, int)
                       for group in case["truth_surface_patch_indices_refined"]]
    if case["deep_index"] is not None:
        expected_groups.append(np.asarray([case["deep_index"]], int))
    if len(groups) != len(expected_groups) or \
            any(not np.array_equal(a, b) for a, b in zip(groups, expected_groups)):
        raise RuntimeError(f"{name}: fine truth groups与78映射不一致")

    projected_truth = np.zeros_like(selected)
    projected_groups = []
    for group, mapping in zip(
            groups[:len(case["surface_centers"])],
            case["evaluation_surface_projection"]["inverse_indices_by_patch"]):
        mapping = np.asarray(mapping, int)
        if mapping.shape != group.shape or np.any(mapping < 0) or np.any(mapping >= n_surf):
            raise ValueError(f"{name}: surface fine→coarse AUC映射非法")
        for fine_index, coarse_index in zip(group, mapping):
            projected_truth[coarse_index] += truth[fine_index]
        projected_groups.append(np.unique(mapping))
    if case["deep_index"] is not None:
        if n_truth_surf + int(case["truth_deep_fine_index"]) != \
                int(case["deep_index"]):
            raise ValueError(f"{name}: deep truth fine identity非法")
        coarse_deep = int(case["evaluation_deep_index"])
        if not n_surf <= coarse_deep < n_sources:
            raise ValueError(f"{name}: deep fine→coarse AUC映射非法")
        projected_truth[coarse_deep] += truth[int(case["deep_index"])]
        projected_groups.append(np.asarray([coarse_deep], int))
    mapped_values = metrics.evaluate_estimate(
        selected, projected_truth, coarse_vertices_head,
        projected_groups, n_surf, active_samples, auc_cortex_head,
        baseline=baseline)

    energy = np.sum(selected[:, active_samples] ** 2, axis=1)
    support = np.zeros(n_sources, bool)
    for start, stop in ((0, n_surf), (n_surf, n_sources)):
        peak = float(energy[start:stop].max(initial=0.))
        if peak > 0:
            support[start:stop] = energy[start:stop] > .1 * peak
    surface_groups = [group for group in groups if np.all(group < n_truth_surf)]
    deep_groups = [group for group in groups if np.all(group >= n_truth_surf)]
    surface_sd = surface_dle = deep_sd = deep_dle = np.nan
    if surface_groups and np.any(support[:n_surf]):
        surface_sd, surface_dle = refined_sd_dle(
            selected[:n_surf, active_samples], support[:n_surf],
            coarse_surface_head, fine_vertices_head, surface_groups)
    if deep_groups and np.any(support[n_surf:]):
        deep_sd, deep_dle = refined_sd_dle(
            selected[n_surf:, active_samples], support[n_surf:],
            coarse_deep_head, fine_vertices_head, deep_groups)

    surface_component_dle = []
    if surface_groups and np.any(support[:n_surf]):
        true_indices = np.concatenate(surface_groups)
        true_owner = np.concatenate([
            np.full(len(group), index, int)
            for index, group in enumerate(surface_groups)])
        owner = true_owner[cKDTree(fine_vertices_head[true_indices]).query(
            coarse_surface_head)[1]]
        surface_energy = energy[:n_surf]
        for index, group in enumerate(surface_groups):
            region = np.flatnonzero((owner == index) & support[:n_surf])
            if region.size:
                peak = int(region[np.argmax(surface_energy[region])])
                center = fine_vertices_head[group].mean(axis=0)
                surface_component_dle.append(float(
                    np.linalg.norm(coarse_surface_head[peak] - center) * 1000.))
            else:
                surface_component_dle.append(None)
    else:
        surface_component_dle = [None] * len(surface_groups)
    finite_component = [value for value in surface_component_dle
                        if value is not None and np.isfinite(value)]
    component_max = (max(finite_component, default=None) if
                     len(finite_component) == len(surface_groups) else None)

    amplitude = metrics.source_amplitude(selected, active_samples, baseline)
    amplitude_peak = float(amplitude.max(initial=0.))
    deep_peak = float(amplitude[n_surf:].max(initial=0.))
    deep_score = deep_peak / amplitude_peak if amplitude_peak > 0 else 0.
    true_has_deep = bool(deep_groups)
    deep_distance = np.nan
    if deep_peak > 0 and true_has_deep:
        predicted = coarse_deep_head[int(np.argmax(amplitude[n_surf:]))]
        true_positions = np.vstack([fine_vertices_head[group] for group in deep_groups])
        deep_distance = float(
            np.min(np.linalg.norm(true_positions - predicted, axis=1)) * 1000.)
    deep_positive = bool(deep_score >= metrics.DEEP_DETECTION_THRESHOLD)
    deep_detected = bool(
        true_has_deep and deep_positive and np.isfinite(deep_distance)
        and deep_distance <= 10. + 1e-6)

    clean_metrics = {}
    for key in ("auc", "auc_tie_corrected", "surface_auc_tie_corrected",
                "deep_auc_tie_corrected", "rmse", "active_count",
                "surface_active_count", "deep_active_count"):
        value = mapped_values[key]
        clean_metrics[key] = (None if isinstance(value, (float, np.floating))
                              and not np.isfinite(value)
                              else value.item() if isinstance(value, np.generic) else value)
    clean_metrics.update({
        "surface_sd_mm_true_fine_head": None if not np.isfinite(surface_sd) else surface_sd,
        "surface_dle_mm_true_fine_head": None if not np.isfinite(surface_dle) else surface_dle,
        "surface_component_dle_mm_true_fine_head": surface_component_dle,
        "surface_component_dle_max_mm_true_fine_head": component_max,
        "deep_sd_mm_true_fine_head": None if not np.isfinite(deep_sd) else deep_sd,
        "deep_dle_mm_true_fine_head": None if not np.isfinite(deep_dle) else deep_dle,
        "has_deep_true": int(true_has_deep), "deep_score": deep_score,
        "deep_peak_distance_mm_true_fine_head": (
            None if not np.isfinite(deep_distance) else deep_distance),
        "deep_detected": int(deep_detected),
        "deep_false_positive": int((not true_has_deep) and deep_positive),
    })
    posthoc = {
        "complete": True, "stage": "posthoc_truth_unsealed_after_final_map",
        "case_number": number, "case_id": name,
        "identity_sha256": identity_sha256,
        "decision_sha256": decision_sha256,
        "final_map_sha256": decision["final_map_sha256"],
        "selected_family": decision["selected_family"],
        "selected_mrf_strength": decision["selected_mrf_strength"],
        "gate_statistic": decision["gate_statistic"],
        "gate_threshold": decision["gate_threshold"],
        "scenario": case["scenario"], "h0_role": case["h0_role"],
        "alias_tier": case["alias_tier"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "truth_grid": case["truth_grid"],
        "auc_truth_projection": "78 frozen per-fine-source nearest coarse mapping",
        "distance_metric_coordinate_frame": "HEAD; no projected truth positions",
        "metrics": clean_metrics,
    }
    payload = (json.dumps(posthoc, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = posthoc_path.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    os.replace(temporary, posthoc_path)
    print(name, case["scenario"],
          f"AUC={clean_metrics['auc_tie_corrected']:.3f}", flush=True)


# %% 8. 从不可变checkpoint汇总；calibration H0与audit H0严格分开报告。
rows = []
details = []
for case in cases:
    number = int(case["case_number"])
    decision_path = output / "final_cases" / f"case_{number:02d}" / "decision.json"
    posthoc_path = output / "posthoc" / f"case_{number:02d}.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
    values = posthoc["metrics"]
    row = {
        "case_number": number, "case_id": case["case_id"],
        "scenario": case["scenario"], "h0_role": case["h0_role"],
        "alias_tier": case["alias_tier"],
        "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "gate_statistic": decision["gate_statistic"],
        "gate_threshold": decision["gate_threshold"],
        "deep_present_decision": int(decision["deep_present_decision"]),
        "selected_family": decision["selected_family"],
        "selected_mrf_strength": decision["selected_mrf_strength"],
        "mrf_05_heldout_loss": next(
            item["total_heldout_squared_loss"] for item in decision["mrf_candidates"]
            if item["mrf_strength"] == .5),
        "mrf_08_heldout_loss": next(
            item["total_heldout_squared_loss"] for item in decision["mrf_candidates"]
            if item["mrf_strength"] == .8),
        "final_map_sha256": decision["final_map_sha256"],
        "global_An_auc": values["auc_tie_corrected"],
        "requested_An_cal_auc": values["auc"],
        "surface_An_auc": values["surface_auc_tie_corrected"],
        "deep_An_auc": values["deep_auc_tie_corrected"],
        "surface_SD_mm": values["surface_sd_mm_true_fine_head"],
        "surface_DLE_mm": values["surface_dle_mm_true_fine_head"],
        "surface_component_DLE_max_mm":
            values["surface_component_dle_max_mm_true_fine_head"],
        "deep_SD_mm": values["deep_sd_mm_true_fine_head"],
        "deep_DLE_mm": values["deep_dle_mm_true_fine_head"],
        "deep_peak_distance_mm": values["deep_peak_distance_mm_true_fine_head"],
        "has_deep_true": values["has_deep_true"],
        "deep_detected": values["deep_detected"],
        "deep_false_positive": values["deep_false_positive"],
        "surface_active_count": values["surface_active_count"],
        "deep_active_count": values["deep_active_count"],
    }
    rows.append(row)
    details.append({
        "decision": decision, "posthoc": posthoc,
        "gate_score_sha256": gate_score_sha256[number],
        "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
        "posthoc_sha256": hashlib.sha256(posthoc_path.read_bytes()).hexdigest(),
    })

calibration_h0 = [row for row in rows if row["h0_role"] == "gate_calibration"]
audit_h0 = [row for row in rows if row["h0_role"] == "gate_audit"]
h1 = [row for row in rows if row["has_deep_true"]]
summary = {
    "complete": True, "development_only": True,
    "formal_or_blind_validation": False, "case_count": len(rows),
    "gate_threshold": gate_threshold,
    "gate_threshold_calibration_h0_count": len(calibration_h0),
    "gate_calibration_h0_false_positive_count": sum(
        row["deep_present_decision"] for row in calibration_h0),
    "gate_audit_h0_count": len(audit_h0),
    "gate_audit_h0_false_positive_count": sum(
        row["deep_present_decision"] for row in audit_h0),
    "h1_count": len(h1),
    "h1_family_detection_count": sum(row["deep_present_decision"] for row in h1),
    "h1_localization_detection_count": sum(row["deep_detected"] for row in h1),
    "minimum_global_An_auc": min(row["global_An_auc"] for row in rows),
    "median_global_An_auc": float(np.median(
        [row["global_An_auc"] for row in rows])),
    "mrf_selection_counts": dict(Counter(
        str(row["selected_mrf_strength"]) for row in rows)),
    "all_gate_and_final_solvers_strictly_converged": True,
    "audit_h0_and_h1_truth_unsealed_only_after_each_final_map_hash": True,
    "calibration_h0_roles_unsealed_before_final_maps_for_threshold": True,
    "distance_metrics_use_true_fine_HEAD_coordinates": True,
    "auc_only_uses_frozen_fine_to_coarse_projection": True,
    "wall_seconds_this_invocation": float(time.perf_counter() - started),
}
metadata = {
    **identity, "identity_sha256": identity_sha256,
    "evaluation_script_sha256": script_sha256,
    "frozen_evaluation_resume": bool(args.evaluate_frozen),
    "threshold_sha256": threshold_sha256,
    "threshold_calibration_role": "6 predeclared gate_calibration pure-surface cases only",
    "gate_audit_and_h1_role": "not used by threshold; posthoc development audit only",
    "checkpoint_layout": {
        "gate_scores": "one sealed score JSON per case",
        "gate_threshold": "one frozen marker bound to all score hashes",
        "final_cases": "one atomic directory per case with truth-free decision and map",
        "posthoc": "one JSON per case created only after map hash verification",
    },
}

for path, value in ((output / "metadata.json", metadata),
                    (output / "details.json", details),
                    (output / "summary.json", summary)):
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
rows_path = output / "rows.csv"
temporary_rows = rows_path.with_suffix(".csv.tmp")
with temporary_rows.open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
os.replace(temporary_rows, rows_path)

report = [
    "# v3 全新开发结果", "",
    "本目录仅是 fresh development，不是正式校准或 blind validation。", "",
    f"- gate statistic：`{gate_statistic_rule}`。",
    f"- 阈值：6个预声明 calibration H0 的最大值 `{gate_threshold:.9g}`；严格 `>` 才报H1。",
    f"- calibration H0误报：{summary['gate_calibration_h0_false_positive_count']}/6；"
    f"audit H0误报：{summary['gate_audit_h0_false_positive_count']}/6。",
    f"- H1 family检出：{summary['h1_family_detection_count']}/27；"
    f"同时满足10 mm深源定位：{summary['h1_localization_detection_count']}/27。",
    f"- 全局An_auc最小/中位：{summary['minimum_global_An_auc']:.3f}/"
    f"{summary['median_global_An_auc']:.3f}。",
    "- AUC使用78冻结的fine→coarse映射；SD、DLE和深源距离直接使用真实fine HEAD坐标。", "",
]
(output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
completion = {
    "complete": True, "development_only": True,
    "case_count": 39, "gate_score_checkpoint_count": 39,
    "final_map_checkpoint_count": 39, "posthoc_checkpoint_count": 39,
    "threshold_sha256": threshold_sha256,
    "evaluation_script_sha256": script_sha256,
    "frozen_evaluation_resume": bool(args.evaluate_frozen),
    "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
    "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
}
(output / "completion.json").write_text(
    json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True,
               allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
