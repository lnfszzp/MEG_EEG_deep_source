"""# %% 层级 OASTER 正式流水线：两次独立校准后一次性验证。"""

# %% 1. 命令行、执行锁和不可变输入。正式运行没有病例筛选参数。
import argparse
import atexit
from collections import Counter
import csv
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import methods, metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import (
    _smooth_temporal_basis,
    reconstruct_evoked_oaster_v5_from_whitened,
)
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--stage",
    choices=("gate-calibration", "surface-calibration", "validation"),
    required=True,
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--resume", action="store_true", help="只续跑已严格核验的完整病例前缀")
parser.add_argument(
    "--lock",
    type=Path,
    default=root / "results/erp_whole_head/adaptive_v6/protocol/hierarchical_v1_execution_lock.json",
)
args = parser.parse_args()
if not __debug__:
    raise RuntimeError("正式运行禁止 python -O；输入与哈希核验依赖 assert 语义之外的调试状态")

stage = args.stage
output = args.output.resolve()
lock_path = args.lock if args.lock.is_absolute() else root / args.lock
lock_path = lock_path.resolve()
if output.exists() and not args.resume:
    raise FileExistsError(f"不覆盖已有正式输出：{output}")
if args.resume and not output.is_dir():
    raise FileNotFoundError(f"--resume 要求已有正式输出目录：{output}")
if not lock_path.is_file():
    raise FileNotFoundError(f"执行锁不存在：{lock_path}")
lock_sidecar = Path(str(lock_path) + ".sha256")
if not lock_sidecar.is_file():
    raise FileNotFoundError(f"执行锁缺少 SHA256 旁车：{lock_sidecar}")
lock_bytes = lock_path.read_bytes()
lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
if lock_sidecar.read_text(encoding="ascii").split()[0] != lock_sha256:
    raise ValueError("执行锁与 SHA256 旁车不一致")
execution_lock = json.loads(lock_bytes)
recovery = execution_lock.get("interrupted_prefix_recovery")
if execution_lock.get("schema_version") != 1 or \
        execution_lock.get("protocol") != "erp-v6-hierarchical-v1-execution" or \
        execution_lock.get("status") not in {
            "algorithm_frozen_calibration_allowed",
            "interrupted_prefix_recovery_allowed",
        } or \
        execution_lock.get("formal_cases_run_by_generator") is not False:
    raise ValueError("不是允许一次性正式运行的 hierarchical-v1 执行锁")
if args.resume and execution_lock.get("status") != "interrupted_prefix_recovery_allowed":
    raise ValueError("--resume 只能使用显式绑定中断前缀的恢复执行锁")
if args.resume and (not isinstance(recovery, dict) or recovery.get("stage") != stage):
    raise ValueError("--resume 的 stage 必须与恢复锁冻结的中断 stage 完全一致")

manifests = execution_lock.get("manifests", {})
if set(manifests) != {"gate-calibration", "surface-calibration", "validation"}:
    raise ValueError("执行锁必须且只能登记三个正式 stage")
locked_markers = []
locked_outputs = []
for locked_stage in manifests.values():
    marker = Path(locked_stage.get("consumed_marker", ""))
    marker = marker if marker.is_absolute() else root / marker
    locked_markers.append(marker.resolve())
    locked_stage_output = Path(locked_stage.get("output_dir", ""))
    locked_stage_output = (locked_stage_output if locked_stage_output.is_absolute()
                           else root / locked_stage_output)
    locked_outputs.append(locked_stage_output.resolve())
if len(set(locked_markers)) != 3 or len(set(locked_outputs)) != 3:
    raise ValueError("三个 stage 必须使用互不相同的 marker 和输出目录")
stage_lock = manifests[stage]
expected_case_count = {
    "gate-calibration": 57,
    "surface-calibration": 57,
    "validation": 21,
}[stage]
if stage_lock.get("case_count") != expected_case_count:
    raise ValueError(f"{stage} 必须锁定 {expected_case_count} 例")
locked_output = Path(stage_lock.get("output_dir", ""))
locked_output = locked_output if locked_output.is_absolute() else root / locked_output
if output != locked_output.resolve():
    raise ValueError(f"--output 必须与执行锁一致：{locked_output.resolve()}")

run_lock_path = output.with_name(output.name + ".run.lock")
run_lock_stream = run_lock_path.open("a+b")
run_lock_stream.seek(0, os.SEEK_END)
if run_lock_stream.tell() == 0:
    run_lock_stream.write(b"\0")
    run_lock_stream.flush()
run_lock_stream.seek(0)
try:
    msvcrt.locking(run_lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
except OSError as error:
    run_lock_stream.close()
    raise RuntimeError(f"同一正式 stage 已有进程运行：{run_lock_path}") from error
atexit.register(run_lock_path.unlink, missing_ok=True)
atexit.register(run_lock_stream.close)
atexit.register(msvcrt.locking, run_lock_stream.fileno(), msvcrt.LK_UNLCK, 1)

for locked_stage_name, locked_stage in manifests.items():
    locked_manifest = locked_stage.get("manifest", {})
    locked_manifest_path = Path(locked_manifest.get("path", ""))
    locked_manifest_path = (locked_manifest_path if locked_manifest_path.is_absolute()
                            else root / locked_manifest_path)
    locked_manifest_sidecar = Path(locked_manifest.get("sidecar_path", ""))
    locked_manifest_sidecar = (
        locked_manifest_sidecar if locked_manifest_sidecar.is_absolute()
        else root / locked_manifest_sidecar)
    if not locked_manifest_path.is_file() or \
            hashlib.sha256(locked_manifest_path.read_bytes()).hexdigest() != \
            locked_manifest.get("sha256"):
        raise ValueError(f"执行锁后 manifest 已变化：{locked_stage_name}")
    if not locked_manifest_sidecar.is_file() or hashlib.sha256(
            locked_manifest_sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() != \
            locked_manifest.get("sidecar_text_sha256") or \
            locked_manifest_sidecar.read_text(encoding="ascii").split()[0] != \
            locked_manifest.get("sha256"):
        raise ValueError(f"执行锁后 manifest 旁车已变化：{locked_stage_name}")

protocol_inputs = execution_lock.get("protocol_inputs", {})
if set(protocol_inputs) != {"protocol_lock", "protocol_document"}:
    raise ValueError("执行锁必须同时保护 protocol lock 与 protocol document")
for item_name, locked in protocol_inputs.items():
    path = Path(locked.get("path", ""))
    path = path if path.is_absolute() else root / path
    sidecar = Path(locked.get("sidecar_path", ""))
    sidecar = sidecar if sidecar.is_absolute() else root / sidecar
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != locked.get("sha256"):
        raise ValueError(f"执行锁保护输入已变化：{item_name}")
    if not sidecar.is_file() or hashlib.sha256(
            sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() != \
            locked.get("sidecar_text_sha256"):
        raise ValueError(f"执行锁保护旁车已变化：{item_name}")
    if sidecar.read_text(encoding="ascii").split()[0] != locked.get("sha256"):
        raise ValueError(f"执行锁保护旁车没有绑定正文：{item_name}")

locked_code_hashes = execution_lock.get("code_sha256", {})
if not isinstance(locked_code_hashes, dict) or not locked_code_hashes:
    raise ValueError("执行锁缺少代码哈希")
required_code_hashes = {
    "作者风格版/64-层级盲定位正式清单.py",
    "作者风格版/65-层级盲定位正式运行.py",
    "candidates/oaster_predictive.py",
    "candidates/oaster_balanced.py",
    "candidates/graph_reweight_solver.py",
    "candidates/graph_irls.py",
    "benchmark/protocol.py",
    "benchmark/erp_protocol.py",
    "benchmark/erp_replicates.py",
    "benchmark/erp_trial_covariance.py",
    "benchmark/metrics.py",
    "benchmark/methods.py",
    "benchmark/deep_acceptance.py",
    "run_erp_whole_head_matrix.py",
    "run_strict_oaster.py",
    "algorithms/spatial_fused_fusion.py",
    "protected_multilayer.py",
    "metrics/user_metrics/An_auc.py",
    "metrics/user_metrics/DLE_an.py",
    "metrics/user_metrics/SD.py",
    "metrics/user_metrics/_common.py",
}
if not required_code_hashes.issubset(locked_code_hashes):
    missing = sorted(required_code_hashes - set(locked_code_hashes))
    raise ValueError(f"执行锁遗漏正式链路代码哈希：{missing}")
for relative, expected in locked_code_hashes.items():
    path = Path(relative)
    path = path if path.is_absolute() else root / path
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"执行锁后代码已变化：{path}")

algorithm = execution_lock.get("algorithm", {})
solver_settings = algorithm.get("source_model_settings", {})
expected_solver_settings = {
    "solver_kind": "admm",
    "mrf_strength": .8,
    "outer_iterations": 40,
    "max_inner_retries": 20,
    "epsilon_fraction": .05,
    "tolerance": .001,
    "outer_tolerance": .01,
    "surface_reweight_floor": .5,
    "deep_reweight_floor": .5,
    "edge_weight_floor": .5,
}
if algorithm.get("covariance") != "trial" or solver_settings != expected_solver_settings:
    raise ValueError("正式算法必须是 trial covariance + outer40/retry20 bounded-MM")
conditional_surface_settings = algorithm.get("conditional_surface_settings", {})
expected_conditional_surface_settings = {
    "require_one": False,
    "edge_fraction": .5,
    "noise_multiplier": 1.,
    "calibration": "layer",
    "temporal_mode": "smooth",
    "max_iter": 2000,
    "rho": 1.,
    "adaptive_rho": True,
    "ridge_fraction": 0.,
    "edge_penalty_mode": "group",
    "source_penalty_mode": "group",
    "deep_alias_penalty": False,
}
if conditional_surface_settings != expected_conditional_surface_settings:
    raise ValueError("条件表层模型的结构与数值设置未完整锁定")
if algorithm.get("replicates") != {
        "training_trials": 20, "confirmation_trials": 20, "combined_refit_trials": 40}:
    raise ValueError("正式 train/check/combined 试次数与执行锁不一致")

gate_lock = algorithm.get("gate", {})
gate_weights = np.asarray(gate_lock.get("weights"), float)
gate_calibration_rule = gate_lock.get("calibration", {})
if gate_lock.get("score_kind") != "fixed_3_to_1_linear_fusion" or \
        gate_lock.get("modality_order") != ["EEG", "MEG"] or \
        gate_weights.shape != (2,) or not np.array_equal(gate_weights, [.75, .25]) or \
        gate_lock.get("uses_confirmation_only") is not True or \
        gate_lock.get("test_true_snr_used") is not False or \
        gate_calibration_rule.get("score_count") != 57 or \
        gate_calibration_rule.get("statistical_claim") != \
        "pooled fixed-design finite-rank engineering threshold" or \
        gate_calibration_rule.get("iid_claim") is not False or \
        gate_calibration_rule.get("fpr_guarantee") is not False or \
        gate_calibration_rule.get("required_complete_finite_scores") != 57 or \
        gate_calibration_rule.get("invalid_if_any_solver_unconverged") is not True or \
        gate_calibration_rule.get("equivalent_boundary") != \
        "T strictly greater than max(0, second-largest calibration score)" or \
        gate_calibration_rule.get("ties_count_against_acceptance") is not True:
    raise ValueError("一级门的固定 3:1 分数或校准规则不一致")

surface_gate_lock = algorithm.get("surface_presence", {})
surface_calibration_rule = surface_gate_lock.get("calibration", {})
if surface_gate_lock.get("score_kind") != "conjunctive_modality_noise_gain" or \
        surface_gate_lock.get("formula") != "min(EEG, MEG) held-out D versus D+S improvement" or \
        surface_gate_lock.get("closed_primary_score") != 0 or \
        surface_calibration_rule.get("score_count") != 57 or \
        surface_calibration_rule.get("statistical_claim") != \
        "pooled fixed-design finite-rank engineering threshold" or \
        surface_calibration_rule.get("iid_claim") is not False or \
        surface_calibration_rule.get("fpr_guarantee") is not False or \
        surface_calibration_rule.get("required_complete_finite_scores") != 57 or \
        surface_calibration_rule.get("invalid_if_any_solver_unconverged") is not True or \
        surface_calibration_rule.get("equivalent_boundary") != \
        "T strictly greater than max(0, maximum calibration score)" or \
        surface_calibration_rule.get("ties_count_against_acceptance") is not True or \
        surface_calibration_rule.get("test_true_snr_used") is not False:
    raise ValueError("二级表层门的分数或保守统一阈值规则不一致")

hierarchy_lock = algorithm.get("hierarchy", {})
for key, expected in {
    "deep_location": "argmax sum over EEG,MEG modality-normalized leave-one-deep-out confirmation improvements",
    "h0_localization": "combined40 MRF-innovation BIC dipoles then R^-1 physical surface map",
    "h1_surface_selection": "training20 fixed selected deep; W=1 bounded-MM surface; frozen surface calibration",
    "final_refit": "freeze family/deep/surface-presence first; combined40 refits amplitudes only",
    "truth_access": "only after final estimate is frozen",
}.items():
    if hierarchy_lock.get(key) != expected:
        raise ValueError(f"层级算法锁字段不一致：{key}")

acceptance = execution_lock.get("acceptance", {})
required_acceptance = {
    "minimum_global_auc", "minimum_surface_auc", "maximum_surface_dle_mm",
    "maximum_surface_component_dle_mm", "maximum_surface_sd_mm",
    "maximum_deep_distance_mm", "h0_false_positives_per_snr_max",
    "h1_detected_total_min", "h1_case_count", "h1_detected_per_snr_min",
    "deep_only_surface_false_count_max",
    "all_selection_and_final_solvers_converged",
}
if set(acceptance) != required_acceptance or \
        not all(np.isfinite(float(acceptance[key])) for key in required_acceptance
                if key != "all_selection_and_final_solvers_converged") or \
        acceptance.get("all_selection_and_final_solvers_converged") is not True or \
        acceptance.get("h1_case_count") != 9:
    raise ValueError("执行锁 acceptance 字段不完整或含未审计规则")

current_environment = original._environment_versions()
current_environment["python_debug"] = __debug__
if execution_lock.get("environment") != current_environment:
    raise ValueError("正式运行环境与执行锁不一致")

shared_lock = execution_lock.get("shared", {})
generated_dir = Path(shared_lock.get("generated_dir", ""))
generated_dir = generated_dir if generated_dir.is_absolute() else root / generated_dir
sample_path = Path(shared_lock.get("sample_path", ""))
sample_path = sample_path if sample_path.is_absolute() else root / sample_path
shared = protocol.load_shared(generated_dir, sample_path)
if original._shared_fingerprint(shared) != shared_lock.get("fingerprint"):
    raise ValueError("共享 forward 与执行锁不一致")
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"], float)
adjacency = shared["adjacency"]


# %% 2. 当前 manifest、旧校准封存和一次性 consumed marker。
manifest_lock = stage_lock.get("manifest", {})
manifest = Path(manifest_lock.get("path", ""))
manifest = manifest if manifest.is_absolute() else root / manifest
manifest_sidecar = Path(manifest_lock.get("sidecar_path", ""))
manifest_sidecar = manifest_sidecar if manifest_sidecar.is_absolute() else root / manifest_sidecar
if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_lock.get("sha256"):
    raise ValueError("当前 stage manifest 与执行锁不一致")
if not manifest_sidecar.is_file() or hashlib.sha256(
        manifest_sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() != \
        manifest_lock.get("sidecar_text_sha256"):
    raise ValueError("当前 stage manifest 旁车与执行锁不一致")
if manifest_sidecar.read_text(encoding="ascii").split()[0] != manifest_lock.get("sha256"):
    raise ValueError("当前 stage manifest 旁车没有绑定 manifest")

cases = json.loads(manifest.read_text(encoding="utf-8"))
if not isinstance(cases, list) or len(cases) != expected_case_count or \
        len({case.get("case_id") for case in cases}) != len(cases):
    raise ValueError("manifest 病例数或 case_id 唯一性错误")
seed_roots = stage_lock.get("seed_roots")
if not isinstance(seed_roots, dict) or set(seed_roots) != {"fit", "check"} or \
        seed_roots["check"] != seed_roots["fit"] + 1:
    raise ValueError("manifest 必须锁定相邻的 fit/check roots")
if any(case.get("seed", [None])[0] != seed_roots["fit"] or
       case.get("replica_seed_roots") != seed_roots for case in cases):
    raise ValueError("manifest 病例没有使用锁定 seed roots")

snr_pairs = [tuple(pair) for pair in execution_lock.get("snr_pairs_eeg_meg_db", [])]
if len(snr_pairs) != 3 or len(set(snr_pairs)) != 3:
    raise ValueError("执行锁必须登记三个 EEG/MEG SNR 组合")
cases_per_snr = stage_lock.get("cases_per_snr")
expected_manifest_shape = {
    "gate-calibration": {
        "cases_per_snr": 19, "configuration_count": 19,
        "configuration_repetition": 3,
    },
    "surface-calibration": {
        "cases_per_snr": 19, "configuration_count": 57,
        "configuration_repetition": 1,
    },
    "validation": {
        "cases_per_snr": 7, "configuration_count": 21,
        "configuration_repetition": 1,
    },
}[stage]
if any(stage_lock.get(key) != value
       for key, value in expected_manifest_shape.items()):
    raise ValueError("manifest 的病例/SNR/固定几何重复结构与正式协议不一致")
if Counter((case.get("eeg_snr_db"), case.get("meg_snr_db")) for case in cases) != \
        Counter({pair: cases_per_snr for pair in snr_pairs}):
    raise ValueError("manifest 的 SNR 分层与执行锁不一致")
configurations = Counter(case.get("configuration_id") for case in cases)
if len(configurations) != stage_lock.get("configuration_count") or \
        any(value != stage_lock.get("configuration_repetition")
            for value in configurations.values()):
    raise ValueError("manifest 的配置独立性/重复结构与执行锁不一致")
if stage == "gate-calibration" and not all(
        case.get("scenario") == "surface_only" and case.get("deep_index") is None
        and bool(case.get("surface_centers")) for case in cases):
    raise ValueError("一级门校准必须是 57 个纯表层病例")
if stage == "gate-calibration" and any(
        Counter((case.get("eeg_snr_db"), case.get("meg_snr_db"))
                for case in cases if case.get("configuration_id") == configuration)
        != Counter(snr_pairs) for configuration in configurations):
    raise ValueError("一级门的19个固定几何必须各自完整跨越三个SNR")
if stage == "surface-calibration" and not all(
        case.get("scenario") == "deep_only" and case.get("deep_index") is not None
        and not case.get("surface_centers") for case in cases):
    raise ValueError("二级表层门校准必须是 57 个纯深层病例")
if stage == "validation" and Counter(case.get("scenario") for case in cases) != \
        Counter(stage_lock.get("scenario_counts", {})):
    raise ValueError("validation 场景组成与执行锁不一致")
if stage == "validation" and any(
        Counter(case.get("scenario") for case in cases
                if (case.get("eeg_snr_db"), case.get("meg_snr_db")) == pair)
        != Counter({"surface_only": 4, "deep_only": 1,
                    "deep_plus_surface": 1, "deep_plus_two_surface": 1})
        for pair in snr_pairs):
    raise ValueError("validation 每个SNR必须固定为4个H0和3个H1")

calibrations = {}
prior_stages = (() if stage == "gate-calibration" else
                ("gate-calibration",) if stage == "surface-calibration" else
                ("gate-calibration", "surface-calibration"))
frozen_names = {
    "gate-calibration": "frozen_gate_calibration.json",
    "surface-calibration": "frozen_surface_calibration.json",
}
for prior_stage in prior_stages:
    prior_lock = manifests[prior_stage]
    prior_manifest_path = Path(prior_lock.get("manifest", {}).get("path", ""))
    prior_manifest_path = (prior_manifest_path if prior_manifest_path.is_absolute()
                           else root / prior_manifest_path).resolve()
    marker_path = Path(prior_lock.get("consumed_marker", ""))
    marker_path = marker_path if marker_path.is_absolute() else root / marker_path
    if not marker_path.is_file():
        raise FileNotFoundError(f"缺少前序 consumed marker：{marker_path}")
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    expected_prior_output = Path(prior_lock.get("output_dir", ""))
    expected_prior_output = (expected_prior_output if expected_prior_output.is_absolute()
                             else root / expected_prior_output).resolve()
    expected_marker_lock_sha256 = (
        recovery.get("predecessor_execution_lock_sha256")
        if isinstance(recovery, dict) and prior_stage == recovery.get("stage")
        else lock_sha256)
    if marker.get("stage") != prior_stage or \
            marker.get("execution_lock_sha256") != expected_marker_lock_sha256 or \
            marker.get("manifest_sha256") != prior_lock.get("manifest", {}).get("sha256") or \
            Path(marker.get("output", "")).resolve() != expected_prior_output:
        raise ValueError(f"前序 {prior_stage} marker 未绑定当前执行锁/manifest/output")
    expected_prior_hashes = {
        name: value["sha256"] for name, value in calibrations.items()}
    if marker.get("prior_calibration_sha256") != expected_prior_hashes:
        raise ValueError(f"前序 {prior_stage} marker 没有绑定当时已有的校准")
    seal_path = expected_prior_output / "calibration_seal.json"
    seal_sidecar = Path(str(seal_path) + ".sha256")
    if not seal_path.is_file() or not seal_sidecar.is_file():
        raise FileNotFoundError(f"前序 {prior_stage} 没有一次性校准 seal")
    seal_bytes = seal_path.read_bytes()
    seal_sha256 = hashlib.sha256(seal_bytes).hexdigest()
    if seal_sidecar.read_text(encoding="ascii").split()[0] != seal_sha256:
        raise ValueError(f"前序 {prior_stage} seal 与旁车不一致")
    seal = json.loads(seal_bytes)
    expected_output_paths = {
        "metadata": expected_prior_output / "metadata.json",
        "evidence": expected_prior_output / "evidence.csv",
        "completion": expected_prior_output / "completion.json",
        "frozen_calibration": expected_prior_output / frozen_names[prior_stage],
    }
    if seal.get("schema_version") != 1 or \
            seal.get("protocol") != "erp-v6-hierarchical-v1-calibration-seal" or \
            seal.get("stage") != prior_stage or \
            seal.get("status") != "calibration_frozen_next_stage_allowed" or \
            seal.get("execution_lock") != {
                "path": str(lock_path), "sha256": lock_sha256} or \
            seal.get("manifest") != {
                "path": str(prior_manifest_path),
                "sha256": prior_lock.get("manifest", {}).get("sha256")} or \
            seal.get("consumed_marker") != {
                "path": str(marker_path),
                "sha256": hashlib.sha256(marker_bytes).hexdigest()} or \
            seal.get("prior_calibration_sha256") != expected_prior_hashes or \
            set(seal.get("outputs", {})) != set(expected_output_paths):
        raise ValueError(f"前序 {prior_stage} seal 没有完整绑定正式输入")
    for output_name, expected_path in expected_output_paths.items():
        sealed_output = seal["outputs"][output_name]
        if Path(sealed_output.get("path", "")).resolve() != expected_path.resolve() or \
                not expected_path.is_file() or \
                hashlib.sha256(expected_path.read_bytes()).hexdigest() != \
                sealed_output.get("sha256"):
            raise ValueError(f"前序 {prior_stage} seal 输出已变化：{output_name}")
    frozen_path = expected_prior_output / frozen_names[prior_stage]
    frozen_sidecar = Path(str(frozen_path) + ".sha256")
    if not frozen_path.is_file() or not frozen_sidecar.is_file():
        raise FileNotFoundError(f"前序 {prior_stage} 没有有效冻结校准")
    frozen_bytes = frozen_path.read_bytes()
    frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
    if frozen_sidecar.read_text(encoding="ascii").split()[0] != frozen_sha256:
        raise ValueError(f"前序 {prior_stage} 校准与旁车不一致")
    frozen = json.loads(frozen_bytes)
    scores = np.asarray(frozen.get("scores", []), float)
    prior_cases = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    prior_case_ids = [case.get("case_id") for case in prior_cases]
    evidence_path = expected_prior_output / "evidence.csv"
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as stream:
        prior_evidence = list(csv.DictReader(stream))
    evidence_score_key = ("gate_score" if prior_stage == "gate-calibration"
                          else "secondary_surface_score")
    try:
        evidence_scores = np.asarray(
            [float(row[evidence_score_key]) for row in prior_evidence], float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"前序 {prior_stage} evidence 分数不可复算") from error
    if prior_stage == "gate-calibration":
        expected_threshold = max(0., float(np.sort(scores)[-2])) if scores.size == 57 else np.nan
    else:
        expected_threshold = max(0., float(scores.max())) if scores.size == 57 else np.nan
    if frozen.get("valid") is not True or frozen.get("complete") is not True or \
            frozen.get("stage") != prior_stage or frozen.get("case_count") != 57 or \
            scores.shape != (57,) or not np.isfinite(scores).all() or \
            frozen.get("execution_lock_sha256") != lock_sha256 or \
            frozen.get("manifest_sha256") != prior_lock.get("manifest", {}).get("sha256") or \
            frozen.get("seed_roots") != prior_lock.get("seed_roots") or \
            frozen.get("case_ids") != prior_case_ids or len(set(prior_case_ids)) != 57 or \
            [row.get("case_id") for row in prior_evidence] != prior_case_ids or \
            any(row.get("status") != "ok" or row.get("all_converged") != "1"
                for row in prior_evidence) or \
            not np.array_equal(scores, evidence_scores) or \
            frozen.get("code_sha256") != locked_code_hashes or \
            frozen.get("shared_fingerprint") != shared_lock.get("fingerprint") or \
            frozen.get("environment") != current_environment or \
            frozen.get("consumed_marker_sha256") != hashlib.sha256(marker_bytes).hexdigest() or \
            not np.isclose(float(frozen.get("threshold", np.nan)), expected_threshold,
                           rtol=0., atol=np.finfo(float).eps * 8) or \
            frozen.get("comparison") != "strict_greater":
        raise ValueError(f"前序 {prior_stage} 校准不完整、已变化或阈值不可复算")
    expected_checks = {
        "case_count": 57,
        "complete": True,
        "valid": True,
        "all_converged": True,
        "finite_score_count": 57,
    }
    if seal.get("checks") != expected_checks or seal.get("threshold") != {
            "value": expected_threshold,
            "construction": frozen.get("threshold_construction"),
            "comparison": "strict_greater",
    }:
        raise ValueError(f"前序 {prior_stage} seal 的校准结论不可复算")
    calibrations[prior_stage] = {
        "payload": frozen,
        "path": str(frozen_path),
        "sha256": frozen_sha256,
        "seal_path": str(seal_path),
        "seal_sha256": seal_sha256,
        "threshold": expected_threshold,
    }

prior_case_ids = set()
for calibration in calibrations.values():
    overlap = prior_case_ids & set(calibration["payload"]["case_ids"])
    if overlap:
        raise ValueError("两套校准病例发生重叠")
    prior_case_ids.update(calibration["payload"]["case_ids"])
if prior_case_ids & {case["case_id"] for case in cases}:
    raise ValueError("当前 manifest 与前序校准病例发生重叠")

marker_path = Path(stage_lock.get("consumed_marker", ""))
marker_path = marker_path if marker_path.is_absolute() else root / marker_path
if not marker_path.parent.is_dir():
    raise FileNotFoundError(f"consumed marker 目录不存在：{marker_path.parent}")
resumed_evidence_rows = []
if args.resume:
    if not marker_path.is_file():
        raise FileNotFoundError(f"--resume 缺少原 consumed marker：{marker_path}")
    forbidden_final = [
        output / "completion.json", output / "calibration_seal.json",
        output / "invalid_calibration.json",
        output / frozen_names[stage],
    ]
    if any(path.exists() for path in forbidden_final):
        raise FileExistsError("已有 completion/frozen/seal 的 stage 禁止续跑")
    marker_bytes = marker_path.read_bytes()
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    marker_payload = json.loads(marker_bytes)
    recovery_stage = isinstance(recovery, dict) and stage == recovery.get("stage")
    expected_marker_lock = (
        recovery.get("predecessor_execution_lock_sha256")
        if recovery_stage else lock_sha256)
    if marker_payload.get("stage") != stage or \
            marker_payload.get("execution_lock_sha256") != expected_marker_lock or \
            marker_payload.get("manifest_sha256") != manifest_lock["sha256"] or \
            Path(marker_payload.get("output", "")).resolve() != output or \
            marker_payload.get("prior_calibration_sha256") != {
                name: value["sha256"] for name, value in calibrations.items()}:
        raise ValueError("原 consumed marker 未绑定当前 stage/manifest/output/前序校准")
    evidence_path = output / "evidence.csv"
    if not evidence_path.is_file():
        raise FileNotFoundError("--resume 缺少 evidence.csv")
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        raw_evidence_rows = list(reader)
        evidence_columns = reader.fieldnames
    expected_evidence_columns = [
        "case_id", "scenario", "eeg_snr_db", "meg_snr_db", "gate_score",
        "gate_eeg_score", "gate_meg_score", "gate_threshold",
        "deep_present_decision", "selected_deep_index", "secondary_surface_score",
        "surface_threshold", "surface_present_decision", "null_converged",
        "full_converged", "selection_surface_converged", "final_surface_converged",
        "all_converged", "status", "error", "elapsed_seconds",
    ]
    prefix_count = len(raw_evidence_rows)
    if evidence_columns != expected_evidence_columns or not 0 < prefix_count < len(cases) or \
            [row.get("case_id") for row in raw_evidence_rows] != [
                case["case_id"] for case in cases[:prefix_count]]:
        raise ValueError("evidence 必须是 manifest 的非空严格完整前缀")
    if recovery_stage:
        frozen_count = int(recovery.get("completed_prefix_count", -1))
        if prefix_count != frozen_count or \
                int(recovery.get("remaining_case_count", -1)) != len(cases) - frozen_count or \
                marker_sha256 != recovery.get("consumed_marker_sha256"):
            raise ValueError("当前前缀长度/剩余病例数或 marker 与恢复锁不一致")
        expected_snapshot_paths = {
            "metadata": output / "metadata.interrupted_prefix.json",
            "evidence": output / "evidence.interrupted_prefix.csv",
        }
        if set(recovery.get("snapshots", {})) != set(expected_snapshot_paths):
            raise ValueError("恢复锁必须且只能登记 metadata/evidence 两份中断快照")
        for snapshot_name in ("metadata", "evidence"):
            snapshot = recovery.get("snapshots", {}).get(snapshot_name, {})
            snapshot_path = Path(snapshot.get("path", ""))
            snapshot_path = snapshot_path if snapshot_path.is_absolute() else root / snapshot_path
            if snapshot_path.resolve() != expected_snapshot_paths[snapshot_name].resolve() or \
                    not snapshot_path.is_file() or hashlib.sha256(
                    snapshot_path.read_bytes()).hexdigest() != snapshot.get("sha256"):
                raise ValueError(f"中断快照已变化：{snapshot_name}")
        snapshot_path = Path(recovery["snapshots"]["evidence"]["path"])
        snapshot_path = snapshot_path if snapshot_path.is_absolute() else root / snapshot_path
        with snapshot_path.open("r", encoding="utf-8-sig", newline="") as stream:
            frozen_prefix = list(csv.DictReader(stream))
        if len(frozen_prefix) != frozen_count or raw_evidence_rows != frozen_prefix:
            raise ValueError("当前 evidence 不再包含恢复锁冻结的原始前缀")
        expected_detail_ids = {case["case_id"] for case in cases[:frozen_count]}
        case_details = recovery.get("case_details", {})
        if set(case_details) != expected_detail_ids:
            raise ValueError("恢复锁必须逐例登记完整中断前缀的详情哈希")
        for case_id, specification in case_details.items():
            detail_path = Path(specification.get("path", ""))
            detail_path = detail_path if detail_path.is_absolute() else root / detail_path
            expected_detail_path = output / f"{case_id}.json"
            if detail_path.resolve() != expected_detail_path.resolve() or \
                    not detail_path.is_file() or hashlib.sha256(
                    detail_path.read_bytes()).hexdigest() != specification.get("sha256"):
                raise ValueError(f"中断前病例详情已变化：{case_id}")
    existing_metadata_path = output / "metadata.json"
    existing_metadata_bytes = existing_metadata_path.read_bytes()
    existing_metadata_sha256 = hashlib.sha256(existing_metadata_bytes).hexdigest()
    frozen_metadata_sha256 = (
        recovery.get("snapshots", {}).get("metadata", {}).get("sha256")
        if recovery_stage else None)
    if existing_metadata_sha256 != frozen_metadata_sha256:
        existing_metadata = json.loads(existing_metadata_bytes)
        if existing_metadata.get("execution_lock_sha256") != lock_sha256 or \
                existing_metadata.get("stage") != stage or \
                (recovery_stage and existing_metadata.get("resumed") is not True):
            raise ValueError("已有 metadata 既不是冻结中断快照，也不是当前恢复锁续跑记录")
    integer_fields = {
        "eeg_snr_db", "meg_snr_db", "deep_present_decision", "selected_deep_index",
        "surface_present_decision", "null_converged", "full_converged",
        "selection_surface_converged", "final_surface_converged", "all_converged",
    }
    float_fields = {
        "gate_score", "gate_eeg_score", "gate_meg_score", "gate_threshold",
        "secondary_surface_score", "surface_threshold", "elapsed_seconds",
    }
    for raw_row, case in zip(raw_evidence_rows, cases):
        row = dict(raw_row)
        for key in integer_fields:
            row[key] = None if row[key] == "" else int(row[key])
        for key in float_fields:
            row[key] = None if row[key] == "" else float(row[key])
        row["error"] = None if row["error"] == "" else row["error"]
        detail_path = output / (case["case_id"] + ".json")
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        score_key = "gate_score" if stage == "gate-calibration" else "secondary_surface_score"
        if row["status"] != "ok" or row["all_converged"] != 1 or row["error"] is not None or \
                detail.get("case_id") != case["case_id"] or detail.get("stage") != stage or \
                detail.get("complete") is not True or \
                detail.get("truth_used_by_decisions") is not False or \
                not np.isclose(float(detail.get(score_key, np.nan)), float(row[score_key]),
                               rtol=0., atol=0., equal_nan=False):
            raise ValueError(f"已有完整前缀不可复核：{case['case_id']}")
        resumed_evidence_rows.append(row)
else:
    if marker_path.exists():
        raise FileExistsError(f"该正式 stage 已消费，禁止换目录重跑：{marker_path}")
    marker_payload = {
        "schema_version": 1,
        "protocol": "erp-v6-hierarchical-v1-consumed-marker",
        "stage": stage,
        "execution_lock": str(lock_path),
        "execution_lock_sha256": lock_sha256,
        "manifest": str(manifest),
        "manifest_sha256": manifest_lock["sha256"],
        "seed_roots": seed_roots,
        "output": str(output),
        "prior_calibration_sha256": {
            name: value["sha256"] for name, value in calibrations.items()},
    }
    marker_serialized = (json.dumps(
        marker_payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")) + "\n").encode("utf-8")
    with marker_path.open("xb") as stream:
        stream.write(marker_serialized)
    marker_bytes = marker_path.read_bytes()
    marker_sha256 = hashlib.sha256(marker_bytes).hexdigest()
    output.mkdir(parents=True)

gate_threshold = (None if stage == "gate-calibration"
                  else calibrations["gate-calibration"]["threshold"])
surface_threshold = (calibrations.get("surface-calibration", {}).get("threshold")
                     if stage == "validation" else None)
metadata = {
    "schema_version": 1,
    "protocol": "erp-v6-hierarchical-v1-formal-run",
    "stage": stage,
    "execution_lock": str(lock_path),
    "execution_lock_sha256": lock_sha256,
    "manifest": str(manifest),
    "manifest_sha256": manifest_lock["sha256"],
    "seed_roots": seed_roots,
    "case_ids": [case["case_id"] for case in cases],
    "consumed_marker": str(marker_path),
    "consumed_marker_sha256": marker_sha256,
    "code_sha256": locked_code_hashes,
    "shared_fingerprint": shared_lock["fingerprint"],
    "environment": current_environment,
    "covariance": algorithm["covariance"],
    "solver_settings": solver_settings,
    "gate_weights_EEG_MEG": gate_weights.tolist(),
    "gate_calibration": calibrations.get("gate-calibration"),
    "surface_calibration": calibrations.get("surface-calibration"),
    "gate_threshold": gate_threshold,
    "surface_threshold": surface_threshold,
    "truth_access": "metrics only after the blind final estimate is frozen",
    "resumed": bool(args.resume),
    "resumed_prefix_case_count": len(resumed_evidence_rows),
    "interrupted_prefix_recovery": recovery if args.resume else None,
}
(output / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    encoding="utf-8")

surface_graph = sparse.csr_matrix(adjacency[:n_surf, :n_surf])
surface_graph = surface_graph.maximum(surface_graph.T)
surface_graph.setdiag(0)
surface_graph.eliminate_zeros()
surface_degree = np.asarray(surface_graph.sum(axis=1)).ravel()
surface_transition = sparse.diags(1 / np.maximum(surface_degree, 1)) @ surface_graph
surface_mrf = sparse.eye(n_surf) - solver_settings["mrf_strength"] * surface_transition
surface_mrf_factor = splu(surface_mrf.tocsc())


# %% 3. 每例先完成盲决策；validation 到最终图冻结后才读取真值与指标。
evidence_rows = resumed_evidence_rows
rows = []
failures = []
started = time.perf_counter()
for case in cases[len(evidence_rows):]:
    tick = time.perf_counter()
    name = case["case_id"]
    evidence_row = {
        "case_id": name,
        "scenario": case.get("scenario"),
        "eeg_snr_db": case.get("eeg_snr_db"),
        "meg_snr_db": case.get("meg_snr_db"),
        "gate_score": None,
        "gate_eeg_score": None,
        "gate_meg_score": None,
        "gate_threshold": gate_threshold,
        "deep_present_decision": None,
        "selected_deep_index": None,
        "secondary_surface_score": None,
        "surface_threshold": surface_threshold,
        "surface_present_decision": None,
        "null_converged": 0,
        "full_converged": 0,
        "selection_surface_converged": None,
        "final_surface_converged": None,
        "all_converged": 0,
        "status": "error",
        "error": None,
        "elapsed_seconds": None,
    }
    detail = {
        "case_id": name,
        "stage": stage,
        "complete": False,
        "truth_used_by_decisions": False,
        "truth_used_only_for_posthoc_metrics": stage == "validation",
    }
    try:
        for relative, expected in locked_code_hashes.items():
            path = Path(relative)
            path = path if path.is_absolute() else root / path
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"运行中代码发生变化：{path}")

        observation = prepare_trial_covariance_case(
            shared, case, seed_root=seed_roots["fit"])
        baseline = np.asarray(observation["baseline"], bool)
        active_window = np.asarray(observation["active_windows"][0], bool)
        active_samples = np.asarray(observation["active"], int)
        gain = np.asarray(observation["gain"], float)
        weights = np.asarray(observation["channel_weights"], float)
        modality_sizes = tuple(
            int(value) for value in observation["metadata"]["retained_channels"])

        null, full, fitting = fit_predictive_models(
            observation["training"], gain, n_surf,
            adjacency=adjacency, baseline=baseline, active=active_window,
            channel_weights=weights, solver_settings=solver_settings)
        model_solvers = {
            family: fitting[family + "_model"]["windows"][0]["solver"]
            for family in ("null", "full")}
        model_convergence = {
            family: bool(
                solver["converged"] and solver["outer_converged"]
                and solver["final_inner_converged"]
                and solver["final_stationarity_gap_relative"]
                <= solver_settings["tolerance"] + 1e-12)
            for family, solver in model_solvers.items()}
        evidence_row["null_converged"] = int(model_convergence["null"])
        evidence_row["full_converged"] = int(model_convergence["full"])
        if not all(model_convergence.values()):
            raise RuntimeError("一级 H0/H1 模型没有全部到达固定点")

        _, gate_info = score_predictive_models(
            observation["confirmation"], gain, null, full,
            baseline=baseline, active=active_window, channel_weights=weights,
            modality_sizes=modality_sizes)
        gate_modality_scores = np.asarray(gate_info["modality_noise_scores"], float)
        if gate_modality_scores.shape != (2,) or not np.isfinite(gate_modality_scores).all():
            raise RuntimeError("一级 EEG/MEG 分数缺失或非有限")
        gate_score = float(gate_weights @ gate_modality_scores)
        if not np.isfinite(gate_score):
            raise RuntimeError("一级融合分数非有限")
        evidence_row.update({
            "gate_score": gate_score,
            "gate_eeg_score": float(gate_modality_scores[0]),
            "gate_meg_score": float(gate_modality_scores[1]),
        })
        detail.update({
            "gate_score": gate_score,
            "gate_modality_scores": gate_modality_scores.tolist(),
            "gate_model_convergence": model_convergence,
        })

        if stage == "gate-calibration":
            evidence_row.update({"all_converged": 1, "status": "ok"})
            detail["complete"] = True
        else:
            deep_present = bool(gate_score > gate_threshold)
            evidence_row["deep_present_decision"] = int(deep_present)
            detail["deep_present_decision"] = deep_present
            selected_deep = None
            amplitude_deep = None
            loo_candidate_indices = None
            loo_by_modality = None
            loo_sum = None
            secondary_score = 0. if not deep_present else None
            secondary_modalities = None
            surface_solver = None
            final_surface_solver = None
            surface_solver_ok = True
            h0_innovation_indices = None

            if not deep_present:
                if stage == "surface-calibration":
                    evidence_row.update({
                        "secondary_surface_score": 0.,
                        "surface_present_decision": 0,
                        "all_converged": 1,
                        "status": "ok",
                    })
                    detail.update({
                        "secondary_surface_score": 0.,
                        "secondary_rule": "primary gate closed: end-to-end null score is zero",
                        "complete": True,
                    })
                else:
                    combined = (observation["training"] + observation["confirmation"]) / 2
                    innovation_gain = surface_mrf_factor.solve(
                        gain[:, :n_surf].T, trans="T").T
                    innovation = methods.dipole_fit(
                        combined, innovation_gain, active_samples)
                    h0_innovation_indices = np.flatnonzero(
                        np.linalg.norm(innovation[:, active_samples], axis=1) > 0)
                    surface = surface_mrf_factor.solve(innovation)
                    np.testing.assert_allclose(
                        gain[:, :n_surf] @ surface,
                        innovation_gain @ innovation,
                        rtol=2e-10, atol=2e-10)
                    selected = np.zeros_like(full)
                    selected[:n_surf] = surface
                    selected_family = "H0 combined40 MRF-innovation dipole"
                    evidence_row.update({
                        "secondary_surface_score": 0.,
                        "surface_present_decision": 0,
                        "all_converged": 1,
                    })
            else:
                basis, _ = _smooth_temporal_basis(
                    observation["training"], baseline, active_window)
                full_modes = full @ basis.T
                loo_candidate_indices = n_surf + np.flatnonzero(
                    np.linalg.norm(full_modes[n_surf:], axis=1) > 0)
                if not loo_candidate_indices.size:
                    raise RuntimeError("一级 H1 图中没有活动深源候选")
                loo_by_modality = np.zeros(
                    (len(modality_sizes), len(loo_candidate_indices)))
                for local, deep_index in enumerate(loo_candidate_indices):
                    without = full.copy()
                    without[deep_index] = 0
                    _, contribution = score_predictive_models(
                        observation["confirmation"], gain, without, full,
                        baseline=baseline, active=active_window,
                        channel_weights=weights, modality_sizes=modality_sizes)
                    loo_by_modality[:, local] = contribution["modality_noise_scores"]
                if not np.isfinite(loo_by_modality).all():
                    raise RuntimeError("深源 LOO 模态分数非有限")
                loo_sum = loo_by_modality.sum(axis=0)
                selected_deep = int(loo_candidate_indices[np.argmax(loo_sum)])
                full_amplitude = metrics.source_amplitude(full, active_samples, baseline)
                amplitude_deep = n_surf + int(np.argmax(full_amplitude[n_surf:]))
                evidence_row["selected_deep_index"] = selected_deep

                centered = (observation["training"]
                            - observation["training"][:, baseline].mean(
                                axis=1, keepdims=True))
                weighted_response = (weights[:, None] * centered) @ basis.T
                weighted_deep_gain = weights * gain[:, selected_deep]
                denominator = float(weighted_deep_gain @ weighted_deep_gain)
                if not np.isfinite(denominator) or denominator <= 0:
                    raise RuntimeError("所选深源在训练坐标中没有灵敏度")
                deep_modes = (weighted_deep_gain @ weighted_response) / denominator
                deep = np.zeros_like(full)
                deep[selected_deep] = deep_modes @ basis
                np.testing.assert_allclose(
                    deep[selected_deep] @ basis.T, deep_modes,
                    rtol=1e-10, atol=1e-10)

                residual = (observation["training"]
                            - gain[:, [selected_deep]] @ deep[[selected_deep]])
                surface, surface_info = reconstruct_evoked_oaster_v5_from_whitened(
                    residual, gain[:, :n_surf], n_surf, kernels=(),
                    adjacency=adjacency[:n_surf, :n_surf], baseline=baseline,
                    active_windows=(active_window,),
                    window_channel_weights=(np.ones(gain.shape[0]),),
                    require_one=conditional_surface_settings["require_one"],
                    edge_fraction=conditional_surface_settings["edge_fraction"],
                    noise_multiplier=conditional_surface_settings["noise_multiplier"],
                    mrf_strength=solver_settings["mrf_strength"],
                    calibration=conditional_surface_settings["calibration"],
                    temporal_mode=conditional_surface_settings["temporal_mode"],
                    solver_kind=solver_settings["solver_kind"],
                    surface_reweight_floor=solver_settings["surface_reweight_floor"],
                    deep_reweight_floor=solver_settings["deep_reweight_floor"],
                    ridge_fraction=conditional_surface_settings["ridge_fraction"],
                    edge_penalty_mode=conditional_surface_settings["edge_penalty_mode"],
                    surface_penalty_multiplier=None,
                    source_penalty_mode=conditional_surface_settings["source_penalty_mode"],
                    deep_alias_penalty=conditional_surface_settings["deep_alias_penalty"],
                    outer_iterations=solver_settings["outer_iterations"],
                    max_iter=conditional_surface_settings["max_iter"],
                    tolerance=solver_settings["tolerance"],
                    epsilon_fraction=solver_settings["epsilon_fraction"],
                    rho=conditional_surface_settings["rho"],
                    outer_tolerance=solver_settings["outer_tolerance"],
                    adaptive_rho=conditional_surface_settings["adaptive_rho"],
                    max_inner_retries=solver_settings["max_inner_retries"],
                    edge_weight_floor=solver_settings["edge_weight_floor"])
                surface_solver = surface_info["windows"][0]["solver"]
                surface_solver_ok = bool(
                    surface_solver["converged"]
                    and surface_solver["outer_converged"]
                    and surface_solver["final_inner_converged"]
                    and surface_solver["final_stationarity_gap_relative"]
                    <= solver_settings["tolerance"] + 1e-12)
                evidence_row["selection_surface_converged"] = int(surface_solver_ok)
                if not surface_solver_ok:
                    raise RuntimeError("train20 条件表层拟合未到固定点")
                surface_full = np.zeros_like(deep)
                surface_full[:n_surf] = surface
                mixed = deep + surface_full
                _, secondary = score_predictive_models(
                    observation["confirmation"], gain, deep, mixed,
                    baseline=baseline, active=active_window,
                    channel_weights=np.ones(gain.shape[0]),
                    modality_sizes=modality_sizes)
                secondary_modalities = np.asarray(
                    secondary["modality_noise_scores"], float)
                secondary_score = float(secondary["conjunctive_modality_score"])
                if secondary_modalities.shape != (2,) or \
                        not np.isfinite(secondary_modalities).all() or \
                        not np.isfinite(secondary_score):
                    raise RuntimeError("二级表层分数缺失或非有限")
                evidence_row["secondary_surface_score"] = secondary_score

                if stage == "surface-calibration":
                    evidence_row.update({
                        "surface_present_decision": None,
                        "all_converged": 1,
                        "status": "ok",
                    })
                    detail["complete"] = True
                else:
                    surface_present = bool(secondary_score > surface_threshold)
                    evidence_row["surface_present_decision"] = int(surface_present)
                    combined = (observation["training"] + observation["confirmation"]) / 2
                    final_basis, _ = _smooth_temporal_basis(
                        combined, baseline, active_window)
                    final_centered = combined - combined[:, baseline].mean(
                        axis=1, keepdims=True)
                    final_response = (weights[:, None] * final_centered) @ final_basis.T
                    final_deep_modes = (weighted_deep_gain @ final_response) / denominator
                    final_deep = np.zeros_like(full)
                    final_deep[selected_deep] = final_deep_modes @ final_basis
                    selected = final_deep
                    if surface_present:
                        final_residual = (combined - gain[:, [selected_deep]]
                                          @ final_deep[[selected_deep]])
                        final_surface, final_surface_info = \
                            reconstruct_evoked_oaster_v5_from_whitened(
                                final_residual, gain[:, :n_surf], n_surf, kernels=(),
                                adjacency=adjacency[:n_surf, :n_surf], baseline=baseline,
                                active_windows=(active_window,),
                                window_channel_weights=(np.ones(gain.shape[0]),),
                                require_one=conditional_surface_settings["require_one"],
                                edge_fraction=conditional_surface_settings["edge_fraction"],
                                noise_multiplier=conditional_surface_settings["noise_multiplier"],
                                mrf_strength=solver_settings["mrf_strength"],
                                calibration=conditional_surface_settings["calibration"],
                                temporal_mode=conditional_surface_settings["temporal_mode"],
                                solver_kind=solver_settings["solver_kind"],
                                surface_reweight_floor=solver_settings["surface_reweight_floor"],
                                deep_reweight_floor=solver_settings["deep_reweight_floor"],
                                ridge_fraction=conditional_surface_settings["ridge_fraction"],
                                edge_penalty_mode=conditional_surface_settings["edge_penalty_mode"],
                                surface_penalty_multiplier=None,
                                source_penalty_mode=conditional_surface_settings["source_penalty_mode"],
                                deep_alias_penalty=conditional_surface_settings["deep_alias_penalty"],
                                outer_iterations=solver_settings["outer_iterations"],
                                max_iter=conditional_surface_settings["max_iter"],
                                tolerance=solver_settings["tolerance"],
                                epsilon_fraction=solver_settings["epsilon_fraction"],
                                rho=conditional_surface_settings["rho"],
                                outer_tolerance=solver_settings["outer_tolerance"],
                                adaptive_rho=conditional_surface_settings["adaptive_rho"],
                                max_inner_retries=solver_settings["max_inner_retries"],
                                edge_weight_floor=solver_settings["edge_weight_floor"])
                        final_surface_solver = final_surface_info["windows"][0]["solver"]
                        final_surface_solver_ok = bool(
                            final_surface_solver["converged"]
                            and final_surface_solver["outer_converged"]
                            and final_surface_solver["final_inner_converged"]
                            and final_surface_solver["final_stationarity_gap_relative"]
                            <= solver_settings["tolerance"] + 1e-12)
                        evidence_row["final_surface_converged"] = int(
                            final_surface_solver_ok)
                        if not final_surface_solver_ok:
                            raise RuntimeError("combined40 条件表层重拟合未到固定点")
                        selected[:n_surf] = final_surface
                    selected_family = (
                        "H1 deep+surface combined40 refit" if surface_present
                        else "H1 deep-only combined40 refit")
                    evidence_row["all_converged"] = 1

            if deep_present:
                detail.update({
                    "selected_deep_index": selected_deep,
                    "amplitude_deep_index": amplitude_deep,
                    "deep_loo_candidate_indices": loo_candidate_indices.tolist(),
                    "deep_loo_modality_scores": loo_by_modality.tolist(),
                    "deep_loo_sum_scores": loo_sum.tolist(),
                    "secondary_surface_score": secondary_score,
                    "secondary_modality_scores": secondary_modalities.tolist(),
                    "surface_selection_solver": surface_solver,
                    "surface_final_solver": final_surface_solver,
                })

            if stage == "validation":
                # 盲 family、深源位置和是否保留表层已冻结；从这里起真值只进指标。
                values = metrics.evaluate_estimate(
                    selected, observation["truth"], vertices,
                    observation["groups"], n_surf, active_samples,
                    shared["auc_cortex"], baseline=baseline)
                surface_groups = [
                    np.asarray(group, int) for group in observation["groups"]
                    if np.asarray(group).size
                    and np.all(np.asarray(group) < n_surf)]
                surface_component_dle = []
                surface_patch_hits = []
                if surface_groups:
                    surface_energy = np.sum(
                        selected[:n_surf, active_samples] ** 2, axis=1)
                    truth_indices = np.concatenate(surface_groups)
                    truth_owner = np.concatenate([
                        np.full(len(group), index, int)
                        for index, group in enumerate(surface_groups)])
                    owner = truth_owner[cKDTree(vertices[truth_indices]).query(
                        vertices[:n_surf])[1]]
                    support = surface_energy > .1 * surface_energy.max(initial=0.)
                    for index, group in enumerate(surface_groups):
                        region = np.flatnonzero(owner == index)
                        supported_region = region[support[region]]
                        if not supported_region.size:
                            surface_component_dle.append(None)
                        else:
                            peak = int(supported_region[
                                np.argmax(surface_energy[supported_region])])
                            center = vertices[group].mean(axis=0)
                            surface_component_dle.append(float(
                                np.linalg.norm(vertices[peak] - center) * 1000))
                        surface_patch_hits.append(int(np.count_nonzero(support[group])))
                finite_components = [
                    value for value in surface_component_dle if value is not None]
                component_max = (np.nan if len(finite_components) != len(surface_groups)
                                 else max(finite_components, default=np.nan))
                row = {
                    "case_id": name,
                    "case_number": case.get("case_number"),
                    "configuration_id": case.get("configuration_id"),
                    "scenario": case.get("scenario"),
                    "eeg_snr_db": case.get("eeg_snr_db"),
                    "meg_snr_db": case.get("meg_snr_db"),
                    "status": "ok",
                    "gate_score": gate_score,
                    "gate_threshold": gate_threshold,
                    "deep_present_decision": int(deep_present),
                    "selected_family": selected_family,
                    "selected_deep_index": ("" if selected_deep is None
                                            else selected_deep),
                    "secondary_surface_score": secondary_score,
                    "surface_threshold": surface_threshold,
                    "surface_present_decision": int(
                        bool(deep_present and secondary_score > surface_threshold)),
                    "h0_innovation_count": ("" if h0_innovation_indices is None
                                            else len(h0_innovation_indices)),
                    "surface_component_dle_max_mm": component_max,
                    "surface_patch_hit_count": int(sum(
                        value > 0 for value in surface_patch_hits)),
                    "surface_patch_count": len(surface_patch_hits),
                    "all_converged": evidence_row["all_converged"],
                    **values,
                }
                rows.append(row)
                detail.update({
                    "selected_family": selected_family,
                    "h0_innovation_indices": (
                        None if h0_innovation_indices is None
                        else h0_innovation_indices.tolist()),
                    "surface_component_dle_mm": surface_component_dle,
                    "surface_patch_support_hits": surface_patch_hits,
                    "metrics": {
                        key: (None if isinstance(value, float) and not np.isfinite(value)
                              else value)
                        for key, value in values.items()},
                })
                np.savez_compressed(
                    output / (name + ".npz"),
                    truth=np.asarray(observation["truth"], np.float32),
                    selected=np.asarray(selected, np.float32),
                    vertices=vertices.astype(np.float32),
                    times=np.asarray(shared["times"]),
                    active=active_samples,
                    baseline=baseline)
                evidence_row["status"] = "ok"
                detail["complete"] = True

        detail["elapsed_seconds"] = time.perf_counter() - tick
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        evidence_row["error"] = message
        failures.append({"case_id": name, "error": message})
        detail.update({"error": message, "elapsed_seconds": time.perf_counter() - tick})
    evidence_row["elapsed_seconds"] = time.perf_counter() - tick
    evidence_rows.append(evidence_row)
    (output / (name + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    archive._atomic_csv(output / "evidence.csv", evidence_rows, evidence_rows[0].keys())
    if rows:
        archive._atomic_csv(output / "rows.csv", rows, rows[0].keys())
    print(stage, name, evidence_row["status"],
          "gate=", evidence_row["gate_score"],
          "surface=", evidence_row["secondary_surface_score"], flush=True)


# %% 4. 两次校准只在 57 例全部有限、全部收敛时生成冻结文件。
if stage in {"gate-calibration", "surface-calibration"}:
    score_key = ("gate_score" if stage == "gate-calibration"
                 else "secondary_surface_score")
    scores = np.asarray([
        row[score_key] for row in evidence_rows if row["status"] == "ok"], float)
    valid = bool(
        not failures and len(evidence_rows) == 57 and scores.shape == (57,)
        and np.isfinite(scores).all()
        and all(row["all_converged"] for row in evidence_rows))
    threshold = None
    if valid:
        threshold = (max(0., float(np.sort(scores)[-2]))
                     if stage == "gate-calibration"
                     else max(0., float(scores.max())))
    completion = {
        "stage": stage,
        "complete": len(evidence_rows) == 57,
        "valid": valid,
        "case_count": len(evidence_rows),
        "finite_score_count": int(np.isfinite(scores).sum()),
        "all_converged": bool(all(row["all_converged"] for row in evidence_rows)),
        "failure_count": len(failures),
        "failures": failures,
        "wall_seconds": time.perf_counter() - started,
    }
    (output / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    if valid:
        frozen_payload = {
            **metadata,
            **completion,
            "valid": True,
            "scores": scores.tolist(),
            "threshold": threshold,
            "comparison": "strict_greater",
            "threshold_construction": (
                "max(0, second-largest of 57 pure-surface scores)"
                if stage == "gate-calibration"
                else "max(0, maximum of 57 pure-deep end-to-end scores)"),
            "case_ids": [row["case_id"] for row in evidence_rows],
        }
        frozen_path = output / frozen_names[stage]
        frozen_path.write_text(
            json.dumps(frozen_payload, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8")
        digest = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
        frozen_sidecar = Path(str(frozen_path) + ".sha256")
        frozen_sidecar.write_text(
            f"{digest}  {frozen_path.name}\n", encoding="ascii")
        sealed_outputs = {}
        for output_name, path in {
                "metadata": output / "metadata.json",
                "evidence": output / "evidence.csv",
                "completion": output / "completion.json",
                "frozen_calibration": frozen_path,
        }.items():
            sealed_outputs[output_name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        seal_payload = {
            "schema_version": 1,
            "protocol": "erp-v6-hierarchical-v1-calibration-seal",
            "stage": stage,
            "execution_lock": {"path": str(lock_path), "sha256": lock_sha256},
            "manifest": {"path": str(manifest.resolve()),
                         "sha256": manifest_lock["sha256"]},
            "consumed_marker": {"path": str(marker_path.resolve()),
                                "sha256": marker_sha256},
            "prior_calibration_sha256": {
                name: value["sha256"] for name, value in calibrations.items()},
            "outputs": sealed_outputs,
            "checks": {
                "case_count": 57,
                "complete": True,
                "valid": True,
                "all_converged": True,
                "finite_score_count": 57,
            },
            "threshold": {
                "value": threshold,
                "construction": frozen_payload["threshold_construction"],
                "comparison": "strict_greater",
            },
            "status": "calibration_frozen_next_stage_allowed",
        }
        seal_path = output / "calibration_seal.json"
        seal_serialized = (json.dumps(
            seal_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        with seal_path.open("xb") as stream:
            stream.write(seal_serialized)
        seal_digest = hashlib.sha256(seal_serialized).hexdigest()
        seal_sidecar = Path(str(seal_path) + ".sha256")
        with seal_sidecar.open("xb") as stream:
            stream.write(f"{seal_digest}  {seal_path.name}\n".encode("ascii"))
        print(json.dumps({"valid": True, "stage": stage, "threshold": threshold,
                          "sha256": digest, "seal_sha256": seal_digest},
                         ensure_ascii=False, indent=2), flush=True)
    else:
        (output / "invalid_calibration.json").write_text(
            json.dumps({**metadata, **completion}, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
        raise SystemExit(2)


# %% 5. validation 主指标硬报告；requested AUC 与精确 patch 命中只作补充。
if stage == "validation":
    surface_rows = [row for row in rows if row["scenario"] != "deep_only"]
    deep_rows = [row for row in rows if row["scenario"] != "surface_only"]
    global_auc = np.asarray([row["auc_tie_corrected"] for row in rows], float)
    surface_auc = np.asarray(
        [row["surface_auc_tie_corrected"] for row in surface_rows], float)
    surface_dle = np.asarray([row["surface_dle_mm"] for row in surface_rows], float)
    component_dle = np.asarray(
        [row["surface_component_dle_max_mm"] for row in surface_rows], float)
    surface_sd = np.asarray([row["surface_sd_mm"] for row in surface_rows], float)
    deep_distance = np.asarray(
        [row["deep_peak_distance_mm"] for row in deep_rows], float)
    requested_auc = np.asarray([row["auc"] for row in rows], float)
    family_error_case_ids = [
        row["case_id"] for row in rows
        if row["deep_present_decision"] != int(row["scenario"] != "surface_only")]
    h0_deep_false_case_ids = [
        row["case_id"] for row in rows
        if row["scenario"] == "surface_only" and row["deep_present_decision"]]
    h1_detected_case_ids = [
        row["case_id"] for row in deep_rows if row["deep_present_decision"]]
    deep_only_surface_false_case_ids = [
        row["case_id"] for row in rows
        if row["scenario"] == "deep_only" and row["surface_present_decision"]]
    patch_miss_case_ids = [
        row["case_id"] for row in surface_rows
        if row["surface_patch_hit_count"] != row["surface_patch_count"]]
    finite_global = global_auc[np.isfinite(global_auc)]
    finite_surface_auc = surface_auc[np.isfinite(surface_auc)]
    finite_surface_dle = surface_dle[np.isfinite(surface_dle)]
    finite_component = component_dle[np.isfinite(component_dle)]
    finite_surface_sd = surface_sd[np.isfinite(surface_sd)]
    finite_deep_distance = deep_distance[np.isfinite(deep_distance)]
    finite_requested = requested_auc[np.isfinite(requested_auc)]
    h0_false_count_by_snr = {}
    h1_detected_count_by_snr = {}
    h1_case_count_by_snr = {}
    for eeg_snr, meg_snr in snr_pairs:
        snr_name = f"eeg{int(eeg_snr):+d}_meg{int(meg_snr):+d}"
        h0_cell = [row for row in rows
                   if row["scenario"] == "surface_only"
                   and (row["eeg_snr_db"], row["meg_snr_db"])
                   == (eeg_snr, meg_snr)]
        h1_cell = [row for row in deep_rows
                   if (row["eeg_snr_db"], row["meg_snr_db"])
                   == (eeg_snr, meg_snr)]
        h0_false_count_by_snr[snr_name] = int(sum(
            row["deep_present_decision"] for row in h0_cell))
        h1_detected_count_by_snr[snr_name] = int(sum(
            row["deep_present_decision"] for row in h1_cell))
        h1_case_count_by_snr[snr_name] = len(h1_cell)
    summary = {
        "stage": stage,
        "complete": len(rows) == len(cases) and not failures,
        "case_count": len(rows),
        "expected_case_count": len(cases),
        "failure_count": len(failures),
        "failures": failures,
        "family_error_case_ids": family_error_case_ids,
        "minimum_global_An_auc": (
            None if not finite_global.size else float(finite_global.min())),
        "minimum_surface_An_auc": (
            None if not finite_surface_auc.size else float(finite_surface_auc.min())),
        "maximum_surface_DLE_mm": (
            None if not finite_surface_dle.size else float(finite_surface_dle.max())),
        "maximum_surface_component_DLE_mm": (
            None if not finite_component.size else float(finite_component.max())),
        "maximum_surface_SD_mm": (
            None if not finite_surface_sd.size else float(finite_surface_sd.max())),
        "maximum_deep_peak_distance_mm": (
            None if not finite_deep_distance.size else float(finite_deep_distance.max())),
        "h0_deep_false_case_ids": h0_deep_false_case_ids,
        "h0_deep_false_count": len(h0_deep_false_case_ids),
        "h0_deep_false_count_by_snr": h0_false_count_by_snr,
        "h1_detected_case_ids": h1_detected_case_ids,
        "h1_detected_count": len(h1_detected_case_ids),
        "h1_case_count": len(deep_rows),
        "h1_detected_count_by_snr": h1_detected_count_by_snr,
        "h1_case_count_by_snr": h1_case_count_by_snr,
        "deep_only_surface_false_case_ids": deep_only_surface_false_case_ids,
        "deep_only_surface_false_count": len(deep_only_surface_false_case_ids),
        "all_selection_and_final_solvers_converged": bool(
            len(rows) == len(cases) and all(row["all_converged"] for row in rows)),
        "supplemental_minimum_requested_parcel_AUC": (
            None if not finite_requested.size else float(finite_requested.min())),
        "supplemental_exact_patch_miss_case_ids": patch_miss_case_ids,
        "gate_calibration_sha256": calibrations["gate-calibration"]["sha256"],
        "surface_calibration_sha256": calibrations["surface-calibration"]["sha256"],
        "gate_threshold": gate_threshold,
        "surface_threshold": surface_threshold,
        "wall_seconds": time.perf_counter() - started,
    }
    metric_completeness = bool(
        global_auc.size == len(cases) and np.isfinite(global_auc).all()
        and surface_auc.size == len(surface_rows) and np.isfinite(surface_auc).all()
        and surface_dle.size == len(surface_rows) and np.isfinite(surface_dle).all()
        and component_dle.size == len(surface_rows) and np.isfinite(component_dle).all()
        and surface_sd.size == len(surface_rows) and np.isfinite(surface_sd).all()
        and deep_distance.size == len(deep_rows) and np.isfinite(deep_distance).all())
    gates = {
        "complete_finite_metrics": metric_completeness,
        "global_An_auc": summary["minimum_global_An_auc"] is not None
        and summary["minimum_global_An_auc"] >= acceptance["minimum_global_auc"],
        "surface_An_auc": summary["minimum_surface_An_auc"] is not None
        and summary["minimum_surface_An_auc"] >= acceptance["minimum_surface_auc"],
        "surface_DLE": summary["maximum_surface_DLE_mm"] is not None
        and summary["maximum_surface_DLE_mm"] <= acceptance["maximum_surface_dle_mm"],
        "surface_component_DLE": summary["maximum_surface_component_DLE_mm"] is not None
        and summary["maximum_surface_component_DLE_mm"]
        <= acceptance["maximum_surface_component_dle_mm"],
        "surface_SD": summary["maximum_surface_SD_mm"] is not None
        and summary["maximum_surface_SD_mm"] <= acceptance["maximum_surface_sd_mm"],
        "deep_distance": summary["maximum_deep_peak_distance_mm"] is not None
        and summary["maximum_deep_peak_distance_mm"]
        <= acceptance["maximum_deep_distance_mm"],
        "h0_deep_false_positive_per_snr": bool(
            len(h0_false_count_by_snr) == len(snr_pairs)
            and max(h0_false_count_by_snr.values(), default=np.inf)
            <= acceptance["h0_false_positives_per_snr_max"]),
        "h1_case_count": summary["h1_case_count"] == acceptance["h1_case_count"],
        "h1_detected_total": summary["h1_detected_count"]
        >= acceptance["h1_detected_total_min"],
        "h1_detected_per_snr": bool(
            len(h1_detected_count_by_snr) == len(snr_pairs)
            and min(h1_detected_count_by_snr.values(), default=-1)
            >= acceptance["h1_detected_per_snr_min"]),
        "deep_only_surface_false_positive": summary["deep_only_surface_false_count"]
        <= acceptance["deep_only_surface_false_count_max"],
        "all_selection_and_final_solvers_converged":
        summary["all_selection_and_final_solvers_converged"]
        == acceptance["all_selection_and_final_solvers_converged"],
    }
    summary["gates"] = gates
    summary["passed"] = bool(summary["complete"] and all(gates.values()))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "completion.json").write_text(
        json.dumps({
            "stage": stage,
            "complete": summary["complete"],
            "passed": summary["passed"],
            "case_count": len(rows),
            "failure_count": len(failures),
            "wall_seconds": summary["wall_seconds"],
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")

    lines = [
        "# hierarchical-v1 正式 validation", "",
        f"最终结论：`{'PASS' if summary['passed'] else 'FAIL'}`。", "",
        "| 病例 | 场景 | SNR EEG/MEG | family | An_auc | surface AUC | surface DLE/component/SD mm | deep mm | H0 deep FP | deep-only surface FP |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        surface_values = ("—" if not np.isfinite(row["surface_dle_mm"]) else
                          f"{row['surface_dle_mm']:.2f}/{row['surface_component_dle_max_mm']:.2f}/{row['surface_sd_mm']:.2f}")
        deep_value = ("—" if not np.isfinite(row["deep_peak_distance_mm"])
                      else f"{row['deep_peak_distance_mm']:.2f}")
        surface_auc_value = ("—" if not np.isfinite(row["surface_auc_tie_corrected"])
                             else f"{row['surface_auc_tie_corrected']:.3f}")
        lines.append(
            f"| {row['case_number']} | {row['scenario']} | {row['eeg_snr_db']:+d}/{row['meg_snr_db']:+d} | "
            f"{row['selected_family']} | {row['auc_tie_corrected']:.3f} | {surface_auc_value} | "
            f"{surface_values} | {deep_value} | "
            f"{int(row['scenario'] == 'surface_only' and row['deep_present_decision'])} | "
            f"{int(row['scenario'] == 'deep_only' and row['surface_present_decision'])} |")
    lines += [
        "", "## 硬门", "",
        *[f"- {name}: `{value}`" for name, value in gates.items()],
        f"- H0 每SNR深源误报数：`{h0_false_count_by_snr}`",
        f"- H1 总检出：`{summary['h1_detected_count']}/{summary['h1_case_count']}`",
        f"- H1 每SNR检出数：`{h1_detected_count_by_snr}`",
        "", "## 补充指标（不替代硬门）", "",
        f"- family 判断不一致病例：`{family_error_case_ids or '无'}`",
        f"- requested parcel AUC 最低值：`{summary['supplemental_minimum_requested_parcel_AUC']}`",
        f"- 精确真值 patch 未命中病例：`{patch_miss_case_ids or '无'}`",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(3)
