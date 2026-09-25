"""# %% 冻结 hierarchical-v1 正式执行锁；不运行任何正式病例。"""

# %% 1. 固定路径；正式 consumed marker、输出目录或校准 seal 一旦存在就拒绝建锁。
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if not __debug__:
    raise RuntimeError("冻结执行锁禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="只逐字节核验，不创建执行锁")
args = parser.parse_args()

adaptive_root = root / "results/erp_whole_head/adaptive_v6"
protocol_dir = adaptive_root / "protocol"
output_path = protocol_dir / "hierarchical_v1_execution_lock.json"
output_sidecar = Path(str(output_path) + ".sha256")
stage_outputs = {
    "gate-calibration": adaptive_root / "formal_hierarchical_v1_gate_calibration",
    "surface-calibration": adaptive_root / "formal_hierarchical_v1_surface_calibration",
    "validation": adaptive_root / "formal_hierarchical_v1_validation",
}
stage_markers = {
    "gate-calibration": protocol_dir / "hierarchical_v1_gate_calibration_manifest_consumed.json",
    "surface-calibration": protocol_dir / "hierarchical_v1_surface_calibration_manifest_consumed.json",
    "validation": protocol_dir / "hierarchical_v1_validation_manifest_consumed.json",
}
forbidden_markers = sorted(protocol_dir.glob("hierarchical_v1*consumed*.json"))
forbidden_outputs = sorted(path for path in adaptive_root.glob("*hierarchical_v1*")
                           if path.is_dir())
forbidden_seals = sorted(protocol_dir.glob("hierarchical_v1*seal*.json"))
for stage_output in stage_outputs.values():
    seal = stage_output / "calibration_seal.json"
    if seal.exists() or Path(str(seal) + ".sha256").exists():
        forbidden_seals.append(seal)
if forbidden_markers or forbidden_outputs or forbidden_seals:
    raise FileExistsError(
        "hierarchical-v1 已开始正式消费，禁止新建或复核执行锁："
        f"markers={list(map(str, forbidden_markers))}, "
        f"outputs={list(map(str, forbidden_outputs))}, "
        f"seals={list(map(str, forbidden_seals))}"
    )


# %% 2. 核验64号脚本冻结的五个文件及标准SHA256旁车。
input_paths = {
    "gate_manifest": protocol_dir / "hierarchical_v1_gate_calibration_manifest.json",
    "surface_manifest": protocol_dir / "hierarchical_v1_surface_calibration_manifest.json",
    "validation_manifest": protocol_dir / "hierarchical_v1_validation_manifest.json",
    "protocol_lock": protocol_dir / "hierarchical_v1_protocol_audit.json",
    "protocol_document": protocol_dir / "HIERARCHICAL_V1_PROTOCOL.md",
}
verified_inputs = {}
for name, path in input_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"缺少64号脚本冻结输入：{path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"冻结输入或旁车不一致：{path}")
    verified_inputs[name] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "sidecar_path": sidecar.relative_to(root).as_posix(),
        "sidecar_text_sha256": hashlib.sha256(expected_sidecar).hexdigest(),
    }

audit = json.loads(input_paths["protocol_lock"].read_text(encoding="utf-8"))
gate_cases = json.loads(input_paths["gate_manifest"].read_text(encoding="utf-8"))
surface_cases = json.loads(input_paths["surface_manifest"].read_text(encoding="utf-8"))
validation_cases = json.loads(input_paths["validation_manifest"].read_text(encoding="utf-8"))
if audit.get("protocol") != "erp-v6-hierarchical-v1-formal-manifests" or \
        audit.get("formal_results_present") is not False or \
        audit.get("truth_firewall", {}).get("generator_runs_simulation_or_inverse") is not False or \
        audit.get("truth_firewall", {}).get("generator_reads_result_metrics") is not False:
    raise ValueError("64号协议audit不是未执行的hierarchical-v1正式清单")


# %% 3. 三个manifest的病例数、roots、SNR、配置重复和场景必须与65号入口完全一致。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
seed_roots = {
    "gate-calibration": {"fit": 2026093001, "check": 2026093002},
    "surface-calibration": {"fit": 2026093003, "check": 2026093004},
    "validation": {"fit": 2026100101, "check": 2026100102},
}
case_groups = {
    "gate-calibration": gate_cases,
    "surface-calibration": surface_cases,
    "validation": validation_cases,
}
expected_shape = {
    "gate-calibration": {"case_count": 57, "configuration_count": 19,
                         "configuration_repetition": 3, "cases_per_snr": 19},
    "surface-calibration": {"case_count": 57, "configuration_count": 57,
                            "configuration_repetition": 1, "cases_per_snr": 19},
    "validation": {"case_count": 21, "configuration_count": 21,
                   "configuration_repetition": 1, "cases_per_snr": 7},
}
all_case_ids = []
for stage, cases in case_groups.items():
    shape = expected_shape[stage]
    if not isinstance(cases, list) or len(cases) != shape["case_count"]:
        raise ValueError(f"{stage}病例数错误")
    case_ids = [case.get("case_id") for case in cases]
    if None in case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"{stage} case_id不唯一")
    all_case_ids.extend(case_ids)
    if any(case.get("replica_seed_roots") != seed_roots[stage] or
           case.get("seed", [None])[0] != seed_roots[stage]["fit"] for case in cases):
        raise ValueError(f"{stage}没有使用冻结fit/check roots")
    if Counter((case.get("eeg_snr_db"), case.get("meg_snr_db")) for case in cases) != \
            Counter({pair: shape["cases_per_snr"] for pair in snr_pairs}):
        raise ValueError(f"{stage}的三个SNR分层错误")
    configurations = Counter(case.get("configuration_id") for case in cases)
    if len(configurations) != shape["configuration_count"] or \
            set(configurations.values()) != {shape["configuration_repetition"]}:
        raise ValueError(f"{stage}配置数或跨SNR重复结构错误")
if len(all_case_ids) != len(set(all_case_ids)):
    raise ValueError("三个正式manifest之间复用了case_id")

for pair in snr_pairs:
    cell = [case for case in gate_cases
            if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    if Counter(len(case.get("surface_centers", [])) for case in cell) != {1: 10, 2: 9}:
        raise ValueError(f"gate {pair}不是10单表层+9双表层")
if not all(case.get("scenario") == "surface_only" and case.get("deep_index") is None
           and bool(case.get("surface_centers")) for case in gate_cases):
    raise ValueError("gate校准必须全是纯表层")
if any(case.get("surface_component_sensor_balance") is not True
       for case in gate_cases if len(case.get("surface_centers", [])) == 2):
    raise ValueError("gate双表层病例没有启用传感器分量平衡")
for configuration in {case["configuration_id"] for case in gate_cases}:
    repeated = [case for case in gate_cases if case["configuration_id"] == configuration]
    geometry = {(tuple(case["surface_centers"]), case["correlation"]) for case in repeated}
    if len(geometry) != 1 or Counter(
            (case["eeg_snr_db"], case["meg_snr_db"]) for case in repeated) != Counter(snr_pairs):
        raise ValueError(f"gate固定几何没有原样跨三个SNR：{configuration}")

if not all(case.get("scenario") == "deep_only" and case.get("deep_index") is not None
           and not case.get("surface_centers") for case in surface_cases):
    raise ValueError("surface校准必须全是纯深层")
validation_scenarios = {
    "surface_only": 12,
    "deep_only": 3,
    "deep_plus_surface": 3,
    "deep_plus_two_surface": 3,
}
if Counter(case.get("scenario") for case in validation_cases) != Counter(validation_scenarios):
    raise ValueError("validation总场景组成错误")
for pair in snr_pairs:
    cell = [case for case in validation_cases
            if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    if Counter(case["scenario"] for case in cell) != Counter({
            "surface_only": 4, "deep_only": 1,
            "deep_plus_surface": 1, "deep_plus_two_surface": 1}):
        raise ValueError(f"validation {pair}不是4个H0+3个H1")
if any(case.get("surface_component_sensor_balance") is not True
       for case in validation_cases if len(case.get("surface_centers", [])) == 2):
    raise ValueError("validation双表层病例没有启用传感器分量平衡")
if any(case.get("deep_surface_sensor_amplitude_ratio") != .5
       for case in validation_cases
       if case.get("deep_index") is not None and case.get("surface_centers")):
    raise ValueError("validation混合病例的深浅传感器比不是0.5")

expected_audit_roots = {
    "gate": seed_roots["gate-calibration"],
    "surface": seed_roots["surface-calibration"],
    "validation": seed_roots["validation"],
}
if audit.get("replica_seed_roots") != expected_audit_roots or \
        audit.get("snr_pairs_eeg_meg_db") != [list(pair) for pair in snr_pairs] or \
        audit.get("gate_calibration", {}).get("case_count") != 57 or \
        audit.get("gate_calibration", {}).get("configuration_count") != 19 or \
        audit.get("gate_calibration", {}).get("configuration_repeated_across_snr") is not True or \
        audit.get("surface_calibration", {}).get("case_count") != 57 or \
        audit.get("surface_calibration", {}).get("configuration_count") != 57 or \
        audit.get("validation", {}).get("case_count") != 21 or \
        audit.get("validation", {}).get("configuration_count") != 21 or \
        audit.get("validation", {}).get("configuration_repeats_across_snr") != 0:
    raise ValueError("64号audit与三个manifest的正式结构不一致")


# %% 4. 当前HEAD必须包含全部冻结输入与代码，而且整个仓库不得有tracked diff。
code_paths = [
    "作者风格版/63-层级浅深盲定位开发.py",
    "作者风格版/64-层级盲定位正式清单.py",
    "作者风格版/65-层级盲定位正式运行.py",
    "作者风格版/66-冻结层级盲定位执行锁.py",
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
]
for relative in code_paths:
    if not (root / relative).is_file():
        raise FileNotFoundError(f"正式链路缺少代码：{relative}")

git_status = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=no"],
    cwd=root, capture_output=True, text=True, encoding="utf-8",
)
if git_status.returncode or git_status.stdout:
    raise RuntimeError("冻结执行锁前tracked工作树和index必须完全clean")
git_head_result = subprocess.run(
    ["git", "rev-parse", "--verify", "HEAD"],
    cwd=root, capture_output=True, text=True, encoding="ascii",
)
if git_head_result.returncode:
    raise RuntimeError("无法解析当前Git HEAD")
git_head = git_head_result.stdout.strip()
tracked_relatives = code_paths + [
    item for path in input_paths.values()
    for item in (path.relative_to(root).as_posix(),
                 Path(str(path) + ".sha256").relative_to(root).as_posix())
]
tracked_result = subprocess.run(
    ["git", "ls-files", "--error-unmatch", "--", *tracked_relatives],
    cwd=root, capture_output=True, text=True, encoding="utf-8",
)
if tracked_result.returncode:
    raise RuntimeError("64/65/66、正式输入或旁车尚未全部提交到当前Git HEAD")
code_sha256 = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in code_paths
}
script63 = root / "作者风格版/63-层级浅深盲定位开发.py"
if Path(audit.get("algorithm_path", "")).name != script63.name or \
        audit.get("algorithm_sha256") != code_sha256["作者风格版/63-层级浅深盲定位开发.py"]:
    raise ValueError("64号audit绑定的63号开发算法已经变化")


# %% 5. 共享forward指纹、运行环境和65号入口所需完整算法设置。
generated_dir = root / "corrected_v2/generated"
sample_path = original.DEFAULT_SAMPLE_PATH.resolve()
shared = protocol.load_shared(generated_dir, sample_path)
shared_fingerprint = original._shared_fingerprint(shared)
if shared_fingerprint != audit.get("shared_fingerprint"):
    raise ValueError("共享forward已偏离64号清单")
environment = original._environment_versions()
environment["python_debug"] = __debug__

source_model_settings = {
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
conditional_surface_settings = {
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
algorithm = {
    "covariance": "trial",
    "replicates": {
        "training_trials": 20,
        "confirmation_trials": 20,
        "combined_refit_trials": 40,
    },
    "source_model_settings": source_model_settings,
    "conditional_surface_settings": conditional_surface_settings,
    "gate": {
        "score_kind": "fixed_3_to_1_linear_fusion",
        "modality_order": ["EEG", "MEG"],
        "weights": [.75, .25],
        "uses_confirmation_only": True,
        "test_true_snr_used": False,
        "calibration": {
            "score_count": 57,
            "statistical_claim":
                "pooled fixed-design finite-rank engineering threshold",
            "iid_claim": False,
            "fpr_guarantee": False,
            "required_complete_finite_scores": 57,
            "invalid_if_any_solver_unconverged": True,
            "equivalent_boundary":
                "T strictly greater than max(0, second-largest calibration score)",
            "ties_count_against_acceptance": True,
        },
    },
    "surface_presence": {
        "score_kind": "conjunctive_modality_noise_gain",
        "formula": "min(EEG, MEG) held-out D versus D+S improvement",
        "closed_primary_score": 0,
        "calibration": {
            "score_count": 57,
            "statistical_claim":
                "pooled fixed-design finite-rank engineering threshold",
            "iid_claim": False,
            "fpr_guarantee": False,
            "required_complete_finite_scores": 57,
            "invalid_if_any_solver_unconverged": True,
            "equivalent_boundary":
                "T strictly greater than max(0, maximum calibration score)",
            "ties_count_against_acceptance": True,
            "test_true_snr_used": False,
        },
    },
    "hierarchy": {
        "deep_location":
            "argmax sum over EEG,MEG modality-normalized leave-one-deep-out confirmation improvements",
        "h0_localization":
            "combined40 MRF-innovation BIC dipoles then R^-1 physical surface map",
        "h1_surface_selection":
            "training20 fixed selected deep; W=1 bounded-MM surface; frozen surface calibration",
        "final_refit":
            "freeze family/deep/surface-presence first; combined40 refits amplitudes only",
        "truth_access": "only after final estimate is frozen",
    },
}

acceptance = {
    "minimum_global_auc": .90,
    "minimum_surface_auc": .90,
    "maximum_surface_dle_mm": 15.,
    "maximum_surface_component_dle_mm": 15.,
    "maximum_surface_sd_mm": 20.,
    "maximum_deep_distance_mm": 10.,
    "h0_false_positives_per_snr_max": 0,
    "h1_detected_total_min": 8,
    "h1_case_count": 9,
    "h1_detected_per_snr_min": 2,
    "deep_only_surface_false_count_max": 0,
    "all_selection_and_final_solvers_converged": True,
}


# %% 6. 预登记三阶段路径；未来校准值与seal哈希绝不提前写入执行锁。
manifests = {}
for stage, input_name in {
        "gate-calibration": "gate_manifest",
        "surface-calibration": "surface_manifest",
        "validation": "validation_manifest",
}.items():
    manifests[stage] = {
        "manifest": verified_inputs[input_name],
        **expected_shape[stage],
        "seed_roots": seed_roots[stage],
        "scenario_counts": dict(Counter(
            case["scenario"] for case in case_groups[stage])),
        "consumed_marker": stage_markers[stage].relative_to(root).as_posix(),
        "output_dir": stage_outputs[stage].relative_to(root).as_posix(),
    }

for stage, frozen_name in {
        "gate-calibration": "frozen_gate_calibration.json",
        "surface-calibration": "frozen_surface_calibration.json",
}.items():
    frozen_path = stage_outputs[stage] / frozen_name
    seal_path = stage_outputs[stage] / "calibration_seal.json"
    manifests[stage]["frozen_calibration"] = {
        "path": frozen_path.relative_to(root).as_posix(),
        "sidecar_path": Path(str(frozen_path) + ".sha256").relative_to(root).as_posix(),
    }
    manifests[stage]["calibration_seal"] = {
        "path": seal_path.relative_to(root).as_posix(),
        "sidecar_path": Path(str(seal_path) + ".sha256").relative_to(root).as_posix(),
    }

execution_lock = {
    "schema_version": 1,
    "protocol": "erp-v6-hierarchical-v1-execution",
    "status": "algorithm_frozen_calibration_allowed",
    "formal_cases_run_by_generator": False,
    "consumed_marker_created_by_generator": False,
    "stage_order": ["gate-calibration", "surface-calibration", "validation"],
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "protocol_inputs": {
        "protocol_lock": verified_inputs["protocol_lock"],
        "protocol_document": verified_inputs["protocol_document"],
    },
    "manifests": manifests,
    "algorithm": algorithm,
    "acceptance": acceptance,
    "shared": {
        "generated_dir": generated_dir.relative_to(root).as_posix(),
        "sample_path": str(sample_path),
        "fingerprint": shared_fingerprint,
    },
    "environment": environment,
    "repository": {
        "git_head": git_head,
        "tracked_worktree_and_index_clean": True,
        "all_frozen_inputs_and_code_tracked_at_head": True,
    },
    "code_sha256": code_sha256,
    "generator_sha256": code_sha256["作者风格版/66-冻结层级盲定位执行锁.py"],
}


# %% 7. 首次用xb创建；已有锁只允许逐字节相同，--check绝不创建。
output_payload = (json.dumps(
    execution_lock, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")) + "\n").encode("utf-8")
output_sha256 = hashlib.sha256(output_payload).hexdigest()
sidecar_payload = f"{output_sha256}  {output_path.name}\n".encode("ascii")
if output_path.exists() or output_sidecar.exists():
    if not output_path.is_file() or not output_sidecar.is_file() or \
            output_path.read_bytes() != output_payload or \
            output_sidecar.read_bytes() != sidecar_payload:
        raise ValueError("已有hierarchical-v1执行锁不同或不完整，拒绝覆盖")
elif args.check:
    raise FileNotFoundError("--check要求执行锁及旁车已经存在")
else:
    with output_path.open("xb") as stream:
        stream.write(output_payload)
    with output_sidecar.open("xb") as stream:
        stream.write(sidecar_payload)

print(json.dumps({
    "checked": bool(args.check or output_path.exists()),
    "path": str(output_path),
    "sha256": output_sha256,
    "git_head": git_head,
    "stage_case_counts": {
        stage: specification["case_count"] for stage, specification in manifests.items()},
    "formal_cases_run": False,
}, ensure_ascii=False, indent=2))
