"""# %% 冻结 hierarchical-v2 正式执行锁；不运行任何正式病例。"""

# %% 1. 固定路径；任何正式 marker、输出、run-lock 或 seal 已存在时都拒绝建锁。
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
output_path = protocol_dir / "hierarchical_v2_execution_lock.json"
output_sidecar = Path(str(output_path) + ".sha256")
stage_outputs = {
    "gate-calibration": adaptive_root / "formal_hierarchical_v2_gate_calibration",
    "surface-calibration": adaptive_root / "formal_hierarchical_v2_surface_calibration",
    "validation": adaptive_root / "formal_hierarchical_v2_validation",
}
stage_markers = {
    "gate-calibration": protocol_dir / "hierarchical_v2_gate_calibration_manifest_consumed.json",
    "surface-calibration": protocol_dir / "hierarchical_v2_surface_calibration_manifest_consumed.json",
    "validation": protocol_dir / "hierarchical_v1_validation_manifest_consumed.json",
}
marker_candidates = [*protocol_dir.glob("hierarchical_v2*consumed*.json"),
                     stage_markers["validation"]]
forbidden_markers = sorted({path for path in marker_candidates
                            if path.exists() or path.is_symlink()}, key=str)
output_candidates = [*adaptive_root.glob("formal_hierarchical_v2*"),
                     adaptive_root / "formal_hierarchical_v1_validation"]
forbidden_outputs = sorted({path for path in output_candidates
                            if path.exists() or path.is_symlink()}, key=str)
run_lock_candidates = [path.with_name(path.name + ".run.lock")
                       for path in stage_outputs.values()]
run_lock_candidates.append(adaptive_root / "formal_hierarchical_v1_validation.run.lock")
forbidden_run_locks = [path for path in run_lock_candidates
                       if path.exists() or path.is_symlink()]
forbidden_seals = sorted(protocol_dir.glob("hierarchical_v2*seal*.json"))
for stage_output in stage_outputs.values():
    for name in ("calibration_seal.json", "invalid_calibration_seal.json"):
        seal = stage_output / name
        if seal.exists() or Path(str(seal) + ".sha256").exists():
            forbidden_seals.append(seal)
if forbidden_markers or forbidden_outputs or forbidden_run_locks or forbidden_seals:
    raise FileExistsError(
        "hierarchical-v2 已开始正式消费，禁止新建或复核执行锁："
        f"markers={list(map(str, forbidden_markers))}, "
        f"outputs={list(map(str, forbidden_outputs))}, "
        f"run_locks={list(map(str, forbidden_run_locks))}, "
        f"seals={list(map(str, forbidden_seals))}"
    )


# %% 2. 核验71号冻结的两套新校准、协议文件和 untouched v1 validation。
input_paths = {
    "gate_manifest": protocol_dir / "hierarchical_v2_gate_calibration_manifest.json",
    "surface_manifest": protocol_dir / "hierarchical_v2_surface_calibration_manifest.json",
    "validation_manifest": protocol_dir / "hierarchical_v1_validation_manifest.json",
    "protocol_lock": protocol_dir / "hierarchical_v2_protocol_audit.json",
    "protocol_document": protocol_dir / "HIERARCHICAL_V2_PROTOCOL.md",
}
verified_inputs = {}
for name, path in input_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"缺少71号脚本冻结输入：{path}")
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
protected_validation_sha256 = "6ef97341434bca99bc7c75d205c8dc6d3815b3260ca3c62c10c615ce2efb0eff"
if verified_inputs["validation_manifest"]["sha256"] != protected_validation_sha256:
    raise ValueError("validation 必须逐字节引用 untouched hierarchical-v1 manifest")
if audit.get("protocol") != "erp-v6-hierarchical-v2-formal-manifests" or \
        audit.get("formal_results_present") is not False or \
        audit.get("simulation_or_inverse_run_by_generator") is not False or \
        audit.get("validation_results_read") is not False or \
        audit.get("protected_validation_copied_or_modified") is not False or \
        audit.get("truth_firewall", {}).get("generator_runs_simulation_or_inverse") is not False or \
        audit.get("truth_firewall", {}).get("generator_reads_validation_result_metrics") is not False:
    raise ValueError("71号协议audit不是未执行的 hierarchical-v2 正式清单")
if audit.get("manifest_sha256") != {
        "gate_calibration": verified_inputs["gate_manifest"]["sha256"],
        "surface_calibration": verified_inputs["surface_manifest"]["sha256"],
        "protected_v1_validation_reference": protected_validation_sha256,
}:
    raise ValueError("71号audit没有逐字节绑定两套新校准和旧validation")
history_sha256 = audit.get("history_manifest_sha256", {})
auxiliary_sha256 = audit.get("auxiliary_input_sha256", {})
if len(history_sha256) != 28 or len(auxiliary_sha256) != 3:
    raise ValueError("71号audit没有完整列出28份历史清单和3份辅助几何输入")
for name, expected in history_sha256.items():
    path = protocol_dir / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"71号audit绑定的历史清单已变化：{path}")
for relative, expected in auxiliary_sha256.items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"71号audit绑定的辅助几何输入已变化：{path}")


# %% 3. 三个manifest的病例数、SNR、固定配置重复和场景必须与72号入口一致。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
seed_roots = {
    "gate-calibration": {"fit": 2026100201, "check": 2026100202},
    "surface-calibration": {"fit": 2026100203, "check": 2026100204},
    "validation": {"fit": 2026100101, "check": 2026100102},
}
case_groups = {
    "gate-calibration": gate_cases,
    "surface-calibration": surface_cases,
    "validation": validation_cases,
}
expected_shape = {
    "gate-calibration": {
        "case_count": 57, "configuration_count": 19,
        "configuration_repetition": 3, "cases_per_snr": 19,
    },
    "surface-calibration": {
        "case_count": 57, "configuration_count": 19,
        "configuration_repetition": 3, "cases_per_snr": 19,
    },
    "validation": {
        "case_count": 21, "configuration_count": 21,
        "configuration_repetition": 1, "cases_per_snr": 7,
    },
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
    raise ValueError("三套正式manifest之间复用了case_id")

for pair in snr_pairs:
    gate_cell = [case for case in gate_cases
                 if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    if Counter(len(case.get("surface_centers", [])) for case in gate_cell) != {1: 10, 2: 9}:
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
for configuration in {case["configuration_id"] for case in surface_cases}:
    repeated = [case for case in surface_cases if case["configuration_id"] == configuration]
    geometry = {case.get("deep_local") for case in repeated}
    if len(geometry) != 1 or Counter(
            (case["eeg_snr_db"], case["meg_snr_db"]) for case in repeated) != Counter(snr_pairs):
        raise ValueError(f"surface固定深点没有原样跨三个SNR：{configuration}")

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

if audit.get("replica_seed_roots") != {
        "gate": seed_roots["gate-calibration"],
        "surface": seed_roots["surface-calibration"],
} or audit.get("snr_pairs_eeg_meg_db") != [list(pair) for pair in snr_pairs] or \
        audit.get("gate_calibration", {}).get("case_count") != 57 or \
        audit.get("gate_calibration", {}).get("configuration_count") != 19 or \
        audit.get("gate_calibration", {}).get("configuration_repetition_across_snr") != 3 or \
        audit.get("surface_calibration", {}).get("case_count") != 57 or \
        audit.get("surface_calibration", {}).get("configuration_count") != 19 or \
        audit.get("surface_calibration", {}).get("configuration_repetition_across_snr") != 3 or \
        audit.get("protected_validation", {}).get("path") != \
        verified_inputs["validation_manifest"]["path"] or \
        audit.get("protected_validation", {}).get("sha256") != protected_validation_sha256 or \
        audit.get("protected_validation", {}).get("sidecar_path") != \
        verified_inputs["validation_manifest"]["sidecar_path"] or \
        audit.get("protected_validation", {}).get("case_count") != 21 or \
        audit.get("protected_validation", {}).get("seed_roots") != seed_roots["validation"] or \
        audit.get("protected_validation", {}).get("shared_consumed_marker") != \
        stage_markers["validation"].relative_to(root).as_posix() or \
        audit.get("protected_validation", {}).get("manifest_created_by_v2_generator") is not False or \
        audit.get("protected_validation", {}).get("manifest_bytes_written_by_v2_generator") is not False:
    raise ValueError("71号audit与三个manifest的正式结构不一致")


# %% 4. 锁定70号选择summary、开发证据及其内嵌输入，不把开发表现冒充正式结果。
selection_dir = (
    adaptive_root / "development_diagnosis/hierarchical_v2_blind_reliability_fusion"
)
selection_artifact_paths = {
    name: selection_dir / filename for name, filename in {
        "summary": "summary.json",
        "candidate_scores": "candidate_scores.csv",
        "case_scores": "case_scores.csv",
        "report": "REPORT.md",
        "current15_plot": "current15_candidate_points.png",
        "candidate_auc_plot": "candidate_auc_lines.png",
    }.items()
}
missing_selection = [str(path) for path in selection_artifact_paths.values()
                     if not path.is_file()]
if missing_selection:
    raise FileNotFoundError(f"缺少70号选择证据：{missing_selection}")
selection_artifacts = {
    name: {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for name, path in selection_artifact_paths.items()
}
expected_selection_sha256 = "5c2a7e535a50e882d130cddf0afccebe3c873fc0b7f8c2f6651482eb3195bba8"
if selection_artifacts["summary"]["sha256"] != expected_selection_sha256:
    raise ValueError("70号已审计选择summary已经变化")
selection = json.loads(selection_artifact_paths["summary"].read_text(encoding="utf-8"))
selected_specs = [item for item in selection.get("all_candidates", [])
                  if item.get("candidate") == selection.get("selected_candidate")]
selected_metrics = [item for item in selection.get("candidate_metrics", [])
                    if item.get("candidate") == selection.get("selected_candidate")]
if selection.get("complete") is not True or selection.get("phase") != "development" or \
        selection.get("formal_claim_allowed") is not False or \
        selection.get("validation_results_read") is not False or \
        selection.get("truth_or_snr_used_by_score") is not False or \
        selection.get("selected_candidate") != "adaptive_balance_p1" or \
        len(selected_specs) != 1 or len(selected_metrics) != 1 or \
        selected_specs[0].get("family") != "adaptive_balance" or \
        selected_specs[0].get("exponent") != .5 or \
        selected_specs[0].get("balance_power") != 1. or \
        selected_metrics[0].get("passes_frozen_rule") != 1 or \
        selected_metrics[0].get("selected") != 1:
    raise ValueError("70号summary不是冻结的 adaptive p=1 开发选择")
for relative, expected in selection.get("input_sha256", {}).items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"70号summary的开发输入已变化：{path}")
for relative, expected in selection.get("code_sha256", {}).items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"70号summary的开发代码已变化：{path}")
selected_audit = audit.get("selected_gate_score", {})
if selected_audit.get("candidate") != "adaptive_balance_p1" or \
        selected_audit.get("operational_formula") != (
            "h_m=g_m/sqrt(1+q_m); b=min(1+q_EEG,1+q_MEG)/max(1+q_EEG,1+q_MEG); "
            "w_EEG=0.5+0.25*b; score=w_EEG*h_EEG+(1-w_EEG)*h_MEG") or \
        selected_audit.get("primary_fit_channel_weights") != \
        "all ones after per-channel whitening" or \
        selected_audit.get("scope") != (
            "primary H0/H1 fits and gate score use all-one weights after per-channel whitening; "
            "deep LOO selection and deep-amplitude estimation use original training-derived "
            "observation weights; conditional-surface train20 and combined40 fits use frozen W=1") or \
        selected_audit.get("selection_summary_sha256") != expected_selection_sha256 or \
        selected_audit.get("formal_performance_claim_from_development") is not False or \
        selected_audit.get("test_true_snr_used") is not False:
    raise ValueError("71号audit没有绑定70号选择及其非正式性质")


# %% 5. 当前HEAD必须包含全部冻结输入与代码，而且tracked工作树不得有diff。
code_paths = [
    "作者风格版/70-v2盲可靠性融合开发.py",
    "作者风格版/71-层级盲定位v2正式清单.py",
    "作者风格版/72-层级盲定位v2正式运行.py",
    "作者风格版/73-冻结层级盲定位v2执行锁.py",
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
    "metrics/user_metrics/An_roc.py",
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

selection_input_relatives = [Path(relative).as_posix()
                             for relative in selection.get("input_sha256", {})]
history_relatives = [(protocol_dir / name).relative_to(root).as_posix()
                     for name in history_sha256]
auxiliary_relatives = [Path(relative).as_posix() for relative in auxiliary_sha256]
tracked_relatives = (code_paths + selection_input_relatives + history_relatives
                     + auxiliary_relatives + [
    item for path in input_paths.values()
    for item in (path.relative_to(root).as_posix(),
                 Path(str(path) + ".sha256").relative_to(root).as_posix())
] + [item["path"] for item in selection_artifacts.values()])
tracked_result = subprocess.run(
    ["git", "ls-files", "--error-unmatch", "--", *tracked_relatives],
    cwd=root, capture_output=True, text=True, encoding="utf-8",
)
if tracked_result.returncode:
    raise RuntimeError("71/72/73、正式输入、70号证据或旁车尚未全部提交到当前Git HEAD")
code_sha256 = {
    relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
    for relative in code_paths
}
audit_inputs = audit.get("input_sha256", {})
for audit_name, relative in {
        "manifest_generator_71": "作者风格版/71-层级盲定位v2正式清单.py",
        "planned_formal_runner_72": "作者风格版/72-层级盲定位v2正式运行.py",
        "blind_reliability_selection": selection_artifacts["summary"]["path"],
        "predictive_score_helper": "candidates/oaster_predictive.py",
        "protocol_geometry_helper": "benchmark/protocol.py",
        "shared_forward_loader": "run_erp_whole_head_matrix.py",
}.items():
    item = audit_inputs.get(audit_name, {})
    expected = (selection_artifacts["summary"]["sha256"]
                if audit_name == "blind_reliability_selection"
                else code_sha256[relative])
    if item.get("path") != relative or item.get("sha256") != expected:
        raise ValueError(f"71号audit冻结代码或选择证据已变化：{audit_name}")


# %% 6. 共享forward、环境、三级权重范围和两级校准/验收规则逐项冻结。
generated_dir = root / "corrected_v2/generated"
sample_path = original.DEFAULT_SAMPLE_PATH.resolve()
shared = protocol.load_shared(generated_dir, sample_path)
shared_fingerprint = original._shared_fingerprint(shared)
if shared_fingerprint != audit.get("shared_fingerprint"):
    raise ValueError("共享forward已偏离71号清单")
environment = original._environment_versions()
environment["python_debug"] = __debug__

source_model_settings = {
    "solver_kind": "admm",
    "mrf_strength": .8,
    "outer_iterations": 80,
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
gate_formula = (
    "h_m=g_m/sqrt(1+q_m); b=min(1+q_E,1+q_M)/max(1+q_E,1+q_M); "
    "w_E=0.5+0.25*b; score=w_E*h_E+(1-w_E)*h_M"
)
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
        "score_kind": "snr_blind_adaptive_reliability_p1",
        "formula": gate_formula,
        "modality_order": ["EEG", "MEG"],
        "primary_channel_weights":
            "all whitened channels exactly one for primary H0/H1 fit and score",
        "deep_channel_weights":
            "training-derived modality evidence weights for deep LOO and amplitude",
        "conditional_surface_channel_weights":
            "all whitened channels exactly one for train20 and combined40 frozen surface fits",
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
    "gate_power": {
        "h1_source": "57 pure-deep surface-calibration cases",
        "h1_case_count": 57,
        "h1_cases_per_snr": 19,
        "detected_total_min": 52,
        "detected_per_snr_min": 17,
        "pooled_gate_auc_min": .9,
        "per_snr_gate_auc_min": .9,
        "auc_metric": "metrics/user_metrics/An_auc.py",
        "evaluated_after_gate_threshold_frozen": True,
        "used_to_fit_gate_threshold": False,
        "validation_opened_on_failure": False,
        "failure_action": "retire_protocol_without_validation",
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


# %% 7. 预登记三阶段路径；未来阈值、校准值和seal哈希绝不提前写入执行锁。
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
    "protocol": "erp-v6-hierarchical-v2-execution",
    "status": "algorithm_frozen_calibration_allowed",
    "formal_cases_run_by_generator": False,
    "consumed_marker_created_by_generator": False,
    "validation_manifest_created_copied_or_modified_by_generator": False,
    "stage_order": ["gate-calibration", "surface-calibration", "validation"],
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "protocol_inputs": {
        "protocol_lock": verified_inputs["protocol_lock"],
        "protocol_document": verified_inputs["protocol_document"],
    },
    "development_selection": {
        "selected_candidate": "adaptive_balance_p1",
        "phase": "development",
        "formal_claim_allowed": False,
        "validation_results_read": False,
        "artifacts": selection_artifacts,
        "embedded_input_sha256": selection["input_sha256"],
        "embedded_code_sha256": selection["code_sha256"],
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
        "all_frozen_inputs_code_and_development_evidence_tracked_at_head": True,
    },
    "code_sha256": code_sha256,
    "generator_sha256": code_sha256["作者风格版/73-冻结层级盲定位v2执行锁.py"],
}


# %% 8. 写锁前复核HEAD与全部冻结字节，封住生成期间的tracked输入变化。
final_git_status = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=no"],
    cwd=root, capture_output=True, text=True, encoding="utf-8",
)
final_git_head = subprocess.run(
    ["git", "rev-parse", "--verify", "HEAD"],
    cwd=root, capture_output=True, text=True, encoding="ascii",
)
if final_git_status.returncode or final_git_status.stdout or \
        final_git_head.returncode or final_git_head.stdout.strip() != git_head:
    raise RuntimeError("生成期间Git HEAD、tracked工作树或index发生变化")
for relative, expected in code_sha256.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"生成期间正式代码发生变化：{relative}")
for name, path in input_paths.items():
    sidecar = Path(str(path) + ".sha256")
    if hashlib.sha256(path.read_bytes()).hexdigest() != verified_inputs[name]["sha256"] or \
            hashlib.sha256(sidecar.read_bytes()).hexdigest() != \
            verified_inputs[name]["sidecar_text_sha256"]:
        raise RuntimeError(f"生成期间正式输入或旁车发生变化：{path}")
for name, item in selection_artifacts.items():
    if hashlib.sha256(selection_artifact_paths[name].read_bytes()).hexdigest() != item["sha256"]:
        raise RuntimeError(f"生成期间70号选择证据发生变化：{item['path']}")
for relative, expected in selection.get("input_sha256", {}).items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"生成期间70号开发输入发生变化：{relative}")
for name, expected in history_sha256.items():
    if hashlib.sha256((protocol_dir / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"生成期间历史清单发生变化：{name}")
for relative, expected in auxiliary_sha256.items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"生成期间辅助几何输入发生变化：{relative}")
late_forbidden_markers = sorted({
    path for path in [*protocol_dir.glob("hierarchical_v2*consumed*.json"),
                      stage_markers["validation"]]
    if path.exists() or path.is_symlink()
}, key=str)
late_forbidden_outputs = sorted({
    path for path in [*adaptive_root.glob("formal_hierarchical_v2*"),
                      adaptive_root / "formal_hierarchical_v1_validation"]
    if path.exists() or path.is_symlink()
}, key=str)
late_forbidden_run_locks = sorted({
    path for path in run_lock_candidates if path.exists() or path.is_symlink()
}, key=str)
late_forbidden_seals = sorted(protocol_dir.glob("hierarchical_v2*seal*.json"))
for stage_output in stage_outputs.values():
    for name in ("calibration_seal.json", "invalid_calibration_seal.json"):
        seal = stage_output / name
        if seal.exists() or Path(str(seal) + ".sha256").exists():
            late_forbidden_seals.append(seal)
if late_forbidden_markers or late_forbidden_outputs or \
        late_forbidden_run_locks or late_forbidden_seals:
    raise FileExistsError(
        "生成期间 hierarchical-v2 正式消费状态发生变化，拒绝写锁："
        f"markers={list(map(str, late_forbidden_markers))}, "
        f"outputs={list(map(str, late_forbidden_outputs))}, "
        f"run_locks={list(map(str, late_forbidden_run_locks))}, "
        f"seals={list(map(str, late_forbidden_seals))}"
    )


# %% 9. 首次用xb创建；已有锁只允许逐字节相同，--check绝不创建。
output_payload = (json.dumps(
    execution_lock, ensure_ascii=False, sort_keys=True,
    separators=(",", ":")) + "\n").encode("utf-8")
output_sha256 = hashlib.sha256(output_payload).hexdigest()
sidecar_payload = f"{output_sha256}  {output_path.name}\n".encode("ascii")
if output_path.exists() or output_sidecar.exists():
    if not output_path.is_file() or not output_sidecar.is_file() or \
            output_path.read_bytes() != output_payload or \
            output_sidecar.read_bytes() != sidecar_payload:
        raise ValueError("已有 hierarchical-v2 执行锁不同或不完整，拒绝覆盖")
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
    "selected_gate_score": algorithm["gate"]["score_kind"],
    "stage_case_counts": {
        stage: specification["case_count"] for stage, specification in manifests.items()},
    "formal_cases_run": False,
    "validation_manifest_written": False,
}, ensure_ascii=False, indent=2))
