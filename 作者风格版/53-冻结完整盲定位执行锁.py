"""# %% 冻结完整盲定位正式执行锁；只核对开发证据，不运行正式病例。"""

# %% 1. 路径与不可变输入。23/54 可并行修改，所以只在本脚本真正执行时读取最终哈希。
from collections import Counter
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
if not __debug__:
    raise RuntimeError("冻结执行锁禁止使用python -O；否则assert完整性检查会被关闭")
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
development_dir = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/blind_gate_combined40_localization_adaptive_mix_balanced_full20"
development_run_dir = root / "results/erp_whole_head/adaptive_v6/dev_component_balanced_excess_retry5_full20"
output_path = protocol_dir / "component_balanced_v2_execution_lock.json"
output_sidecar = output_path.with_suffix(output_path.suffix + ".sha256")

input_paths = {
    "protocol_lock": protocol_dir / "component_balanced_v2_lock.json",
    "calibration_manifest": protocol_dir / "calibration_component_balanced_v2_manifest.json",
    "validation_manifest": protocol_dir / "validation_component_balanced_v2_manifest.json",
}
expected_sha256 = {
    "protocol_lock": "114e4216270108eecdf4a89c03c682a9e60bd38cc43fa6ec35a0448600a1f466",
    "calibration_manifest": "4d091c2cbc12d1ec2165ff27e754bf1e3d540c2b4f00c89d3967ac6a3e89eefb",
    "validation_manifest": "68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7",
}
verified_inputs = {}
for name, path in input_paths.items():
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_text = sidecar.read_text(encoding="ascii")
    assert digest == expected_sha256[name], f"冻结输入已变化：{path}"
    assert sidecar_text == f"{digest}  {path.name}\n", f"旁车不匹配：{sidecar}"
    verified_inputs[name] = {
        "path": path.relative_to(root).as_posix(), "sha256": digest,
        "sidecar_path": sidecar.relative_to(root).as_posix(),
        "sidecar_text_sha256": hashlib.sha256(sidecar_text.encode("ascii")).hexdigest(),
    }

protocol_lock = json.loads(input_paths["protocol_lock"].read_text(encoding="utf-8"))
calibration_manifest = json.loads(input_paths["calibration_manifest"].read_text(encoding="utf-8"))
validation_manifest = json.loads(input_paths["validation_manifest"].read_text(encoding="utf-8"))


# %% 2. 正式病例协议仍未消费，清单身份、数量和独立 seed roots 均与协议锁一致。
assert protocol_lock["protocol"] == "erp-v6-component-balanced-formal-v2"
assert protocol_lock["status"] == "algorithm_not_yet_frozen"
assert not protocol_lock["formal_calibration_allowed"] and not protocol_lock["formal_validation_allowed"]
assert protocol_lock["protected_manifest_sha256"] == {
    input_paths["calibration_manifest"].name: expected_sha256["calibration_manifest"],
    input_paths["validation_manifest"].name: expected_sha256["validation_manifest"],
}
assert not (protocol_dir / "calibration_manifest_consumed.json").exists()
assert not (protocol_dir / "validation_manifest_consumed.json").exists()
assert len(calibration_manifest) == protocol_lock["calibration"]["case_count"] == 57
assert len(validation_manifest) == protocol_lock["validation"]["case_count"] == 24
assert len({case["case_id"] for case in calibration_manifest}) == 57
assert len({case["case_id"] for case in validation_manifest}) == 24
assert Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in calibration_manifest) == Counter(
    {(-10, -10): 19, (-10, 20): 19, (20, -10): 19})
assert Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in validation_manifest) == Counter(
    {(-10, -10): 8, (-10, 20): 8, (20, -10): 8})
calibration_seed_roots = {"fit": 2026092401, "check": 2026092402}
validation_seed_roots = {"fit": 2026092403, "check": 2026092404}
assert protocol_lock["calibration"]["replica_seed_roots"] == calibration_seed_roots
assert protocol_lock["validation"]["replica_seed_roots"] == validation_seed_roots
assert all(case["replica_seed_roots"] == calibration_seed_roots for case in calibration_manifest)
assert all(case["replica_seed_roots"] == validation_seed_roots for case in validation_manifest)


# %% 3. 开发闭环必须完整通过；0.12 只证明开发闭环，不进入正式决策。
development_summary_path = development_dir / "summary.json"
development_summary_payload = development_summary_path.read_bytes()
development_summary_sha256 = hashlib.sha256(development_summary_payload).hexdigest()
assert development_summary_sha256 == "8fdd244885f84045a8b97d97be55a53ae1ead6726b8001f89b626288e878b8c6"
development_summary = json.loads(development_summary_payload)
expected_solver = {"solver_kind": "admm", "mrf_strength": 0.8,
                   "outer_iterations": 1, "max_inner_retries": 5}
assert development_summary["complete"] and development_summary["phase"] == "development"
assert development_summary["case_count"] == 20
assert development_summary["solver_settings"] == expected_solver
assert development_summary["covariance"] == "trial"
assert development_summary["score_kind"] == "excess"
assert development_summary["family_correct_posthoc_count"] == 20
assert development_summary["pure_surface_false_positive_count"] == 0
assert development_summary["deep_true_detected_count"] == 12
assert development_summary["all_selected_families_converged"]
assert development_summary["all_conditional_local_AUC_at_least_0p9"]
assert development_summary["minimum_local_AUC"] >= 0.90
assert development_summary["maximum_surface_DLE_mm"] <= 15.0
assert development_summary["maximum_surface_SD_mm"] <= 20.0
assert development_summary["all_deep_DLE_zero_when_deep_true"]
assert max(item["maximum_deep_DLE_mm"] for item in development_summary["snr_summary"]) == 0.0

script48 = root / "作者风格版/48-独立门控后40试次盲定位.py"
script48_sha256 = hashlib.sha256(script48.read_bytes()).hexdigest()
assert script48_sha256 == development_summary["localization_script_sha256"], \
    "48 号定位实现与已通过的开发结果不一致"


# %% 4. 冻结正式入口、定位器和已有 code_sha256 全集；除正式入口外不得偏离开发运行。
development_metadata_path = development_run_dir / "metadata.json"
development_metadata_payload = development_metadata_path.read_bytes()
development_metadata = json.loads(development_metadata_payload)
assert development_metadata["phase"] == "development"
assert development_metadata["manifest_sha256"] == \
    "c64d02429fd5e9e9c2a7831a36b7f86c2a0fe614ecb2c837fa0284fe96fb5425"
assert development_metadata["seed_root"] == 2026092206
assert development_metadata["solver_settings"] == expected_solver
assert development_metadata["covariance"] == "trial" and development_metadata["score_kind"] == "excess"
assert development_metadata["shared_fingerprint"] == protocol_lock["shared_fingerprint"]

development_code_sha256 = development_metadata["code_sha256"]
formal_entry = str(Path("作者风格版/23-独立观测浅深检出仿真.py"))
for relative, expected in development_code_sha256.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if relative != formal_entry:
        assert actual == expected, f"开发后算法/仿真/指标代码已变化：{relative}"

formal_files = set(development_code_sha256) | {
    formal_entry,
    str(Path("作者风格版/48-独立门控后40试次盲定位.py")),
    str(Path("作者风格版/54-完整盲定位正式验收.py")),
}
code_sha256 = {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                  for relative in sorted(formal_files)}
assert code_sha256[str(Path("作者风格版/48-独立门控后40试次盲定位.py"))] == script48_sha256


# %% 5. 固定统计量、求解器、定位、校准决策及一次性盲验证硬门限。
localization_rule = dict(development_summary["localization_parameters"])
localization_rule.pop("amplitude_evidence_mix_range", None)
assert localization_rule == {
    "svd_modes": 2, "sensitivity_floor_fraction": 0.1, "depth_exponent": 0.6,
    "graph_steps": 2, "support_seed_fraction": 0.1, "support_radius_m": 0.015,
    "amplitude_evidence_mix_rule": "0.5 * training_MEG / max(training_EEG, training_MEG)",
}
calibration_decision = protocol_lock["calibration"]["decision"]
assert calibration_decision["alpha"] == 0.05
assert calibration_decision["required_complete_scores_per_stratum"] == 19
assert calibration_decision["strata_count"] == 3
assert calibration_decision["test_true_snr_used_to_select_calibration_stratum"] is False
assert calibration_decision["missing_nonfinite_or_unconverged_score_action"] == \
    "calibration_invalid_do_not_open_validation"

acceptance = {
    "required_false_positives_each_snr": 0,
    "required_localized_deep_detections_of_12": 10,
    "all_selected_families_converged": True,
    "minimum_local_AUC": 0.90,
    "maximum_surface_DLE_mm": 15.0,
    "maximum_surface_SD_mm": 20.0,
    "maximum_deep_DLE_mm": 10.0,
    "existing_deep_acceptance_passed_required": True,
}


# %% 6. 新锁只允许校准；验证必须由同一锁的完整校准结果另行解锁，锁本身永不改写。
execution_lock = {
    "protocol": "erp-v6-component-balanced-formal-v2-execution",
    "status": "algorithm_frozen_calibration_allowed",
    "formal_calibration_allowed": True,
    "formal_validation_allowed": False,
    "validation_unlock_rule": (
        "Require a complete, finite, all-converged frozen calibration carrying this execution-lock "
        "SHA256; never edit this lock to open validation."),
    "formal_cases_run_by_generator": False,
    "consumed_marker_created_by_generator": False,
    "protocol_inputs": verified_inputs,
    "manifests": {
        "calibration": {**verified_inputs["calibration_manifest"], "case_count": 57,
                        "seed_roots": calibration_seed_roots},
        "validation": {**verified_inputs["validation_manifest"], "case_count": 24,
                       "seed_roots": validation_seed_roots},
    },
    "development_evidence": {
        "summary_path": development_summary_path.relative_to(root).as_posix(),
        "summary_sha256": development_summary_sha256,
        "metadata_path": development_metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": hashlib.sha256(development_metadata_payload).hexdigest(),
        "case_count": 20, "gate_correct": "20/20", "false_positives": "0/8",
        "true_positives": "12/12", "minimum_local_AUC": development_summary["minimum_local_AUC"],
        "maximum_surface_DLE_mm": development_summary["maximum_surface_DLE_mm"],
        "maximum_surface_SD_mm": development_summary["maximum_surface_SD_mm"],
        "maximum_deep_DLE_mm": 0.0,
        "development_threshold_not_used_formally": development_summary["development_threshold"],
    },
    "algorithm": {
        "covariance": "trial",
        "score_kind": "excess",
        "decision_statistic": "independent confirmation excess-loss fraction",
        "excess_loss_denominator_floor_fraction": 0.05,
        "solver_settings": expected_solver,
        "localization_input": "combine train20 and confirmation20 only after the blind family decision",
        "localization_family": "combined40 H1 when all frozen null strata accept deep presence, otherwise combined40 H0",
        "localization_rule": localization_rule,
        "calibration_decision": calibration_decision,
    },
    "acceptance": acceptance,
    "benchmark_deep_acceptance": {
        "implementation": "benchmark/deep_acceptance.py",
        "passed_required": True,
        "protocol_thresholds": protocol_lock["validation"]["acceptance"],
    },
    "shared_fingerprint": protocol_lock["shared_fingerprint"],
    "environment": protocol_lock["environment"],
    "development_code_sha256": development_code_sha256,
    "code_sha256": code_sha256,
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


# %% 7. 相同内容可复核，不同内容永不覆盖；写出标准 SHA256 旁车。
output_payload = (json.dumps(execution_lock, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")) + "\n").encode("utf-8")
output_sha256 = hashlib.sha256(output_payload).hexdigest()
sidecar_payload = f"{output_sha256}  {output_path.name}\n".encode("ascii")
if output_path.exists() or output_sidecar.exists():
    assert output_path.read_bytes() == output_payload, f"冻结内容不同，拒绝覆盖：{output_path}"
    assert output_sidecar.read_bytes() == sidecar_payload, f"冻结旁车不同，拒绝覆盖：{output_sidecar}"
else:
    with output_path.open("xb") as stream:
        stream.write(output_payload)
    with output_sidecar.open("xb") as stream:
        stream.write(sidecar_payload)
print(output_path.name, output_sha256)
print("完整盲定位算法已冻结；只允许正式校准，未运行任何正式病例。")
