"""# %% 冻结 v3 完整盲定位正式执行锁；只核对开发证据，不运行正式病例。"""

# %% 1. 路径与不可变输入。23/48/54 的最终版本由本脚本执行时直接冻结。
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

root = Path(__file__).resolve().parents[1]
if not __debug__:
    raise RuntimeError("冻结执行锁禁止使用 python -O；否则 assert 完整性检查会被关闭")
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
development_dir = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/blind_gate_combined40_localization_adaptive_mix_balanced_full20"
development_run_dir = root / "results/erp_whole_head/adaptive_v6/dev_component_balanced_excess_retry5_full20"
output_path = protocol_dir / "component_balanced_v3_execution_lock.json"
output_sidecar = output_path.with_suffix(output_path.suffix + ".sha256")

input_paths = {
    "protocol_lock": protocol_dir / "component_balanced_v3_protocol_lock.json",
    "calibration_manifest": protocol_dir / "calibration_component_balanced_v3_manifest.json",
    "v2_retirement": protocol_dir / "component_balanced_v2_calibration_retirement.json",
    "validation_manifest": protocol_dir / "validation_component_balanced_v2_manifest.json",
    "protocol_document": protocol_dir / "V3_PROTOCOL.md",
}
expected_sha256 = {
    "protocol_lock": "6652685b66bea4b9edd3fa81785afc944103a70aaa4a30ccc6eff921a7bae306",
    "calibration_manifest": "bb9f1a7d949efa6ba91dc8aceff604edca79752ecfb507134d06e4ccf56421d2",
    "v2_retirement": "8874a883e34037ad13cd28e820bc64b3b0f1245e95162c1fea44210886f07114",
    "validation_manifest": "68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7",
    "protocol_document": "59a1b6bc94088625d908e359df8b543f54ccdbf21313eb50d5bfab22f1648583",
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
retirement = json.loads(input_paths["v2_retirement"].read_text(encoding="utf-8"))
validation_manifest = json.loads(input_paths["validation_manifest"].read_text(encoding="utf-8"))


# %% 2. v2 失败校准永久保留且只供诊断；v3 校准和原 validation 都尚未消费。
legacy_marker_path = protocol_dir / "calibration_manifest_consumed.json"
legacy_marker_payload = legacy_marker_path.read_bytes()
legacy_marker_sha256 = hashlib.sha256(legacy_marker_payload).hexdigest()
legacy_marker_key = legacy_marker_path.relative_to(root).as_posix()
legacy_marker = json.loads(legacy_marker_payload)
assert retirement["protocol"] == "erp-v6-component-balanced-v2-calibration-retirement-v1"
assert retirement["status"] == "retired_invalid_solver_convergence"
assert retirement["formal_calibration_reuse_allowed"] is False
assert retirement["allowed_use"] == "development_solver_diagnosis_only"
assert retirement["preserved_artifacts_sha256"][legacy_marker_key] == legacy_marker_sha256
assert legacy_marker["manifest_sha256"] == \
    retirement["preserved_artifacts_sha256"]["results/erp_whole_head/adaptive_v6/protocol/calibration_component_balanced_v2_manifest.json"]
assert legacy_marker["execution_lock_sha256"] == \
    retirement["preserved_artifacts_sha256"]["results/erp_whole_head/adaptive_v6/protocol/component_balanced_v2_execution_lock.json"]
assert not (protocol_dir / "calibration_component_balanced_v3_manifest_consumed.json").exists()
assert not (protocol_dir / "validation_manifest_consumed.json").exists()

assert protocol_lock["protocol"] == "erp-v6-component-balanced-formal-v3-independent-calibration"
assert protocol_lock["status"] == "algorithm_not_yet_frozen"
assert not protocol_lock["formal_calibration_allowed"] and not protocol_lock["formal_validation_allowed"]
assert protocol_lock["calibration_or_validation_run_by_generator"] is False
assert protocol_lock["validation_results_read_by_generator"] is False
assert protocol_lock["protected_calibration_manifest"] == {
    "path": input_paths["calibration_manifest"].name,
    "sha256": expected_sha256["calibration_manifest"],
}
assert protocol_lock["protected_validation_manifest"]["path"] == input_paths["validation_manifest"].name
assert protocol_lock["protected_validation_manifest"]["sha256"] == expected_sha256["validation_manifest"]
assert protocol_lock["protected_validation_manifest"]["consumed_marker_present_at_freeze"] is False
assert protocol_lock["retired_v2_calibration"] == {
    "path": input_paths["v2_retirement"].name,
    "sha256": expected_sha256["v2_retirement"],
    "legacy_consumed_marker_preserved": True,
}
assert protocol_lock["protocol_document"] == input_paths["protocol_document"].name
assert protocol_lock["protocol_document_sha256"] == expected_sha256["protocol_document"]


# %% 3. v3 校准是 57 个独立纯表层配置；每个 SNR 均为 10 单源、9 双源且分量平衡。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
calibration_seed_roots = {"fit": 2026092501, "check": 2026092502}
validation_seed_roots = {"fit": 2026092403, "check": 2026092404}
assert len(calibration_manifest) == protocol_lock["calibration"]["case_count"] == 57
assert len({case["case_id"] for case in calibration_manifest}) == 57
assert len({case["configuration_id"] for case in calibration_manifest}) == 57
assert all(case["scenario"] == "surface_only" and case["deep_index"] is None
           for case in calibration_manifest)
assert all(case["replica_seed_roots"] == calibration_seed_roots for case in calibration_manifest)
assert Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in calibration_manifest) == \
    Counter({pair: 19 for pair in snr_pairs})
for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in calibration_manifest
            if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    assert Counter(case["configuration_kind"] for case in cell) == \
        {"single_surface": 10, "two_surface_only": 9}
    assert sum(len(case["surface_centers"]) == 1 for case in cell) == 10
    assert sum(len(case["surface_centers"]) == 2 for case in cell) == 9
    assert all(case.get("surface_component_sensor_balance") is True
               for case in cell if len(case["surface_centers"]) == 2)
    expected_correlations = {0.0: 5, .5: 4} if pair_index != 1 else {0.0: 4, .5: 5}
    assert Counter(case["correlation"] for case in cell
                   if len(case["surface_centers"]) == 2) == expected_correlations
selected_centers = [center for case in calibration_manifest for center in case["surface_centers"]]
assert len(selected_centers) == len(set(selected_centers)) == 84
assert protocol_lock["calibration"]["replica_seed_roots"] == calibration_seed_roots
assert protocol_lock["calibration"]["configuration_count"] == 57
assert protocol_lock["calibration"]["configurations_repeated_across_snr"] is False
assert protocol_lock["calibration"]["two_surface_cases_sensor_balanced"] == 27
assert protocol_lock["geometry"]["selected_center_count"] == 84
assert protocol_lock["geometry"]["selected_lh_count"] == protocol_lock["geometry"]["selected_rh_count"] == 42
assert protocol_lock["geometry"]["protected_support_overlap_count"] == 0
assert protocol_lock["geometry"]["within_v3_support_overlap_count"] == 0
assert protocol_lock["geometry"]["minimum_double_center_distance_mm"] >= 50.0
assert protocol_lock["geometry"]["selected_centers_by_case_id"] == {
    case["case_id"]: case["surface_centers"] for case in calibration_manifest}


# %% 4. 受保护 validation 逐字节沿用 v2 清单，病例、平衡和 seed 均不改变。
assert len(validation_manifest) == protocol_lock["validation"]["case_count"] == 24
assert len({case["case_id"] for case in validation_manifest}) == 24
assert Counter(case["configuration_id"] for case in validation_manifest) == Counter(
    {configuration: 3 for configuration in {case["configuration_id"] for case in validation_manifest}})
assert len({case["configuration_id"] for case in validation_manifest}) == 8
assert all(case["component_balance_revision"] == "formal_component_balanced_v2"
           for case in validation_manifest)
assert all(case["replica_seed_roots"] == validation_seed_roots for case in validation_manifest)
assert Counter((case["eeg_snr_db"], case["meg_snr_db"], case["deep_index"] is not None)
               for case in validation_manifest) == Counter(
    {(eeg, meg, present): 4 for eeg, meg in snr_pairs for present in (False, True)})
validation_two_surface = [case for case in validation_manifest if len(case["surface_centers"]) == 2]
validation_mixed = [case for case in validation_manifest
                    if case["deep_index"] is not None and case["surface_centers"]]
assert len(validation_two_surface) == 9
assert all(case.get("surface_component_sensor_balance") is True for case in validation_two_surface)
assert len(validation_mixed) == 6
assert all(case["deep_surface_sensor_amplitude_ratio"] == .5 for case in validation_mixed)
assert protocol_lock["validation"]["replica_seed_roots"] == validation_seed_roots
assert protocol_lock["protected_validation_manifest"]["case_count"] == 24


# %% 5. 开发闭环完整通过；retry 5→10 只延长未收敛任务的预算，不改变 20 个已收敛结果。
development_summary_path = development_dir / "summary.json"
development_metadata_path = development_run_dir / "metadata.json"
development_evidence_path = development_run_dir / "evidence.csv"
development_summary_payload = development_summary_path.read_bytes()
development_metadata_payload = development_metadata_path.read_bytes()
development_evidence_payload = development_evidence_path.read_bytes()
development_summary_sha256 = hashlib.sha256(development_summary_payload).hexdigest()
development_metadata_sha256 = hashlib.sha256(development_metadata_payload).hexdigest()
development_evidence_sha256 = hashlib.sha256(development_evidence_payload).hexdigest()
assert development_summary_sha256 == "8fdd244885f84045a8b97d97be55a53ae1ead6726b8001f89b626288e878b8c6"
assert development_metadata_sha256 == "8e5747a51543cc97155a4c14362129a6789bc508f2eb87851b9e6d9712735766"
assert development_evidence_sha256 == "ab699aadf8d61165d4559e10f9697e3957bb2458c246e6919e7b6534f8e1eed9"
development_summary = json.loads(development_summary_payload)
development_metadata = json.loads(development_metadata_payload)
development_evidence = list(csv.DictReader(
    development_evidence_payload.decode("utf-8-sig").splitlines()))
development_solver = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 1, "max_inner_retries": 5,
}
formal_solver = {
    "solver_kind": "admm", "mrf_strength": .8,
    "outer_iterations": 1, "max_inner_retries": 10,
}
assert development_metadata["solver_settings"] == development_summary["solver_settings"] == development_solver
assert {key: value for key, value in formal_solver.items() if key != "max_inner_retries"} == \
    {key: value for key, value in development_solver.items() if key != "max_inner_retries"}
assert formal_solver["max_inner_retries"] > development_solver["max_inner_retries"]
assert len(development_evidence) == 20
assert {row["case_id"] for row in development_evidence} == set(development_metadata["case_ids"])
assert all(row["null_converged"] == row["full_converged"] == row["all_converged"] == "1"
           for row in development_evidence)
assert development_summary["complete"] and development_summary["phase"] == "development"
assert development_summary["case_count"] == 20
assert development_summary["covariance"] == "trial" and development_summary["score_kind"] == "excess"
assert development_summary["family_correct_posthoc_count"] == 20
assert development_summary["pure_surface_false_positive_count"] == 0
assert development_summary["deep_true_detected_count"] == 12
assert development_summary["all_selected_families_converged"]
assert development_summary["all_conditional_local_AUC_at_least_0p9"]
assert development_summary["minimum_local_AUC"] >= .90
assert development_summary["maximum_surface_DLE_mm"] <= 15.0
assert development_summary["maximum_surface_SD_mm"] <= 20.0
assert development_summary["all_deep_DLE_zero_when_deep_true"]
assert max(item["maximum_deep_DLE_mm"] for item in development_summary["snr_summary"]) == 0.0

script48 = root / "作者风格版/48-独立门控后40试次盲定位.py"
script48_sha256 = hashlib.sha256(script48.read_bytes()).hexdigest()
assert script48_sha256 == development_summary["localization_script_sha256"], \
    "48 号定位实现与已通过的开发结果不一致"


# %% 6. 冻结正式入口、定位器和已有 code_sha256 全集；求解器实现不得偏离开发运行。
assert development_metadata["phase"] == "development"
assert development_metadata["manifest_sha256"] == \
    "c64d02429fd5e9e9c2a7831a36b7f86c2a0fe614ecb2c837fa0284fe96fb5425"
assert development_metadata["seed_root"] == 2026092206
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


# %% 7. 固定统计量、求解器、定位、pooled 校准决策及一次性盲验证硬门限。
localization_rule = dict(development_summary["localization_parameters"])
localization_rule.pop("amplitude_evidence_mix_range", None)
assert localization_rule == {
    "svd_modes": 2, "sensitivity_floor_fraction": .1, "depth_exponent": .6,
    "graph_steps": 2, "support_seed_fraction": .1, "support_radius_m": .015,
    "amplitude_evidence_mix_rule": "0.5 * training_MEG / max(training_EEG, training_MEG)",
}
calibration_decision = protocol_lock["calibration"]["decision"]
assert calibration_decision == {
    "accept_deep": "T > 0 AND p(T) <= 0.05",
    "alpha": .05,
    "calibration_score_count": 57,
    "claim_scope": "marginal engineering FPR for the equal-weight three-SNR mixture only; no per-SNR or population guarantee",
    "equivalent_boundary": "T must be strictly greater than the second-largest of all 57 calibration scores",
    "final_validation_requirement": "retain the predeclared zero false positives among four H0 cases in every SNR cell",
    "ideal_continuous_exchangeable_tail_bound": 2 / 58,
    "maximum_calibration_scores_greater_or_equal": 1,
    "missing_nonfinite_or_unconverged_score_action": "calibration_invalid_do_not_open_validation",
    "rank_formula": "p(T) = (1 + count(T_cal >= T)) / 58",
    "required_complete_finite_scores": 57,
    "strata_count": 1,
    "test_true_snr_used": False,
    "ties_count_against_acceptance": True,
}
acceptance = {
    "required_false_positives_each_snr": 0,
    "required_localized_deep_detections_of_12": 10,
    "all_selected_families_converged": True,
    "minimum_local_AUC": .90,
    "maximum_surface_DLE_mm": 15.0,
    "maximum_surface_SD_mm": 20.0,
    "maximum_deep_DLE_mm": 10.0,
    "existing_deep_acceptance_passed_required": True,
}


# %% 8. 新锁只允许校准；完整且全收敛的同锁校准才可另行解锁 validation。
execution_lock = {
    "protocol": "erp-v6-component-balanced-formal-v3-execution",
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
    "retired_v2_calibration": {
        **verified_inputs["v2_retirement"],
        "legacy_consumed_marker_path": legacy_marker_key,
        "legacy_consumed_marker_sha256": legacy_marker_sha256,
        "formal_reuse_allowed": False,
        "allowed_use": "development_solver_diagnosis_only",
    },
    "development_evidence": {
        "summary_path": development_summary_path.relative_to(root).as_posix(),
        "summary_sha256": development_summary_sha256,
        "metadata_path": development_metadata_path.relative_to(root).as_posix(),
        "metadata_sha256": development_metadata_sha256,
        "evidence_path": development_evidence_path.relative_to(root).as_posix(),
        "evidence_sha256": development_evidence_sha256,
        "case_count": 20, "gate_correct": "20/20", "false_positives": "0/8",
        "true_positives": "12/12", "null_models_converged": "20/20",
        "full_models_converged": "20/20",
        "minimum_local_AUC": development_summary["minimum_local_AUC"],
        "maximum_surface_DLE_mm": development_summary["maximum_surface_DLE_mm"],
        "maximum_surface_SD_mm": development_summary["maximum_surface_SD_mm"],
        "maximum_deep_DLE_mm": 0.0,
        "development_threshold_not_used_formally": development_summary["development_threshold"],
        "retry_cap_extension": {
            "development_max_inner_retries": 5,
            "formal_max_inner_retries": 10,
            "only_solver_setting_changed": "max_inner_retries",
            "completed_development_results_preserved": True,
            "basis": "all 20 H0 and all 20 H1 fits converged under cap 5; cap 10 only adds unchanged-weight budgets after the old cap for future hard cases",
        },
    },
    "algorithm": {
        "covariance": "trial",
        "score_kind": "excess",
        "decision_statistic": "independent confirmation excess-loss fraction",
        "excess_loss_denominator_floor_fraction": .05,
        "solver_settings": formal_solver,
        "localization_input": "combine train20 and confirmation20 only after the blind family decision",
        "localization_family": "combined40 H1 when the frozen pooled conformal decision accepts deep presence, otherwise combined40 H0",
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


# %% 9. 相同内容可复核，不同内容永不覆盖；写出标准 SHA256 旁车。
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
print("v3 完整盲定位算法已冻结；只允许正式校准，未运行任何正式病例。")
