"""# %% 冻结 v3 独立纯表层校准协议；不运行校准、验证或任何反演。"""

# %% 1. 路径、冻结输入和只读保护。
from collections import Counter
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys

if not __debug__:
    raise RuntimeError("协议生成/核验禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol
import run_erp_whole_head_matrix as simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="只核对已冻结文件，不创建或覆盖")
args = parser.parse_args()
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
v2_result_dir = root / "results/erp_whole_head/adaptive_v6/formal_component_balanced_v2_calibration"
input_paths = {
    "lock.json": protocol_dir / "lock.json",
    "development_component_balanced_all_snr.json": protocol_dir / "development_component_balanced_all_snr.json",
    "calibration_component_balanced_v2_manifest.json": protocol_dir / "calibration_component_balanced_v2_manifest.json",
    "validation_component_balanced_v2_manifest.json": protocol_dir / "validation_component_balanced_v2_manifest.json",
    "component_balanced_v2_lock.json": protocol_dir / "component_balanced_v2_lock.json",
    "component_balanced_v2_execution_lock.json": protocol_dir / "component_balanced_v2_execution_lock.json",
    "calibration_manifest_consumed.json": protocol_dir / "calibration_manifest_consumed.json",
    "v2_completion.json": v2_result_dir / "completion.json",
    "v2_frozen_calibration.json": v2_result_dir / "frozen_calibration.json",
    "v2_evidence.csv": v2_result_dir / "evidence.csv",
}
expected_sha256 = {
    "lock.json": "eb1625ef1be95553ebf5094bf3c08a4df24bcb007038fb18fab31b4f3a685bc3",
    "development_component_balanced_all_snr.json": "c64d02429fd5e9e9c2a7831a36b7f86c2a0fe614ecb2c837fa0284fe96fb5425",
    "calibration_component_balanced_v2_manifest.json": "4d091c2cbc12d1ec2165ff27e754bf1e3d540c2b4f00c89d3967ac6a3e89eefb",
    "validation_component_balanced_v2_manifest.json": "68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7",
    "component_balanced_v2_lock.json": "114e4216270108eecdf4a89c03c682a9e60bd38cc43fa6ec35a0448600a1f466",
    "component_balanced_v2_execution_lock.json": "ff2bd8382f6bb9bd3fa17083cb36940a1bb9840164917160cd8150586be0a706",
    "calibration_manifest_consumed.json": "6effab634d56f6e4256183996396fd5776b0a859abb5062c2a40a291c26c506e",
    "v2_completion.json": "57cdd1fc17e470323c96819a84c20acaca72e19d90fe074376b5c9cbb99f5ae9",
    "v2_frozen_calibration.json": "03a7970f0618259b85ce4b41095186aee94c0ebc0b4a849556d2cd1735e94bda",
    "v2_evidence.csv": "c34000f9d7df12717811dc853767cc56c559c67f20f4de860c1b150d5a8b2b5b",
}
payloads = {}
for name, path in input_paths.items():
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256[name]:
        raise ValueError(f"冻结输入已变化：{path}")
    payloads[name] = payload
for name in ("lock.json", "calibration_component_balanced_v2_manifest.json",
             "validation_component_balanced_v2_manifest.json", "component_balanced_v2_lock.json",
             "component_balanced_v2_execution_lock.json"):
    path = input_paths[name]
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="ascii") != f"{expected_sha256[name]}  {path.name}\n":
        raise ValueError(f"冻结输入旁车已变化：{sidecar}")
if (protocol_dir / "validation_manifest_consumed.json").exists():
    raise FileExistsError("受保护 validation 已消费，不能再建 v3 协议")

old_lock = json.loads(payloads["lock.json"])
development = json.loads(payloads["development_component_balanced_all_snr.json"])
v2_calibration = json.loads(payloads["calibration_component_balanced_v2_manifest.json"])
protected_validation = json.loads(payloads["validation_component_balanced_v2_manifest.json"])
v2_protocol_lock = json.loads(payloads["component_balanced_v2_lock.json"])
v2_execution_lock = json.loads(payloads["component_balanced_v2_execution_lock.json"])
v2_consumed = json.loads(payloads["calibration_manifest_consumed.json"])
v2_completion = json.loads(payloads["v2_completion.json"])
if v2_protocol_lock["protected_manifest_sha256"]["validation_component_balanced_v2_manifest.json"] != \
        expected_sha256["validation_component_balanced_v2_manifest.json"] or \
        v2_execution_lock["protocol"] != "erp-v6-component-balanced-formal-v2-execution":
    raise ValueError("v2 协议/执行锁与受保护 validation 不符")
if len(protected_validation) != 24 or len({case["case_id"] for case in protected_validation}) != 24:
    raise ValueError("受保护 v2 validation manifest 不完整")
if not (v2_completion["complete"] and v2_completion["case_count"] == 57 and not v2_completion["all_converged"]):
    raise ValueError("v2 校准退役原因与冻结 completion 不符")
if v2_consumed["manifest_sha256"] != expected_sha256["calibration_component_balanced_v2_manifest.json"] or \
        v2_consumed["execution_lock_sha256"] != expected_sha256["component_balanced_v2_execution_lock.json"]:
    raise ValueError("v2 consumed marker 与冻结清单/执行锁不符")

evidence_rows = list(csv.DictReader(payloads["v2_evidence.csv"].decode("utf-8-sig").splitlines()))
nonconverged = [row for row in evidence_rows if row["all_converged"] != "1"]
if len(evidence_rows) != 57 or len(nonconverged) != 6 or \
        any(row["null_converged"] != "1" or row["full_converged"] != "0" or
            (int(row["eeg_snr_db"]), int(row["meg_snr_db"])) != (-10, 20) for row in nonconverged):
    raise ValueError("v2 校准收敛遥测与退役记录不符")


# %% 2. 只按冻结几何选 57 个不重复配置；不使用任何校准分数或 validation 结果。
shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
if not shared_fingerprint == old_lock["shared_fingerprint"] == v2_protocol_lock["shared_fingerprint"]:
    raise ValueError("共享几何指纹已变化")
n_surf = int(shared["n_surf"])
n_lh = len(shared["src_surface"][0]["vertno"])
vertices = np.asarray(shared["vertices"], dtype=float)
surface_patches = [set(map(int, protocol._surface_patch(shared, center)[0])) for center in range(n_surf)]

historical_centers = set(map(int, old_lock["previously_used_surface_centers"]))
for cases in (development, v2_calibration, protected_validation):
    historical_centers.update(int(center) for case in cases for center in case["surface_centers"])
protected_panel_centers = {
    "historical_before_current_development": set(map(int, old_lock["previously_used_surface_centers"])),
    "development": {int(center) for case in development for center in case["surface_centers"]},
    "consumed_v2_calibration": {int(center) for case in v2_calibration for center in case["surface_centers"]},
    "protected_v2_validation": {int(center) for case in protected_validation for center in case["surface_centers"]},
}
protected_panel_supports = {}
for panel, centers in protected_panel_centers.items():
    protected_panel_supports[panel] = set()
    for center in sorted(centers):
        protected_panel_supports[panel].update(surface_patches[center])
protected_support = set()
for center in sorted(historical_centers):
    protected_support.update(surface_patches[center])
available = {center for center in range(n_surf) if surface_patches[center].isdisjoint(protected_support)}
available_before_selection = len(available)

geometry_inputs = [expected_sha256[name] for name in (
    "lock.json", "development_component_balanced_all_snr.json",
    "calibration_component_balanced_v2_manifest.json", "validation_component_balanced_v2_manifest.json")]
geometry_material = "ERP-v6-component-balanced-v3-57-independent-calibration-geometry-v2\0" + "\0".join(geometry_inputs)
assignment_material = "ERP-v6-component-balanced-v3-57-independent-calibration-snr-assignment-v1\0" + "\0".join(geometry_inputs)
geometry_digest = hashlib.sha256(geometry_material.encode("ascii")).hexdigest()
assignment_digest = hashlib.sha256(assignment_material.encode("ascii")).hexdigest()
geometry_seed = int.from_bytes(bytes.fromhex(geometry_digest)[:4], "little")
assignment_seed = int.from_bytes(bytes.fromhex(assignment_digest)[:4], "little")
if geometry_seed != 3514841820 or assignment_seed != 3963286106:
    raise ValueError("几何/SNR 派生 seed 与预先登记值不符")
geometry_rng = np.random.default_rng(geometry_seed)
assignment_rng = np.random.default_rng(assignment_seed)
templates = ["S-L"] * 15 + ["S-R"] * 15 + ["D-LL"] * 9 + ["D-RR"] * 9 + ["D-LR"] * 9
geometry_rng.shuffle(templates)
selected_support = set()
configurations = []
for selection_number, template in enumerate(templates):
    hemispheres = [template[-1]] if template.startswith("S-") else list(template[-2:])
    centers = []
    for component_number, hemisphere in enumerate(hemispheres):
        possible = [center for center in sorted(available)
                    if (center < n_lh) == (hemisphere == "L") and
                    (component_number == 0 or np.linalg.norm(vertices[center] - vertices[centers[0]]) >= .050)]
        if not possible:
            raise RuntimeError(f"第 {selection_number} 个配置无满足隔离约束的候选中心")
        center = int(geometry_rng.choice(possible))
        centers.append(center)
        selected_support.update(surface_patches[center])
        available = {candidate for candidate in available
                     if surface_patches[candidate].isdisjoint(selected_support)}
    configurations.append({"selection_number": selection_number, "template": template,
                           "surface_centers": centers})

# 几何全部选完后才平衡分配 SNR，避免某一 SNR 总是拿剩余位置。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
for template in ("S-L", "S-R", "D-LL", "D-RR", "D-LR"):
    indices = np.array([number for number, item in enumerate(configurations) if item["template"] == template])
    assignment_rng.shuffle(indices)
    per_snr = 5 if template.startswith("S-") else 3
    for pair_index, pair in enumerate(snr_pairs):
        for number in indices[pair_index * per_snr:(pair_index + 1) * per_snr]:
            configurations[int(number)]["pair_index"] = pair_index
            configurations[int(number)]["snr_pair"] = pair
for pair_index, pair in enumerate(snr_pairs):
    double_indices = np.array([number for number, item in enumerate(configurations)
                               if item["snr_pair"] == pair and item["template"].startswith("D-")])
    assignment_rng.shuffle(double_indices)
    zero_count = 5 if pair_index != 1 else 4
    for rank, number in enumerate(double_indices):
        configurations[int(number)]["correlation"] = 0.0 if rank < zero_count else .5
for item in configurations:
    item.setdefault("correlation", 0.0)


# %% 3. 生成 57 个一次性病例；每个配置只出现在一个 SNR。
replica_seed_roots = {"fit": 2026092501, "check": 2026092502}
manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    selected = sorted((item for item in configurations if item["pair_index"] == pair_index),
                      key=lambda item: (len(item["surface_centers"]), item["selection_number"]))
    for item in selected:
        number = int(item["selection_number"])
        centers = list(map(int, item["surface_centers"]))
        case = {
            "case_id": f"erp-v6-component-balanced-v3-calibration-{number:03d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": len(manifest), "configuration_id": f"erp-v6-component-balanced-v3-calibration-{number:03d}",
            "configuration_number": number, "configuration_kind": "single_surface" if len(centers) == 1 else "two_surface_only",
            "panel": "erp_v6_component_balanced_v3_calibration", "pair_index": pair_index,
            "scenario": "surface_only", "scenario_code": protocol.SCENARIO_CODES["surface_only"],
            "surface_centers": centers, "deep_local": None, "deep_index": None,
            "location": None, "replicate": 0, "deep_surface_ratio": None,
            "correlation": float(item["correlation"]), "snr_db": eeg_snr,
            "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
            "seed": [replica_seed_roots["fit"], pair_index, number],
            "replica_seed_roots": dict(replica_seed_roots),
            "component_balance_revision": "formal_component_balanced_v3_independent_calibration",
        }
        if len(centers) == 2:
            case["surface_component_sensor_balance"] = True
        manifest.append(case)

if len(manifest) != 57 or len({case["case_id"] for case in manifest}) != 57 or \
        len({case["configuration_id"] for case in manifest}) != 57:
    raise ValueError("v3 必须是 57 个唯一病例和 57 个唯一空间配置")
for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in manifest if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    if len(cell) != 19 or Counter(case["configuration_kind"] for case in cell) != {"single_surface": 10, "two_surface_only": 9}:
        raise ValueError(f"SNR {pair} 的 10/9 配置不完整")
    if sum(center < n_lh for case in cell for center in case["surface_centers"]) != 14:
        raise ValueError(f"SNR {pair} 左右半球不平衡")
    selected_numbers = {case["configuration_number"] for case in cell}
    if Counter(configurations[number]["template"] for number in selected_numbers) != \
            {"S-L": 5, "S-R": 5, "D-LL": 3, "D-RR": 3, "D-LR": 3}:
        raise ValueError(f"SNR {pair} 的半球配置类型不平衡")
    expected_correlations = {0.0: 5, .5: 4} if pair_index != 1 else {0.0: 4, .5: 5}
    if Counter(case["correlation"] for case in cell if len(case["surface_centers"]) == 2) != expected_correlations:
        raise ValueError(f"SNR {pair} 的双源相关性分配不平衡")
if sum(case.get("surface_component_sensor_balance", False) for case in manifest) != 27 or \
        any(case.get("surface_component_sensor_balance", False) != (len(case["surface_centers"]) == 2) for case in manifest):
    raise ValueError("双表层分量平衡字段不完整")

selected_centers = [center for case in manifest for center in case["surface_centers"]]
pair_distances_mm = [float(np.linalg.norm(vertices[case["surface_centers"][0]] -
                                          vertices[case["surface_centers"][1]]) * 1000)
                     for case in manifest if len(case["surface_centers"]) == 2]
internal_overlap = sum(len(surface_patches[first] & surface_patches[second])
                       for number, first in enumerate(selected_centers)
                       for second in selected_centers[number + 1:])
if len(selected_centers) != 84 or len(set(selected_centers)) != 84 or internal_overlap or \
        selected_support & protected_support or min(pair_distances_mm) < 50 or \
        sum(center < n_lh for center in selected_centers) != 42:
    raise ValueError("表层中心的独立性、patch 隔离、双源距离或半球平衡失败")
old_case_ids = {case["case_id"] for cases in (development, v2_calibration, protected_validation) for case in cases}
old_seed_roots = {int(value) for cases in (development, v2_calibration, protected_validation)
                  for case in cases for value in case["replica_seed_roots"].values()}
if {case["case_id"] for case in manifest} & old_case_ids or set(replica_seed_roots.values()) & old_seed_roots:
    raise ValueError("v3 case ID 或 fit/check seed root 与历史协议重复")


# %% 4. 退役 v2 校准；原 marker/结果不删除，只允许用于求解器诊断。
retirement = {
    "protocol": "erp-v6-component-balanced-v2-calibration-retirement-v1",
    "status": "retired_invalid_solver_convergence",
    "formal_calibration_reuse_allowed": False,
    "allowed_use": "development_solver_diagnosis_only",
    "score_values_used_to_select_v3_geometry": False,
    "validation_results_read_by_generator": False,
    "validation_consumed_marker_present": False,
    "reason": {"complete": True, "case_count": 57, "all_converged": False,
               "nonconverged_case_count": 6, "nonconverged_model": "full_model",
               "nonconverged_snr_eeg_meg_db": [-10, 20], "null_models_all_converged": True,
               "action": "do not drop cases or reuse scores; require a new independent calibration"},
    "preserved_artifacts_sha256": {
        str(input_paths[name].relative_to(root)).replace("\\", "/"): expected_sha256[name]
        for name in ("component_balanced_v2_execution_lock.json", "calibration_manifest_consumed.json",
                     "calibration_component_balanced_v2_manifest.json", "v2_completion.json",
                     "v2_frozen_calibration.json", "v2_evidence.csv")},
    "nonconverged_case_ids": [row["case_id"] for row in nonconverged],
    "successor_protocol": "erp-v6-component-balanced-formal-v3-independent-calibration",
}


# %% 5. 冻结 pooled conformal-style 工程规则、协议说明和 algorithm_not_yet_frozen 锁。
decision = {
    "alpha": .05, "calibration_score_count": 57, "strata_count": 1,
    "rank_formula": "p(T) = (1 + count(T_cal >= T)) / 58",
    "accept_deep": "T > 0 AND p(T) <= 0.05",
    "equivalent_boundary": "T must be strictly greater than the second-largest of all 57 calibration scores",
    "maximum_calibration_scores_greater_or_equal": 1,
    "ties_count_against_acceptance": True,
    "test_true_snr_used": False,
    "required_complete_finite_scores": 57,
    "missing_nonfinite_or_unconverged_score_action": "calibration_invalid_do_not_open_validation",
    "ideal_continuous_exchangeable_tail_bound": 2 / 58,
    "claim_scope": "marginal engineering FPR for the equal-weight three-SNR mixture only; no per-SNR or population guarantee",
    "final_validation_requirement": "retain the predeclared zero false positives among four H0 cases in every SNR cell",
}
manifest_name = "calibration_component_balanced_v3_manifest.json"
retirement_name = "component_balanced_v2_calibration_retirement.json"
lock_name = "component_balanced_v3_protocol_lock.json"
document_name = "V3_PROTOCOL.md"
manifest_payload = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
retirement_payload = (json.dumps(retirement, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
retirement_sha256 = hashlib.sha256(retirement_payload).hexdigest()
document = f"""# ERP-v6 component-balanced v3 独立校准协议

状态：`algorithm_not_yet_frozen`。本生成器只冻结病例和决策规则，不运行校准、validation 或反演。

## 病例与几何

- 57 个纯表层病例对应 57 个不同空间配置；每个配置只分配一个 SNR。
- EEG/MEG SNR 为 `(-10,-10)`、`(-10,20)`、`(20,-10)` dB，每层 19 例：10 单表层、9 双表层。
- 27 个双表层病例均启用 sensor balance；双源中心距离至少 50 mm。
- 84 个源中心左右半球各 42 个。全部二阶邻接 patch 在 v3 内互不相交，且与历史开发、已消费 v2 校准和受保护 v2 validation 的 patch 交集为 0。
- fit/check roots 为 `2026092501/2026092502`；几何和 SNR 分配 seed 由四个冻结输入哈希分域派生，不由结果挑选。

## 一个 pooled 决策

`p(T)=(1+count(T_cal>=T))/58`。只有 `T>0` 且 `p<=0.05` 才报深源；tie 保守计数。这等价于 T 严格超过 57 个校准分数的第二大值，完全不使用测试真实 SNR。

该规则只是三个 SNR 等权混合分布下的边际工程 FPR 校准。几何受保护隔离且共享同一前向模型，不宣称严格 iid/conformal 或每 SNR 总体保证。最终 validation 仍必须每个 SNR 对 4 个 H0 均 `0/4` 假阳性。

## v2 退役与一次性消费

v2 校准虽完成 57 例，但 6 个 full model 未收敛，因此其分数禁止用于正式阈值，只能用于求解器诊断。原 marker、执行锁和结果保留不变。

v3 校准使用独立 marker `calibration_component_balanced_v3_manifest_consumed.json`，不被旧的 `calibration_manifest_consumed.json` 阻断。受保护 validation 继续直接引用原 v2 manifest SHA256 `{expected_sha256['validation_component_balanced_v2_manifest.json']}`，且仍由全局 `validation_manifest_consumed.json` 保证只消费一次。

下一步必须先完成算法修改和开发验证，再生成不可覆盖的 v3 execution lock。任一 v3 校准分数缺失、非有限或求解器未收敛，整个 v3 校准作废，不得删除病例后继续。
"""
document_payload = document.encode("utf-8")
document_sha256 = hashlib.sha256(document_payload).hexdigest()
protocol_lock = {
    "protocol": "erp-v6-component-balanced-formal-v3-independent-calibration",
    "status": "algorithm_not_yet_frozen",
    "formal_calibration_allowed": False, "formal_validation_allowed": False,
    "calibration_or_validation_run_by_generator": False,
    "consumed_marker_created_by_generator": False,
    "validation_results_read_by_generator": False,
    "input_sha256": {name: expected_sha256[name] for name in expected_sha256},
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "protocol_document": document_name, "protocol_document_sha256": document_sha256,
    "protected_calibration_manifest": {"path": manifest_name, "sha256": manifest_sha256},
    "protected_validation_manifest": {
        "path": "validation_component_balanced_v2_manifest.json",
        "sha256": expected_sha256["validation_component_balanced_v2_manifest.json"],
        "case_count": 24, "consumed_marker": "validation_manifest_consumed.json",
        "consumed_marker_present_at_freeze": False,
    },
    "validation": {
        "case_count": 24, "manifest_reused_without_opening_results": True,
        "replica_seed_roots": v2_protocol_lock["validation"]["replica_seed_roots"],
        "acceptance": v2_protocol_lock["validation"]["acceptance"],
    },
    "retired_v2_calibration": {"path": retirement_name, "sha256": retirement_sha256,
                               "legacy_consumed_marker_preserved": True},
    "shared_fingerprint": shared_fingerprint, "environment": simulation._environment_versions(),
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "calibration": {
        "case_count": 57, "configuration_count": 57, "configurations_repeated_across_snr": False,
        "cases_per_snr": 19, "single_configurations_per_snr": 10, "double_configurations_per_snr": 9,
        "replica_seed_roots": replica_seed_roots,
        "consumed_marker": "calibration_component_balanced_v3_manifest_consumed.json",
        "all_cases_have_no_deep_truth": True, "two_surface_cases_sensor_balanced": 27,
        "decision": decision,
    },
    "component_balance": {
        "revision": "formal_component_balanced_v3_independent_calibration",
        "surface_components": "equal joint EEG+MEG generation-noise-whitened sensor energy after per-modality rank normalization",
        "implementation": "benchmark/erp_replicates.py",
    },
    "geometry": {
        "selection_rule": "uniform seeded draws from sorted eligible centers; all two-hop patches are disjoint from protected and already selected patches",
        "geometry_seed_derivation": geometry_material, "geometry_seed_sha256": geometry_digest,
        "geometry_seed_uint32": geometry_seed,
        "snr_assignment_seed_derivation": assignment_material, "snr_assignment_seed_sha256": assignment_digest,
        "snr_assignment_seed_uint32": assignment_seed,
        "historical_center_count": len(historical_centers), "protected_support_size": len(protected_support),
        "eligible_center_count_before_selection": available_before_selection,
        "eligible_center_count_after_selection": len(available),
        "selected_center_count": len(selected_centers), "selected_support_size": len(selected_support),
        "selected_lh_count": sum(center < n_lh for center in selected_centers),
        "selected_rh_count": sum(center >= n_lh for center in selected_centers),
        "minimum_double_center_distance_mm": min(pair_distances_mm),
        "protected_support_overlap_count": len(selected_support & protected_support),
        "within_v3_support_overlap_count": internal_overlap,
        "protected_panels": {panel: {"center_count": len(protected_panel_centers[panel]),
                                      "support_size": len(support),
                                      "selected_support_overlap_count": len(selected_support & support)}
                              for panel, support in protected_panel_supports.items()},
        "selected_centers_by_case_id": {case["case_id"]: case["surface_centers"] for case in manifest},
    },
    "claim_scope": decision["claim_scope"],
    "required_next_step": "Freeze a new development-only v3 algorithm execution lock before consuming the v3 calibration manifest; never edit this protocol lock.",
}
lock_payload = (json.dumps(protocol_lock, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


# %% 6. 新文件不存在才创建；--check 只逐字节和旁车复核。
artifacts = {
    manifest_name: manifest_payload,
    retirement_name: retirement_payload,
    lock_name: lock_payload,
    document_name: document_payload,
}
for name, payload in artifacts.items():
    path = protocol_dir / name
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {name}\n"
    if args.check:
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"缺少待核验的冻结文件：{path}")
        if path.read_bytes() != payload or sidecar.read_text(encoding="ascii") != expected_sidecar:
            raise ValueError(f"冻结文件或旁车不一致：{path}")
    elif path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file() or path.read_bytes() != payload or \
                sidecar.read_text(encoding="ascii") != expected_sidecar:
            raise FileExistsError(f"拒绝覆盖内容不同的冻结文件：{path}")
    else:
        with path.open("xb") as stream:
            stream.write(payload)
        with sidecar.open("x", encoding="ascii") as stream:
            stream.write(expected_sidecar)
    print(name, digest)
print("v3 独立校准协议已冻结/核验；algorithm_not_yet_frozen，未运行 calibration/validation。")
