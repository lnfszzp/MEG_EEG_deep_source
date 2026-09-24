"""# %% 生成分量平衡 v2 正式清单；不运行校准、验证或任何反演。"""

# %% 1. 路径、冻结输入和线程环境。算法尚未冻结，因此本脚本只准生成病例协议。
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import sys

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
calibration_consumed_path = protocol_dir / "calibration_manifest_consumed.json"
validation_consumed_path = protocol_dir / "validation_manifest_consumed.json"
v2_lock_path = protocol_dir / "component_balanced_v2_lock.json"
if not v2_lock_path.exists():
    assert not calibration_consumed_path.exists() and not validation_consumed_path.exists(), \
        "旧正式病例已经消费，不能再把 v2 标成未见协议"
expected_inputs = {
    "development_component_balanced_all_snr.json": "c64d02429fd5e9e9c2a7831a36b7f86c2a0fe614ecb2c837fa0284fe96fb5425",
    "calibration_manifest.json": "459b7b983b9ad4bbbc8b84e3d0a494e8466dbe093a32af02b78e7d871119a313",
    "validation_manifest.json": "2da4682fbd23b591a20e56fd9612f5324ecb9cf68ed744f85c51400d11759fb7",
    "lock.json": "eb1625ef1be95553ebf5094bf3c08a4df24bcb007038fb18fab31b4f3a685bc3",
    "calibration_lock.json": "30fc911220f421583318e64ac4be9bffb88c28cf92ec415dbfcdec58195b0791",
}
inputs = {}
for name, expected in expected_inputs.items():
    payload = (protocol_dir / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected, f"冻结输入已变化：{name}"
    inputs[name] = json.loads(payload)

development = [dict(case) for case in inputs["development_component_balanced_all_snr.json"]]
calibration_source = [dict(case) for case in inputs["calibration_manifest.json"]]
validation_source = [dict(case) for case in inputs["validation_manifest.json"]]
old_lock = inputs["lock.json"]
old_calibration_lock = inputs["calibration_lock.json"]
assert len(development) == 20 and len(calibration_source) == 57 and len(validation_source) == 24


# %% 2. 复用未消费的 19/8 个空间配置，只更新协议身份、独立噪声和分量平衡字段。
calibration_roots = {"fit": 2026092401, "check": 2026092402}
validation_roots = {"fit": 2026092403, "check": 2026092404}
calibration = []
for source_case in calibration_source:
    case = dict(source_case)
    case["case_id"] = source_case["case_id"].replace(
        "erp-v6-calibration-", "erp-v6-component-balanced-v2-calibration-", 1)
    case["configuration_id"] = source_case["configuration_id"].replace(
        "erp-v6-calibration-", "erp-v6-component-balanced-v2-calibration-", 1)
    case["panel"] = "erp_v6_component_balanced_v2_calibration"
    case["seed"] = [calibration_roots["fit"], case["pair_index"], case["configuration_number"]]
    case["replica_seed_roots"] = dict(calibration_roots)
    case["component_balance_revision"] = "formal_component_balanced_v2"
    if len(case["surface_centers"]) == 2:
        case["surface_component_sensor_balance"] = True
    calibration.append(case)

validation = []
for source_case in validation_source:
    case = dict(source_case)
    case["case_id"] = source_case["case_id"].replace(
        "erp-v6-validation-", "erp-v6-component-balanced-v2-validation-", 1)
    case["configuration_id"] = source_case["configuration_id"].replace(
        "erp-v6-validation-", "erp-v6-component-balanced-v2-validation-", 1)
    case["panel"] = "erp_v6_component_balanced_v2_validation"
    case["seed"] = [validation_roots["fit"], case["pair_index"], case["configuration_number"]]
    case["replica_seed_roots"] = dict(validation_roots)
    case["component_balance_revision"] = "formal_component_balanced_v2"
    if len(case["surface_centers"]) == 2:
        case["surface_component_sensor_balance"] = True
    if case["deep_index"] is not None and case["surface_centers"]:
        case["deep_surface_sensor_amplitude_ratio"] = 0.5
    validation.append(case)


# %% 3. 病例结构自检。三个 SNR 层都完整；不存在偷偷删掉的高分校准病例。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
assert Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in calibration) == Counter(
    {pair: 19 for pair in snr_pairs})
assert Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in validation) == Counter(
    {pair: 8 for pair in snr_pairs})
assert len({case["case_id"] for case in calibration}) == len(calibration) == 57
assert len({case["configuration_id"] for case in calibration}) == 19
assert Counter(case["configuration_kind"] for case in calibration) == {
    "single_surface": 30, "two_surface_only": 27}
assert all(case["deep_index"] is None for case in calibration)
assert sum(case.get("surface_component_sensor_balance", False) for case in calibration) == 27
assert all(case.get("surface_component_sensor_balance", False) ==
           (len(case["surface_centers"]) == 2) for case in calibration)
assert len({case["case_id"] for case in validation}) == len(validation) == 24
assert len({case["configuration_id"] for case in validation}) == 8
assert Counter(case["configuration_kind"] for case in validation) == {
    "single_surface": 6, "two_surface_only": 6, "deep_only": 6,
    "deep_plus_surface": 3, "deep_plus_two_surface": 3}
assert sum(case.get("surface_component_sensor_balance", False) for case in validation) == 9
assert all(case.get("surface_component_sensor_balance", False) ==
           (len(case["surface_centers"]) == 2) for case in validation)
assert sum("deep_surface_sensor_amplitude_ratio" in case for case in validation) == 6
assert all((case.get("deep_surface_sensor_amplitude_ratio") == 0.5) ==
           (case["deep_index"] is not None and bool(case["surface_centers"])) for case in validation)
for pair in snr_pairs:
    cell = [case for case in validation if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
    assert sum(case["deep_index"] is None for case in cell) == 4
    assert sum(case["deep_index"] is not None for case in cell) == 4


# %% 4. 开发、校准、验证位置/种子审计；连片支持按正式仿真的同一函数计算。
shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
assert shared_fingerprint == old_lock["shared_fingerprint"] == old_calibration_lock["shared_fingerprint"]
panels = {"development": development, "calibration": calibration, "validation": validation}
center_sets = {}
support_sets = {}
deep_sets = {}
case_id_sets = {}
seed_root_sets = {}
for panel, cases in panels.items():
    center_sets[panel] = {int(center) for case in cases for center in case["surface_centers"]}
    support_sets[panel] = set()
    for center in sorted(center_sets[panel]):
        support_sets[panel].update(map(int, protocol._surface_patch(shared, center)[0]))
    deep_sets[panel] = {int(case["deep_index"]) for case in cases if case["deep_index"] is not None}
    case_id_sets[panel] = {case["case_id"] for case in cases}
    seed_root_sets[panel] = {int(value) for case in cases for value in case["replica_seed_roots"].values()}

panel_pairs = [("development", "calibration"), ("development", "validation"),
               ("calibration", "validation")]
for first, second in panel_pairs:
    assert not (center_sets[first] & center_sets[second]), f"{first}/{second} 表层中心重叠"
    assert not (case_id_sets[first] & case_id_sets[second]), f"{first}/{second} case_id 重叠"
    assert not (seed_root_sets[first] & seed_root_sets[second]), f"{first}/{second} seed root 重叠"
assert not (support_sets["development"] & support_sets["calibration"])
assert not (support_sets["development"] & support_sets["validation"])
assert not (deep_sets["development"] & deep_sets["validation"])
calibration_validation_support_overlap = sorted(
    support_sets["calibration"] & support_sets["validation"])
assert len(calibration_validation_support_overlap) == 9

calibration_configurations = {}
validation_configurations = {}
for case in calibration:
    calibration_configurations.setdefault(case["configuration_id"], case)
for case in validation:
    validation_configurations.setdefault(case["configuration_id"], case)
overlap_pairs = []
for calibration_id, calibration_case in sorted(calibration_configurations.items()):
    calibration_support = set()
    for center in calibration_case["surface_centers"]:
        calibration_support.update(map(int, protocol._surface_patch(shared, center)[0]))
    for validation_id, validation_case in sorted(validation_configurations.items()):
        validation_support = set()
        for center in validation_case["surface_centers"]:
            validation_support.update(map(int, protocol._surface_patch(shared, center)[0]))
        overlap = sorted(calibration_support & validation_support)
        if overlap:
            overlap_pairs.append({"calibration_configuration_id": calibration_id,
                                  "validation_configuration_id": validation_id,
                                  "vertex_count": len(overlap), "vertices": overlap})
assert sum(item["vertex_count"] for item in overlap_pairs) == 9


# %% 5. 先算清单摘要，再冻结协议说明和状态为 algorithm_not_yet_frozen 的锁。
calibration_name = "calibration_component_balanced_v2_manifest.json"
validation_name = "validation_component_balanced_v2_manifest.json"
lock_name = "component_balanced_v2_lock.json"
document_name = "COMPONENT_BALANCED_V2_PROTOCOL.md"
calibration_payload = (json.dumps(calibration, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n").encode("utf-8")
validation_payload = (json.dumps(validation, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("utf-8")
manifest_sha256 = {
    calibration_name: hashlib.sha256(calibration_payload).hexdigest(),
    validation_name: hashlib.sha256(validation_payload).hexdigest(),
}
document = """# ERP-v6 component-balanced v2 正式协议

状态：`algorithm_not_yet_frozen`。本协议只冻结病例身份、真值缩放规则和独立噪声；尚未冻结算法、统计量、求解器或定位规则，因此禁止运行正式校准和验证。

## 病例

| 面板 | 空间配置 | SNR 层 | 病例数 |
|---|---:|---:|---:|
| 纯表层校准 | 19（10 单表层、9 双表层） | 3 | 57 |
| 一次性验证 | 8（每层 4 无深源、4 有深源） | 3 | 24 |

SNR 均为 EEG/MEG `(-10,-10)`、`(-10,20)`、`(20,-10)` dB。每个校准层恰有 19 个完整纯表层病例，保留原三层交集排名规则。旧 v1 的 19/8 个空间配置仅作为几何来源，旧文件不修改、也没有被运行。

## 分量平衡

所有双表层病例均设置 `surface_component_sensor_balance=true`，使两个表层分量在生成噪声协方差白化并按模态秩归一化后的 EEG+MEG 联合空间具有相同能量。所有浅深混合验证病例均设置 `deep_surface_sensor_amplitude_ratio=0.5`；这是联合传感器空间的幅度比，能量比为 0.25。双表层加深源病例先平衡两个表层分量，再把深层相对整个表层活动缩放到该幅度比。

## 独立性和限制

开发、校准和验证的 case ID、表层中心及 fit/check seed roots 两两不相交。开发与校准、开发与验证的完整表层连片支持也不相交；开发深源和验证深源索引不相交。新 case ID 与新 seed roots 改变正式观测噪声，但保留原配置编号决定的真值空间和 ERP 波形分配。

校准和验证的中心不相交，但连片支持有 9 个顶点重叠。校准只能向决策器提供冻结的标量零分布，不能传入位置、真值或源图，因此这不是直接的真值泄漏；它仍然意味着本验证不能被描述为严格的 cortical-support holdout，也不提供总体 FPR 的统计保证。

## 冻结和消费

下一步必须只用开发结果另建不可覆盖的算法执行锁，固定统计量、协方差、求解器、盲门控和门控后定位实现及全部代码摘要。只有锁状态从 `algorithm_not_yet_frozen` 进入独立执行锁后，才允许一次性运行完整 57 例校准；校准完成且全部收敛后，才允许一次性运行 24 例验证。生成本协议不会创建 consumed marker，也不会调用仿真或反演。
"""
document_payload = document.encode("utf-8")
document_sha256 = hashlib.sha256(document_payload).hexdigest()
lock = {
    "protocol": "erp-v6-component-balanced-formal-v2",
    "status": "algorithm_not_yet_frozen",
    "formal_calibration_allowed": False,
    "formal_validation_allowed": False,
    "calibration_or_validation_run_by_generator": False,
    "consumed_marker_created_by_generator": False,
    "historical_metric_values_used": False,
    "input_sha256": expected_inputs,
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "protocol_document": document_name,
    "protocol_document_sha256": document_sha256,
    "protected_manifest_sha256": manifest_sha256,
    "shared_fingerprint": shared_fingerprint,
    "environment": simulation._environment_versions(),
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "calibration": {
        "configuration_count": 19, "case_count": 57,
        "single_surface_configurations": 10, "two_surface_configurations": 9,
        "cases_per_snr": 19, "replica_seed_roots": calibration_roots,
        "all_cases_have_no_deep_truth": True,
        "two_surface_cases_sensor_balanced": 27,
        "decision": old_calibration_lock["decision"],
    },
    "validation": {
        "configuration_count": 8, "case_count": 24,
        "cases_per_snr": 8, "negative_cases_per_snr": 4,
        "positive_cases_per_snr": 4, "replica_seed_roots": validation_roots,
        "two_surface_cases_sensor_balanced": 9,
        "mixed_surface_deep_cases_sensor_balanced": 6,
        "deep_surface_sensor_amplitude_ratio": 0.5,
        "acceptance": old_lock["acceptance"],
    },
    "component_balance": {
        "revision": "formal_component_balanced_v2",
        "surface_components": "equal joint EEG+MEG generation-noise-whitened sensor energy after per-modality rank normalization",
        "mixed_layers": "deep/surface joint sensor amplitude ratio 0.5 after surface-component balancing",
        "implementation": "benchmark/erp_replicates.py",
    },
    "disjoint_audit": {
        "surface_centers": {panel: sorted(values) for panel, values in center_sets.items()},
        "surface_support_size": {panel: len(values) for panel, values in support_sets.items()},
        "deep_indices": {panel: sorted(values) for panel, values in deep_sets.items()},
        "seed_roots": {panel: sorted(values) for panel, values in seed_root_sets.items()},
        "case_ids_pairwise_disjoint": True,
        "surface_centers_pairwise_disjoint": True,
        "seed_roots_pairwise_disjoint": True,
        "development_calibration_surface_support_overlap": 0,
        "development_validation_surface_support_overlap": 0,
        "development_validation_deep_index_overlap": 0,
        "calibration_validation_surface_support_overlap_vertices": calibration_validation_support_overlap,
        "calibration_validation_surface_support_overlap_count": 9,
        "calibration_validation_surface_support_overlap_pairs": overlap_pairs,
        "overlap_interpretation": "No direct truth leakage because calibration may expose only frozen scalar null scores, but validation is not a strict cortical-support holdout.",
    },
    "predecessor_v1": {
        "files_unchanged": True,
        "calibration_consumed_marker_present_at_freeze": False,
        "validation_consumed_marker_present_at_freeze": False,
        "spatial_configurations_reused_without_metric_selection": True,
        "new_case_ids_and_noise_roots": True,
    },
    "required_next_step": "Freeze one development-only algorithm/statistic execution lock before any formal calibration; do not edit this protocol lock.",
}
assert not lock["predecessor_v1"]["calibration_consumed_marker_present_at_freeze"]
assert not lock["predecessor_v1"]["validation_consumed_marker_present_at_freeze"]


# %% 6. 新文件不存在才创建；存在时只逐字节核对，永不覆盖旧 v1 或 v2 冻结文件。
artifacts = {
    calibration_name: calibration,
    validation_name: validation,
    lock_name: lock,
}
for name, contents in artifacts.items():
    payload = (json.dumps(contents, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = protocol_dir / name
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if args.check or path.exists():
        assert path.read_bytes() == payload, f"冻结内容不同，拒绝覆盖：{path}"
        assert sidecar.read_text(encoding="ascii") == f"{digest}  {name}\n"
    else:
        protocol.save_manifest(path, contents, expected_digest=digest)
    print(name, digest)

document_path = protocol_dir / document_name
document_sidecar = document_path.with_suffix(document_path.suffix + ".sha256")
if args.check or document_path.exists():
    assert document_path.read_bytes() == document_payload, f"冻结内容不同，拒绝覆盖：{document_path}"
    assert document_sidecar.read_text(encoding="ascii") == f"{document_sha256}  {document_name}\n"
else:
    with document_path.open("xb") as stream:
        stream.write(document_payload)
    with document_sidecar.open("x", encoding="ascii") as stream:
        stream.write(f"{document_sha256}  {document_name}\n")
print(document_name, document_sha256)
print("已冻结 component-balanced v2 病例协议；algorithm_not_yet_frozen，未运行校准/验证。")
