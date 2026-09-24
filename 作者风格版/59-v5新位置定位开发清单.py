"""# %% 生成 v5 新位置定位开发清单；不运行仿真、反演或正式验证。"""

# %% 1. 固定输入和输出；旧 validation 必须已经消费并冻结结论。
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("清单核验禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol
import run_erp_whole_head_matrix as simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", action="store_true", help="只核对，不创建或覆盖")
args = parser.parse_args()

protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
manifest_path = protocol_dir / "development_localization_v5_new_positions.json"
audit_path = protocol_dir / "development_localization_v5_new_positions_audit.json"
report_path = protocol_dir / "V5_LOCALIZATION_DEVELOPMENT.md"
input_paths = {
    "old_development": protocol_dir / "development_component_balanced_all_snr.json",
    "v2_calibration": protocol_dir / "calibration_component_balanced_v2_manifest.json",
    "v3_calibration": protocol_dir / "calibration_component_balanced_v3_manifest.json",
    "v4_calibration": protocol_dir / "calibration_component_balanced_v4_manifest.json",
    "consumed_validation": protocol_dir / "validation_component_balanced_v2_manifest.json",
    "validation_marker": protocol_dir / "validation_manifest_consumed.json",
    "v4_acceptance": root / "results/erp_whole_head/adaptive_v6/formal_component_balanced_v4_acceptance/acceptance.json",
}
for name, path in input_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结输入 {name}: {path}")
payloads = {name: path.read_bytes() for name, path in input_paths.items()}
input_sha256 = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
if json.loads(payloads["validation_marker"])["manifest_sha256"] != \
        hashlib.sha256(payloads["consumed_validation"]).hexdigest():
    raise ValueError("旧 validation 消费标记与清单不一致")
if json.loads(payloads["v4_acceptance"])["passed"] is not False:
    raise ValueError("这里只能在已冻结且未通过的 v4 周期之后建立新开发面板")


# %% 2. 新位置不复用历史中心；新 patch 内部互斥，并避开旧开发和已消费 validation patch。
shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
n_lh = len(shared["src_surface"][0]["vertno"])
positions = np.asarray(shared["vertices"], float)
surface_patches = [set(map(int, protocol._surface_patch(shared, center)[0]))
                   for center in range(n_surf)]
old_manifests = {
    name: json.loads(payloads[name]) for name in (
        "old_development", "v2_calibration", "v3_calibration", "v4_calibration",
        "consumed_validation")
}
historical_centers = {
    int(center) for cases in old_manifests.values() for case in cases
    for center in case["surface_centers"]
}
focused_support = set()
for name in ("old_development", "consumed_validation"):
    for center in sorted({int(center) for case in old_manifests[name]
                          for center in case["surface_centers"]}):
        focused_support.update(surface_patches[center])

available = {center for center in range(n_surf)
             if center not in historical_centers
             and surface_patches[center].isdisjoint(focused_support)}
available_before = len(available)
geometry_material = (
    "ERP-v6-localization-v5-new-position-development-v2-paired-SNR\0"
    + "\0".join(input_sha256[name] for name in (
        "old_development", "v2_calibration", "v3_calibration", "v4_calibration",
        "consumed_validation"))
)
geometry_digest = hashlib.sha256(geometry_material.encode("ascii")).hexdigest()
geometry_seed = int.from_bytes(bytes.fromhex(geometry_digest)[:8], "little")
rng = np.random.default_rng(geometry_seed)

hemisphere_templates = [
    ("surface_only", ["L"], 0.0),
    ("surface_only", ["L", "R"], 0.0),
    ("deep_only", [], 0.0),
    ("deep_plus_surface", ["R"], 0.0),
    ("deep_plus_two_surface", ["L", "R"], 0.5),
]
selected_support = set()
configurations = []
for slot, (scenario, hemispheres, correlation) in enumerate(hemisphere_templates):
    centers = []
    for component_number, hemisphere in enumerate(hemispheres):
        possible = [center for center in sorted(available)
                    if (center < n_lh) == (hemisphere == "L")
                    and (component_number == 0 or
                         np.linalg.norm(positions[center] - positions[centers[0]]) >= .050)]
        if not possible:
            raise RuntimeError(f"slot={slot} 没有满足隔离约束的表层中心")
        center = int(rng.choice(possible))
        centers.append(center)
        selected_support.update(surface_patches[center])
        available = {candidate for candidate in available
                     if surface_patches[candidate].isdisjoint(selected_support)}
    configurations.append({"slot": slot, "scenario": scenario,
                           "surface_centers": centers, "correlation": correlation})

used_deep_local = {
    int(case["deep_local"]) for name in ("old_development", "consumed_validation")
    for case in old_manifests[name] if case.get("deep_local") is not None
}
deep_candidates = np.array(sorted(set(range(len(positions) - n_surf)) - used_deep_local), int)
if deep_candidates.size < 3:
    raise RuntimeError("没有足够的全新深层候选")
rng.shuffle(deep_candidates)
development_deep_local = list(map(int, deep_candidates[:3]))


# %% 3. 五个固定几何跨三个 SNR 配对，只改变噪声条件。
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
replica_seed_roots = {"fit": 2026092701, "check": 2026092702}
manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for item in configurations:
        slot = item["slot"]
        scenario = item["scenario"]
        centers = item["surface_centers"]
        has_deep = scenario != "surface_only"
        deep_local = development_deep_local[slot - 2] if has_deep else None
        deep_index = n_surf + deep_local if has_deep else None
        number = len(manifest)
        configuration_kind = ("single_surface" if scenario == "surface_only" and len(centers) == 1
                              else "two_surface_only" if scenario == "surface_only"
                              else scenario)
        case = {
            "case_id": f"erp-v6-localization-v5-development-{number:02d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": number,
            "component_balance_revision": "development_localization_v5_new_positions_paired_snr",
            "configuration_id": f"erp-v6-localization-v5-geometry-{slot:02d}",
            "configuration_kind": configuration_kind,
            "configuration_number": slot,
            "correlation": float(item["correlation"]),
            "deep_index": deep_index,
            "deep_local": deep_local,
            "deep_surface_ratio": 0.5 if has_deep and centers else None,
            "eeg_snr_db": eeg_snr,
            "location": None,
            "meg_snr_db": meg_snr,
            "pair_index": pair_index,
            "panel": "erp_v6_localization_v5_new_position_development",
            "replica_seed_roots": dict(replica_seed_roots),
            "replicate": 0,
            "scenario": scenario,
            "scenario_code": protocol.SCENARIO_CODES[scenario],
            "seed": [replica_seed_roots["fit"], pair_index, slot],
            "snr_db": eeg_snr,
            "surface_centers": centers,
        }
        if has_deep and centers:
            case["deep_surface_sensor_amplitude_ratio"] = 0.5
        if len(centers) == 2:
            case["surface_component_sensor_balance"] = True
        manifest.append(case)


# %% 4. 独立性、平衡和 schema 自检。
selected_centers = [center for item in configurations for center in item["surface_centers"]]
surface_center_occurrences = sum(len(case["surface_centers"]) for case in manifest)
internal_overlap = sum(len(surface_patches[first] & surface_patches[second])
                       for number, first in enumerate(selected_centers)
                       for second in selected_centers[number + 1:])
pair_distances_mm = [float(np.linalg.norm(positions[item["surface_centers"][0]]
                                          - positions[item["surface_centers"][1]]) * 1000)
                     for item in configurations if len(item["surface_centers"]) == 2]
if len(manifest) != 15 or len({case["case_id"] for case in manifest}) != 15 or \
        [case["case_number"] for case in manifest] != list(range(15)):
    raise ValueError("v5 新位置开发清单必须含 15 个唯一病例")
configuration_counts = Counter(case["configuration_id"] for case in manifest)
if len(configuration_counts) != 5 or set(configuration_counts.values()) != {3}:
    raise ValueError("五个固定几何必须分别跨三个 SNR 配对")
for configuration_id in configuration_counts:
    paired = [case for case in manifest if case["configuration_id"] == configuration_id]
    geometry = {(tuple(case["surface_centers"]), case["deep_local"], case["correlation"])
                for case in paired}
    if len(geometry) != 1 or {(case["eeg_snr_db"], case["meg_snr_db"])
                             for case in paired} != set(snr_pairs):
        raise ValueError(f"{configuration_id} 的跨 SNR 几何不一致")
for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in manifest if case["pair_index"] == pair_index]
    if len(cell) != 5 or Counter(case["scenario"] for case in cell) != {
            "surface_only": 2, "deep_only": 1, "deep_plus_surface": 1,
            "deep_plus_two_surface": 1}:
        raise ValueError(f"SNR {pair} 的五种开发情况不完整")
if len(selected_centers) != 6 or len(set(selected_centers)) != 6 or \
        surface_center_occurrences != 18 or internal_overlap or \
        set(selected_centers) & historical_centers or selected_support & focused_support or \
        min(pair_distances_mm) < 50 or sum(center < n_lh for center in selected_centers) != 3:
    raise ValueError("新位置的唯一性、patch 隔离、双源距离或半球平衡失败")
if len({case["deep_local"] for case in manifest if case["deep_local"] is not None}) != 3 or \
        any(case["deep_local"] in used_deep_local for case in manifest
            if case["deep_local"] is not None):
    raise ValueError("深层开发位置与旧开发/validation 重复")

audit = {
    "protocol": "erp-v6-localization-v5-new-position-development-v2-paired-SNR",
    "purpose": "development only; no formal calibration or validation",
    "formal_acceptance_allowed": False,
    "validation_results_used_to_select_geometry": False,
    "geometry_seed_material": geometry_material,
    "geometry_seed_sha256": geometry_digest,
    "geometry_seed_uint64": geometry_seed,
    "input_sha256": input_sha256,
    "case_count": len(manifest),
    "configuration_count": len(configuration_counts),
    "cases_per_snr": 5,
    "surface_center_count": len(selected_centers),
    "surface_center_occurrence_count": surface_center_occurrences,
    "surface_lh_rh": [sum(center < n_lh for center in selected_centers),
                       sum(center >= n_lh for center in selected_centers)],
    "eligible_center_count_before_selection": available_before,
    "eligible_center_count_after_selection": len(available),
    "within_panel_patch_overlap_count": internal_overlap,
    "old_development_and_validation_patch_overlap_count": len(selected_support & focused_support),
    "historical_exact_center_overlap_count": len(set(selected_centers) & historical_centers),
    "minimum_two_surface_distance_mm": min(pair_distances_mm),
    "development_deep_local": development_deep_local,
    "development_deep_global": [n_surf + value for value in development_deep_local],
    "replica_seed_roots": replica_seed_roots,
}
manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
audit_text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
report_text = f"""# v5 新位置定位开发面板

本面板只用于开发定位后处理，不用于正式校准或验收。几何选择不读取 v4 validation 数值。

- 5 个固定几何跨三个 EEG/MEG SNR 配对，共 15 例；SNR 比较不混入位置差异。
- 每个 SNR 都含：单表层、双表层、纯深层、深层+单表层、深层+双表层。
- 6 个唯一表层中心左右半球各 3 个；面板内二阶 patch 零重叠。
- 新 patch 与旧开发和已消费 validation 的 patch 零重叠；历史精确中心零复用。
- 双表层中心最小欧氏距离：{min(pair_distances_mm):.3f} mm。
- 新深层 local 索引：{development_deep_local}；不复用旧开发或已消费 validation 深层位置。
- fit/check seed roots：{replica_seed_roots['fit']}/{replica_seed_roots['check']}。

任何候选定位规则都必须同时报告 15 例，不得删除困难病例；后续正式 validation 必须重新生成未见位置。
"""


# %% 5. 原子语义：首次生成，之后只能 --check。
outputs = {manifest_path: manifest_text.encode("utf-8"), audit_path: audit_text.encode("utf-8"),
           report_path: report_text.encode("utf-8")}
if args.check:
    for path, expected in outputs.items():
        sidecar = Path(str(path) + ".sha256")
        digest = hashlib.sha256(expected).hexdigest()
        expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
        if not path.is_file() or path.read_bytes() != expected or \
                not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
            raise ValueError(f"冻结开发协议已变化：{path}")
else:
    for path in outputs:
        if path.exists() or Path(str(path) + ".sha256").exists():
            raise FileExistsError(f"拒绝覆盖：{path}")
    for path, payload in outputs.items():
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        Path(str(path) + ".sha256").write_bytes(f"{digest}  {path.name}\n".encode("ascii"))

print(json.dumps({"manifest": str(manifest_path),
                  "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
                  "case_count": len(manifest), "surface_centers": len(selected_centers),
                  "deep_local": development_deep_local,
                  "minimum_pair_distance_mm": min(pair_distances_mm)},
                 ensure_ascii=False, indent=2))
