"""# %% 冻结层级盲定位的新校准与 untouched validation 清单；不运行仿真或反演。"""

# %% 1. 固定输入、输出和新噪声域。旧文件只读，新文件拒绝覆盖。
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
parser.add_argument("--check", action="store_true", help="只逐字节核对，不创建或覆盖")
args = parser.parse_args()

protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
algorithm_path = root / "作者风格版/63-层级浅深盲定位开发.py"
if not algorithm_path.is_file():
    raise FileNotFoundError(f"缺少待冻结算法：{algorithm_path}")
output_paths = {
    "gate": protocol_dir / "hierarchical_v1_gate_calibration_manifest.json",
    "surface": protocol_dir / "hierarchical_v1_surface_calibration_manifest.json",
    "validation": protocol_dir / "hierarchical_v1_validation_manifest.json",
    "audit": protocol_dir / "hierarchical_v1_protocol_audit.json",
    "report": protocol_dir / "HIERARCHICAL_V1_PROTOCOL.md",
}
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
seed_roots = {
    "gate": {"fit": 2026093001, "check": 2026093002},
    "surface": {"fit": 2026093003, "check": 2026093004},
    "validation": {"fit": 2026100101, "check": 2026100102},
}


# %% 2. 只读历史清单身份和几何；不读取任何旧结果、分数或指标。
history = {}
for path in sorted(protocol_dir.glob("*.json")):
    historical_name = (
        path.name.startswith("development")
        or path.name.startswith("calibration") and "manifest" in path.name
        or path.name in {"validation_manifest.json", "validation_component_balanced_v2_manifest.json"}
    )
    if not historical_name:
        continue
    payload = path.read_bytes()
    try:
        cases = json.loads(payload)
    except json.JSONDecodeError:
        continue
    if isinstance(cases, list) and cases and isinstance(cases[0], dict) and "case_id" in cases[0]:
        history[path.name] = {
            "path": path,
            "payload": payload,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "cases": cases,
        }

required_history = {
    "development_manifest.json",
    "development_component_balanced_all_snr.json",
    "development_localization_v5_new_positions.json",
    "calibration_manifest.json",
    "calibration_component_balanced_v2_manifest.json",
    "calibration_component_balanced_v3_manifest.json",
    "calibration_component_balanced_v4_manifest.json",
    "validation_manifest.json",
    "validation_component_balanced_v2_manifest.json",
}
missing_history = sorted(required_history - history.keys())
if missing_history:
    raise FileNotFoundError(f"历史清单不完整：{missing_history}")
validation_marker_path = protocol_dir / "validation_manifest_consumed.json"
validation_marker_payload = validation_marker_path.read_bytes()
validation_marker = json.loads(validation_marker_payload)
if validation_marker.get("manifest_sha256") != history[
        "validation_component_balanced_v2_manifest.json"]["sha256"]:
    raise ValueError("旧 validation 消费标记与受保护清单不一致")
legacy_lock_path = protocol_dir / "lock.json"
legacy_lock_payload = legacy_lock_path.read_bytes()
legacy_lock = json.loads(legacy_lock_payload)
legacy_lock_sha256 = hashlib.sha256(legacy_lock_payload).hexdigest()
legacy_surface_centers = {
    int(center) for center in legacy_lock.get("previously_used_surface_centers", [])}
if not legacy_surface_centers or legacy_lock.get("historical_metric_values_used") is not False:
    raise ValueError("旧 lock 缺少可审计的历史表层几何防火墙")

shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
n_surf = int(shared["n_surf"])
n_deep = int(shared["n_deep"])
n_lh = len(shared["src_surface"][0]["vertno"])
vertices = np.asarray(shared["vertices"], float)
deep_labels = np.asarray(shared["deep_aseg_labels"], int)
if n_deep != 16 or deep_labels.shape != (n_deep,) or set(np.unique(deep_labels)) != {10, 49}:
    raise ValueError("本协议只接受当前16点双侧丘脑深源网格")
if legacy_lock.get("shared_fingerprint") != shared_fingerprint:
    raise ValueError("旧 lock 与当前 forward 指纹不一致")

surface_patches = [set(map(int, protocol._surface_patch(shared, center)[0]))
                   for center in range(n_surf)]
historical_cases = [case for item in history.values() for case in item["cases"]]
historical_case_ids = {case["case_id"] for case in historical_cases}
historical_centers = {int(center) for case in historical_cases
                      for center in case.get("surface_centers", [])}
historical_centers.update(legacy_surface_centers)
historical_support = set()
for center in sorted(historical_centers):
    historical_support.update(surface_patches[center])
historical_guard = set(historical_support)
for index in historical_support:
    historical_guard.update(protocol._neighbors(shared["adjacency"], index, n_surf))
historical_deep_local = {int(case["deep_local"]) for case in historical_cases
                         if case.get("deep_local") is not None}
historical_seed_roots = {int(value) for case in historical_cases
                         for value in case.get("replica_seed_roots", {}).values()}
new_seed_values = {value for roots in seed_roots.values() for value in roots.values()}
if len(new_seed_values) != 6 or new_seed_values & historical_seed_roots:
    raise ValueError("新 fit/check root 必须两两不同且不复用历史 root")


# %% 3. 三个几何 seed 都由当前算法、forward 指纹和列明的历史清单哈希分域派生。
algorithm_sha256 = hashlib.sha256(algorithm_path.read_bytes()).hexdigest()
history_sha256 = {name: item["sha256"] for name, item in sorted(history.items())}
history_sha256[legacy_lock_path.name] = legacy_lock_sha256
seed_input = "\0".join([
    algorithm_sha256,
    shared_fingerprint,
    *[f"{name}:{digest}" for name, digest in history_sha256.items()],
])
geometry_domains = {
    "validation": "ERP-v6-hierarchical-v1-untouched-validation-geometry-v1",
    "gate": "ERP-v6-hierarchical-v1-gate-calibration-geometry-v1",
    "surface": "ERP-v6-hierarchical-v1-surface-calibration-deep-assignment-v1",
}
geometry_digests = {
    name: hashlib.sha256((domain + "\0" + seed_input).encode("utf-8")).hexdigest()
    for name, domain in geometry_domains.items()
}
geometry_seeds = {name: int.from_bytes(bytes.fromhex(digest)[:8], "little")
                  for name, digest in geometry_digests.items()}
validation_rng = np.random.default_rng(geometry_seeds["validation"])
gate_rng = np.random.default_rng(geometry_seeds["gate"])
surface_rng = np.random.default_rng(geometry_seeds["surface"])


# %% 4. Validation 先选：每个SNR 7个独立几何；跨SNR不复用位置。
available_surface = {
    center for center in range(n_surf)
    if center not in historical_centers
    and center not in historical_guard
}
available_surface_before = len(available_surface)
eligible_surface = sorted(available_surface)
surface_conflicts = {center: {center} for center in eligible_surface}
for position, first in enumerate(eligible_surface):
    for second in eligible_surface[position + 1:]:
        if not surface_patches[first].isdisjoint(surface_patches[second]):
            surface_conflicts[first].add(second)
            surface_conflicts[second].add(first)
validation_geometries = []
new_surface_support = set()
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    mixed_single_hemisphere = "L"
    validation_templates = [
        ("single_surface", "surface_only", ["L"], 0.0),
        ("single_surface", "surface_only", ["R"], 0.0),
        ("two_surface_only", "surface_only", ["L", "R"], 0.0),
        ("two_surface_only", "surface_only", ["L", "R"], 0.5),
        ("deep_only", "deep_only", [], 0.0),
        ("deep_plus_surface", "deep_plus_surface", [mixed_single_hemisphere], 0.0),
        ("deep_plus_two_surface", "deep_plus_two_surface", ["L", "R"], 0.5),
    ]
    for within_pair, (kind, scenario, hemispheres, correlation) in enumerate(validation_templates):
        slot = len(validation_geometries)
        centers = []
        for component_number, hemisphere in enumerate(hemispheres):
            possible = [
                center for center in sorted(available_surface)
                if (center < n_lh) == (hemisphere == "L")
                and (component_number == 0
                     or np.linalg.norm(vertices[center] - vertices[centers[0]]) >= .050)
            ]
            if not possible:
                raise RuntimeError(f"validation slot={slot} 没有满足隔离约束的表层中心")
            conflict_counts = np.array([
                len(surface_conflicts[candidate] & available_surface) for candidate in possible
            ])
            least_conflict = np.flatnonzero(conflict_counts == conflict_counts.min())
            center = int(possible[int(validation_rng.choice(least_conflict))])
            centers.append(center)
            new_surface_support.update(surface_patches[center])
            available_surface = {
                candidate for candidate in available_surface
                if surface_patches[candidate].isdisjoint(new_surface_support)
            }
        validation_geometries.append({
            "slot": slot, "within_pair": within_pair, "pair_index": pair_index,
            "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
            "configuration_kind": kind, "scenario": scenario,
            "surface_centers": centers, "correlation": float(correlation),
            "deep_local": None,
        })

unseen_deep = sorted(set(range(n_deep)) - historical_deep_local)
validation_deep_by_label = {}
for label in (10, 49):
    candidates = np.array([index for index in unseen_deep if deep_labels[index] == label], int)
    validation_rng.shuffle(candidates)
    if len(candidates) != 5:
        raise RuntimeError(f"label={label} 应恰有5个相对当前protocol未见深点")
    validation_deep_by_label[label] = list(map(int, candidates))
# 当前16点网格的label 10/49为9/7；validation取5/4后，校准恰保留4/3。
extra_label = 10
other_label = 49
validation_deep_label_pattern = [
    extra_label, other_label, extra_label,
    other_label, extra_label, other_label,
    extra_label, other_label, extra_label,
]
validation_deep_local = []
for item, label in zip(
        [item for item in validation_geometries if item["scenario"] != "surface_only"],
        validation_deep_label_pattern):
    deep_local = validation_deep_by_label[label].pop()
    item["deep_local"] = deep_local
    validation_deep_local.append(deep_local)
validation_deep_reserve_current = sorted(set(unseen_deep) - set(validation_deep_local))

validation_manifest = []
for item in validation_geometries:
    centers = list(map(int, item["surface_centers"]))
    deep_local = item["deep_local"]
    number = len(validation_manifest)
    case = {
        "case_id": f"erp-v6-hierarchical-v1-validation-{number:02d}-eeg{item['eeg_snr_db']:+03d}-meg{item['meg_snr_db']:+03d}",
        "case_number": number,
        "component_balance_revision": "formal_hierarchical_v1_untouched_validation",
        "configuration_id": f"erp-v6-hierarchical-v1-validation-geometry-{number:02d}",
        "configuration_kind": item["configuration_kind"],
        "configuration_number": number,
        "correlation": item["correlation"],
        "deep_index": None if deep_local is None else n_surf + deep_local,
        "deep_local": deep_local,
        "deep_surface_ratio": 0.5 if deep_local is not None and centers else None,
        "eeg_snr_db": item["eeg_snr_db"],
        "location": None,
        "meg_snr_db": item["meg_snr_db"],
        "pair_index": item["pair_index"],
        "panel": "erp_v6_hierarchical_v1_untouched_validation",
        "replica_seed_roots": dict(seed_roots["validation"]),
        "replicate": 0,
        "scenario": item["scenario"],
        "scenario_code": protocol.SCENARIO_CODES[item["scenario"]],
        "seed": [seed_roots["validation"]["fit"], item["pair_index"], number],
        "snr_db": item["eeg_snr_db"],
        "surface_centers": centers,
    }
    if len(centers) == 2:
        case["surface_component_sensor_balance"] = True
    if deep_local is not None and centers:
        case["deep_surface_sensor_amplitude_ratio"] = 0.5
    validation_manifest.append(case)


# %% 5. 一级gate校准：19个纯表层固定几何跨3个SNR；先排除全部validation patch。
gate_templates = ["S-L"] * 6 + ["S-R"] * 4 + ["D-LL"] * 3 + ["D-RR"] * 3 + ["D-LR"] * 3
gate_center_counts = {"L": 15, "R": 13}
gate_center_bank = {}
for hemisphere in ("L", "R"):
    pool = {
        center for center in available_surface
        if (center < n_lh) == (hemisphere == "L")
    }
    selected = []
    while pool and len(selected) < gate_center_counts[hemisphere]:
        possible = sorted(pool)
        conflict_counts = np.array([
            len(surface_conflicts[candidate] & pool) for candidate in possible
        ])
        least_conflict = np.flatnonzero(conflict_counts == conflict_counts.min())
        center = int(possible[int(gate_rng.choice(least_conflict))])
        selected.append(center)
        pool -= surface_conflicts[center]
    if len(selected) != gate_center_counts[hemisphere]:
        raise RuntimeError(f"gate {hemisphere}半球只能取得{len(selected)}个互斥patch中心")
    gate_center_bank[hemisphere] = selected

for center in gate_center_bank["L"] + gate_center_bank["R"]:
    new_surface_support.update(surface_patches[center])
available_surface = {
    candidate for candidate in available_surface
    if surface_patches[candidate].isdisjoint(new_surface_support)
}

gate_same_pairs = {"LL": [], "RR": []}
gate_remaining = {hemisphere: set(gate_center_bank[hemisphere]) for hemisphere in ("L", "R")}
for hemisphere, template in (("L", "LL"), ("R", "RR")):
    for pair_number in range(3):
        choices = [
            (float(np.linalg.norm(vertices[first] - vertices[second])), first, second)
            for position, first in enumerate(sorted(gate_remaining[hemisphere]))
            for second in sorted(gate_remaining[hemisphere])[position + 1:]
        ]
        distance, first, second = max(choices)
        if distance < .050:
            raise RuntimeError(f"gate {template}第{pair_number}对中心无法满足50 mm")
        gate_same_pairs[template].append([first, second])
        gate_remaining[hemisphere] -= {first, second}

gate_cross_pairs = []
for pair_number in range(3):
    choices = [
        (float(np.linalg.norm(vertices[left] - vertices[right])), left, right)
        for left in sorted(gate_remaining["L"])
        for right in sorted(gate_remaining["R"])
    ]
    distance, left, right = max(choices)
    if distance < .050:
        raise RuntimeError(f"gate LR第{pair_number}对中心无法满足50 mm")
    gate_cross_pairs.append([left, right])
    gate_remaining["L"].remove(left)
    gate_remaining["R"].remove(right)
gate_singles = {hemisphere: list(gate_remaining[hemisphere]) for hemisphere in ("L", "R")}
for hemisphere in ("L", "R"):
    gate_rng.shuffle(gate_singles[hemisphere])
    gate_rng.shuffle(gate_same_pairs[hemisphere * 2])
gate_rng.shuffle(gate_cross_pairs)

gate_manifest = []
gate_geometries = []
templates = np.array(gate_templates, object)
gate_rng.shuffle(templates)
double_number = 0
for configuration_number, template in enumerate(templates):
    if template == "S-L":
        centers = [gate_singles["L"].pop()]
    elif template == "S-R":
        centers = [gate_singles["R"].pop()]
    elif template == "D-LL":
        centers = gate_same_pairs["LL"].pop()
    elif template == "D-RR":
        centers = gate_same_pairs["RR"].pop()
    else:
        centers = gate_cross_pairs.pop()
    correlation = 0.0
    if len(centers) == 2:
        correlation = 0.5 if double_number % 2 else 0.0
        double_number += 1
    gate_geometries.append({
        "configuration_number": configuration_number,
        "template": str(template), "surface_centers": centers,
        "correlation": correlation,
    })

for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for item in gate_geometries:
        centers = list(map(int, item["surface_centers"]))
        number = len(gate_manifest)
        case = {
            "case_id": f"erp-v6-hierarchical-v1-gate-calibration-{number:03d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": number,
            "component_balance_revision": "formal_hierarchical_v1_gate_calibration",
            "configuration_id": f"erp-v6-hierarchical-v1-gate-geometry-{item['configuration_number']:02d}",
            "configuration_kind": "single_surface" if len(centers) == 1 else "two_surface_only",
            "configuration_number": item["configuration_number"],
            "correlation": item["correlation"],
            "deep_index": None,
            "deep_local": None,
            "deep_surface_ratio": None,
            "eeg_snr_db": eeg_snr,
            "location": None,
            "meg_snr_db": meg_snr,
            "pair_index": pair_index,
            "panel": "erp_v6_hierarchical_v1_gate_calibration",
            "replica_seed_roots": dict(seed_roots["gate"]),
            "replicate": 0,
            "scenario": "surface_only",
            "scenario_code": protocol.SCENARIO_CODES["surface_only"],
            "seed": [seed_roots["gate"]["fit"], pair_index, item["configuration_number"]],
            "snr_db": eeg_snr,
            "surface_centers": centers,
        }
        if len(centers) == 2:
            case["surface_component_sensor_balance"] = True
        gate_manifest.append(case)


# %% 6. 二级surface校准：57个纯深层独立噪声病例；均衡覆盖validation排除后的7点。
surface_deep_candidates = sorted(set(range(n_deep)) - set(validation_deep_local))
if len(surface_deep_candidates) != 7 or any(
        sum(deep_labels[index] == label for index in surface_deep_candidates) < 1
        for label in (10, 49)):
    raise RuntimeError("排除validation后必须恰有7个左右丘脑深点用于二级校准")
surface_manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    assignments = []
    while len(assignments) < 19:
        block = np.array(surface_deep_candidates, int)
        surface_rng.shuffle(block)
        assignments.extend(map(int, block))
    for within_pair, deep_local in enumerate(assignments[:19]):
        number = len(surface_manifest)
        surface_manifest.append({
            "case_id": f"erp-v6-hierarchical-v1-surface-calibration-{number:03d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": number,
            "component_balance_revision": "formal_hierarchical_v1_surface_calibration",
            "configuration_id": f"erp-v6-hierarchical-v1-surface-calibration-{number:03d}",
            "configuration_kind": "deep_only",
            "configuration_number": number,
            "correlation": 0.0,
            "deep_index": n_surf + deep_local,
            "deep_local": deep_local,
            "deep_surface_ratio": None,
            "eeg_snr_db": eeg_snr,
            "location": None,
            "meg_snr_db": meg_snr,
            "pair_index": pair_index,
            "panel": "erp_v6_hierarchical_v1_surface_calibration",
            "replica_seed_roots": dict(seed_roots["surface"]),
            "replicate": within_pair,
            "scenario": "deep_only",
            "scenario_code": protocol.SCENARIO_CODES["deep_only"],
            "seed": [seed_roots["surface"]["fit"], pair_index, number],
            "snr_db": eeg_snr,
            "surface_centers": [],
        })


# %% 7. 病例schema、平衡、空间隔离和噪声独立性硬核验。
if len(gate_manifest) != 57 or len(surface_manifest) != 57 or len(validation_manifest) != 21:
    raise ValueError("三个正式面板必须分别为57、57、21例")
all_new_cases = gate_manifest + surface_manifest + validation_manifest
new_case_ids = [case["case_id"] for case in all_new_cases]
if len(new_case_ids) != len(set(new_case_ids)) or set(new_case_ids) & historical_case_ids:
    raise ValueError("新case_id必须互异且不能复用历史case_id")
required_case_keys = {
    "case_id", "case_number", "component_balance_revision", "configuration_id",
    "configuration_kind", "configuration_number", "correlation", "deep_index",
    "deep_local", "deep_surface_ratio", "eeg_snr_db", "location", "meg_snr_db",
    "pair_index", "panel", "replica_seed_roots", "replicate", "scenario",
    "scenario_code", "seed", "snr_db", "surface_centers",
}
if any(not required_case_keys <= case.keys() for case in all_new_cases):
    raise ValueError("新病例缺少prepare_trial_covariance_case所需schema字段")
if any(case["replica_seed_roots"] != seed_roots[panel]
       for panel, cases in (("gate", gate_manifest), ("surface", surface_manifest),
                            ("validation", validation_manifest)) for case in cases):
    raise ValueError("病例noise root与面板冻结值不一致")

for pair_index, pair in enumerate(snr_pairs):
    gate_cell = [case for case in gate_manifest if case["pair_index"] == pair_index]
    surface_cell = [case for case in surface_manifest if case["pair_index"] == pair_index]
    validation_cell = [case for case in validation_manifest if case["pair_index"] == pair_index]
    if {(case["eeg_snr_db"], case["meg_snr_db"]) for case in gate_cell + surface_cell + validation_cell} != {pair}:
        raise ValueError(f"pair_index={pair_index}与SNR不一致")
    if len(gate_cell) != 19 or Counter(len(case["surface_centers"]) for case in gate_cell) != {1: 10, 2: 9}:
        raise ValueError(f"一级校准SNR={pair}不是10单源+9双源")
    if len({case["configuration_id"] for case in gate_cell}) != 19:
        raise ValueError("一级校准每个SNR必须完整包含19个固定配置")
    if len(surface_cell) != 19 or any(case["scenario"] != "deep_only" for case in surface_cell):
        raise ValueError(f"二级校准SNR={pair}必须为19个纯深层病例")
    if len(validation_cell) != 7 or Counter(case["scenario"] for case in validation_cell) != {
            "surface_only": 4, "deep_only": 1,
            "deep_plus_surface": 1, "deep_plus_two_surface": 1}:
        raise ValueError(f"validation SNR={pair}的7种唯一几何不完整")
    if Counter(case["deep_index"] is not None for case in validation_cell) != {False: 4, True: 3}:
        raise ValueError("validation每个SNR必须4个H0和3个H1")

gate_configuration_counts = Counter(case["configuration_id"] for case in gate_manifest)
if len(gate_configuration_counts) != 19 or set(gate_configuration_counts.values()) != {3} or \
        len({case["configuration_id"] for case in surface_manifest}) != 57 or \
        len({case["configuration_id"] for case in validation_manifest}) != 21:
    raise ValueError("配置ID重复规则不符合冻结协议")
for configuration_id in gate_configuration_counts:
    repeated = [case for case in gate_manifest if case["configuration_id"] == configuration_id]
    truth = {(tuple(case["surface_centers"]), case["correlation"]) for case in repeated}
    if len(truth) != 1 or {
            (case["eeg_snr_db"], case["meg_snr_db"]) for case in repeated} != set(snr_pairs):
        raise ValueError(f"gate固定配置没有原样跨三个SNR：{configuration_id}")
validation_truth = {
    (tuple(case["surface_centers"]), case["deep_local"], case["correlation"])
    for case in validation_manifest
}
if len(validation_truth) != 21:
    raise ValueError("validation的21个几何必须跨SNR全部唯一")

new_surface_centers = [center for item in gate_geometries + validation_geometries
                       for center in item["surface_centers"]]
new_surface_patches = [surface_patches[center] for center in new_surface_centers]
new_surface_union = set().union(*new_surface_patches)
new_surface_core = set(new_surface_centers)
for center in new_surface_centers:
    new_surface_core.update(protocol._neighbors(shared["adjacency"], center, n_surf))
pair_distances_mm = [
    float(np.linalg.norm(vertices[item["surface_centers"][0]]
                         - vertices[item["surface_centers"][1]]) * 1000)
    for item in gate_geometries + validation_geometries
    if len(item["surface_centers"]) == 2
]
if len(new_surface_centers) != 55 or len(set(new_surface_centers)) != 55:
    raise ValueError("一级校准28个中心加validation 27个中心必须全部唯一")
if sum(map(len, new_surface_patches)) != len(new_surface_union) or \
        new_surface_core & historical_support:
    raise ValueError("新二阶patch必须内部互斥，且新中心及一阶邻域必须避开历史二阶patch")
if min(pair_distances_mm) < 50:
    raise ValueError("双表层中心距离不得小于50 mm")
if sum(center < n_lh for center in new_surface_centers) != 30:
    raise ValueError("新表层中心应为validation 15/12加gate 15/13的半球分配")
if any(case.get("surface_component_sensor_balance") is not True
       for case in gate_manifest + validation_manifest if len(case["surface_centers"]) == 2):
    raise ValueError("所有双表层病例必须启用传感器分量平衡")
if any(case.get("deep_surface_sensor_amplitude_ratio") != .5
       for case in validation_manifest
       if case["deep_index"] is not None and case["surface_centers"]):
    raise ValueError("所有validation混合病例必须固定深浅传感器幅度比0.5")
if set(validation_deep_local) & historical_deep_local or \
        set(surface_deep_candidates) & set(validation_deep_local):
    raise ValueError("validation深点必须相对当前protocol未见且从二级校准排除")
if len(validation_deep_local) != 9 or len(set(validation_deep_local)) != 9 or \
        sorted(Counter(deep_labels[validation_deep_local]).values()) != [4, 5]:
    raise ValueError("validation必须使用9个唯一未见深点，左右分配为4/5")
if len(validation_deep_reserve_current) != 1 or \
        set(validation_deep_reserve_current) != set(surface_deep_candidates) - historical_deep_local:
    raise ValueError("二级校准应保留恰1个相对当前protocol未见的深点")
if sorted(Counter(deep_labels[surface_deep_candidates]).values()) != [3, 4]:
    raise ValueError("二级校准的7个非validation深点应左右分配为3/4")


# %% 8. 冻结audit与说明；明确真值禁用和16点深网格的历史限制。
gate_centers = [center for item in gate_geometries for center in item["surface_centers"]]
validation_centers = [center for item in validation_geometries for center in item["surface_centers"]]
audit = {
    "protocol": "erp-v6-hierarchical-v1-formal-manifests",
    "purpose": "freeze manifests only; no simulation, inverse, score, metric, calibration or validation run",
    "formal_results_present": False,
    "algorithm_path": str(algorithm_path.relative_to(root)),
    "algorithm_sha256": algorithm_sha256,
    "shared_fingerprint": shared_fingerprint,
    "history_manifest_sha256": history_sha256,
    "legacy_lock_surface_center_count": len(legacy_surface_centers),
    "legacy_lock_sha256": legacy_lock_sha256,
    "validation_consumed_marker_sha256": hashlib.sha256(validation_marker_payload).hexdigest(),
    "geometry_domains": geometry_domains,
    "geometry_seed_sha256": geometry_digests,
    "geometry_seed_uint64": geometry_seeds,
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "replica_seed_roots": seed_roots,
    "historical_surface_center_count": len(historical_centers),
    "historical_surface_support_vertex_count": len(historical_support),
    "historical_surface_one_ring_guard_vertex_count": len(historical_guard),
    "eligible_one_ring_buffered_surface_centers_before_selection": available_surface_before,
    "new_surface_center_count": len(new_surface_centers),
    "new_surface_core_vertex_count": len(new_surface_core),
    "new_surface_patch_vertex_count": len(new_surface_union),
    "new_surface_lh_rh": [sum(center < n_lh for center in new_surface_centers),
                           sum(center >= n_lh for center in new_surface_centers)],
    "within_new_panels_patch_overlap_count": sum(map(len, new_surface_patches)) - len(new_surface_union),
    "new_core_vs_history_patch_overlap_count": len(new_surface_core & historical_support),
    "new_vs_history_patch_overlap_count": len(new_surface_union & historical_support),
    "minimum_two_surface_distance_mm": min(pair_distances_mm),
    "gate_calibration": {
        "case_count": len(gate_manifest), "configuration_count": 19,
        "surface_center_count": len(gate_centers), "single_per_snr": 10,
        "double_per_snr": 9, "configuration_repeated_across_snr": True,
        "scores_per_configuration": 3,
        "calibration_claim": (
            "pooled fixed-design engineering calibration across 19 geometries and three SNRs; "
            "not an iid population conformal guarantee"
        ),
    },
    "surface_calibration": {
        "case_count": len(surface_manifest), "configuration_count": 57,
        "deep_geometry_count": len(surface_deep_candidates),
        "deep_local": surface_deep_candidates,
        "deep_global": [n_surf + index for index in surface_deep_candidates],
        "deep_geometry_seen_in_listed_history": sorted(
            set(surface_deep_candidates) & historical_deep_local),
        "deep_geometry_new_relative_to_listed_history": validation_deep_reserve_current,
        "case_and_noise_identity_unique": True,
        "truth_use": "predeclared deep-only null for surface-presence calibration; true deep index must never enter inverse, localization candidate choice or score",
    },
    "validation": {
        "case_count": len(validation_manifest), "configuration_count": 21,
        "configuration_repeats_across_snr": 0,
        "surface_center_count": len(validation_centers),
        "surface_centers": validation_centers,
        "deep_local": validation_deep_local,
        "deep_global": [n_surf + index for index in validation_deep_local],
        "deep_new_relative_to_listed_history": True,
        "h0_h1_per_snr": [4, 3],
    },
    "truth_firewall": {
        "generator_reads_result_metrics": False,
        "generator_runs_simulation_or_inverse": False,
        "validation_truth_may_be_read_by_algorithm": False,
        "truth_allowed_use": "posthoc metrics and predeclared null-label calibration only",
    },
    "deep_grid_limit": (
        "The repository's older pre-v6 experiments had already used all 16 deep grid points. "
        "The selected deep points are untouched only relative to the listed current development, "
        "calibration and consumed-validation manifests; global historical novelty is impossible "
        "without rebuilding the deep source space and forward model."
    ),
    "sampling_limit": (
        "The 21 validation cases contain 21 independent geometries, seven at each of three SNRs; "
        "they are a one-use engineering panel, not proof of a population error rate."
    ),
}

report = f"""# Hierarchical v1 正式清单

状态：只冻结病例，尚未运行仿真、反演、校准或 validation。

## 三个新面板

- 一级 deep-presence gate 校准：19 个固定纯表层几何跨三个 EEG/MEG SNR，共 57 个独立噪声病例；每档 10 单表层、9 双表层。这是 pooled fixed-design 工程校准，不声称 i.i.d. 总体 conformal 保证。
- 二级 surface-presence 校准：57 个纯深层病例；三个 SNR 各 19 例。病例 ID 和噪声身份唯一，均衡重复 validation 排除后的 7 个深点；其中只有 1 个点相对列明的当前 protocol 未见，另 6 个在历史清单出现过。真值深点禁止进入反演、候选选择或分数。
- untouched validation：每个 SNR 7 个独立几何，共 21 例，跨 SNR 不复用位置。每档含 2 个单表层、2 个双表层、1 个纯深层、1 个深层+单表层、1 个深层+双表层，即 4 个 H0 和 3 个 H1。

SNR 固定为 `(-10,-10)`、`(-10,+20)`、`(+20,-10)` dB，不做参数网格。新 roots 分别为 gate `{seed_roots['gate']['fit']}/{seed_roots['gate']['check']}`、surface `{seed_roots['surface']['fit']}/{seed_roots['surface']['check']}`、validation `{seed_roots['validation']['fit']}/{seed_roots['validation']['check']}`。

## 空间隔离

- Validation 先选并保留 27 个表层中心和 9 个相对当前 protocol 未见的深点；后续两套校准显式排除这些位置。
- Gate 校准再选 28 个唯一表层中心并固定跨 SNR。三个新面板的 55 个表层二阶 patch 内部互斥；每个新中心及其一阶邻域与列明的全部历史 development、calibration、已消费 validation 及旧 `lock.json` 的 272 个中心二阶 patch 零交集。由于当前源空间的历史覆盖过密，新二阶 patch 的最外圈允许与历史二阶 patch 边界重叠，精确数量写入 audit，不声称完整 patch 全历史未见。
- 所有双表层中心相距至少 50 mm，并启用传感器分量平衡；validation 混合病例的深浅传感器幅度比固定为 0.5。
- 二级校准均衡使用 validation 排除后的 7 个深点，其中 1 个相对当前 protocol 未见；validation 使用 9 个唯一未见深点，左右丘脑按 5/4 分配，并留下 1 个未见点给校准。

## 诚实限制

更早的 pre-v6 仿真已使用过当前全部 16 个深层网格点。因此这里的深点只能称为相对列明的当前 development、calibration 和已消费 validation 未见；若要求全项目历史绝对新深点，必须重建深源空间与 forward。

21 例 validation 是 21 个独立几何，每个 SNR 各 7 个；这是一次性工程验收，不证明总体误报率。必须先冻结最终算法和两个校准决策，再一次性消费 validation；不得按 validation 结果修改算法、阈值或病例。
"""


# %% 9. 首次生成全部文件；之后--check逐字节核验，任何文件都不覆盖。
payloads = {
    output_paths["gate"]: (json.dumps(gate_manifest, ensure_ascii=False, indent=2,
                                      sort_keys=True) + "\n").encode("utf-8"),
    output_paths["surface"]: (json.dumps(surface_manifest, ensure_ascii=False, indent=2,
                                         sort_keys=True) + "\n").encode("utf-8"),
    output_paths["validation"]: (json.dumps(validation_manifest, ensure_ascii=False,
                                            indent=2, sort_keys=True) + "\n").encode("utf-8"),
    output_paths["audit"]: (json.dumps(audit, ensure_ascii=False, indent=2,
                                       sort_keys=True) + "\n").encode("utf-8"),
    output_paths["report"]: report.encode("utf-8"),
}
if args.check:
    for path, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = Path(str(path) + ".sha256")
        expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
        if not path.is_file() or path.read_bytes() != payload or \
                not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
            raise ValueError(f"冻结正式协议已变化：{path}")
else:
    occupied = [str(path) for path in payloads
                if path.exists() or Path(str(path) + ".sha256").exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有正式协议：{occupied}")
    for path, payload in payloads.items():
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        Path(str(path) + ".sha256").write_bytes(
            f"{digest}  {path.name}\n".encode("ascii"))

print(json.dumps({
    "checked": bool(args.check),
    "outputs": {str(path): hashlib.sha256(payload).hexdigest()
                for path, payload in payloads.items()},
    "case_counts": {"gate": len(gate_manifest), "surface": len(surface_manifest),
                    "validation": len(validation_manifest)},
    "new_surface_centers": len(new_surface_centers),
    "validation_deep_local": validation_deep_local,
    "surface_calibration_deep_local": surface_deep_candidates,
}, ensure_ascii=False, indent=2))
