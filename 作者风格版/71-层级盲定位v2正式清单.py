"""# %% 冻结 hierarchical-v2 两级新校准清单；只引用 untouched v1 validation。"""

# %% 1. 固定入口、输出和不可跨越的 validation 防火墙。
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

adaptive_root = root / "results/erp_whole_head/adaptive_v6"
protocol_dir = adaptive_root / "protocol"
output_paths = {
    "gate": protocol_dir / "hierarchical_v2_gate_calibration_manifest.json",
    "surface": protocol_dir / "hierarchical_v2_surface_calibration_manifest.json",
    "audit": protocol_dir / "hierarchical_v2_protocol_audit.json",
    "report": protocol_dir / "HIERARCHICAL_V2_PROTOCOL.md",
}
if not args.check:
    occupied = [str(path) for path in output_paths.values()
                if path.exists() or Path(str(path) + ".sha256").exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有 hierarchical-v2 正式协议：{occupied}")

shared_validation_marker = protocol_dir / "hierarchical_v1_validation_manifest_consumed.json"
validation_firewall_paths = [
    shared_validation_marker,
    adaptive_root / "formal_hierarchical_v1_validation",
    adaptive_root / "formal_hierarchical_v1_validation.run.lock",
    adaptive_root / "formal_hierarchical_v2_validation",
    adaptive_root / "formal_hierarchical_v2_validation.run.lock",
]
started_validation = [str(path) for path in validation_firewall_paths
                      if path.exists() or path.is_symlink()]
if started_validation:
    raise FileExistsError(
        "共享 validation 已被消费、已产生输出或正在运行，禁止生成/核对 v2 清单："
        f"{started_validation}"
    )


# %% 2. 71、计划中的72、冻结选择结果和评分helper必须全部存在并锁定。
generator_path = Path(__file__).resolve()
runner_path = root / "作者风格版/72-层级盲定位v2正式运行.py"
selection_path = (
    adaptive_root / "development_diagnosis/hierarchical_v2_blind_reliability_fusion/summary.json"
)
helper_path = root / "candidates/oaster_predictive.py"
protocol_helper_path = root / "benchmark/protocol.py"
shared_loader_path = root / "run_erp_whole_head_matrix.py"
input_paths = {
    "manifest_generator_71": generator_path,
    "planned_formal_runner_72": runner_path,
    "blind_reliability_selection": selection_path,
    "predictive_score_helper": helper_path,
    "protocol_geometry_helper": protocol_helper_path,
    "shared_forward_loader": shared_loader_path,
}
missing_inputs = [str(path) for path in input_paths.values() if not path.is_file()]
if missing_inputs:
    raise FileNotFoundError(
        "hierarchical-v2 清单拒绝占位或放宽输入锁；请先提供计划中的72号正式runner。"
        f" 缺少：{missing_inputs}"
    )
input_sha256 = {
    name: {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    for name, path in input_paths.items()
}
expected_selection_sha256 = "5c2a7e535a50e882d130cddf0afccebe3c873fc0b7f8c2f6651482eb3195bba8"
expected_helper_sha256 = "575a660382fa0330720d9e1d3bcd253f08a3727c4358b7d919a67e33812f1775"
if input_sha256["blind_reliability_selection"]["sha256"] != expected_selection_sha256:
    raise ValueError("70号已审计的盲可靠性选择summary已经变化")
if input_sha256["predictive_score_helper"]["sha256"] != expected_helper_sha256:
    raise ValueError("已审计的 predictive score helper 已经变化")

selection = json.loads(selection_path.read_text(encoding="utf-8"))
selected_specs = [item for item in selection.get("all_candidates", [])
                  if item.get("candidate") == selection.get("selected_candidate")]
selected_metrics = [item for item in selection.get("candidate_metrics", [])
                    if item.get("candidate") == selection.get("selected_candidate")]
if selection.get("complete") is not True or selection.get("phase") != "development" or \
        selection.get("formal_claim_allowed") is not False or \
        selection.get("validation_results_read") is not False or \
        selection.get("inverse_refitted") is not False or \
        selection.get("truth_or_snr_used_by_score") is not False or \
        selection.get("candidate_count") != 14 or \
        selection.get("selected_candidate") != "adaptive_balance_p1" or \
        len(selected_specs) != 1 or len(selected_metrics) != 1 or \
        selected_specs[0].get("family") != "adaptive_balance" or \
        selected_specs[0].get("exponent") != .5 or \
        selected_specs[0].get("balance_power") != 1. or \
        selected_metrics[0].get("passes_frozen_rule") != 1 or \
        selected_metrics[0].get("selected") != 1:
    raise ValueError("70号summary不是已审计通过的 adaptive p=1 开发选择")


# %% 3. 只读历史病例清单和旧几何锁；不读取任何结果、分数或validation输出。
history_names = [
    "calibration_component_balanced_v2_manifest.json",
    "calibration_component_balanced_v3_manifest.json",
    "calibration_component_balanced_v4_manifest.json",
    "calibration_manifest.json",
    "development_component_balanced_all_snr.json",
    "development_component_balanced_all_snr_mixed_only.json",
    "development_component_balanced_m10_m10.json",
    "development_component_balanced_m10_m10_case00.json",
    "development_component_balanced_m10_m10_case01.json",
    "development_component_balanced_m10_m10_case02.json",
    "development_component_balanced_m10_m10_case03.json",
    "development_component_balanced_m10_m10_case04.json",
    "development_component_balanced_m10_m10_cases00_02_03.json",
    "development_component_balanced_m10_m10_cases01_04.json",
    "development_component_balanced_new_snr_mixed_only.json",
    "development_localization_v5_mm_floor_probe.json",
    "development_localization_v5_new_positions.json",
    "development_manifest.json",
    "development_sensor_balanced_m10_m10.json",
    "development_sensor_balanced_m10_m10_case01.json",
    "development_sensor_balanced_m10_m10_case04.json",
    "development_sensor_balanced_m10_m10_cases00_01.json",
    "development_sensor_balanced_m10_m10_cases02_04.json",
    "hierarchical_v1_gate_calibration_manifest.json",
    "hierarchical_v1_surface_calibration_manifest.json",
    "hierarchical_v1_validation_manifest.json",
    "validation_component_balanced_v2_manifest.json",
    "validation_manifest.json",
]
history = {}
for name in history_names:
    path = protocol_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少列明的历史病例清单：{path}")
    payload = path.read_bytes()
    try:
        cases = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"历史清单不是合法JSON：{path}") from error
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict) or \
            "case_id" not in cases[0]:
        raise ValueError(f"列明的历史文件不是非空病例清单：{path}")
    history[path.name] = {
        "path": path,
        "payload": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "cases": cases,
    }
if set(history) != set(history_names) or len(history) != 28:
    raise RuntimeError("历史病例清单allow-list必须固定为28个文件")

protected_validation_name = "hierarchical_v1_validation_manifest.json"
protected_validation_path = protocol_dir / protected_validation_name
protected_validation_sha256 = "6ef97341434bca99bc7c75d205c8dc6d3815b3260ca3c62c10c615ce2efb0eff"
protected_validation_payload = history[protected_validation_name]["payload"]
if hashlib.sha256(protected_validation_payload).hexdigest() != protected_validation_sha256:
    raise ValueError("受保护的 v1 validation manifest 与冻结SHA256不一致")
protected_validation_sidecar = Path(str(protected_validation_path) + ".sha256")
expected_validation_sidecar = (
    f"{protected_validation_sha256}  {protected_validation_path.name}\n".encode("ascii")
)
if not protected_validation_sidecar.is_file() or \
        protected_validation_sidecar.read_bytes() != expected_validation_sidecar:
    raise ValueError("受保护的 v1 validation manifest 旁车不一致")
validation_manifest = history[protected_validation_name]["cases"]

snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
validation_seed_roots = {"fit": 2026100101, "check": 2026100102}
if len(validation_manifest) != 21 or \
        len({case["case_id"] for case in validation_manifest}) != 21 or \
        len({case["configuration_id"] for case in validation_manifest}) != 21 or \
        any(case.get("replica_seed_roots") != validation_seed_roots
            or case.get("seed", [None])[0] != validation_seed_roots["fit"]
            for case in validation_manifest):
    raise ValueError("受保护的 v1 validation 病例数、ID或2026100101/02 seeds不一致")
for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in validation_manifest if case.get("pair_index") == pair_index]
    if len(cell) != 7 or \
            {(case.get("eeg_snr_db"), case.get("meg_snr_db")) for case in cell} != {pair} or \
            Counter(case.get("deep_index") is not None for case in cell) != {False: 4, True: 3}:
        raise ValueError(f"受保护 validation 的 SNR={pair} 结构不一致")

legacy_lock_path = protocol_dir / "lock.json"
v1_audit_path = protocol_dir / "hierarchical_v1_protocol_audit.json"
if not legacy_lock_path.is_file() or not v1_audit_path.is_file():
    raise FileNotFoundError("缺少旧几何lock或hierarchical-v1 protocol audit")
legacy_lock_payload = legacy_lock_path.read_bytes()
v1_audit_payload = v1_audit_path.read_bytes()
legacy_lock = json.loads(legacy_lock_payload)
v1_audit = json.loads(v1_audit_payload)
legacy_surface_centers = {
    int(center) for center in legacy_lock.get("previously_used_surface_centers", [])
}
if not legacy_surface_centers or legacy_lock.get("historical_metric_values_used") is not False:
    raise ValueError("旧 lock 缺少只基于几何的历史表层防火墙")


# %% 4. 锁定共享forward，并把validation位置与全部历史patch转换为空间防火墙。
shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
expected_shared_fingerprint = "a97e6991fde1cedfac66857d5c07683c43af62f16ad19bfa31ea7199a9083556"
if shared_fingerprint != expected_shared_fingerprint or \
        v1_audit.get("shared_fingerprint") != expected_shared_fingerprint or \
        legacy_lock.get("shared_fingerprint") != expected_shared_fingerprint:
    raise ValueError("共享forward不再是受保护v1 validation使用的指纹")

n_surf = int(shared["n_surf"])
n_deep = int(shared["n_deep"])
n_lh = len(shared["src_surface"][0]["vertno"])
vertices = np.asarray(shared["vertices"], float)
deep_labels = np.asarray(shared["deep_aseg_labels"], int)
expected_deep_labels = np.asarray(
    [10, 10, 10, 49, 49, 10, 10, 49, 49, 10, 49, 10, 10, 49, 10, 49], int)
if n_surf != 7498 or n_lh != 3732 or n_deep != 16 or \
        not np.array_equal(deep_labels, expected_deep_labels):
    raise ValueError("本协议只接受冻结的7498表层点和当前16点双侧丘脑深源网格")

surface_patches = [set(map(int, protocol._surface_patch(shared, center)[0]))
                   for center in range(n_surf)]
historical_cases = [case for item in history.values() for case in item["cases"]]
nonvalidation_cases = [case for name, item in history.items()
                       if name != protected_validation_name for case in item["cases"]]
historical_case_ids = {case["case_id"] for case in historical_cases}
historical_centers = {
    int(center) for case in historical_cases for center in case.get("surface_centers", [])
}
historical_centers.update(legacy_surface_centers)
nonvalidation_centers = {
    int(center) for case in nonvalidation_cases for center in case.get("surface_centers", [])
}
nonvalidation_centers.update(legacy_surface_centers)
validation_centers = [
    int(center) for case in validation_manifest for center in case.get("surface_centers", [])
]
if any(center < 0 or center >= n_surf
       for center in historical_centers | set(validation_centers)):
    raise ValueError("历史清单含越界表层中心")
if len(validation_centers) != 27 or len(set(validation_centers)) != 27:
    raise ValueError("受保护 validation 必须含27个唯一表层中心")

historical_support = set()
for center in sorted(historical_centers):
    historical_support.update(surface_patches[center])
nonvalidation_support = set()
for center in sorted(nonvalidation_centers):
    nonvalidation_support.update(surface_patches[center])
validation_support = set()
for center in validation_centers:
    validation_support.update(surface_patches[center])

validation_deep_local = [
    int(case["deep_local"]) for case in validation_manifest if case.get("deep_local") is not None
]
if len(validation_deep_local) != 9 or len(set(validation_deep_local)) != 9 or \
        any(index < 0 or index >= n_deep for index in validation_deep_local):
    raise ValueError("受保护 validation 必须含9个唯一合法深点")
surface_deep_candidates = sorted(set(range(n_deep)) - set(validation_deep_local))
if len(surface_deep_candidates) != 7 or \
        sorted(Counter(deep_labels[surface_deep_candidates]).values()) != [3, 4]:
    raise ValueError("排除 validation 后必须恰有左右3/4分配的7个深点")

historical_deep_local = {
    int(case["deep_local"]) for case in historical_cases if case.get("deep_local") is not None
}
historical_seed_roots = {
    int(value) for case in historical_cases
    for value in case.get("replica_seed_roots", {}).values()
}
seed_roots = {
    "gate": {"fit": 2026100201, "check": 2026100202},
    "surface": {"fit": 2026100203, "check": 2026100204},
}
new_seed_values = {value for roots in seed_roots.values() for value in roots.values()}
if len(new_seed_values) != 4 or new_seed_values & historical_seed_roots:
    raise ValueError("v2 fit/check roots必须两两不同且不复用任何历史root")


# %% 5. 几何seed绑定71/72、70选择、helper、shared和全部列明历史清单哈希。
history_sha256 = {name: item["sha256"] for name, item in sorted(history.items())}
auxiliary_sha256 = {
    legacy_lock_path.relative_to(root).as_posix(): hashlib.sha256(legacy_lock_payload).hexdigest(),
    v1_audit_path.relative_to(root).as_posix(): hashlib.sha256(v1_audit_payload).hexdigest(),
    protected_validation_sidecar.relative_to(root).as_posix():
        hashlib.sha256(expected_validation_sidecar).hexdigest(),
}
seed_input = "\0".join([
    shared_fingerprint,
    selection["selected_candidate"],
    *[f"{name}:{item['sha256']}" for name, item in sorted(input_sha256.items())],
    *[f"{name}:{digest}" for name, digest in sorted(history_sha256.items())],
    *[f"{name}:{digest}" for name, digest in sorted(auxiliary_sha256.items())],
])
geometry_domains = {
    "gate": "ERP-v6-hierarchical-v2-fresh-gate-calibration-geometry-v1",
    "surface": "ERP-v6-hierarchical-v2-fresh-surface-calibration-assignment-v1",
}
geometry_digests = {
    name: hashlib.sha256((domain + "\0" + seed_input).encode("utf-8")).hexdigest()
    for name, domain in geometry_domains.items()
}
geometry_seeds = {
    name: int.from_bytes(bytes.fromhex(digest)[:8], "little")
    for name, digest in geometry_digests.items()
}
gate_rng = np.random.default_rng(geometry_seeds["gate"])
surface_rng = np.random.default_rng(geometry_seeds["surface"])


# %% 6. 一级gate：28个中心不落历史patch，且新patch避开validation并彼此互斥。
eligible_gate = {
    center for center in range(n_surf)
    if center not in historical_support
    and surface_patches[center].isdisjoint(validation_support)
}
eligible_gate_lh_rh = [sum(center < n_lh for center in eligible_gate),
                       sum(center >= n_lh for center in eligible_gate)]
gate_center_counts = {"L": 15, "R": 13}
gate_center_bank = {}
for hemisphere in ("L", "R"):
    eligible = sorted(center for center in eligible_gate
                      if (center < n_lh) == (hemisphere == "L"))
    conflicts = {center: {center} for center in eligible}
    for position, first in enumerate(eligible):
        for second in eligible[position + 1:]:
            if not surface_patches[first].isdisjoint(surface_patches[second]):
                conflicts[first].add(second)
                conflicts[second].add(first)
    pool = set(eligible)
    selected = []
    while pool and len(selected) < gate_center_counts[hemisphere]:
        possible = sorted(pool)
        conflict_counts = np.asarray([len(conflicts[center] & pool) for center in possible])
        least_conflict = np.flatnonzero(conflict_counts == conflict_counts.min())
        center = int(possible[int(gate_rng.choice(least_conflict))])
        selected.append(center)
        pool -= conflicts[center]
    if len(selected) != gate_center_counts[hemisphere]:
        raise RuntimeError(
            f"gate {hemisphere}半球只能取得{len(selected)}个满足历史/validation隔离的互斥patch中心"
        )
    gate_center_bank[hemisphere] = selected

gate_same_pairs = {"LL": [], "RR": []}
gate_remaining = {hemisphere: set(gate_center_bank[hemisphere]) for hemisphere in ("L", "R")}
for hemisphere, template in (("L", "LL"), ("R", "RR")):
    for pair_number in range(3):
        ordered = sorted(gate_remaining[hemisphere])
        choices = [
            (float(np.linalg.norm(vertices[first] - vertices[second])), first, second)
            for position, first in enumerate(ordered) for second in ordered[position + 1:]
        ]
        if not choices:
            raise RuntimeError(f"gate {template}第{pair_number}对没有候选")
        distance, first, second = max(choices)
        if distance < .050:
            raise RuntimeError(f"gate {template}第{pair_number}对中心无法满足50 mm")
        gate_same_pairs[template].append([first, second])
        gate_remaining[hemisphere] -= {first, second}

gate_cross_pairs = []
for pair_number in range(3):
    choices = [
        (float(np.linalg.norm(vertices[left] - vertices[right])), left, right)
        for left in sorted(gate_remaining["L"]) for right in sorted(gate_remaining["R"])
    ]
    if not choices:
        raise RuntimeError(f"gate LR第{pair_number}对没有候选")
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

gate_templates = np.array(
    ["S-L"] * 6 + ["S-R"] * 4 + ["D-LL"] * 3 + ["D-RR"] * 3 + ["D-LR"] * 3,
    object,
)
gate_rng.shuffle(gate_templates)
gate_geometries = []
double_number = 0
for configuration_number, template in enumerate(gate_templates):
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
    correlation = 0.
    if len(centers) == 2:
        correlation = .5 if double_number % 2 else 0.
        double_number += 1
    gate_geometries.append({
        "configuration_number": configuration_number,
        "template": str(template),
        "surface_centers": list(map(int, centers)),
        "correlation": correlation,
    })

gate_manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for item in gate_geometries:
        centers = list(item["surface_centers"])
        number = len(gate_manifest)
        case = {
            "case_id": f"erp-v6-hierarchical-v2-gate-calibration-{number:03d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": number,
            "component_balance_revision": "formal_hierarchical_v2_gate_calibration",
            "configuration_id": f"erp-v6-hierarchical-v2-gate-geometry-{item['configuration_number']:02d}",
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
            "panel": "erp_v6_hierarchical_v2_gate_calibration",
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


# %% 7. 二级surface：19个固定深点配置槽跨SNR复用，只用validation未占的7点。
surface_assignments = []
while len(surface_assignments) < 19:
    block = np.asarray(surface_deep_candidates, int)
    surface_rng.shuffle(block)
    surface_assignments.extend(map(int, block))
surface_assignments = surface_assignments[:19]

surface_manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for configuration_number, deep_local in enumerate(surface_assignments):
        number = len(surface_manifest)
        surface_manifest.append({
            "case_id": f"erp-v6-hierarchical-v2-surface-calibration-{number:03d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": number,
            "component_balance_revision": "formal_hierarchical_v2_surface_calibration",
            "configuration_id": f"erp-v6-hierarchical-v2-surface-geometry-{configuration_number:02d}",
            "configuration_kind": "deep_only",
            "configuration_number": configuration_number,
            "correlation": 0.,
            "deep_index": n_surf + deep_local,
            "deep_local": deep_local,
            "deep_surface_ratio": None,
            "eeg_snr_db": eeg_snr,
            "location": None,
            "meg_snr_db": meg_snr,
            "pair_index": pair_index,
            "panel": "erp_v6_hierarchical_v2_surface_calibration",
            "replica_seed_roots": dict(seed_roots["surface"]),
            "replicate": 0,
            "scenario": "deep_only",
            "scenario_code": protocol.SCENARIO_CODES["deep_only"],
            "seed": [seed_roots["surface"]["fit"], pair_index, configuration_number],
            "snr_db": eeg_snr,
            "surface_centers": [],
        })


# %% 8. 病例schema、重复规则、位置隔离和新噪声身份硬核验。
if len(gate_manifest) != 57 or len(surface_manifest) != 57:
    raise ValueError("v2两套正式校准必须各57例")
all_new_cases = gate_manifest + surface_manifest
new_case_ids = [case["case_id"] for case in all_new_cases]
if len(new_case_ids) != len(set(new_case_ids)) or set(new_case_ids) & historical_case_ids or \
        any(not case_id.startswith("erp-v6-hierarchical-v2-") for case_id in new_case_ids):
    raise ValueError("v2 case_id必须使用新前缀、互异且不复用历史ID")
required_case_keys = {
    "case_id", "case_number", "component_balance_revision", "configuration_id",
    "configuration_kind", "configuration_number", "correlation", "deep_index",
    "deep_local", "deep_surface_ratio", "eeg_snr_db", "location", "meg_snr_db",
    "pair_index", "panel", "replica_seed_roots", "replicate", "scenario",
    "scenario_code", "seed", "snr_db", "surface_centers",
}
if any(not required_case_keys <= case.keys() for case in all_new_cases):
    raise ValueError("v2病例缺少prepare_trial_covariance_case所需schema字段")
if any(case["replica_seed_roots"] != seed_roots[panel]
       for panel, cases in (("gate", gate_manifest), ("surface", surface_manifest))
       for case in cases):
    raise ValueError("v2病例noise root与面板冻结值不一致")
if len({(case["panel"], tuple(case["seed"])) for case in all_new_cases}) != 114:
    raise ValueError("v2两套校准的fit噪声身份必须逐例唯一")

for pair_index, pair in enumerate(snr_pairs):
    gate_cell = [case for case in gate_manifest if case["pair_index"] == pair_index]
    surface_cell = [case for case in surface_manifest if case["pair_index"] == pair_index]
    if {(case["eeg_snr_db"], case["meg_snr_db"])
            for case in gate_cell + surface_cell} != {pair}:
        raise ValueError(f"pair_index={pair_index}与SNR不一致")
    if len(gate_cell) != 19 or Counter(
            len(case["surface_centers"]) for case in gate_cell) != {1: 10, 2: 9}:
        raise ValueError(f"gate SNR={pair}不是10单源+9双源")
    if len(surface_cell) != 19 or any(case["scenario"] != "deep_only" for case in surface_cell):
        raise ValueError(f"surface SNR={pair}必须为19个纯深层病例")

for cases, label in ((gate_manifest, "gate"), (surface_manifest, "surface")):
    configuration_counts = Counter(case["configuration_id"] for case in cases)
    if len(configuration_counts) != 19 or set(configuration_counts.values()) != {3}:
        raise ValueError(f"{label}必须是19个固定配置各跨三个SNR")
    for configuration_id in configuration_counts:
        repeated = [case for case in cases if case["configuration_id"] == configuration_id]
        truth = {(tuple(case["surface_centers"]), case["deep_local"], case["correlation"])
                 for case in repeated}
        observed_snr = {(case["eeg_snr_db"], case["meg_snr_db"]) for case in repeated}
        if len(truth) != 1 or observed_snr != set(snr_pairs):
            raise ValueError(f"{configuration_id}没有原样跨三个SNR")

gate_centers = [center for item in gate_geometries for center in item["surface_centers"]]
gate_patches = [surface_patches[center] for center in gate_centers]
gate_patch_union = set().union(*gate_patches)
gate_center_vs_history_support = set(gate_centers) & historical_support
gate_patch_internal_overlap_count = sum(map(len, gate_patches)) - len(gate_patch_union)
gate_patch_vs_validation_overlap = gate_patch_union & validation_support
gate_patch_vs_nonvalidation_history_overlap = gate_patch_union & nonvalidation_support
pair_distances_mm = [
    float(np.linalg.norm(vertices[item["surface_centers"][0]]
                         - vertices[item["surface_centers"][1]]) * 1000)
    for item in gate_geometries if len(item["surface_centers"]) == 2
]
if len(gate_centers) != 28 or len(set(gate_centers)) != 28 or \
        sum(center < n_lh for center in gate_centers) != 15 or \
        gate_center_vs_history_support or gate_patch_internal_overlap_count != 0 or \
        gate_patch_vs_validation_overlap or min(pair_distances_mm) < 50.:
    raise ValueError("v2 gate未满足28中心、历史core、validation patch、内部patch或50 mm隔离")
if any(case.get("surface_component_sensor_balance") is not True
       for case in gate_manifest if len(case["surface_centers"]) == 2):
    raise ValueError("v2 gate所有双表层病例必须启用传感器分量平衡")
if set(surface_assignments) != set(surface_deep_candidates) or \
        set(surface_assignments) & set(validation_deep_local) or \
        set(Counter(surface_assignments).values()) != {2, 3}:
    raise ValueError("v2 surface的19配置必须只均衡复用validation未占的7个深点")


# %% 9. 先冻结两个manifest，再写清约束、诚实限制和validation只读引用。
gate_payload = (json.dumps(gate_manifest, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n").encode("utf-8")
surface_payload = (json.dumps(surface_manifest, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n").encode("utf-8")
manifest_sha256 = {
    "gate_calibration": hashlib.sha256(gate_payload).hexdigest(),
    "surface_calibration": hashlib.sha256(surface_payload).hexdigest(),
    "protected_v1_validation_reference": protected_validation_sha256,
}

audit = {
    "protocol": "erp-v6-hierarchical-v2-formal-manifests",
    "purpose": "freeze two fresh calibration manifests; reference but never copy or run untouched v1 validation",
    "formal_results_present": False,
    "simulation_or_inverse_run_by_generator": False,
    "development_selection_summary_read": True,
    "validation_results_read": False,
    "protected_validation_copied_or_modified": False,
    "input_sha256": input_sha256,
    "history_manifest_sha256": history_sha256,
    "auxiliary_input_sha256": auxiliary_sha256,
    "shared_fingerprint": shared_fingerprint,
    "numpy_version_for_deterministic_geometry": np.__version__,
    "manifest_sha256": manifest_sha256,
    "selected_gate_score": {
        "candidate": selection["selected_candidate"],
        "family": selected_specs[0]["family"],
        "formula": selected_specs[0]["formula"],
        "operational_formula": (
            "h_m=g_m/sqrt(1+q_m); b=min(1+q_EEG,1+q_MEG)/max(1+q_EEG,1+q_MEG); "
            "w_EEG=0.5+0.25*b; score=w_EEG*h_EEG+(1-w_EEG)*h_MEG"
        ),
        "primary_fit_channel_weights": "all ones after per-channel whitening",
        "scope": (
            "primary H0/H1 fits and gate score use all-one weights after per-channel whitening; "
            "deep LOO selection and deep-amplitude estimation use original training-derived "
            "observation weights; conditional-surface train20 and combined40 fits use frozen W=1"
        ),
        "test_true_snr_used": False,
        "selection_summary_sha256": expected_selection_sha256,
        "formal_performance_claim_from_development": False,
    },
    "geometry_domains": geometry_domains,
    "geometry_seed_sha256": geometry_digests,
    "geometry_seed_uint64": geometry_seeds,
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "replica_seed_roots": seed_roots,
    "gate_calibration": {
        "case_count": 57,
        "configuration_count": 19,
        "configuration_repetition_across_snr": 3,
        "single_per_snr": 10,
        "double_per_snr": 9,
        "surface_center_count": 28,
        "surface_centers": gate_centers,
        "surface_lh_rh": [sum(center < n_lh for center in gate_centers),
                          sum(center >= n_lh for center in gate_centers)],
        "eligible_centers_lh_rh_before_selection": eligible_gate_lh_rh,
        "centers_inside_any_historical_patch_support": len(gate_center_vs_history_support),
        "within_v2_gate_patch_overlap_count": gate_patch_internal_overlap_count,
        "gate_patch_vs_protected_validation_patch_overlap_count":
            len(gate_patch_vs_validation_overlap),
        "gate_patch_vs_old_nonvalidation_patch_overlap_vertex_count":
            len(gate_patch_vs_nonvalidation_history_overlap),
        "old_nonvalidation_patch_fringe_overlap_allowed": True,
        "full_all_history_patch_novelty_claim": False,
        "minimum_two_surface_distance_mm": min(pair_distances_mm),
        "calibration_claim": (
            "pooled fixed-design engineering calibration across 19 geometries and three SNRs; "
            "not an iid population conformal guarantee"
        ),
    },
    "surface_calibration": {
        "case_count": 57,
        "configuration_count": 19,
        "configuration_repetition_across_snr": 3,
        "unique_deep_geometry_count": 7,
        "deep_local": surface_deep_candidates,
        "deep_global": [n_surf + index for index in surface_deep_candidates],
        "configuration_deep_assignment": surface_assignments,
        "uses_only_validation_unoccupied_deep_local": True,
        "overlap_with_protected_validation_deep_local": 0,
        "deep_geometry_seen_in_any_listed_history": sorted(
            set(surface_deep_candidates) & historical_deep_local),
        "globally_new_deep_geometry_claim": False,
        "fresh_case_and_noise_identity": True,
        "truth_use": (
            "predeclared deep-only null for surface-presence calibration; true deep index must never "
            "enter inverse, localization candidate choice or score"
        ),
    },
    "protected_validation": {
        "path": protected_validation_path.relative_to(root).as_posix(),
        "sha256": protected_validation_sha256,
        "sidecar_path": protected_validation_sidecar.relative_to(root).as_posix(),
        "case_count": 21,
        "seed_roots": validation_seed_roots,
        "surface_center_count": 27,
        "deep_local": validation_deep_local,
        "shared_consumed_marker": shared_validation_marker.relative_to(root).as_posix(),
        "firewall_paths_required_absent": [path.relative_to(root).as_posix()
                                            for path in validation_firewall_paths],
        "manifest_created_by_v2_generator": False,
        "manifest_bytes_written_by_v2_generator": False,
    },
    "truth_firewall": {
        "generator_reads_development_selection_summary": True,
        "generator_reads_validation_result_metrics": False,
        "generator_runs_simulation_or_inverse": False,
        "validation_truth_used_for_algorithm_or_score": False,
        "validation_manifest_use": "geometry exclusion, seed/schema audit and immutable SHA reference only",
    },
    "limits": {
        "surface_history": (
            "gate centers avoid every listed historical patch support and complete gate patches avoid "
            "protected validation patches, but gate patch fringe may overlap old nonvalidation patches"
        ),
        "deep_history": (
            "the current 16-point deep grid is historically exhausted; surface calibration uses the "
            "seven points excluded from protected validation and makes no all-history novelty claim"
        ),
        "statistics": (
            "both calibration panels are repeated fixed designs across SNR and support engineering "
            "thresholds only, not iid/conformal or clinical population error guarantees"
        ),
    },
}

report = f"""# Hierarchical v2 正式清单

状态：只冻结两套新校准病例；未运行仿真、逆解、校准或 validation。

## 冻结输入和分数

- 71 号清单、计划中的 72 号正式 runner、70 号盲可靠性选择 summary、predictive helper、共享 forward 指纹和所有列明历史清单都以 SHA256 绑定。
- 一级门控固定为 70 号预选的 `adaptive_balance_p1`：`h_m=g_m/sqrt(1+q_m)`，`b=min(1+q)/max(1+q)`，`w_EEG=0.5+0.25*b`，最终为 `w_EEG*h_EEG+(1-w_EEG)*h_MEG`。逐白化后的 primary H0/H1 拟合与门控固定 `W=1`；deep LOO 候选选择和深源幅度估计使用训练阶段得到的原 observation weights；conditional surface 的 train20 与 combined40 均固定 `W=1`。
- 这些开发选择不构成独立性能结果；正式功效只能由全新校准和一次性 untouched validation 检验。

## 两套新校准

- Gate：19 个固定纯表层配置跨 `(-10,-10)`、`(-10,+20)`、`(+20,-10)` 三个 EEG/MEG SNR，共 57 例；每档 10 单源、9 双源。roots 为 `{seed_roots['gate']['fit']}/{seed_roots['gate']['check']}`。
- Surface：19 个固定纯深层配置槽跨三个 SNR，共 57 例，只均衡复用 validation 未占的 7 个 deep_local `{surface_deep_candidates}`。roots 为 `{seed_roots['surface']['fit']}/{seed_roots['surface']['check']}`；case ID 和噪声身份全新。

## 空间隔离和诚实边界

- Gate 共 28 个唯一中心。中心落入任一列明历史 patch support 的数量为 0；新 gate patch 内部重叠为 0；与受保护 validation patch 重叠为 0；所有双源中心距离至少 `{min(pair_distances_mm):.6f}` mm。
- Gate patch 允许与旧非 validation 历史 patch 的最外圈重叠，本次精确重叠顶点数为 `{len(gate_patch_vs_nonvalidation_history_overlap)}`。因此不声称完整 patch 相对全历史绝对未见。
- 当前 16 点深网格已被历史耗尽。Surface 校准的 7 点只保证完全排除 validation 使用的 9 点，不声称相对全项目历史为新位置。

## Untouched validation 只读引用

- 继续引用原 `hierarchical_v1_validation_manifest.json`，SHA256 固定为 `{protected_validation_sha256}`；71 号不复制、不改写、也不生成第二份 validation manifest。
- 原 validation roots 必须保持 `2026100101/2026100102`。共享 consumed marker、v1/v2 validation 输出和对应 run-lock 必须全部不存在，否则本脚本立即拒绝生成或核对。
- 21 例 validation 是一次性固定工程面板，不证明总体或临床误报率。
"""


# %% 10. 前后哈希一致后首次写出；以后--check逐字节核对，绝不覆盖。
for name, path in input_paths.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != input_sha256[name]["sha256"]:
        raise RuntimeError(f"生成期间冻结输入发生变化：{path}")
for name, item in history.items():
    if hashlib.sha256(item["path"].read_bytes()).hexdigest() != history_sha256[name]:
        raise RuntimeError(f"生成期间历史清单发生变化：{item['path']}")
if hashlib.sha256(legacy_lock_path.read_bytes()).hexdigest() != auxiliary_sha256[
        legacy_lock_path.relative_to(root).as_posix()] or \
        hashlib.sha256(v1_audit_path.read_bytes()).hexdigest() != auxiliary_sha256[
            v1_audit_path.relative_to(root).as_posix()] or \
        protected_validation_sidecar.read_bytes() != expected_validation_sidecar:
    raise RuntimeError("生成期间旧几何lock、v1 audit或validation旁车发生变化")

payloads = {
    output_paths["gate"]: gate_payload,
    output_paths["surface"]: surface_payload,
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
            raise ValueError(f"冻结 hierarchical-v2 正式协议已变化：{path}")
else:
    occupied = [str(path) for path in payloads
                if path.exists() or Path(str(path) + ".sha256").exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有 hierarchical-v2 正式协议：{occupied}")
    for path, payload in payloads.items():
        with path.open("xb") as stream:
            stream.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = Path(str(path) + ".sha256")
        with sidecar.open("xb") as stream:
            stream.write(f"{digest}  {path.name}\n".encode("ascii"))

print(json.dumps({
    "checked": bool(args.check),
    "outputs": {str(path): hashlib.sha256(payload).hexdigest()
                for path, payload in payloads.items()},
    "case_counts": {"gate": len(gate_manifest), "surface": len(surface_manifest)},
    "protected_validation_sha256": protected_validation_sha256,
    "protected_validation_written": False,
    "gate_surface_centers": len(gate_centers),
    "surface_calibration_deep_local": surface_deep_candidates,
}, ensure_ascii=False, indent=2), flush=True)
