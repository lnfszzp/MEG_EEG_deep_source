"""# %% 生成 v3 全新开发面板：13 套几何跨 3 个 EEG/MEG SNR，共 39 例。"""

# %% 1. 固定入口、作者风格、输出和全新噪声根；preflight 绝不触发 forward 构建。
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("v3 开发面板核验禁止使用 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

parser = argparse.ArgumentParser(description=__doc__)
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--preflight", action="store_true", help="只核验；缓存不存在时也绝不构建")
mode.add_argument("--check", action="store_true", help="重算并逐字节核对已冻结输出")
args = parser.parse_args()

script_path = Path(__file__).resolve()
script_text = script_path.read_text(encoding="utf-8")
script_tree = ast.parse(script_text)
if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
       for node in ast.walk(script_tree)):
    raise RuntimeError("作者风格自检失败：78 号脚本不应定义 def/class")
if script_text.count("# %%") < 8:
    raise RuntimeError("作者风格自检失败：78 号脚本缺少顺序式 # %% 分段")

adaptive_root = root / "results/erp_whole_head/adaptive_v6"
protocol_dir = adaptive_root / "protocol"
output_paths = {
    "manifest": protocol_dir / "hierarchical_v3_fresh_development_manifest.json",
    "audit": protocol_dir / "hierarchical_v3_fresh_development_audit.json",
    "report": protocol_dir / "HIERARCHICAL_V3_FRESH_DEVELOPMENT.md",
}
if not args.preflight and not args.check:
    occupied = [str(path) for path in output_paths.values()
                if path.exists() or Path(str(path) + ".sha256").exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有 v3 全新开发面板：{occupied}")

seed_roots = {"fit": 2026100301, "check": 2026100302}
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
expected_refined_keys = {
    "gain_eeg", "gain_meg", "vertices", "surface_xyz_head", "surface_xyz_mri",
    "deep_xyz_head", "deep_xyz_mri", "surface_vertex_numbers", "surface_hemi",
    "deep_aseg_labels", "eeg_ch_names", "meg_ch_names", "head_mri_t",
    "mri_head_t",
    "coordinate_frame", "surface_edges", "n_surf", "n_deep",
    "surface_spacing_mm", "deep_spacing_mm",
}


# %% 2. 动态读取 protocol 中所有历史病例清单；未知或结果 JSON 不冒充 manifest。
history = {}
history_cases = []
for path in sorted(protocol_dir.glob("*.json")):
    if path.resolve() == output_paths["manifest"].resolve():
        continue
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(value, list) and value and isinstance(value[0], dict) and \
            "case_id" in value[0]:
        if any(not isinstance(case, dict) or "case_id" not in case for case in value):
            raise ValueError(f"历史病例清单结构不一致：{path}")
        history[path.name] = {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "cases": value,
        }
        history_cases.extend(value)
if not history or len(history_cases) < 1:
    raise FileNotFoundError("未找到任何历史病例清单，拒绝生成无法审计的新开发面板")

historical_case_ids = {str(case["case_id"]) for case in history_cases}
historical_seed_roots = {
    int(value) for case in history_cases
    for value in case.get("replica_seed_roots", {}).values()
}
historical_seed_tuples = {
    tuple(map(int, case[key])) for case in history_cases
    for key in ("seed", "check_seed")
    if isinstance(case.get(key), list)
}
if set(seed_roots.values()) & historical_seed_roots or \
        any(seed[0] in seed_roots.values() for seed in historical_seed_tuples if seed):
    raise ValueError("2026100301/02 已被历史清单使用")

refined_helper_path = root / "refined_grid.py"
if not refined_helper_path.is_file():
    raise FileNotFoundError(refined_helper_path)
from refined_grid import CACHE as refined_cache

if args.preflight and not refined_cache.is_file():
    print(json.dumps({
        "preflight": "STATIC_PASS_FORWARD_CACHE_MISSING",
        "manifest_generation_allowed": False,
        "reason": "refined oct7/5mm cache is absent; preflight did not build it",
        "cache": str(refined_cache),
        "expected_cache_keys": sorted(expected_refined_keys),
        "history_manifest_count": len(history),
        "history_case_count": len(history_cases),
        "seed_roots": seed_roots,
        "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
        "planned_case_count": 39,
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)


# %% 3. 载入 coarse inverse 和 refined truth forward；通道、坐标系和缓存字段全部 fail closed。
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from benchmark import protocol
import run_erp_whole_head_matrix as simulation
from refined_grid import load_refined_forward

shared = protocol.load_shared(root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
shared_fingerprint = simulation._shared_fingerprint(shared)
refined = load_refined_forward(refined_cache)
missing_refined_keys = sorted(expected_refined_keys - refined.keys())
if missing_refined_keys:
    raise ValueError(
        "refined cache 是旧 schema，拒绝猜测通道或坐标顺序；请删除后用当前 refined_grid.py 重建："
        f"{missing_refined_keys}"
    )

n_inverse_surface = int(shared["n_surf"])
n_inverse_deep = int(shared["n_deep"])
n_truth_surface = int(np.asarray(refined["n_surf"]).ravel()[0])
n_truth_deep = int(np.asarray(refined["n_deep"]).ravel()[0])
gain_eeg = np.asarray(refined["gain_eeg"], dtype=float)
gain_meg = np.asarray(refined["gain_meg"], dtype=float)
surface_head = np.asarray(refined["surface_xyz_head"], dtype=float)
surface_mri = np.asarray(refined["surface_xyz_mri"], dtype=float)
deep_head = np.asarray(refined["deep_xyz_head"], dtype=float)
deep_mri = np.asarray(refined["deep_xyz_mri"], dtype=float)
vertices_head = np.asarray(refined["vertices"], dtype=float)
surface_hemi = np.asarray(refined["surface_hemi"], dtype=int).ravel()
surface_vertex_numbers = np.asarray(refined["surface_vertex_numbers"], dtype=int).ravel()
deep_labels = np.asarray(refined["deep_aseg_labels"], dtype=int).ravel()
surface_edges = np.asarray(refined["surface_edges"], dtype=int)
eeg_names = list(map(str, np.asarray(refined["eeg_ch_names"]).ravel()))
meg_names = list(map(str, np.asarray(refined["meg_ch_names"]).ravel()))
coordinate_frame = list(map(str, np.asarray(refined["coordinate_frame"]).ravel()))
head_to_mri = np.asarray(refined["head_mri_t"], dtype=float)
mri_to_head = np.asarray(refined["mri_head_t"], dtype=float)

expected_shapes = {
    "gain_eeg": (len(eeg_names), n_truth_surface + n_truth_deep),
    "gain_meg": (len(meg_names), n_truth_surface + n_truth_deep),
    "surface_head": (n_truth_surface, 3),
    "surface_mri": (n_truth_surface, 3),
    "deep_head": (n_truth_deep, 3),
    "deep_mri": (n_truth_deep, 3),
    "vertices_head": (n_truth_surface + n_truth_deep, 3),
    "surface_hemi": (n_truth_surface,),
    "surface_vertex_numbers": (n_truth_surface,),
    "deep_labels": (n_truth_deep,),
    "head_to_mri": (4, 4),
    "mri_to_head": (4, 4),
}
actual_shapes = {
    "gain_eeg": gain_eeg.shape, "gain_meg": gain_meg.shape,
    "surface_head": surface_head.shape, "surface_mri": surface_mri.shape,
    "deep_head": deep_head.shape, "deep_mri": deep_mri.shape,
    "vertices_head": vertices_head.shape, "surface_hemi": surface_hemi.shape,
    "surface_vertex_numbers": surface_vertex_numbers.shape,
    "deep_labels": deep_labels.shape, "head_to_mri": head_to_mri.shape,
    "mri_to_head": mri_to_head.shape,
}
if actual_shapes != expected_shapes:
    raise ValueError(f"refined forward shape 不一致：{actual_shapes} != {expected_shapes}")
if coordinate_frame != ["HEAD"] or \
        not np.allclose(vertices_head, np.vstack([surface_head, deep_head]), rtol=0., atol=1e-12):
    raise ValueError("refined vertices 必须是统一 HEAD 坐标")
if set(np.unique(surface_hemi)) != {0, 1} or \
        len(set(zip(surface_hemi.tolist(), surface_vertex_numbers.tolist()))) != n_truth_surface:
    raise ValueError("oct7 表层 hemisphere/vertex number 身份不完整")
if not np.all(np.isin(deep_labels, (10, 49))) or set(np.unique(deep_labels)) != {10, 49}:
    raise ValueError("5mm 深层 truth 必须只含双侧丘脑标签 10/49")
if surface_edges.ndim != 2 or surface_edges.shape[1] != 2 or \
        np.any(surface_edges < 0) or np.any(surface_edges >= n_truth_surface) or \
        np.any(surface_edges[:, 0] >= surface_edges[:, 1]):
    raise ValueError("oct7 surface_edges 必须是合法且去重的上三角边")
if eeg_names != list(shared["info_eeg"]["ch_names"]) or \
        meg_names != list(shared["info_meg"]["ch_names"]):
    raise ValueError("refined truth 与 coarse inverse 的 EEG/MEG 行顺序不一致")

surface_mri_from_head = (
    head_to_mri @ np.column_stack([surface_head, np.ones(n_truth_surface)]).T
).T[:, :3]
deep_mri_from_head = (
    head_to_mri @ np.column_stack([deep_head, np.ones(n_truth_deep)]).T
).T[:, :3]
surface_transform_error = float(np.max(np.linalg.norm(surface_mri_from_head - surface_mri, axis=1)))
deep_transform_error = float(np.max(np.linalg.norm(deep_mri_from_head - deep_mri, axis=1)))
if surface_transform_error > 1e-9 or deep_transform_error > 1e-9:
    raise ValueError("refined HEAD→MRI 变换与保存坐标不一致")
if not np.allclose(head_to_mri @ mri_to_head, np.eye(4), rtol=0., atol=1e-12) or \
        not np.allclose(mri_to_head @ head_to_mri, np.eye(4), rtol=0., atol=1e-12):
    raise ValueError("refined HEAD↔MRI 变换不是互逆矩阵")

refined_file_sha256 = hashlib.sha256(refined_cache.read_bytes()).hexdigest()
refined_array_sha256 = {}
for name in sorted(expected_refined_keys):
    values = np.ascontiguousarray(np.asarray(refined[name]))
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    refined_array_sha256[name] = digest.hexdigest()
refined_bundle_sha256 = hashlib.sha256("\n".join(
    f"{name}:{refined_array_sha256[name]}" for name in sorted(refined_array_sha256)
).encode("ascii")).hexdigest()
channel_order_sha256 = hashlib.sha256(
    ("EEG\0" + "\0".join(eeg_names) + "\0MEG\0" + "\0".join(meg_names)).encode("utf-8")
).hexdigest()


# %% 4. 把全部历史中心、patch、seed 和位置转换为空间防火墙。
historical_surface_centers = {
    int(center) for case in history_cases for center in case.get("surface_centers", [])
    if isinstance(center, (int, np.integer)) and 0 <= int(center) < n_inverse_surface
}
historical_deep_local = {
    int(case["deep_local"]) for case in history_cases
    if isinstance(case.get("deep_local"), (int, np.integer))
    and 0 <= int(case["deep_local"]) < n_inverse_deep
}
if not historical_surface_centers or not historical_deep_local:
    raise ValueError("历史清单没有可用于空间防火墙的表层或深层位置")

historical_surface_support = set()
for center in sorted(historical_surface_centers):
    historical_surface_support.update(map(int, protocol._surface_patch(shared, center)[0]))
inverse_surface_head = np.asarray(shared["vertices"], dtype=float)[:n_inverse_surface]
inverse_deep_head = np.asarray(shared["deep_rr"], dtype=float)
historical_surface_xyz = inverse_surface_head[sorted(historical_surface_centers)]
historical_deep_xyz = inverse_deep_head[sorted(historical_deep_local)]
historical_surface_tree = cKDTree(historical_surface_xyz)
inverse_surface_tree = cKDTree(inverse_surface_head)
inverse_deep_tree = cKDTree(inverse_deep_head)

history_sha256 = {name: item["sha256"] for name, item in sorted(history.items())}
geometry_seed_payload = "\0".join([
    "ERP-v3-fresh-development-geometry-v1",
    shared_fingerprint,
    refined_file_sha256,
    refined_bundle_sha256,
    hashlib.sha256(refined_helper_path.read_bytes()).hexdigest(),
    *[f"{name}:{digest}" for name, digest in history_sha256.items()],
])
geometry_seed_sha256 = hashlib.sha256(geometry_seed_payload.encode("utf-8")).hexdigest()
geometry_seed_uint64 = int.from_bytes(bytes.fromhex(geometry_seed_sha256)[:8], "little")
rng = np.random.default_rng(geometry_seed_uint64)


# %% 5. 仅用 forward+noise whitening 定义深源 alias；三等分后做 9 槽唯一匹配。
whiteners = {}
whitening_ranks = {}
for modality, covariance in (
        ("eeg", np.asarray(shared["noise_cov_eeg"], dtype=float)),
        ("meg", np.asarray(shared["noise_cov_meg"], dtype=float))):
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2.)
    keep = values > max(float(values.max()) * 1e-12, np.finfo(float).tiny)
    if keep.sum() < 2:
        raise ValueError(f"{modality.upper()} noise covariance 无法稳定白化")
    whiteners[modality] = (vectors[:, keep] / np.sqrt(values[keep])).T
    whitening_ranks[modality] = int(keep.sum())

inverse_surface_design = np.vstack([
    whiteners["eeg"] @ np.asarray(shared["gain_eeg"], float)[:, :n_inverse_surface]
    / np.sqrt(whitening_ranks["eeg"]),
    whiteners["meg"] @ np.asarray(shared["gain_meg"], float)[:, :n_inverse_surface]
    / np.sqrt(whitening_ranks["meg"]),
])
truth_deep_design = np.vstack([
    whiteners["eeg"] @ gain_eeg[:, n_truth_surface:]
    / np.sqrt(whitening_ranks["eeg"]),
    whiteners["meg"] @ gain_meg[:, n_truth_surface:]
    / np.sqrt(whitening_ranks["meg"]),
])
inverse_surface_norm = np.linalg.norm(inverse_surface_design, axis=0)
truth_deep_norm = np.linalg.norm(truth_deep_design, axis=0)
if np.any(inverse_surface_norm <= 0.) or np.any(truth_deep_norm <= 0.):
    raise ValueError("alias audit 遇到零范数 lead field")
inverse_surface_unit = inverse_surface_design / inverse_surface_norm
truth_deep_unit = truth_deep_design / truth_deep_norm
deep_alias_rho = np.zeros(n_truth_deep, dtype=float)
for start in range(0, n_truth_deep, 64):
    stop = min(start + 64, n_truth_deep)
    deep_alias_rho[start:stop] = np.max(np.abs(
        inverse_surface_unit.T @ truth_deep_unit[:, start:stop]
    ), axis=0)
deep_alias_rho = np.clip(deep_alias_rho, 0., 1.)

deep_inverse_distance, deep_nearest_inverse = inverse_deep_tree.query(deep_head, k=1)
deep_eligible = np.flatnonzero(
    (deep_inverse_distance >= .003) & (deep_inverse_distance <= .008)
    & np.isfinite(deep_alias_rho)
)
if deep_eligible.size < 9:
    raise RuntimeError("距 coarse deep 3–8 mm 的 5mm 丘脑候选不足 9 个")
deep_sorted = deep_eligible[np.argsort(deep_alias_rho[deep_eligible], kind="stable")]
deep_tier_arrays = dict(zip(("low", "mid", "high"), np.array_split(deep_sorted, 3)))
if any(indices.size < 3 for indices in deep_tier_arrays.values()):
    raise RuntimeError("alias rho 三等分后某档不足 3 个候选")

slot_sides = {
    "low": (10, 49, 10),
    "mid": (49, 10, 49),
    "high": (10, 49, 10),
}
deep_slots = [
    (tier, scenario_slot, label)
    for tier in ("low", "mid", "high")
    for scenario_slot, label in enumerate(slot_sides[tier])
]
matching_cost = np.full((len(deep_slots), n_inverse_deep), np.inf)
matching_truth_index = np.full((len(deep_slots), n_inverse_deep), -1, dtype=int)
for row, (tier, scenario_slot, label) in enumerate(deep_slots):
    tier_indices = deep_tier_arrays[tier]
    target_rho = float(np.median(deep_alias_rho[tier_indices]))
    for coarse_local in range(n_inverse_deep):
        candidates = tier_indices[
            (deep_labels[tier_indices] == label)
            & (deep_nearest_inverse[tier_indices] == coarse_local)
        ]
        if not candidates.size:
            continue
        candidate_cost = np.abs(deep_alias_rho[candidates] - target_rho) \
            + 1e-4 * np.abs(deep_inverse_distance[candidates] * 1000. - 5.5)
        best = int(candidates[int(np.argmin(candidate_cost))])
        matching_cost[row, coarse_local] = float(candidate_cost.min())
        matching_truth_index[row, coarse_local] = best
matching_rows, matching_columns = linear_sum_assignment(matching_cost)
if not np.array_equal(matching_rows, np.arange(len(deep_slots))) or \
        np.any(~np.isfinite(matching_cost[matching_rows, matching_columns])):
    raise RuntimeError("无法同时满足 alias 档、左右平衡和最近 coarse deep 唯一")
selected_deep = [
    int(matching_truth_index[row, column])
    for row, column in zip(matching_rows, matching_columns)
]
if len(set(selected_deep)) != 9 or len(set(map(int, matching_columns))) != 9:
    raise RuntimeError("9 个 H1 几何没有取得唯一 truth deep / nearest inverse deep")
deep_by_tier = {
    tier: [selected_deep[row] for row, slot in enumerate(deep_slots) if slot[0] == tier]
    for tier in ("low", "mid", "high")
}


# %% 6. 在 oct7 上选 15 个新中心；patch 两两不重叠，且不贴住任何历史中心。
surface_inverse_distance, surface_nearest_inverse = inverse_surface_tree.query(surface_head, k=1)
surface_history_distance = historical_surface_tree.query(surface_head, k=1)[0]
eligible_surface = np.flatnonzero(
    (surface_inverse_distance >= .002) & (surface_inverse_distance <= .005)
    & (surface_history_distance >= .010)
    & ~np.isin(surface_nearest_inverse, np.asarray(sorted(historical_surface_centers), int))
)
if sum(surface_hemi[eligible_surface] == 0) < 8 or \
        sum(surface_hemi[eligible_surface] == 1) < 7:
    raise RuntimeError("满足 2–5 mm grid mismatch 与 10 mm 历史中心隔离的 oct7 候选不足")

surface_graph = sparse.coo_matrix(
    (np.ones(surface_edges.shape[0] * 2, dtype=np.uint8),
     (np.r_[surface_edges[:, 0], surface_edges[:, 1]],
      np.r_[surface_edges[:, 1], surface_edges[:, 0]])),
    shape=(n_truth_surface, n_truth_surface),
).tocsr()
truth_surface_shared = {
    "n_surf": n_truth_surface,
    "adjacency": surface_graph,
    "vertices": surface_head,
}
random_priority = rng.random(n_truth_surface)
desired_hemispheres = [0, 1] * 7 + [0]
selected_surface = []
selected_surface_patches = []
selected_surface_support = set()
for hemisphere in desired_hemispheres:
    candidates = eligible_surface[surface_hemi[eligible_surface] == hemisphere]
    if selected_surface:
        minimum_distance = np.min(
            np.linalg.norm(
                surface_head[candidates, None, :]
                - surface_head[np.asarray(selected_surface), :][None, :, :],
                axis=2,
            ),
            axis=1,
        )
    else:
        minimum_distance = np.full(candidates.size, np.inf)
    order = np.lexsort((random_priority[candidates], -minimum_distance))
    accepted = None
    accepted_patch = None
    for position in order:
        center = int(candidates[position])
        if selected_surface and minimum_distance[position] < .025:
            break
        patch = np.asarray(protocol._surface_patch(truth_surface_shared, center)[0], int)
        if selected_surface_support.intersection(map(int, patch)):
            continue
        if float(historical_surface_tree.query(surface_head[patch], k=1)[0].min()) < .005:
            continue
        accepted = center
        accepted_patch = patch
        break
    if accepted is None:
        raise RuntimeError("无法选足 15 个历史隔离且内部不重叠的 oct7 patch")
    selected_surface.append(accepted)
    selected_surface_patches.append(accepted_patch)
    selected_surface_support.update(map(int, accepted_patch))

remaining = set(selected_surface)
surface_pairs = []
for pair_number in range(5):
    choices = [
        (float(np.linalg.norm(surface_head[first] - surface_head[second])), first, second)
        for position, first in enumerate(sorted(remaining))
        for second in sorted(remaining)[position + 1:]
    ]
    if not choices:
        raise RuntimeError(f"第 {pair_number + 1} 个双表层几何没有候选")
    distance, first, second = max(choices)
    if distance < .050:
        raise RuntimeError("双表层中心无法全部满足至少 50 mm")
    surface_pairs.append([int(first), int(second)])
    remaining -= {first, second}
surface_singles = sorted(remaining, key=lambda index: selected_surface.index(index))
if len(surface_singles) != 5:
    raise RuntimeError("15 个中心必须分为 5 单源中心和 5 个双源对")


# %% 7. 固定 13 套几何：4 H0 + 9 H1；几何完全独立于 SNR。
configurations = []
configurations.extend([
    {
        "configuration_number": 0, "scenario": "surface_only",
        "surface_centers": [surface_singles[0]], "deep_truth_local": None,
        "alias_tier": None, "h0_role": "gate_calibration", "correlation": 0.,
        "deep_surface_ratio": None,
    },
    {
        "configuration_number": 1, "scenario": "surface_only",
        "surface_centers": [surface_singles[1]], "deep_truth_local": None,
        "alias_tier": None, "h0_role": "gate_audit", "correlation": 0.,
        "deep_surface_ratio": None,
    },
    {
        "configuration_number": 2, "scenario": "surface_only",
        "surface_centers": surface_pairs[0], "deep_truth_local": None,
        "alias_tier": None, "h0_role": "gate_calibration", "correlation": 0.,
        "deep_surface_ratio": None, "surface_component_sensor_balance": True,
    },
    {
        "configuration_number": 3, "scenario": "surface_only",
        "surface_centers": surface_pairs[1], "deep_truth_local": None,
        "alias_tier": None, "h0_role": "gate_audit", "correlation": .5,
        "deep_surface_ratio": None, "surface_component_sensor_balance": True,
    },
])
for tier_number, tier in enumerate(("low", "mid", "high")):
    configurations.append({
        "configuration_number": 4 + tier_number, "scenario": "deep_only",
        "surface_centers": [], "deep_truth_local": deep_by_tier[tier][0],
        "alias_tier": tier, "h0_role": None, "correlation": 0.,
        "deep_surface_ratio": None,
    })
deep_one_design = {"low": (0., .5), "mid": (.5, 1.), "high": (.9, .5)}
deep_two_design = {"low": (.9, 1.), "mid": (0., .5), "high": (.5, 1.)}
for tier_number, tier in enumerate(("low", "mid", "high")):
    correlation, ratio = deep_one_design[tier]
    configurations.append({
        "configuration_number": 7 + tier_number, "scenario": "deep_plus_surface",
        "surface_centers": [surface_singles[2 + tier_number]],
        "deep_truth_local": deep_by_tier[tier][1], "alias_tier": tier,
        "h0_role": None, "correlation": correlation, "deep_surface_ratio": ratio,
        "deep_surface_sensor_amplitude_ratio": ratio,
    })
for tier_number, tier in enumerate(("low", "mid", "high")):
    correlation, ratio = deep_two_design[tier]
    configurations.append({
        "configuration_number": 10 + tier_number,
        "scenario": "deep_plus_two_surface",
        "surface_centers": surface_pairs[2 + tier_number],
        "deep_truth_local": deep_by_tier[tier][2], "alias_tier": tier,
        "h0_role": None, "correlation": correlation, "deep_surface_ratio": ratio,
        "deep_surface_sensor_amplitude_ratio": ratio,
        "surface_component_sensor_balance": True,
    })
if [item["configuration_number"] for item in configurations] != list(range(13)):
    raise RuntimeError("13 套几何编号必须连续且顺序冻结")

for item in configurations:
    centers = list(map(int, item["surface_centers"]))
    item.setdefault("deep_surface_sensor_amplitude_ratio", None)
    item["configuration_id"] = \
        f"erp-v3-fresh-dev-geometry-{item['configuration_number']:02d}"
    item["configuration_kind"] = {
        ("surface_only", 1): "single_surface_h0",
        ("surface_only", 2): "two_surface_h0",
        ("deep_only", 0): "deep_only_h1",
        ("deep_plus_surface", 1): "deep_plus_single_surface_h1",
        ("deep_plus_two_surface", 2): "deep_plus_two_surface_h1",
    }[(item["scenario"], len(centers))]
    item["truth_surface_centers_refined"] = centers
    item["truth_surface_center_xyz_head_m"] = surface_head[centers].tolist()
    item["truth_surface_center_xyz_mri_m"] = surface_mri[centers].tolist()
    item["truth_surface_nearest_inverse"] = \
        surface_nearest_inverse[centers].astype(int).tolist()
    item["truth_surface_nearest_inverse_distance_mm"] = \
        (surface_inverse_distance[centers] * 1000.).tolist()
    patch_parts = [protocol._surface_patch(truth_surface_shared, center)
                   for center in centers]
    item["truth_surface_patch_indices_refined"] = [
        indices.astype(int).tolist() for indices, weights in patch_parts
    ]
    item["truth_surface_patch_weights"] = [
        weights.astype(float).tolist() for indices, weights in patch_parts
    ]
    item["truth_surface_patch_nearest_inverse"] = [
        surface_nearest_inverse[indices].astype(int).tolist()
        for indices, weights in patch_parts
    ]
    item["evaluation_surface_projection"] = {
        "rule": "nearest coarse surface vertex in HEAD coordinates",
        "inverse_indices_by_patch": item["truth_surface_patch_nearest_inverse"],
    }
    item["truth_surface_patch_nearest_inverse_distance_mm"] = [
        (surface_inverse_distance[indices] * 1000.).tolist()
        for indices, weights in patch_parts
    ]
    deep_local = item["deep_truth_local"]
    if deep_local is None:
        item.update({
            "deep_index": None, "deep_local": None,
            "truth_deep_fine_index": None, "evaluation_deep_index": None,
            "truth_deep_xyz_head_m": None, "truth_deep_xyz_mri_m": None,
            "truth_deep_aseg_label": None, "deep_nearest_inverse_local": None,
            "deep_nearest_inverse_global": None,
            "deep_nearest_inverse_distance_mm": None, "deep_alias_rho": None,
        })
    else:
        deep_local = int(deep_local)
        item.update({
            "deep_index": n_truth_surface + deep_local, "deep_local": deep_local,
            "truth_deep_fine_index": deep_local,
            "truth_deep_xyz_head_m": deep_head[deep_local].tolist(),
            "truth_deep_xyz_mri_m": deep_mri[deep_local].tolist(),
            "truth_deep_aseg_label": int(deep_labels[deep_local]),
            "deep_nearest_inverse_local": int(deep_nearest_inverse[deep_local]),
            "deep_nearest_inverse_global":
                n_inverse_surface + int(deep_nearest_inverse[deep_local]),
            "evaluation_deep_index":
                n_inverse_surface + int(deep_nearest_inverse[deep_local]),
            "deep_nearest_inverse_distance_mm":
                float(deep_inverse_distance[deep_local] * 1000.),
            "deep_alias_rho": float(deep_alias_rho[deep_local]),
        })


# %% 8. 跨三个正式 SNR 原样复制几何，并硬核验 39 例、角色、seed 与历史零重叠。
manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for item in configurations:
        number = len(manifest)
        scenario = item["scenario"]
        case = dict(item)
        case.update({
            "case_id": (
                f"erp-v3-fresh-dev-{item['configuration_number']:02d}-{scenario}"
                f"-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}"
            ),
            "case_number": number,
            "component_balance_revision": "v3_refined_truth_fresh_development",
            "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
            "snr_db": eeg_snr, "pair_index": pair_index,
            "panel": "erp_v3_fresh_development",
            "replica_seed_roots": dict(seed_roots), "replicate": 0,
            "seed": [seed_roots["fit"], pair_index, item["configuration_number"]],
            "check_seed": [seed_roots["check"], pair_index, item["configuration_number"]],
            "scenario_code": protocol.SCENARIO_CODES[scenario],
            "surface_centers": list(item["truth_surface_centers_refined"]),
            "location": item["configuration_number"],
            "truth_forward": "refined_grid_oct7_surface_5mm_thalamus",
            "truth_grid": "refined_oct7_5mm",
            "truth_forward_sha256": refined_file_sha256,
            "truth_forward_array_bundle_sha256": refined_bundle_sha256,
            "truth_coordinate_frame": "HEAD",
            "inverse_forward_fingerprint": shared_fingerprint,
            "channel_order_sha256": channel_order_sha256,
        })
        manifest.append(case)

if len(manifest) != 39 or len({case["case_id"] for case in manifest}) != 39 or \
        {case["case_id"] for case in manifest} & historical_case_ids:
    raise ValueError("v3 开发面板必须是 39 个全新 case_id")
if len({tuple(case["seed"]) for case in manifest}) != 39 or \
        len({tuple(case["check_seed"]) for case in manifest}) != 39 or \
        {tuple(case["seed"]) for case in manifest} & historical_seed_tuples or \
        {tuple(case["check_seed"]) for case in manifest} & historical_seed_tuples:
    raise ValueError("v3 fit/check 噪声身份必须逐例唯一且不复用历史 seed")

for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in manifest if case["pair_index"] == pair_index]
    if len(cell) != 13 or \
            {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cell} != {pair} or \
            Counter(case["deep_local"] is None for case in cell) != {True: 4, False: 9} or \
            Counter(case["h0_role"] for case in cell if case["deep_local"] is None) != \
            {"gate_calibration": 2, "gate_audit": 2} or \
            Counter(case["alias_tier"] for case in cell if case["deep_local"] is not None) != \
            {"low": 3, "mid": 3, "high": 3}:
        raise ValueError(f"SNR={pair} 不是预声明的 4 H0 + 9 H1")

for configuration_number in range(13):
    repeated = [case for case in manifest
                if case["configuration_number"] == configuration_number]
    truth_signatures = {
        (
            tuple(case["truth_surface_centers_refined"]), case["deep_local"],
            case["correlation"], case["deep_surface_ratio"],
            case["deep_surface_sensor_amplitude_ratio"], case["alias_tier"],
            case["h0_role"], case["truth_forward_sha256"],
        )
        for case in repeated
    }
    if len(repeated) != 3 or len(truth_signatures) != 1 or \
            {(case["eeg_snr_db"], case["meg_snr_db"]) for case in repeated} != set(snr_pairs):
        raise ValueError(f"configuration {configuration_number:02d} 没有原样跨三个 SNR")

selected_surface_head = surface_head[np.asarray(selected_surface)]
selected_surface_mri = surface_mri[np.asarray(selected_surface)]
selected_deep_head = deep_head[np.asarray(selected_deep)]
surface_history_min_mm = float(
    historical_surface_tree.query(selected_surface_head, k=1)[0].min() * 1000.)
deep_history_min_mm = float(
    cKDTree(historical_deep_xyz).query(selected_deep_head, k=1)[0].min() * 1000.)
if surface_history_min_mm < 10. - 1e-9 or deep_history_min_mm < 3. - 1e-9:
    raise ValueError("新 truth source position 与历史 coarse source 位置隔离不足")
if len(set(selected_surface)) != 15 or \
        sum(map(len, selected_surface_patches)) != len(selected_surface_support):
    raise ValueError("15 个表层 truth 中心或其 oct7 patch 出现复用/重叠")
pair_distances_mm = [
    float(np.linalg.norm(surface_head[first] - surface_head[second]) * 1000.)
    for first, second in surface_pairs
]
if min(pair_distances_mm) < 50.:
    raise ValueError("双表层中心距离小于 50 mm")


# %% 9. 保存 forward/channel/order/hash/坐标和完整 alias audit；只冻结开发面板，不作性能声明。
manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n").encode("utf-8")
alias_tier_audit = {}
for tier, indices in deep_tier_arrays.items():
    selected = deep_by_tier[tier]
    alias_tier_audit[tier] = {
        "candidate_count": int(indices.size),
        "rho_min": float(deep_alias_rho[indices].min()),
        "rho_max": float(deep_alias_rho[indices].max()),
        "rho_median": float(np.median(deep_alias_rho[indices])),
        "selected_truth_deep_local": list(map(int, selected)),
        "selected_rho": deep_alias_rho[selected].tolist(),
        "selected_aseg_labels": deep_labels[selected].astype(int).tolist(),
        "selected_nearest_inverse_local":
            deep_nearest_inverse[selected].astype(int).tolist(),
        "selected_nearest_inverse_distance_mm":
            (deep_inverse_distance[selected] * 1000.).tolist(),
    }

audit = {
    "protocol": "erp-v3-fresh-development-manifest",
    "schema_version": 1,
    "purpose": "fresh v3 development only; no formal or blind validation claim",
    "generator_path": script_path.relative_to(root).as_posix(),
    "generator_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
    "case_count": 39, "configuration_count": 13,
    "configuration_repetition_across_snr": 3,
    "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "per_snr": {"h0": 4, "h1": 9, "gate_calibration_h0": 2,
                "gate_audit_h0": 2},
    "replica_seed_roots": seed_roots,
    "geometry_seed_sha256": geometry_seed_sha256,
    "geometry_seed_uint64": geometry_seed_uint64,
    "numpy_version_for_deterministic_geometry": np.__version__,
    "history": {
        "manifest_count": len(history), "case_count": len(history_cases),
        "manifest_sha256": history_sha256,
        "case_id_overlap": 0, "seed_tuple_overlap": 0,
        "seed_root_overlap": 0,
        "surface_center_count": len(historical_surface_centers),
        "deep_local_count": len(historical_deep_local),
    },
    "truth_forward": {
        "path": refined_cache.relative_to(root).as_posix(),
        "file_sha256": refined_file_sha256,
        "array_sha256": refined_array_sha256,
        "array_bundle_sha256": refined_bundle_sha256,
        "builder_path": refined_helper_path.relative_to(root).as_posix(),
        "builder_sha256": hashlib.sha256(refined_helper_path.read_bytes()).hexdigest(),
        "coordinate_frame": coordinate_frame,
        "head_to_mri_transform": head_to_mri.tolist(),
        "mri_to_head_transform": mri_to_head.tolist(),
        "surface_head_to_mri_max_error_m": surface_transform_error,
        "deep_head_to_mri_max_error_m": deep_transform_error,
        "n_surface": n_truth_surface, "n_deep": n_truth_deep,
        "surface_spacing_mm": float(np.asarray(refined["surface_spacing_mm"]).ravel()[0]),
        "deep_spacing_mm": float(np.asarray(refined["deep_spacing_mm"]).ravel()[0]),
        "surface_vertex_numbers_sha256": refined_array_sha256["surface_vertex_numbers"],
        "surface_hemi_sha256": refined_array_sha256["surface_hemi"],
        "deep_aseg_labels_sha256": refined_array_sha256["deep_aseg_labels"],
        "surface_xyz_head_sha256": refined_array_sha256["surface_xyz_head"],
        "surface_xyz_mri_sha256": refined_array_sha256["surface_xyz_mri"],
        "deep_xyz_head_sha256": refined_array_sha256["deep_xyz_head"],
        "deep_xyz_mri_sha256": refined_array_sha256["deep_xyz_mri"],
    },
    "channel_order": {
        "sha256": channel_order_sha256,
        "eeg": eeg_names, "meg": meg_names,
        "eeg_count": len(eeg_names), "meg_count": len(meg_names),
        "exact_match_to_inverse": True,
    },
    "inverse_forward": {
        "fingerprint": shared_fingerprint,
        "n_surface": n_inverse_surface, "n_deep": n_inverse_deep,
    },
    "surface_geometry": {
        "selected_truth_local": list(map(int, selected_surface)),
        "selected_hemi": surface_hemi[selected_surface].astype(int).tolist(),
        "selected_vertex_numbers":
            surface_vertex_numbers[selected_surface].astype(int).tolist(),
        "selected_xyz_head_m": surface_head[selected_surface].tolist(),
        "selected_xyz_mri_m": selected_surface_mri.tolist(),
        "nearest_inverse_local":
            surface_nearest_inverse[selected_surface].astype(int).tolist(),
        "nearest_inverse_distance_mm":
            (surface_inverse_distance[selected_surface] * 1000.).tolist(),
        "minimum_distance_to_any_historical_center_mm": surface_history_min_mm,
        "patch_count": 15, "patch_overlap_vertex_count": 0,
        "minimum_selected_center_distance_mm": float(min(
            np.linalg.norm(selected_surface_head[i] - selected_surface_head[j]) * 1000.
            for i in range(15) for j in range(i + 1, 15)
        )),
        "double_source_pair_distance_mm": pair_distances_mm,
        "truth_to_inverse_required_mm": [2., 5.],
        "historical_center_separation_required_mm": 10.,
    },
    "deep_alias": {
        "formula": (
            "rho_j=max_i |<normalize([W_E G_Esurf_i/sqrt(r_E); "
            "W_M G_Msurf_i/sqrt(r_M)]), normalize([W_E G_Edeep_j/sqrt(r_E); "
            "W_M G_Mdeep_j/sqrt(r_M)])>|"
        ),
        "uses_data_or_truth_metrics": False,
        "uses_forward_and_noise_covariance_only": True,
        "modality_rank_normalization": whitening_ranks,
        "candidate_distance_to_inverse_deep_mm": [3., 8.],
        "candidate_count": int(deep_eligible.size),
        "partition": "stable sort by rho then numpy.array_split into three count-balanced tiers",
        "tiers": alias_tier_audit,
        "selected_truth_deep_local": list(map(int, selected_deep)),
        "selected_xyz_head_m": selected_deep_head.tolist(),
        "selected_xyz_mri_m": deep_mri[selected_deep].tolist(),
        "selected_nearest_inverse_local": list(map(int, matching_columns)),
        "nearest_inverse_all_unique": True,
        "selected_label_counts": {
            str(label): int(sum(deep_labels[selected_deep] == label)) for label in (10, 49)
        },
        "minimum_distance_to_historical_coarse_deep_mm": deep_history_min_mm,
    },
    "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
    "simulation_or_inverse_run_by_generator": False,
    "formal_performance_claim_allowed": False,
}

report = f"""# v3 全新开发面板

状态：只冻结全新开发病例，不是正式校准或 blind validation，也不含性能结果。

- 13 套固定几何原样跨三个 `(EEG, MEG) SNR`：`(-10,-10)`、`(-10,+20)`、`(+20,-10)`，共 39 例；每档固定 4 个 H0 与 9 个 H1。
- H0 为纯表层单源 2 套、双源 2 套；单/双各 1 套用于 gate calibration，另各 1 套只用于 gate audit。
- H1 为纯深层、深层+单表层、深层+双表层各 3 套；每种情形分别覆盖 low/mid/high forward-alias。
- truth forward 固定为 oct7 表层 + 5 mm 双侧丘脑，缓存 SHA256 为 `{refined_file_sha256}`；inverse 仍是 coarse oct6/16点深网格，指纹为 `{shared_fingerprint}`。
- 15 个表层 patch 两两不重叠，中心距所有历史中心至少 `{surface_history_min_mm:.3f}` mm；5 个双源对均至少 50 mm。truth 到最近 inverse 表层点固定为 2–5 mm。
- 9 个深层 truth 均距最近 inverse 深点 3–8 mm，truth 点和最近 inverse 点都不复用；左右丘脑为 `{Counter(deep_labels[selected_deep].tolist())}`。
- alias 只由 forward 与固定 noise covariance whitening 计算；候选按 rho 排序后三等分，再在 tier、左右标签和最近 coarse 点唯一约束下匹配。
- fit/check roots 固定为 `{seed_roots['fit']}/{seed_roots['check']}`，与全部 `{len(history)}` 个历史病例清单的 ID、seed 和 source positions 均无复用。
"""


# %% 10. preflight 只报告；正式生成首次独占写入，--check 逐字节核对。
payloads = {
    output_paths["manifest"]: manifest_payload,
    output_paths["audit"]: (json.dumps(audit, ensure_ascii=False, indent=2,
                                        sort_keys=True) + "\n").encode("utf-8"),
    output_paths["report"]: report.encode("utf-8"),
}
if args.preflight:
    print(json.dumps({
        "preflight": "PASS", "manifest_generation_allowed": True,
        "case_count": len(manifest), "configuration_count": len(configurations),
        "per_snr_h0_h1": [4, 9], "seed_roots": seed_roots,
        "history_manifest_count": len(history),
        "refined_forward_sha256": refined_file_sha256,
        "refined_array_bundle_sha256": refined_bundle_sha256,
        "channel_order_sha256": channel_order_sha256,
        "surface_history_min_mm": surface_history_min_mm,
        "deep_history_min_mm": deep_history_min_mm,
        "selected_surface_count": len(selected_surface),
        "selected_deep_count": len(selected_deep),
        "alias_tier_counts": {tier: int(indices.size)
                              for tier, indices in deep_tier_arrays.items()},
        "outputs_would_be": {str(path): hashlib.sha256(payload).hexdigest()
                             for path, payload in payloads.items()},
    }, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0)

if args.check:
    for path, payload in payloads.items():
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = Path(str(path) + ".sha256")
        expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
        if not path.is_file() or path.read_bytes() != payload or \
                not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
            raise ValueError(f"冻结的 v3 全新开发面板已变化：{path}")
else:
    occupied = [str(path) for path in payloads
                if path.exists() or Path(str(path) + ".sha256").exists()]
    if occupied:
        raise FileExistsError(f"拒绝覆盖已有 v3 全新开发面板：{occupied}")
    for path, payload in payloads.items():
        with path.open("xb") as stream:
            stream.write(payload)
        digest = hashlib.sha256(payload).hexdigest()
        sidecar = Path(str(path) + ".sha256")
        with sidecar.open("xb") as stream:
            stream.write(f"{digest}  {path.name}\n".encode("ascii"))

print(json.dumps({
    "checked": bool(args.check), "case_count": len(manifest),
    "configuration_count": len(configurations),
    "outputs": {str(path): hashlib.sha256(payload).hexdigest()
                for path, payload in payloads.items()},
}, ensure_ascii=False, indent=2), flush=True)
