"""# %% 冻结条件门控 pilot：60 calibration H0 + 30 audit H0 + 27 H1。"""

# %% 1. 只生成病例身份；forward、MRF=.5、20+20观测和既有逆解不改变。
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

if not __debug__:
    raise RuntimeError("条件门控面板自检禁止 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
source_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source_root))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-root", type=Path,
                    default=Path(r"D:\博士\工作＆汇报\源定位\新建文件夹"))
parser.add_argument("--output", type=Path, default=None, help="精确的manifest JSON路径")
parser.add_argument("--preflight", action="store_true", help="只核验几何与身份，不写文件")
args = parser.parse_args()
input_root = args.input_root.resolve()
protocol_dir = input_root / "results/erp_whole_head/adaptive_v6/protocol"
output = (protocol_dir / "conditional_gate_pilot_manifest.json"
          if args.output is None else args.output.resolve())
sidecar = Path(str(output) + ".sha256")
if output.suffix.lower() != ".json":
    raise ValueError("--output必须为精确的.json文件路径")
if not args.preflight and (output.exists() or sidecar.exists()):
    raise FileExistsError(f"拒绝覆盖冻结面板：{output}")
script_path = Path(__file__).resolve()
script_text = script_path.read_text(encoding="utf-8")
assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for node in ast.walk(ast.parse(script_text)))
assert script_text.count("# %%") >= 8
seed_roots = {"geometry": 2026101000, "fit": 2026101001, "check": 2026101002}
snr_pairs = ((-10, -10), (-10, 20), (20, -10))

# %% 2. 读取全部历史manifest，保留输入hash；历史fine与coarse索引分别解释。
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from benchmark import protocol
import run_erp_whole_head_matrix as simulation
from refined_grid import load_refined_forward

history = {}
history_cases = []
for path in sorted(protocol_dir.glob("*.json")):
    if path.resolve() == output:
        continue
    payload = path.read_bytes()
    value = json.loads(payload)
    records = (value.get("cases", []) if isinstance(value, dict) and
               value.get("protocol") == "erp-conditional-gate-pilot-manifest-v1" else value)
    if not isinstance(records, list) or not records or not isinstance(records[0], dict) \
            or "case_id" not in records[0]:
        continue
    if any(not isinstance(case, dict) or "case_id" not in case for case in records):
        raise ValueError(f"历史病例清单结构不一致：{path}")
    history[path.name] = {"sha256": hashlib.sha256(payload).hexdigest(),
                          "case_count": len(records)}
    history_cases.extend(records)
if not history_cases:
    raise FileNotFoundError("没有历史病例清单，无法审计新几何")
historical_ids = {str(case["case_id"]) for case in history_cases}
historical_seeds = {tuple(case[key]) for case in history_cases
                    for key in ("seed", "check_seed") if isinstance(case.get(key), list)}
historical_roots = {int(value) for case in history_cases
                    for value in case.get("replica_seed_roots", {}).values()}
assert not (set(seed_roots.values()) & historical_roots)
assert not any(seed and seed[0] in seed_roots.values() for seed in historical_seeds)

# %% 3. 只读取已存在的oct7/5mm缓存；核验通道、坐标、图和每个数组身份。
refined_path = input_root / "results/refined_forward_oct7_5mm.npz"
if not refined_path.is_file():
    raise FileNotFoundError("fine forward缓存缺失；本脚本不构建forward")
shared = protocol.load_shared(input_root / "corrected_v2/generated", simulation.DEFAULT_SAMPLE_PATH)
refined = load_refined_forward(refined_path)
required = {
    "gain_eeg", "gain_meg", "vertices", "surface_xyz_head", "surface_xyz_mri",
    "deep_xyz_head", "deep_xyz_mri", "surface_vertex_numbers", "surface_hemi",
    "deep_aseg_labels", "eeg_ch_names", "meg_ch_names", "head_mri_t", "mri_head_t",
    "coordinate_frame", "surface_edges", "n_surf", "n_deep", "surface_spacing_mm",
    "deep_spacing_mm",
}
if not required.issubset(refined):
    raise ValueError(f"fine forward schema缺字段：{sorted(required - refined.keys())}")
ns = int(shared["n_surf"])
nd = int(shared["n_deep"])
fs = int(np.asarray(refined["n_surf"]).ravel()[0])
fd = int(np.asarray(refined["n_deep"]).ravel()[0])
surface = np.asarray(refined["surface_xyz_head"], float)
deep = np.asarray(refined["deep_xyz_head"], float)
surface_mri = np.asarray(refined["surface_xyz_mri"], float)
deep_mri = np.asarray(refined["deep_xyz_mri"], float)
hemi = np.asarray(refined["surface_hemi"], int).ravel()
labels = np.asarray(refined["deep_aseg_labels"], int).ravel()
edges = np.asarray(refined["surface_edges"], int)
inverse_surface = np.asarray(shared["vertices"], float)[:ns]
inverse_deep = np.asarray(shared["deep_rr"], float)
eeg_names = list(map(str, np.asarray(refined["eeg_ch_names"]).ravel()))
meg_names = list(map(str, np.asarray(refined["meg_ch_names"]).ravel()))
assert surface.shape == surface_mri.shape == (fs, 3)
assert deep.shape == deep_mri.shape == (fd, 3)
assert hemi.shape == (fs,) and labels.shape == (fd,)
assert set(hemi) == {0, 1} and set(labels) == {10, 49}
assert list(map(str, np.asarray(refined["coordinate_frame"]).ravel())) == ["HEAD"]
assert np.allclose(refined["vertices"], np.vstack([surface, deep]), rtol=0, atol=1e-12)
assert eeg_names == shared["info_eeg"]["ch_names"]
assert meg_names == shared["info_meg"]["ch_names"]
for modality, names in (("eeg", eeg_names), ("meg", meg_names)):
    assert np.asarray(refined["gain_" + modality]).shape == (len(names), fs + fd)
assert edges.ndim == 2 and edges.shape[1] == 2 and np.all(edges[:, 0] < edges[:, 1])
assert np.all((edges >= 0) & (edges < fs))
assert np.allclose(refined["head_mri_t"] @ refined["mri_head_t"], np.eye(4), rtol=0, atol=1e-12)
for xyz, xyz_mri in ((surface, surface_mri), (deep, deep_mri)):
    converted = (refined["head_mri_t"] @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    assert np.max(np.linalg.norm(converted - xyz_mri, axis=1)) < 1e-9
array_sha256 = {}
for name in sorted(required):
    values = np.ascontiguousarray(np.asarray(refined[name]))
    if values.dtype.kind in "fci":
        assert np.isfinite(values).all(), name
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(np.asarray(values.shape, np.int64).tobytes())
    digest.update(values.tobytes())
    array_sha256[name] = digest.hexdigest()
refined_sha256 = hashlib.sha256(refined_path.read_bytes()).hexdigest()
bundle_sha256 = hashlib.sha256("\n".join(
    f"{name}:{array_sha256[name]}" for name in sorted(array_sha256)).encode("ascii")).hexdigest()
channel_sha256 = hashlib.sha256(
    ("EEG\0" + "\0".join(eeg_names) + "\0MEG\0" + "\0".join(meg_names)).encode()).hexdigest()
inverse_fingerprint = simulation._shared_fingerprint(shared)
graph = sparse.coo_matrix((np.ones(2 * len(edges), dtype=np.uint8),
    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(fs, fs)).tocsr()
fine_surface_shared = {"n_surf": fs, "adjacency": graph, "vertices": surface}

# %% 4. 按历史truth_grid恢复真实HEAD坐标；绝不把fine索引当coarse索引。
old_coarse_surface = set()
old_fine_surface = set()
old_coarse_deep = set()
old_fine_deep = set()
old_fine_patch = set()
for case in history_cases:
    grid = case.get("truth_grid", "inverse_grid")
    if grid == "refined_oct7_5mm":
        if case.get("truth_forward_sha256") != refined_sha256:
            raise ValueError("历史fine病例引用了不同forward，不能猜测索引身份")
        centers = list(map(int, case["truth_surface_centers_refined"]))
        assert all(0 <= center < fs for center in centers)
        assert np.allclose(surface[centers], np.asarray(
            case["truth_surface_center_xyz_head_m"], float).reshape(-1, 3), rtol=0, atol=1e-12)
        old_fine_surface.update(centers)
        for center, indices in zip(centers, case["truth_surface_patch_indices_refined"]):
            assert np.array_equal(protocol._surface_patch(fine_surface_shared, center)[0], indices)
            old_fine_patch.update(map(int, indices))
        local = case.get("truth_deep_fine_index")
        if local is not None:
            local = int(local)
            assert 0 <= local < fd
            assert np.allclose(deep[local], case["truth_deep_xyz_head_m"], rtol=0, atol=1e-12)
            old_fine_deep.add(local)
    elif grid == "inverse_grid":
        centers = list(map(int, case.get("surface_centers", [])))
        assert all(0 <= center < ns for center in centers)
        old_coarse_surface.update(centers)
        local = case.get("deep_local")
        if local is not None:
            assert 0 <= int(local) < nd
            old_coarse_deep.add(int(local))
    else:
        raise ValueError(f"不支持的历史truth_grid：{grid}")
historical_surface = np.vstack([inverse_surface[sorted(old_coarse_surface)],
                                surface[sorted(old_fine_surface)]])
historical_deep = np.vstack([inverse_deep[sorted(old_coarse_deep)], deep[sorted(old_fine_deep)]])
if not len(historical_surface) or not len(historical_deep):
    raise ValueError("历史空间防火墙缺少表层或深层坐标")
surface_history_tree = cKDTree(historical_surface)
surface_history_distance = surface_history_tree.query(surface)[0]
deep_history_distance = cKDTree(historical_deep).query(deep)[0]
surface_inverse_distance, surface_inverse_local = cKDTree(inverse_surface).query(surface)
deep_inverse_distance, deep_inverse_local = cKDTree(inverse_deep).query(deep)

# %% 5. 前向和噪声模型定义alias档；9个全新fine深点、最近coarse位置本轮唯一。
whiteners = {}
for modality in ("eeg", "meg"):
    covariance = np.asarray(shared["noise_cov_" + modality], float)
    eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    keep = eigenvalues > max(float(eigenvalues.max()) * 1e-12, np.finfo(float).tiny)
    assert keep.sum() >= 2
    whiteners[modality] = (vectors[:, keep] / np.sqrt(eigenvalues[keep])).T / np.sqrt(keep.sum())
inverse_design = np.vstack([whiteners[m] @ shared["gain_" + m][:, :ns] for m in ("eeg", "meg")])
deep_design = np.vstack([whiteners[m] @ refined["gain_" + m][:, fs:] for m in ("eeg", "meg")])
assert np.all(np.linalg.norm(inverse_design, axis=0) > 0)
assert np.all(np.linalg.norm(deep_design, axis=0) > 0)
inverse_design /= np.linalg.norm(inverse_design, axis=0)
deep_design /= np.linalg.norm(deep_design, axis=0)
alias = np.zeros(fd)
for start in range(0, fd, 64):
    alias[start:start + 64] = np.max(np.abs(inverse_design.T @ deep_design[:, start:start + 64]), axis=0)
alias = np.clip(alias, 0, 1)
eligible_deep = np.flatnonzero((deep_inverse_distance >= .003) & (deep_inverse_distance <= .008)
    & (deep_history_distance >= .003) & ~np.isin(np.arange(fd), list(old_fine_deep)))
assert len(eligible_deep) >= 9
tier_arrays = dict(zip(("low", "mid", "high"),
    np.array_split(eligible_deep[np.argsort(alias[eligible_deep], kind="stable")], 3)))
slots = [(tier, number, label) for tier, sides in
         (("low", (10, 49, 10)), ("mid", (49, 10, 49)), ("high", (10, 49, 10)))
         for number, label in enumerate(sides)]
cost = np.full((9, nd), np.inf)
matching = np.full((9, nd), -1, int)
for row, (tier, number, label) in enumerate(slots):
    indices = tier_arrays[tier]
    for coarse in range(nd):
        pool = indices[(labels[indices] == label) & (deep_inverse_local[indices] == coarse)]
        if pool.size:
            values = np.abs(alias[pool] - np.median(alias[indices])) \
                + 1e-4 * np.abs(deep_inverse_distance[pool] * 1000 - 5.5)
            best = int(np.argmin(values))
            cost[row, coarse] = values[best]
            matching[row, coarse] = pool[best]
matched_rows, matched_columns = linear_sum_assignment(cost)
assert np.array_equal(matched_rows, np.arange(9)) and np.isfinite(cost[matched_rows, matched_columns]).all()
selected_deep = matching[matched_rows, matched_columns].astype(int).tolist()
assert len(set(selected_deep)) == len(set(matched_columns)) == 9
deep_by_tier = {tier: [selected_deep[row] for row, slot in enumerate(slots) if slot[0] == tier]
                for tier in ("low", "mid", "high")}

# %% 6. 全头分散选54个新fine表层中心，patch无复用；cal/audit/H1角色随后冻结。
rng = np.random.default_rng(seed_roots["geometry"])
priority = rng.random(fs)
eligible_surface = np.flatnonzero((surface_inverse_distance >= .002) & (surface_inverse_distance <= .005)
    & (surface_history_distance >= .010)
    & ~np.isin(surface_inverse_local, list(old_coarse_surface))
    & ~np.isin(np.arange(fs), list(old_fine_surface)))
selected_surface = []
selected_patches = []
selected_support = set()
for position in range(54):
    candidates = eligible_surface[hemi[eligible_surface] == position % 2]
    minimum = (np.full(len(candidates), np.inf) if not selected_surface else
               cKDTree(surface[selected_surface]).query(surface[candidates])[0])
    order = np.lexsort((priority[candidates], -minimum))
    accepted = None
    for index in order:
        if selected_surface and minimum[index] < .025:
            break
        center = int(candidates[index])
        patch = protocol._surface_patch(fine_surface_shared, center)[0]
        if selected_support.intersection(map(int, patch)) or old_fine_patch.intersection(map(int, patch)):
            continue
        if surface_history_tree.query(surface[patch])[0].min() < .005:
            continue
        accepted = center
        selected_patches.append(patch.astype(int).tolist())
        selected_support.update(map(int, patch))
        break
    if accepted is None:
        raise RuntimeError("无法满足54个新中心、25mm间隔和patch防火墙；拒绝放宽约束")
    selected_surface.append(accepted)
remaining = set(selected_surface)
pairs = []
for number in range(18):
    choices = [(float(np.linalg.norm(surface[left] - surface[right])), left, right)
               for left in sorted(remaining) if hemi[left] == 0
               for right in sorted(remaining) if hemi[right] == 1]
    distance, left, right = max(choices)
    assert distance >= .050
    pairs.append([left, right])
    remaining -= {left, right}
pairs = [pairs[index] for index in rng.permutation(18)]
singles_by_hemi = [rng.permutation(sorted(center for center in remaining if hemi[center] == h)).tolist()
                  for h in (0, 1)]
assert tuple(map(len, singles_by_hemi)) == (9, 9)
singles = [center for pair in zip(*singles_by_hemi) for center in pair]
parcels = protocol._parcels(shared)
coarse_parcels = {int(vertex): name for name, indices in parcels.items() for vertex in indices}
parcel_counts = Counter(coarse_parcels.get(int(surface_inverse_local[center]), "unassigned")
                        for center in selected_surface)

# %% 7. 20 calibration H0、10 audit H0、9 H1几何，跨SNR不改变真值。
configurations = []
for number in range(30):
    calibration = number < 20
    single = number < 10 or 20 <= number < 25
    centers = ([singles[number if calibration else number - 10]] if single else
               pairs[number - 10 if calibration else number - 15])
    configurations.append({"configuration_number": number, "scenario": "surface_only",
        "surface_centers": centers, "deep_truth_local": None, "alias_tier": None,
        "h0_role": "gate_calibration" if calibration else "gate_audit",
        "correlation": 0. if single else (0., .5, .9)[number % 3],
        "deep_surface_ratio": None, "deep_surface_sensor_amplitude_ratio": None,
        "surface_component_sensor_balance": not single})
for category, scenario in enumerate(("deep_only", "deep_plus_surface", "deep_plus_two_surface")):
    for tier_number, tier in enumerate(("low", "mid", "high")):
        centers = ([] if category == 0 else [singles[15 + tier_number]] if category == 1
                   else pairs[15 + tier_number])
        correlation, ratio = ((0., None) if category == 0 else
            ((0., .5), (.5, 1.), (.9, .5))[tier_number] if category == 1 else
            ((.9, 1.), (0., .5), (.5, 1.))[tier_number])
        configurations.append({"configuration_number": len(configurations), "scenario": scenario,
            "surface_centers": centers, "deep_truth_local": deep_by_tier[tier][category],
            "alias_tier": tier, "h0_role": None, "correlation": correlation,
            "deep_surface_ratio": ratio, "deep_surface_sensor_amplitude_ratio": ratio,
            "surface_component_sensor_balance": category == 2})
for item in configurations:
    number = item["configuration_number"]
    centers = item["surface_centers"]
    item["configuration_id"] = f"erp-conditional-pilot-geometry-{number:02d}"
    item["configuration_kind"] = {("surface_only", 1): "single_surface_h0",
        ("surface_only", 2): "two_surface_h0", ("deep_only", 0): "deep_only_h1",
        ("deep_plus_surface", 1): "deep_plus_single_surface_h1",
        ("deep_plus_two_surface", 2): "deep_plus_two_surface_h1"}[(item["scenario"], len(centers))]
    patch_parts = [protocol._surface_patch(fine_surface_shared, center) for center in centers]
    item.update({"truth_surface_centers_refined": centers,
        "truth_surface_center_xyz_head_m": surface[centers].tolist(),
        "truth_surface_center_xyz_mri_m": surface_mri[centers].tolist(),
        "truth_surface_nearest_inverse": surface_inverse_local[centers].astype(int).tolist(),
        "truth_surface_nearest_inverse_distance_mm": (surface_inverse_distance[centers] * 1000).tolist(),
        "truth_surface_patch_indices_refined": [indices.tolist() for indices, weights in patch_parts],
        "truth_surface_patch_weights": [weights.tolist() for indices, weights in patch_parts],
        "truth_surface_patch_nearest_inverse": [surface_inverse_local[indices].astype(int).tolist()
                                               for indices, weights in patch_parts],
        "truth_surface_patch_nearest_inverse_distance_mm": [(surface_inverse_distance[indices] * 1000).tolist()
                                                           for indices, weights in patch_parts]})
    item["evaluation_surface_projection"] = {"rule": "nearest coarse surface vertex in HEAD coordinates",
        "inverse_indices_by_patch": item["truth_surface_patch_nearest_inverse"]}
    local = item["deep_truth_local"]
    item.update({"deep_local": local, "deep_index": None if local is None else fs + local,
        "truth_deep_fine_index": local,
        "truth_deep_xyz_head_m": None if local is None else deep[local].tolist(),
        "truth_deep_xyz_mri_m": None if local is None else deep_mri[local].tolist(),
        "truth_deep_aseg_label": None if local is None else int(labels[local]),
        "deep_nearest_inverse_local": None if local is None else int(deep_inverse_local[local]),
        "deep_nearest_inverse_global": None if local is None else ns + int(deep_inverse_local[local]),
        "evaluation_deep_index": None if local is None else ns + int(deep_inverse_local[local]),
        "deep_nearest_inverse_distance_mm": None if local is None else float(deep_inverse_distance[local] * 1000),
        "deep_alias_rho": None if local is None else float(alias[local])})
cases = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for item in configurations:
        number = item["configuration_number"]
        cases.append({**item, "case_number": len(cases),
            "case_id": f"erp-conditional-pilot-{number:02d}-{item['scenario']}-eeg{eeg_snr:+d}-meg{meg_snr:+d}",
            "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr, "snr_db": eeg_snr,
            "pair_index": pair_index, "scenario_code": protocol.SCENARIO_CODES[item["scenario"]],
            "location": number, "replicate": 0, "panel": "conditional_gate_pilot",
            "replica_seed_roots": {key: seed_roots[key] for key in ("fit", "check")},
            "seed": [seed_roots["fit"], pair_index, number],
            "check_seed": [seed_roots["check"], pair_index, number],
            "component_balance_revision": "v3_refined_truth_conditional_pilot",
            "truth_grid": "refined_oct7_5mm", "truth_coordinate_frame": "HEAD",
            "truth_forward": "refined_grid_oct7_surface_5mm_thalamus",
            "truth_forward_sha256": refined_sha256,
            "truth_forward_array_bundle_sha256": bundle_sha256,
            "inverse_forward_fingerprint": inverse_fingerprint,
            "channel_order_sha256": channel_sha256})

# %% 8. 真值几何、角色、噪声身份和历史隔离assert；本脚本无仿真与逆解调用。
assert [item["configuration_number"] for item in configurations] == list(range(39))
assert [case["case_number"] for case in cases] == list(range(117))
assert len({case["case_id"] for case in cases}) == 117
assert not ({case["case_id"] for case in cases} & historical_ids)
assert not ({tuple(case[key]) for case in cases for key in ("seed", "check_seed")} & historical_seeds)
assert Counter(case["h0_role"] for case in cases) == {"gate_calibration": 60, "gate_audit": 30, None: 27}
assert len(set(selected_surface)) == 54 and len(set(selected_deep)) == 9
assert len(selected_support) == sum(map(len, selected_patches))
assert not (selected_support & old_fine_patch) and not (set(selected_deep) & old_fine_deep)
assert cKDTree(surface[selected_surface]).query(surface[selected_surface], k=2)[0][:, 1].min() >= .025
assert surface_history_distance[selected_surface].min() >= .010
assert deep_history_distance[selected_deep].min() >= .003
assert Counter(hemi[selected_surface].tolist()) == {0: 27, 1: 27}
assert len(set(center for item in configurations for center in item["surface_centers"])) == 54
for pair_index, pair in enumerate(snr_pairs):
    cell = [case for case in cases if case["pair_index"] == pair_index]
    assert len(cell) == 39 and {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cell} == {pair}
    assert Counter(case["h0_role"] for case in cell) == {"gate_calibration": 20, "gate_audit": 10, None: 9}
    assert Counter((case["h0_role"], len(case["surface_centers"])) for case in cell if case["h0_role"]) == {
        ("gate_calibration", 1): 10, ("gate_calibration", 2): 10,
        ("gate_audit", 1): 5, ("gate_audit", 2): 5}
for number in range(39):
    repeated = [case for case in cases if case["configuration_number"] == number]
    assert len(repeated) == 3
    assert len({(tuple(case["surface_centers"]), case["deep_index"], case["correlation"],
                 case["deep_surface_ratio"], case["h0_role"]) for case in repeated}) == 1

# %% 9. 保存完整自包含pilot manifest及SHA旁车；条件只能来自观测，HH不足不外推。
code_paths = ("benchmark/protocol.py", "benchmark/erp_protocol.py", "benchmark/erp_replicates.py",
              "benchmark/erp_trial_covariance.py", "refined_grid.py", "run_erp_whole_head_matrix.py")
manifest = {
    "schema_version": 1, "protocol": "erp-conditional-gate-pilot-manifest-v1",
    "pilot": True, "development_only": True, "formal_or_blind_validation": False,
    "purpose": "new fine source geometry conditional-threshold pilot; fixed inverse grid and algorithm",
    "case_count": 117, "configuration_count": 39, "cases": cases,
    "seed_roots": seed_roots, "snr_pairs_eeg_meg_db": [list(pair) for pair in snr_pairs],
    "per_snr": {"gate_calibration_h0": 20, "gate_audit_h0": 10, "h1": 9},
    "fixed_experiment": {"mrf_strength": .5, "n_trials_per_half": 20,
        "noise_model": "independent Gaussian half-mean and training-baseline contrasts",
        "algorithm_change_allowed": False, "gate_mode": "lowrank_studentized"},
    "threshold_rule": {"alpha": .05, "minimum_calibration_per_bin": 19,
        "quality_threshold_linear": 10., "quality_threshold_db": 10.,
        "observable_bins": ["LL", "LH", "HL", "HH"], "bin_order": ["EEG", "MEG"],
        "quality_rule": "q_m=max(<R_A_m,R_B_m>_F/sqrt(N_A_m*N_B_m),0); R=baseline-centered Y Phi.T; N=sum baseline sample variances * (K+baseline-mean mode correction); H iff q_m>=10",
        "p_value_rule": "(1 + count(T_cal >= T)) / (n_cal + 1)",
        "decision_rule": "p <= alpha and T > 0", "positive_gain_required": True,
        "unsupported_bin_rule": "p=1; deep_present=false; unsupported=true; abstention, not a supported negative conclusion; no extrapolation",
        "HH_rule": "unsupported unless its observable calibration count is at least 19",
        "calibration_preflight_rule": "read only 60 calibration H0 observations; count bins before any inverse; do not inspect 57 audit/H1 observations",
        "true_snr_or_source_labels_allowed_for_condition": False,
        "frozen_score_rule": "max_d min(studentized_A_to_B[d],studentized_B_to_A[d]) over jointly identifiable deep points"},
    "inputs": {
        "inverse_root": str(input_root / "corrected_v2/generated"),
        "sample_path": str(simulation.DEFAULT_SAMPLE_PATH.resolve()),
        "inverse_forward": {"fingerprint": inverse_fingerprint, "n_surface": ns, "n_deep": nd},
        "refined_forward": {"path": str(refined_path), "file_sha256": refined_sha256,
            "array_sha256": array_sha256, "array_bundle_sha256": bundle_sha256,
            "n_surface": fs, "n_deep": fd, "coordinate_frame": "HEAD",
            "builder_sha256": hashlib.sha256((source_root / "refined_grid.py").read_bytes()).hexdigest()},
        "channel_order": {"eeg": eeg_names, "meg": meg_names, "sha256": channel_sha256},
        "history_manifest_sha256": {name: record["sha256"] for name, record in history.items()},
        "history_protocol_dir": str(protocol_dir),
        "history_manifest_records": history,
        "code_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in code_paths}},
    "geometry_audit": {
        "history_case_count": len(history_cases),
        "historical_surface_coarse_count": len(old_coarse_surface),
        "historical_surface_fine_count": len(old_fine_surface),
        "historical_deep_coarse_count": len(old_coarse_deep),
        "historical_deep_fine_count": len(old_fine_deep),
        "historical_surface_coarse_indices": sorted(old_coarse_surface),
        "historical_surface_fine_indices": sorted(old_fine_surface),
        "historical_deep_coarse_indices": sorted(old_coarse_deep),
        "historical_deep_fine_indices": sorted(old_fine_deep),
        "historical_fine_surface_patch_indices": sorted(old_fine_patch),
        "selected_surface_centers": selected_surface, "selected_surface_patch_indices": selected_patches,
        "selected_surface_minimum_distance_mm": float(cKDTree(surface[selected_surface]).query(
            surface[selected_surface], k=2)[0][:, 1].min() * 1000),
        "selected_surface_minimum_historical_center_distance_mm": float(
            surface_history_distance[selected_surface].min() * 1000),
        "selected_surface_minimum_patch_to_historical_center_distance_mm": float(
            surface_history_tree.query(surface[sorted(selected_support)])[0].min() * 1000),
        "fine_patch_overlap_with_history_count": len(selected_support & old_fine_patch),
        "new_patch_overlap_count": sum(map(len, selected_patches)) - len(selected_support),
        "selected_deep_local": selected_deep, "selected_deep_nearest_inverse_local": matched_columns.tolist(),
        "selected_deep_minimum_history_distance_mm": float(deep_history_distance[selected_deep].min() * 1000),
        "eligible_surface_count": len(eligible_surface), "eligible_deep_count": len(eligible_deep),
        "deep_alias_rule": "max absolute noise-whitened rank-normalized cortical/deep lead-field correlation; no observations or outcome scores",
        "deep_alias_tier_candidate_indices": {tier: indices.tolist() for tier, indices in tier_arrays.items()},
        "deep_alias_values": alias.tolist(), "coverage_aparc_center_counts": dict(sorted(parcel_counts.items())),
        "coverage_aparc_unrepresented": sorted(set(parcels) - set(parcel_counts)),
        "coverage_note": "distributed whole-head sampling, not exhaustive cortical coverage",
        "coarse_deep_history_note": "all 16 coarse deep points have historical use; only fine truth locations and noise identities are new",
        "history_coordinate_rule": "inverse_grid indices map to coarse HEAD coordinates; refined_oct7_5mm indices map to verified fine HEAD coordinates",
        "surface_history_rule": "centers >=10mm from historical centers; full patches >=5mm from historical centers; fine historical patches and new patches disjoint",
        "role_geometry_disjoint": True, "history_case_id_overlap": 0, "history_seed_overlap": 0},
    "generator": {"path": str(script_path), "sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "simulation_run": False, "inverse_run": False},
}
payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
digest = hashlib.sha256(payload).hexdigest()
summary = {"case_count": 117, "roles": {"calibration_h0": 60, "audit_h0": 30, "h1": 27},
    "surface_centers": 54, "deep_points": 9, "manifest_sha256": digest,
    "preflight": args.preflight, "output": str(output), "pilot": True,
    "represented_aparc_parcels": len(set(parcel_counts) - {"unassigned"}),
    "surface_minimum_history_mm": manifest["geometry_audit"]["selected_surface_minimum_historical_center_distance_mm"],
    "surface_minimum_new_distance_mm": manifest["geometry_audit"]["selected_surface_minimum_distance_mm"],
    "deep_minimum_history_mm": manifest["geometry_audit"]["selected_deep_minimum_history_distance_mm"]}
if not args.preflight:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
    os.replace(temporary, output)
    with sidecar.open("xb") as stream:
        stream.write(f"{digest}  {output.name}\n".encode("utf-8"))
print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
