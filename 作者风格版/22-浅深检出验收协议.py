# %% 1. 只冻结开发/验收病例，不运行算法，也不读取旧结果的指标数值。
from pathlib import Path
from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from benchmark import protocol, metrics
import generate_corrected_v2_manifest as geometry
import run_erp_whole_head_matrix as simulation

parser = argparse.ArgumentParser(description="冻结浅深源开发/验收病例；--check 只检查已冻结文件")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
output = project_root / "results/erp_whole_head/adaptive_v6/protocol"
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]  # EEG、MEG；不看结果再替换难例。
shared = geometry._load_shared(project_root / "corrected_v2/generated", Path(r"D:\mne_data\MNE-sample-data"))
assert metrics.DEEP_DETECTION_THRESHOLD == 0.14
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"])


# %% 2. 旧病例身份清单：只提取 case_id；不使用 AUC、检出或其他结果挑新位置。
manifest_paths = sorted((project_root / "results/erp_whole_head").glob("*/manifest.json"))
manifest_paths = [path for path in manifest_paths if "adaptive_v6" not in path.parts]
manifest_paths += [project_root / "results/corrected_v2/strict_blind/manifest.json"]
lookup = {}
source_manifests = []
for path in manifest_paths:
    payload = path.read_bytes()
    entries = json.loads(payload)
    lookup.update({case["case_id"]: case for case in entries})
    source_manifests.append({"path": path.relative_to(project_root).as_posix(), "sha256": hashlib.sha256(payload).hexdigest()})
used_ids = set()
source_rows = []
for path in sorted((project_root / "results/erp_whole_head").rglob("rows.csv")):
    if "adaptive_v6" in path.parts:
        continue
    with path.open(encoding="utf-8-sig", newline="") as stream:
        ids = sorted({row["case_id"] for row in csv.DictReader(stream) if row.get("case_id")})
    used_ids.update(ids)
    source_rows.append({"path": path.relative_to(project_root).as_posix(), "case_count": len(ids), "case_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})
assert used_ids <= lookup.keys(), "存在无法映射的旧 case_id，停止而非把它当未见病例"
old_cases = [lookup[case_id] for case_id in sorted(used_ids)]
old_centers = {int(center) for case in old_cases for center in case["surface_centers"]}
old_deep = {int(case["deep_index"]) for case in old_cases if case["deep_index"] is not None}
old_support = set()
for center in sorted(old_centers):
    old_support.update(map(int, protocol._surface_patch(shared, center)[0]))
shape_inputs = []
for path in sorted((project_root / "results/erp_whole_head/adaptive_v5/shape_final").glob("shape_*eeg+05_meg+05.json")):
    shape = json.loads(path.read_text(encoding="utf-8"))
    # 这里只保留仿真真值身份，不读取 diagnostics 中的算法结果。
    item = {key: shape[key] for key in ("case_id", "parcel", "shape", "center", "surface_indices", "deep_index")}
    shape_inputs.append(item)
    old_centers.add(int(item["center"]))
    old_support.update(map(int, item["surface_indices"]))


# %% 3. 新表层中心按解剖区域与最大最小距离选；深层 16 点已全用过，只能保留新噪声检验。
parcel_names = ["postcentral-lh", "lateraloccipital-rh", "transversetemporal-lh", "superiorfrontal-rh",
                "superiorparietal-lh", "precentral-rh", "postcentral-rh", "lateraloccipital-lh", "superiortemporal-rh"]
parcels = protocol._parcels(shared)
new_centers = []
position_audit = []
for parcel in parcel_names:
    forbidden = sorted(old_centers | set(new_centers))
    candidates = sorted(set(parcels[parcel]) - set(forbidden))
    assert candidates, f"{parcel} 没有未用中心"
    distance = np.linalg.norm(vertices[candidates, None] - vertices[forbidden][None], axis=2).min(axis=1)
    center = int(candidates[int(np.argmax(distance))])
    patch = set(map(int, protocol._surface_patch(shared, center)[0]))
    new_centers.append(center)
    position_audit.append({"parcel": parcel, "center": center, "head_xyz_m": vertices[center].tolist(),
                           "distance_to_old_center_mm": float(np.linalg.norm(vertices[sorted(old_centers)] - vertices[center], axis=1).min() * 1000),
                           "patch_size": len(patch), "old_active_support_overlap": len(patch & old_support)})
assert not (set(new_centers) & old_centers)
assert len(set(new_centers)) == 9
assert old_deep == set(range(n_surf, n_surf + int(shared["n_deep"])))

# 左/右丘脑各选一个几何中心附近点；不根据深源检出率选点。
deep_local = []
for label in (10, 49):
    indices = np.flatnonzero(np.asarray(shared["deep_aseg_labels"]) == label)
    points = vertices[n_surf + indices]
    deep_local.append(int(indices[np.argmin(np.sum((points - points.mean(axis=0)) ** 2, axis=1))]))


# %% 4. 开发病例沿用已见位置；验收病例固定 8 配置 × 3 SNR，4 个无深源、4 个有深源。
development = [
    ("single_surface", [1317], None, 0.0),
    ("two_surface_only", [1317, 7204], None, 0.0),
    ("deep_only", [], 0, 0.0),
    ("deep_plus_surface", [1317], 0, 0.0),
    ("deep_plus_two_surface", [1317, 7204], 0, 0.0),
]
validation = [
    ("single_surface", [new_centers[0]], None, 0.0),
    ("single_surface", [new_centers[1]], None, 0.0),
    ("two_surface_only", new_centers[2:4], None, 0.0),
    ("two_surface_only", new_centers[4:6], None, 0.5),
    ("deep_only", [], deep_local[0], 0.0),
    ("deep_only", [], deep_local[1], 0.0),
    ("deep_plus_surface", [new_centers[6]], deep_local[0], 0.0),
    ("deep_plus_two_surface", new_centers[7:9], deep_local[1], 0.5),
]
artifacts = {}
for panel, configurations, seed_roots in (
    ("development", development, {"fit": 2026092206, "check": 2026092207}),
    ("validation", validation, {"fit": 2026092208, "check": 2026092209}),
):
    manifest = []
    panel_pairs = snr_pairs + [(5, 5)] if panel == "development" else snr_pairs
    for pair_index, (eeg_snr, meg_snr) in enumerate(panel_pairs):
        for number, (kind, centers, deep, correlation) in enumerate(configurations):
            scenario = "surface_only" if deep is None else "deep_only" if not centers else "deep_plus_surface" if len(centers) == 1 else "deep_plus_two_surface"
            case_id = f"erp-v6-{panel}-{number:02d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}"
            manifest.append({"case_id": case_id, "case_number": len(manifest), "configuration_id": f"erp-v6-{panel}-{number:02d}",
                             "configuration_number": number, "configuration_kind": kind, "panel": f"erp_v6_{panel}",
                             "pair_index": pair_index, "scenario": scenario, "scenario_code": protocol.SCENARIO_CODES[scenario],
                             "surface_centers": centers, "deep_local": deep, "deep_index": None if deep is None else n_surf + deep,
                             "location": None, "replicate": 0, "deep_surface_ratio": 0.5 if centers and deep is not None else None,
                             "correlation": correlation, "snr_db": eeg_snr, "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
                             "seed": [seed_roots["fit"], pair_index, number], "replica_seed_roots": seed_roots})
    assert len(manifest) == len(configurations) * len(panel_pairs)
    assert len({case["case_id"] for case in manifest}) == len(manifest)
    assert not ({case["case_id"] for case in manifest} & used_ids)
    assert all(len(case["surface_centers"]) + (case["deep_index"] is not None) in (1, 2, 3) for case in manifest)
    assert len(simulation._select_pairs(manifest, panel_pairs, None)) == len(panel_pairs)
    artifacts[f"{panel}_manifest.json"] = manifest


# %% 5. 不改变原指标；这是小样本工程验收，不是统计学证明 FPR <= 5%。
validation_cases = artifacts["validation_manifest.json"]
assert Counter(case["configuration_kind"] for case in validation_cases) == {
    "single_surface": 6, "two_surface_only": 6, "deep_only": 6, "deep_plus_surface": 3, "deep_plus_two_surface": 3}
for pair_index in range(3):
    cell = [case for case in validation_cases if case["pair_index"] == pair_index]
    assert sum(case["deep_index"] is None for case in cell) == 4
    assert sum(case["deep_index"] is not None for case in cell) == 4
artifacts["lock.json"] = {
    "protocol": "erp-v6-detection-engineering-pilot-v1", "historical_metric_values_used": False,
    "status": "manifest_frozen_inverse_configuration_must_be_frozen_before_validation",
    "shared_fingerprint": simulation._shared_fingerprint(shared), "environment": simulation._environment_versions(),
    "source_manifests": source_manifests, "old_result_case_id_sources": source_rows,
    "used_case_id_count": len(used_ids), "used_case_ids_sha256": hashlib.sha256("\n".join(sorted(used_ids)).encode()).hexdigest(),
    "previously_used_surface_centers": sorted(old_centers), "previously_used_deep_indices": sorted(old_deep),
    "shape_source_identity": shape_inputs, "new_surface_position_audit": position_audit,
    "new_deep_local_geometry_choices": deep_local,
    "spatial_holdout_limit": "New cortical centers, not necessarily disjoint cortical patches. All 16 deep locations were used previously; deep-only cases are noise-held-out, not position-held-out.",
    "acceptance": {"false_positive_rate_max_each_snr": 0.05, "negative_configurations_each_snr": 4,
                   "required_false_positives_each_snr": 0, "localized_deep_detection_min_overall": 0.80,
                   "required_localized_deep_detections_of_12": 10, "conditional_original_deep_dle_mean_max_mm": 10.0,
                   "all_positive_miss_penalized_deep_dle_mean_max_mm": 60.0,
                   "all_validation_solver_jobs_converged": True, "deep_score_threshold_unchanged": 0.14,
                   "localized_detection_radius_mm": 10.0, "miss_penalty_mm": float(np.linalg.norm(np.ptp(vertices, axis=0)) * 1000),
                   "auc_is_secondary_and_cannot_override_detection_failure": True},
    "sampling_limit": {"independent_spatial_configurations_total": 8, "independent_negative_configurations": 4,
                       "independent_positive_configurations": 4, "snr_repeats_are_not_new_locations": True,
                       "zero_of_four_two_sided_95pct_binomial_upper": 1.0 - 0.025 ** (1.0 / 4.0),
                       "minimum_independent_negative_cases_for_zero_event_95pct_upper_below_5pct": 72},
    "observation_rule": "Fit and check are independent 20-trial-mean Gaussian observations of identical truth. Noise scale is fixed once from expected energy so their combined 40-trial mean has the stated expected SNR; never normalize by realized active noise energy. Each half therefore has expected SNR 3.0103 dB below the combined target. Inverse selection must not use truth, source locations, metric labels or check-observation fitting. Validation results may be used once for acceptance, never retuning.",
    "calibration_limit": "Cortex-only calibration draws random new configurations whereas validation cortical centers use farthest-distance geometry. These are not exchangeable draws: alpha=0.05 is an empirical rank cutoff, not a conformal or population false-positive guarantee. Calibration manifest and thresholds must be frozen before the one-use validation run.",
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}


# %% 6. 已存在的文件只核对，不覆盖；修改协议须用新版本与新噪声，不可覆盖旧验收结果。
for name, contents in artifacts.items():
    payload = (json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = output / name
    if args.check or path.exists():
        assert path.read_bytes() == payload, f"冻结内容不同，拒绝覆盖：{path}"
        assert path.with_suffix(".json.sha256").read_text(encoding="ascii") == f"{digest}  {name}\n"
    else:
        # 复用仓库原子保存函数；JSON 根可以是列表或审计字典，不引入保存框架。
        protocol.save_manifest(path, contents, expected_digest=digest)
    print(name, digest)
print("已冻结/核验：开发 20 病例；验收 24 病例；本脚本未运行任何反演。")
print("注意：每 SNR 只有 4 个无深源配置，即使 0 误报也不是已证明总体 FPR <= 5%。")
