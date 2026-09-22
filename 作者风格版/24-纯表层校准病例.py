# %% 1. 只冻结 19 个纯表层配置 × 3 SNR；不运行反演，不读取任何定位分数。
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from benchmark import protocol
import generate_corrected_v2_manifest as geometry
import run_erp_whole_head_matrix as simulation

parser = argparse.ArgumentParser(description="生成纯表层校准清单；--check 仅核验冻结内容")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
output = project_root / "results/erp_whole_head/adaptive_v6/protocol"
snr_pairs = [(-10, -10), (-10, 20), (20, -10)]
configuration_seed = 2026092210
replica_seed_roots = {"fit": 2026092211, "check": 2026092212}
expected_inputs = {
    "development_manifest.json": "70f376c2c753e5f9d2adc22b067fdb2f0aba797f80bca7eb1d7951f884f3260e",
    "validation_manifest.json": "2da4682fbd23b591a20e56fd9612f5324ecb9cf68ed744f85c51400d11759fb7",
    "lock.json": "eb1625ef1be95553ebf5094bf3c08a4df24bcb007038fb18fab31b4f3a685bc3",
}
inputs = {}
for name, digest in expected_inputs.items():
    payload = (output / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == digest, f"原冻结输入已变化：{name}"
    inputs[name] = json.loads(payload)
excluded_centers = set(inputs["lock.json"]["previously_used_surface_centers"])
for name in ("development_manifest.json", "validation_manifest.json"):
    excluded_centers.update(center for case in inputs[name] for center in case["surface_centers"])
existing_ids = {case["case_id"] for name in ("development_manifest.json", "validation_manifest.json") for case in inputs[name]}


# %% 2. 均匀抽未使用的皮层中心；双源第二点在距第一点至少 50 mm 的剩余点中均匀抽。
shared = geometry._load_shared(project_root / "corrected_v2/generated", Path(r"D:\mne_data\MNE-sample-data"))
assert simulation._shared_fingerprint(shared) == inputs["lock.json"]["shared_fingerprint"]
n_surf = int(shared["n_surf"])
vertices = np.asarray(shared["vertices"])
rng = np.random.default_rng(configuration_seed)
available = set(range(n_surf)) - excluded_centers
configurations = []
position_audit = []
for number in range(19):
    count = 1 if number < 10 else 2
    first = int(rng.choice(sorted(available)))
    available.remove(first)
    centers = [first]
    if count == 2:
        possible = [index for index in sorted(available) if np.linalg.norm(vertices[index] - vertices[first]) >= 0.050]
        assert possible, "双源距离条件无候选点，停止而非放宽条件"
        second = int(rng.choice(possible))
        available.remove(second)
        centers.append(second)
    correlation = 0.0 if count == 1 else (0.0, 0.5)[(number - 10) % 2]
    configurations.append((centers, correlation))
    position_audit.append({"configuration_number": number, "surface_centers": centers,
                           "head_xyz_m": vertices[centers].tolist(),
                           "distance_between_centers_mm": None if count == 1 else float(np.linalg.norm(vertices[centers[0]] - vertices[centers[1]]) * 1000),
                           "correlation": correlation})
selected_centers = [center for centers, _correlation in configurations for center in centers]
assert len(selected_centers) == len(set(selected_centers)) == 28
assert not (set(selected_centers) & excluded_centers)
assert Counter(len(centers) for centers, _correlation in configurations) == {1: 10, 2: 9}


# %% 3. 复用仿真器病例格式；没有深源，SNR 只进入仿真控制和校准分组。
manifest = []
for pair_index, (eeg_snr, meg_snr) in enumerate(snr_pairs):
    for number, (centers, correlation) in enumerate(configurations):
        manifest.append({
            "case_id": f"erp-v6-calibration-{number:02d}-eeg{eeg_snr:+03d}-meg{meg_snr:+03d}",
            "case_number": len(manifest), "configuration_id": f"erp-v6-calibration-{number:02d}",
            "configuration_number": number, "configuration_kind": "single_surface" if len(centers) == 1 else "two_surface_only",
            "panel": "erp_v6_calibration", "pair_index": pair_index,
            "scenario": "surface_only", "scenario_code": protocol.SCENARIO_CODES["surface_only"],
            "surface_centers": centers, "deep_local": None, "deep_index": None,
            "location": None, "replicate": 0, "deep_surface_ratio": None, "correlation": correlation,
            "snr_db": eeg_snr, "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
            "seed": [replica_seed_roots["fit"], pair_index, number], "replica_seed_roots": replica_seed_roots,
        })
assert len(manifest) == len({case["case_id"] for case in manifest}) == 57
assert not ({case["case_id"] for case in manifest} & existing_ids)
assert all(case["deep_index"] is None for case in manifest)
for pair_index in range(3):
    cell = [case for case in manifest if case["pair_index"] == pair_index]
    assert len(cell) == 19
    assert Counter(case["configuration_kind"] for case in cell) == {"single_surface": 10, "two_surface_only": 9}
    assert len({case["configuration_id"] for case in cell}) == 19


# %% 4. 固定保守并列规则：各组 p=(1+sum(Tcal>=T))/20，三个 p 都 <=0.05 且 T>0。
# 19 个校准分数时等价于 T 严格大于全部 57 个分数以及 0；不能按测试真实 SNR 选组。
toy_scores = np.arange(57, dtype=float).reshape(3, 19)
toy_pass = 57.0
toy_tie = 56.0
assert np.all((1 + np.sum(toy_scores >= toy_pass, axis=1)) / 20 <= 0.05)
assert not np.all((1 + np.sum(toy_scores >= toy_tie, axis=1)) / 20 <= 0.05)
lock = {
    "protocol": "erp-v6-cortex-only-empirical-calibration-v1",
    "status": "manifest_frozen_no_calibration_scores_computed",
    "input_sha256": expected_inputs, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "shared_fingerprint": inputs["lock.json"]["shared_fingerprint"],
    "configuration_seed": configuration_seed, "replica_seed_roots": replica_seed_roots,
    "snr_pairs_eeg_meg_db": snr_pairs, "configuration_count": 19, "case_count": 57,
    "single_configurations": 10, "double_configurations": 9, "double_center_min_distance_mm": 50.0,
    "excluded_centers": sorted(excluded_centers), "selected_position_audit": position_audit,
    "sampling": "Uniform without replacement over remaining cortical vertices; a double source's second center is uniform among remaining vertices at least 50 mm from its first. No metric values enter sampling.",
    "decision": {"alpha": 0.05, "required_complete_scores_per_stratum": 19, "strata_count": 3,
                 "rank_formula": "p_s(T) = (1 + count(T_cal_s >= T)) / 20",
                 "accept_deep": "T > 0 AND max(p_0(T), p_1(T), p_2(T)) <= 0.05",
                 "equivalent_with_19_per_stratum": "T > max(0, all 57 calibration scores)",
                 "test_true_snr_used_to_select_calibration_stratum": False,
                 "ties_count_against_acceptance": True, "missing_nonfinite_or_unconverged_score_action": "calibration_invalid_do_not_open_validation",
                 "statistic_and_inverse_must_be_frozen_before_calibration": True},
    "no_formal_guarantee": "Random calibration configurations and farthest-distance validation centers are not exchangeable. Alpha is an empirical conservative rank rule, not a conformal/population FPR guarantee. The same 19 configurations recur at three SNRs; they are not 57 independent spatial configurations.",
    "protected_validation_sha256": expected_inputs["validation_manifest.json"],
    "validation_use": "Validation is consumed once run. Never adjust statistic, calibration sample, alpha or the three-stratum intersection after viewing validation outcomes.",
}


# %% 5. 原协议文件只读；新增文件若已存在则只核验，不覆盖。
for name, contents in (("calibration_manifest.json", manifest), ("calibration_lock.json", lock)):
    payload = (json.dumps(contents, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = output / name
    if args.check or path.exists():
        assert path.read_bytes() == payload, f"冻结内容不同，拒绝覆盖：{path}"
        assert path.with_suffix(".json.sha256").read_text(encoding="ascii") == f"{digest}  {name}\n"
    else:
        protocol.save_manifest(path, contents, expected_digest=digest)
    print(name, digest)
for name, digest in expected_inputs.items():
    assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
print("Frozen/checked: 19 cortex-only configurations x 3 SNR = 57 cases. No inverse or score computation.")
