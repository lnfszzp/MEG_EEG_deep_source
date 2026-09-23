"""开发集决策后重定位：冻结训练试次白化器，合并两份均值后重跑同一 ADMM。"""

# %% 参数和只读输入。不读取校准/验证，不改变原 20/20 presence 结果。
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_predictive import fit_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True,
                    help="已有 trial-covariance ADMM 开发结果目录")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--noise-multiplier", type=float, default=1.0,
                    help="同时缩放浅层和深层经验噪声惩罚；只作开发机制消融")
args = parser.parse_args()
if not np.isfinite(args.noise_multiplier) or args.noise_multiplier <= 0:
    raise ValueError("noise-multiplier 必须为正的有限数")
if args.output.exists():
    raise FileExistsError(f"不覆盖旧结果：{args.output}")
completion = json.loads((args.source / "completion.json").read_text(encoding="utf-8"))
source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
assert completion == {**completion, "complete": True, "case_count": 5, "all_converged": True}
assert source_metadata["phase"] == "development"
assert source_metadata["covariance"] == "trial"
assert source_metadata["solver_settings"] == {"solver_kind": "admm"}

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
case_by_id = {case["case_id"]: case for case in json.loads(manifest.read_text(encoding="utf-8"))}
assert source_metadata["manifest_sha256"] == manifest_hash
cases = [case_by_id[case_id] for case_id in source_metadata["case_ids"]]
assert len(cases) == 5 and all(case["panel"] == "erp_v6_development" for case in cases)
positive_cases = [case for case in cases if case["deep_index"] is not None]
assert len(positive_cases) == 3

shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert source_metadata["shared_fingerprint"] == original._shared_fingerprint(shared)
seed_root = int(source_metadata["seed_root"])
tracked = [Path(__file__), *[root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_balanced.py",
    "candidates/graph_reweight_solver.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py",
    "benchmark/erp_protocol.py", "benchmark/protocol.py", "benchmark/metrics.py")]]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in tracked}
args.output.mkdir(parents=True)

# %% 只对三个已有深源的开发病例做 ungated 定位诊断；最终流程须先过独立 presence 决策。
rows = []
started = time.perf_counter()
for case in positive_cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in tracked}
    tick = time.perf_counter()
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    source_case = json.loads((args.source / (case["case_id"] + ".json")).read_text(encoding="utf-8"))
    assert source_case["case_id"] == case["case_id"]
    for key in ("seed_case_key", "replica_seed_roots", "noise_scale", "actual_snr_db",
                "trial_baseline_covariance", "modality_weights"):
        assert source_case["simulation"][key] == observation["metadata"][key], \
            f"原结果与重建观测不一致：{case['case_id']} {key}"

    combined = (observation["training"] + observation["confirmation"]) / 2
    null, full, fitting = fit_predictive_models(
        combined, observation["gain"], shared["n_surf"], adjacency=shared["adjacency"],
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"], solver_settings={
            "solver_kind": "admm", "noise_multiplier": args.noise_multiplier})
    convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"])
                   for name in ("null", "full")}
    identity = {key: case[key] for key in
                ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    for name, estimate in (("combined40-surface-only", null), ("combined40-full-ungated", full)):
        values = metrics.evaluate_estimate(
            estimate, observation["truth"], shared["vertices"], observation["groups"],
            shared["n_surf"], observation["active"], shared["auc_cortex"],
            baseline=observation["baseline"])
        rows.append({**identity, "method": name, "status": "ok", **values})

    np.savez_compressed(args.output / (case["case_id"] + ".npz"),
        truth=observation["truth"].astype(np.float32), null=null.astype(np.float32),
        full=full.astype(np.float32), vertices=shared["vertices"], times=shared["times"],
        active=observation["active"], baseline=observation["baseline"])
    detail = {"case_id": case["case_id"], "complete": True,
        "input_strategy": "decision remains train20/check20; localization refit uses their mean after decision",
        "whitener": "frozen empirical baseline covariance from the original twenty training trials",
        "raw_means_changed": False, "combined_formula": "(training + confirmation) / 2",
        "confirmation_used_for_presence_fit": False,
        "confirmation_used_for_post_decision_localization_refit": True,
        "truth_used_by_inverse": False, "truth_used_only_for_development_metrics": True,
        "noise_multiplier": args.noise_multiplier,
        "convergence": convergence, "elapsed_seconds": time.perf_counter() - tick,
        "fitting": fitting}
    (args.output / (case["case_id"] + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    archive._atomic_csv(args.output / "rows.csv", rows, rows[0].keys())
    current = rows[-1]
    print(f"{case['case_id']}: An_cal={current['auc']:.4f}, "
          f"deep_An_auc={current['deep_auc_tie_corrected']:.4f}, "
          f"deep_DLE={current['deep_dle_mm']:.2f}, converged={convergence}", flush=True)

# %% 保存失败同样有意义；不把 ungated 开发结果写成验收通过。
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in tracked}
full_rows = [row for row in rows if row["method"] == "combined40-full-ungated"]
summary = {"complete": len(full_rows) == 3, "all_converged": all(
        json.loads((args.output / (case["case_id"] + ".json")).read_text(encoding="utf-8"))["convergence"][name]
        for case in positive_cases for name in ("null", "full")),
    "phase": "development", "calibration_or_validation_read": False,
    "source": str(args.source), "source_completion_sha256": hashlib.sha256(
        (args.source / "completion.json").read_bytes()).hexdigest(),
    "manifest_sha256": manifest_hash, "case_count": len(full_rows),
    "mean_historical_An_cal_AUC": float(np.mean([row["auc"] for row in full_rows])),
    "mean_An_auc": float(np.mean([row["auc_tie_corrected"] for row in full_rows])),
    "mean_deep_An_auc": float(np.mean([row["deep_auc_tie_corrected"] for row in full_rows])),
    "mean_deep_DLE_mm_detected_peak": float(np.nanmean([row["deep_dle_mm"] for row in full_rows])),
    "finite_deep_DLE_count": int(np.count_nonzero(np.isfinite(
        [row["deep_dle_mm"] for row in full_rows]))),
    "mean_surface_DLE_mm": float(np.nanmean([row["surface_dle_mm"] for row in full_rows])),
    "noise_multiplier": args.noise_multiplier,
    "wall_seconds": time.perf_counter() - started, "code_sha256": code_hashes,
    "scope": "three deep-positive development cases at one source directory SNR; ungated localization diagnostic only"}
(args.output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
