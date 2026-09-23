"""只在五个已见开发病例比较基线协方差；不校准，不消费验证集。"""

# %% 参数。默认只跑试次基线协方差；--covariance mean 可复跑原均值基线条件。
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
from benchmark import protocol, metrics
from benchmark.erp_replicates import prepare_replicated_case, simulate_replicated_case
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--covariance", choices=("mean", "trial"), default="trial")
parser.add_argument("--solver-kind", choices=("irls", "admm"), default="irls")
parser.add_argument("--save-sources", action="store_true")
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"不覆盖旧消融结果：{args.output}")
prepare = prepare_replicated_case
if args.covariance == "trial":
    from benchmark.erp_trial_covariance import prepare_trial_covariance_case
    prepare = prepare_trial_covariance_case
settings = {"solver_kind": args.solver_kind}
if args.solver_kind == "irls":
    settings["max_iter"] = 300

# %% 固定旧开发 5/5 dB 五例和原始双均值噪声，不接收验收或校准 manifest。
manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
assert manifest.with_suffix(".json.sha256").read_text(encoding="ascii").split()[0] == manifest_hash
cases = [case for case in json.loads(manifest.read_text(encoding="utf-8"))
         if case["eeg_snr_db"] == case["meg_snr_db"] == 5]
assert len(cases) == 5 and all(case["panel"] == "erp_v6_development" for case in cases)
assert {case["configuration_number"] for case in cases} == set(range(5))
seed_root = 2026092206  # 各病例显式 replica_seed_roots 优先；同时保存实际种子。
paths = [Path(__file__), *[root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_balanced.py", "candidates/graph_irls.py",
    "candidates/graph_reweight_solver.py", "candidates/oaster_rebuilt.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py", "benchmark/erp_protocol.py",
    "benchmark/protocol.py", "benchmark/methods.py", "benchmark/metrics.py",
    "run_strict_oaster.py", "run_erp_whole_head_matrix.py", "protected_multilayer.py", "auc_metric.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
metadata = {"phase": "development", "covariance": args.covariance, "manifest_sha256": manifest_hash,
    "case_ids": [case["case_id"] for case in cases], "solver_settings": settings, "seed_root": seed_root,
    "replica_seed_roots": {case["case_id"]: case["replica_seed_roots"] for case in cases},
    "code_sha256": code_hashes, "shared_fingerprint": original._shared_fingerprint(shared),
    "environment": original._environment_versions(), "complete": False,
    "scope": "same five development cases and raw train/confirmation means; covariance ablation only, no calibration or validation",
    "decision": "raw signed prediction score only; no deep-presence cutoff fitted or applied"}
args.output.mkdir(parents=True)
(args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
rows, evidence_rows = [], []
started = time.perf_counter()

# %% 同一观测下拟合完整 H0/H1；真值只用于原始指标，原始 T 不截断。
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    print(f"{args.covariance}/{args.solver_kind}: {case['case_id']} starting", flush=True)
    observation = prepare(shared, case, seed_root=seed_root)
    reference = simulate_replicated_case(shared, case, seed_root=seed_root)
    raw_hashes = {}
    for key in ("eeg_train", "meg_train", "eeg_confirmation", "meg_confirmation"):
        assert np.array_equal(observation[key], reference[key]), f"协方差消融不允许改变原始均值：{key}"
        raw_hashes[key] = hashlib.sha256(np.ascontiguousarray(observation[key]).tobytes()).hexdigest()
    del reference
    null, full, fitting = fit_predictive_models(observation["training"], observation["gain"], shared["n_surf"],
        adjacency=shared["adjacency"], baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"], solver_settings=settings)
    score, evidence = score_predictive_models(observation["confirmation"], observation["gain"], null, full,
        baseline=observation["baseline"], active=observation["active_windows"][0], channel_weights=observation["channel_weights"])
    convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"])
                   for name in ("null", "full")}
    identity = {key: case[key] for key in ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    evidence_rows.append({**identity, "score": score, "null_converged": int(convergence["null"]),
        "full_converged": int(convergence["full"]), "all_converged": int(all(convergence.values())),
        "elapsed_seconds": time.perf_counter() - tick, "covariance": args.covariance})
    for name, estimate in (("v6-surface-only", null), ("v6-full-ungated", full)):
        values = metrics.evaluate_estimate(estimate, observation["truth"], shared["vertices"], observation["groups"],
            shared["n_surf"], observation["active"], shared["auc_cortex"], baseline=observation["baseline"])
        rows.append({**identity, "method": name, "status": "ok", "has_surface_true": int(bool(case["surface_centers"])), **values})
    if args.save_sources:
        np.savez_compressed(args.output / (case["case_id"] + ".npz"), truth=observation["truth"].astype(np.float32),
            null=null.astype(np.float32), full=full.astype(np.float32), vertices=shared["vertices"],
            times=shared["times"], active=observation["active"], baseline=observation["baseline"])
    (args.output / (case["case_id"] + ".json")).write_text(json.dumps({"case_id": case["case_id"],
        "complete": True, "covariance": args.covariance, "raw_means_unchanged": True, "raw_mean_sha256": raw_hashes,
        "simulation": observation["metadata"], "fitting": fitting, "evidence": evidence},
        ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    archive._atomic_csv(args.output / "rows.csv", rows, rows[0].keys())
    archive._atomic_csv(args.output / "evidence.csv", evidence_rows, evidence_rows[0].keys())
    print(f"  T={score:.6g}, converged={convergence}, seconds={time.perf_counter() - tick:.1f}", flush=True)

# %% 数值未收敛仍完整保存，不能把计算结束写成求解收敛。
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
completion = {"complete": len(evidence_rows) == 5 and len(rows) == 10, "case_count": len(evidence_rows),
    "all_converged": all(row["all_converged"] for row in evidence_rows), "wall_seconds": time.perf_counter() - started}
(args.output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
(args.output / "metadata.json").write_text(json.dumps({**metadata, **completion}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(completion, flush=True)
