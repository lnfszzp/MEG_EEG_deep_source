"""#%% 顺序执行：开发、纯表层校准、一次性冻结验证，三阶段分开保存。"""

# %% 参数与路径。不覆盖旧输出；验证只读取已经冻结的纯表层校准分数。
import argparse
from collections import Counter
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
from benchmark import protocol, metrics, methods
from benchmark.erp_replicates import prepare_replicated_case
from candidates.oaster_predictive import fit_predictive_models, score_predictive_models, conformal_decision
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("development", "calibration", "validation"), required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seed-root", type=int, required=True)
parser.add_argument("--calibration", type=Path)
parser.add_argument("--solver-settings", type=json.loads, default={})
parser.add_argument("--snr-pair", nargs=2, type=int)
parser.add_argument("--limit", type=int)
parser.add_argument("--comparators", action="store_true")
parser.add_argument("--save-sources", action="store_true")
args = parser.parse_args()
if args.phase != "development" and (args.limit is not None or args.snr_pair is not None):
    raise ValueError("校准/验证必须完成整个冻结 manifest，不能挑选病例")
if args.output.exists():
    raise FileExistsError(f"输出已存在，不覆盖或事后重跑验证：{args.output}")
cases = json.loads(args.manifest.read_text(encoding="utf-8"))
assert isinstance(cases, list) and cases and len({case["case_id"] for case in cases}) == len(cases)
if args.snr_pair is not None:
    cases = [case for case in cases if [case["eeg_snr_db"], case["meg_snr_db"]] == args.snr_pair]
if args.limit is not None:
    cases = cases[:args.limit]
assert cases
if args.phase == "validation":
    assert hashlib.sha256(args.manifest.read_bytes()).hexdigest() == "2da4682fbd23b591a20e56fd9612f5324ecb9cf68ed744f85c51400d11759fb7", "不是预先冻结的24例验证集"
    assert len(cases) == 24 and len({case["configuration_id"] for case in cases}) == 8
    assert Counter((case["eeg_snr_db"], case["meg_snr_db"], case.get("deep_index") is not None) for case in cases) == Counter(
        {(eeg, meg, present): 4 for eeg, meg in ((-10, -10), (-10, 20), (20, -10)) for present in (False, True)})
if args.phase == "calibration":
    assert hashlib.sha256(args.manifest.read_bytes()).hexdigest() == "459b7b983b9ad4bbbc8b84e3d0a494e8466dbe093a32af02b78e7d871119a313", "不是冻结的57例校准集"
    assert all(case.get("deep_index") is None for case in cases)
    assert min(Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in cases).values()) >= 19

# %% 冻结算法、指标、仿真、环境指纹。更改其中任何一项都不能沿用旧校准。
paths = [Path(__file__), *[root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_balanced.py", "candidates/graph_irls.py",
    "candidates/graph_reweight_solver.py", "candidates/oaster_rebuilt.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_replicates.py", "benchmark/erp_protocol.py", "benchmark/protocol.py",
    "benchmark/methods.py", "benchmark/metrics.py", "metrics/user_metrics/An_auc.py",
    "run_strict_oaster.py", "run_erp_whole_head_matrix.py")]]
if args.phase != "development":
    paths.append(root / "benchmark/deep_acceptance.py")
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
shared_hash = original._shared_fingerprint(shared)
environment = original._environment_versions()
calibration = None
if args.phase == "validation":
    if args.calibration is None:
        raise ValueError("验证必须指定完成的纯表层校准目录")
    calibration = json.loads((args.calibration / "frozen_calibration.json").read_text(encoding="utf-8"))
    for key, value in (("code_sha256", code_hashes), ("shared_fingerprint", shared_hash),
                       ("environment", environment), ("solver_settings", args.solver_settings)):
        assert calibration[key] == value, f"校准后发生改变：{key}"
    assert calibration["complete"] and calibration["all_converged"] and calibration["alpha"] == .05
    assert not ({case["case_id"] for case in cases} & set(calibration["case_ids"]))
    # 在反演前消费测试标记；不因结果不好重置。只允许另立新的未来验证集。
    consumed_path = args.manifest.with_name(args.manifest.stem + "_consumed.json")
    with consumed_path.open("x", encoding="utf-8") as stream:
        json.dump({"output": str(args.output.resolve()), "calibration": str(args.calibration.resolve()),
                   "code_sha256": code_hashes}, stream, ensure_ascii=False, indent=2)
args.output.mkdir(parents=True)
metadata = {"phase": args.phase, "manifest": str(args.manifest.resolve()),
    "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
    "seed_root": args.seed_root, "case_ids": [case["case_id"] for case in cases],
    "code_sha256": code_hashes, "shared_fingerprint": shared_hash, "environment": environment,
    "solver_settings": args.solver_settings, "alpha": .05,
    "comparison_data": "v6 H0/H1 fitted to train-20 mean, confirmation-20 used only for prediction score; seven comparators reported separately on train-20 and combined-40 means, not treated as identical data usage",
    "scope": "engineering development/calibration/stress validation; no universal or clinical FPR guarantee"}
(args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
rows, evidence_rows = [], []
started = time.perf_counter()

# %% 每例都先拟合纯表层与完整模型，再用独立观测评分。真值仅进入最后指标计算。
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    print(f"{args.phase}: {case['case_id']} starting", flush=True)
    observation = prepare_replicated_case(shared, case, seed_root=args.seed_root)
    null, full, fitting = fit_predictive_models(observation["training"], observation["gain"], shared["n_surf"],
        adjacency=shared["adjacency"], baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"], solver_settings=args.solver_settings)
    score, evidence = score_predictive_models(observation["confirmation"], observation["gain"], null, full,
        baseline=observation["baseline"], active=observation["active_windows"][0], channel_weights=observation["channel_weights"])
    pair_key = f"{int(case['eeg_snr_db'])},{int(case['meg_snr_db'])}"
    model_convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"]) for name in ("null", "full")}
    converged = all(model_convergence.values())
    decision = None
    if calibration is not None:
        # 同一个保守规则应用于所有病例；绝不把测试的真实 SNR 交给决策器选阈值。
        stratum_decisions = [conformal_decision(score, values, alpha=.05)
                            for values in calibration["null_scores_by_snr"].values()]
        decision = {"p_value": max(item["p_value"] for item in stratum_decisions),
                    "deep_present": all(item["deep_present"] for item in stratum_decisions),
                    "rule": "pass every frozen null-calibration stratum; no test-SNR input", "alpha": .05}
    elapsed = time.perf_counter() - tick
    evidence_row = {"case_id": case["case_id"], "configuration_id": case["configuration_id"],
        "scenario": case["scenario"], "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
        "score": score, "null_converged": int(model_convergence["null"]), "full_converged": int(model_convergence["full"]),
        "all_converged": int(converged), "p_value": np.nan if decision is None else decision["p_value"],
        "deep_present_decision": "" if decision is None else int(decision["deep_present"]), "elapsed_seconds": elapsed}
    evidence_rows.append(evidence_row)
    estimates = {"v6-surface-only": null, "v6-full-ungated": full}
    if decision is not None:
        estimates["OASTER-ERP-v6"] = full if decision["deep_present"] else null
    if args.phase != "calibration":
        if args.comparators:
            gain = observation["gain"]
            for condition, data in (("train20", observation["training"]),
                                    ("combined40", (observation["training"] + observation["confirmation"]) / 2)):
                comparison = methods.minimum_norm_family(data, gain)
                comparison["LCMV"] = methods.lcmv(data, gain, observation["active"])
                comparison["Dipole fitting (grid)"] = methods.dipole_fit(data, gain, observation["active"])
                comparison["RAP-MUSIC"] = methods.rap_music(data, gain, observation["active"])
                estimates.update({f"{name} [{condition}]": estimate for name, estimate in comparison.items()})
        for name, estimate in estimates.items():
            values = metrics.evaluate_estimate(estimate, observation["truth"], shared["vertices"], observation["groups"],
                shared["n_surf"], observation["active"], shared["auc_cortex"], baseline=observation["baseline"])
            row = {"case_id": case["case_id"], "configuration_id": case["configuration_id"], "scenario": case["scenario"],
                "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"], "method": name,
                "status": "ok", "has_surface_true": int(bool(case["surface_centers"])), **values}
            rows.append(row)
        archive._atomic_csv(args.output / "rows.csv", rows, rows[0].keys())
    if args.save_sources:
        np.savez_compressed(args.output / (case["case_id"] + ".npz"), truth=observation["truth"].astype(np.float32),
            null=null.astype(np.float32), full=full.astype(np.float32), vertices=shared["vertices"],
            times=shared["times"], active=observation["active"], baseline=observation["baseline"])
    (args.output / (case["case_id"] + ".json")).write_text(json.dumps(
        {"case_id": case["case_id"], "simulation": observation["metadata"], "fitting": fitting, "evidence": evidence,
         "decision": decision}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive._atomic_csv(args.output / "evidence.csv", evidence_rows, evidence_rows[0].keys())
    print(f"  T={score:.5g} converged={model_convergence} deep={None if decision is None else decision['deep_present']} seconds={elapsed:.1f}", flush=True)

# %% 校准使用所有纯表层分数（包含拟合残差），不借用验证真值调整阈值。
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
completion = {"complete": len(evidence_rows) == len(cases), "case_count": len(cases),
              "all_converged": all(row["all_converged"] for row in evidence_rows), "wall_seconds": time.perf_counter() - started}
if args.phase == "calibration":
    null_scores = {}
    for row in evidence_rows:
        key = f"{int(row['eeg_snr_db'])},{int(row['meg_snr_db'])}"
        null_scores.setdefault(key, []).append(row["score"])
    (args.output / "frozen_calibration.json").write_text(json.dumps(
        {**metadata, **completion, "null_scores_by_snr": null_scores}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if args.phase == "validation":
    from benchmark.deep_acceptance import evaluate_acceptance
    penalty = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000)
    acceptance = evaluate_acceptance(cases, rows, evidence_rows, penalty)
    (args.output / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print("硬性验收：", acceptance["gates"], flush=True)
(args.output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
print(completion, flush=True)
