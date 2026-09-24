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
parser.add_argument("--covariance", choices=("mean", "trial"), default="mean")
parser.add_argument("--solver-settings", type=json.loads, default={})
parser.add_argument("--score-kind", choices=("noise", "excess", "conjunctive"), default="noise")
parser.add_argument("--snr-pair", nargs=2, type=int)
parser.add_argument("--limit", type=int)
parser.add_argument("--comparators", action="store_true")
parser.add_argument("--save-sources", action="store_true")
parser.add_argument("--preflight", action="store_true", help="只核对正式冻结链，不消费或运行病例")
args = parser.parse_args()
if not __debug__:
    raise RuntimeError("正式运行禁止使用python -O；否则assert完整性检查会被关闭")
formal = args.phase != "development"
if args.preflight and not formal:
    raise ValueError("--preflight只用于正式calibration/validation")
if formal and (args.limit is not None or args.snr_pair is not None):
    raise ValueError("校准/验证必须完成整个冻结 manifest，不能挑选病例")
if args.output.exists():
    raise FileExistsError(f"输出已存在，不覆盖或事后重跑验证：{args.output}")
cases = json.loads(args.manifest.read_text(encoding="utf-8"))
assert isinstance(cases, list) and cases and len({case["case_id"] for case in cases}) == len(cases)
manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
if args.snr_pair is not None:
    cases = [case for case in cases if [case["eeg_snr_db"], case["meg_snr_db"]] == args.snr_pair]
if args.limit is not None:
    cases = cases[:args.limit]
assert cases
protocol_dir = root / "results/erp_whole_head/adaptive_v6/protocol"
execution_lock = None
execution_lock_sha = None
formal_solver_settings = {"solver_kind": "admm", "mrf_strength": .8, "outer_iterations": 1, "max_inner_retries": 20}
formal_manifest_sha = {"calibration": "09dde6a2c14ceeeaa4a3185d218f8017bc6141b37e4172d19f6f85c22cf73ff1",
                       "validation": "68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7"}
formal_seed_root = {"calibration": 2026092601, "validation": 2026092403}
if formal:
    if manifest_sha != formal_manifest_sha[args.phase]:
        raise ValueError(f"不是冻结的分量平衡 v4 {args.phase} manifest")
    if args.seed_root != formal_seed_root[args.phase]:
        raise ValueError(f"正式{args.phase}的 seed root 必须是 {formal_seed_root[args.phase]}")
    if args.covariance != "trial" or args.score_kind != "excess" or args.solver_settings != formal_solver_settings:
        raise ValueError("正式阶段必须使用 trial covariance、excess score 和冻结 ADMM 设置")
    if args.phase == "calibration" and args.comparators:
        raise ValueError("正式校准只生成零假设分数，不运行对比方法")
    if args.phase == "validation" and (not args.save_sources or not args.comparators):
        raise ValueError("正式验证必须同时指定 --save-sources 和 --comparators")
    expected_revision = ("formal_component_balanced_v4_independent_calibration"
                         if args.phase == "calibration" else "formal_component_balanced_v2")
    if any(case.get("component_balance_revision") != expected_revision for case in cases):
        raise ValueError(f"正式 {args.phase} manifest 的分量平衡版本不符")
    if any(case.get("seed", [None])[0] != args.seed_root or
           case.get("replica_seed_roots") != {"fit": args.seed_root, "check": args.seed_root + 1} for case in cases):
        raise ValueError("正式 manifest 的拟合/复核噪声 seed 与冻结协议不符")
    pairs = ((-10, -10), (-10, 20), (20, -10))
    configurations = Counter(case["configuration_id"] for case in cases)
    two_surface = [case for case in cases if len(case["surface_centers"]) == 2]
    if args.phase == "calibration":
        valid = (len(cases) == 57 and len(configurations) == 57 and set(configurations.values()) == {1} and
                 all(case.get("deep_index") is None for case in cases) and len(two_surface) == 27 and
                 all(case.get("surface_component_sensor_balance") is True for case in two_surface) and
                 Counter((case["eeg_snr_db"], case["meg_snr_db"]) for case in cases) == Counter({pair: 19 for pair in pairs}))
    else:
        mixed = [case for case in cases if case.get("deep_index") is not None and case["surface_centers"]]
        valid = (len(cases) == 24 and len(configurations) == 8 and set(configurations.values()) == {3} and
                 len(two_surface) == 9 and all(case.get("surface_component_sensor_balance") is True for case in two_surface) and
                 len(mixed) == 6 and all(case.get("deep_surface_sensor_amplitude_ratio") == .5 for case in mixed) and
                 Counter((case["eeg_snr_db"], case["meg_snr_db"], case.get("deep_index") is not None) for case in cases) ==
                 Counter({(eeg, meg, present): 4 for eeg, meg in pairs for present in (False, True)}))
    if not valid:
        raise ValueError(f"冻结的{args.phase} manifest 分层或分量平衡计数不符")

    execution_lock_path = protocol_dir / "component_balanced_v4_execution_lock.json"
    execution_lock_sidecar = Path(str(execution_lock_path) + ".sha256")
    if not execution_lock_path.exists() or not execution_lock_sidecar.exists():
        raise FileNotFoundError("正式算法尚未冻结：缺少 component_balanced_v4_execution_lock.json 或其 .sha256 旁车")
    execution_lock_sha = hashlib.sha256(execution_lock_path.read_bytes()).hexdigest()
    if execution_lock_sidecar.read_text(encoding="utf-8").split()[0] != execution_lock_sha:
        raise ValueError("正式执行锁与 .sha256 旁车不一致")
    execution_lock = json.loads(execution_lock_path.read_text(encoding="utf-8"))
    if execution_lock.get("protocol") != "erp-v6-component-balanced-formal-v4-execution" or \
            execution_lock.get("status") != "algorithm_frozen_calibration_allowed" or not execution_lock.get("formal_calibration_allowed"):
        raise ValueError("正式执行锁尚未允许校准")
    locked_algorithm = execution_lock.get("algorithm", {})
    for key, value in (("covariance", args.covariance), ("score_kind", args.score_kind),
                       ("solver_settings", args.solver_settings), ("excess_loss_denominator_floor_fraction", .05)):
        if locked_algorithm.get(key) != value:
            raise ValueError(f"正式执行锁算法设置不一致：{key}")
    locked_decision = locked_algorithm.get("calibration_decision", {})
    for key, value in (("alpha", .05), ("strata_count", 1), ("calibration_score_count", 57),
                       ("required_complete_finite_scores", 57), ("test_true_snr_used", False),
                       ("ties_count_against_acceptance", True),
                       ("accept_deep", "T > 0 AND p(T) <= 0.05"),
                       ("rank_formula", "p(T) = (1 + count(T_cal >= T)) / 58")):
        if locked_decision.get(key) != value:
            raise ValueError(f"正式执行锁校准决策不一致：{key}")
    locked_manifest = execution_lock.get("manifests", {}).get(args.phase, {})
    if locked_manifest.get("sha256") != manifest_sha or locked_manifest.get("case_count") != len(cases) or \
            locked_manifest.get("seed_roots") != {"fit": args.seed_root, "check": args.seed_root + 1}:
        raise ValueError(f"正式执行锁中的 {args.phase} manifest 或 seed 不一致")
    for locked in (execution_lock.get("protocol_inputs", {}).get("protocol_lock", {}), locked_manifest):
        locked_path = root / locked.get("path", "")
        locked_sidecar = root / locked.get("sidecar_path", "")
        if not locked_path.is_file() or hashlib.sha256(locked_path.read_bytes()).hexdigest() != locked.get("sha256"):
            raise ValueError(f"执行锁保护文件已变化：{locked_path}")
        if not locked_sidecar.is_file() or hashlib.sha256(
                locked_sidecar.read_text(encoding="ascii").encode("ascii")).hexdigest() != locked.get("sidecar_text_sha256"):
            raise ValueError(f"执行锁保护旁车已变化：{locked_sidecar}")
    if args.manifest.resolve() != (root / locked_manifest["path"]).resolve():
        raise ValueError("正式阶段必须直接使用执行锁中的 manifest 路径")

consumed_path = protocol_dir / ("calibration_component_balanced_v4_manifest_consumed.json"
                                if args.phase == "calibration" else "validation_manifest_consumed.json")
if formal and consumed_path.exists():
    raise FileExistsError(f"冻结{args.phase}病例已消费，不可换目录重新调参/校准：{consumed_path}")
prepare = prepare_replicated_case
if args.covariance == "trial":
    from benchmark.erp_trial_covariance import prepare_trial_covariance_case
    prepare = prepare_trial_covariance_case

# %% 冻结算法、指标、仿真、环境指纹。更改其中任何一项都不能沿用旧校准。
paths = [Path(__file__), *[root / name for name in (
    "candidates/oaster_predictive.py", "candidates/oaster_balanced.py", "candidates/graph_irls.py",
    "candidates/graph_reweight_solver.py", "candidates/oaster_rebuilt.py", "algorithms/spatial_fused_fusion.py",
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py", "benchmark/erp_protocol.py", "benchmark/protocol.py",
    "benchmark/methods.py", "benchmark/metrics.py", "benchmark/deep_acceptance.py",
    "run_strict_oaster.py", "run_erp_whole_head_matrix.py", "protected_multilayer.py", "auc_metric.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
if formal:
    formal_scripts = [root / "作者风格版/48-独立门控后40试次盲定位.py", root / "作者风格版/54-完整盲定位正式验收.py"]
    missing = [str(path.relative_to(root)) for path in formal_scripts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"正式算法尚未补齐，不能运行：{missing}")
    paths.extend(formal_scripts)
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
if formal:
    locked_hashes = execution_lock.get("code_sha256", {})
    changed = [name for name, digest in code_hashes.items() if locked_hashes.get(name) != digest]
    if changed:
        raise ValueError(f"正式执行锁后代码已变化：{changed}")
calibration = None
calibration_path = None
calibration_sha256 = None
if args.phase == "validation":
    if args.calibration is None:
        raise ValueError("验证必须指定完成的纯表层校准目录")
    calibration_path = (args.calibration / "frozen_calibration.json").resolve()
    calibration_payload = calibration_path.read_bytes()
    calibration_sha256 = hashlib.sha256(calibration_payload).hexdigest()
    calibration = json.loads(calibration_payload)
    if calibration.get("covariance") != args.covariance:
        raise ValueError("校准后不能改变 covariance；缺少该记录的旧校准也不可沿用")
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
shared_hash = original._shared_fingerprint(shared)
environment = original._environment_versions()
if formal:
    for key, value in (("shared_fingerprint", shared_hash), ("environment", environment)):
        if execution_lock.get(key) != value:
            raise ValueError(f"正式执行锁与当前运行环境不一致：{key}")
if args.phase == "validation":
    for key, value in (("code_sha256", code_hashes), ("shared_fingerprint", shared_hash),
                       ("environment", environment), ("solver_settings", args.solver_settings),
                       ("covariance", args.covariance), ("score_kind", args.score_kind),
                       ("execution_lock_sha256", execution_lock_sha)):
        if calibration.get(key) != value:
            raise ValueError(f"校准后发生改变：{key}")
    calibration_valid = (calibration.get("complete") and calibration.get("all_converged") and calibration.get("alpha") == .05 and
        calibration.get("phase") == "calibration" and calibration.get("manifest_sha256") == formal_manifest_sha["calibration"] and
        calibration.get("seed_root") == formal_seed_root["calibration"] and calibration.get("case_count") == 57 and
        len(calibration.get("null_scores", [])) == 57 and
        all(np.isfinite(values) for values in calibration.get("null_scores", [])) and
        not ({case["case_id"] for case in cases} & set(calibration.get("case_ids", []))))
    if not calibration_valid:
        raise ValueError("冻结校准不完整、未收敛或与受保护验证集不独立")
if formal and args.preflight:
    print(json.dumps({"preflight": "passed", "phase": args.phase,
                      "manifest_sha256": manifest_sha,
                      "execution_lock_sha256": execution_lock_sha}, ensure_ascii=False, indent=2))
    raise SystemExit(0)
if formal:
    # 校准和验证均一次性消费；固定位置标记，复制相同 manifest 也不能绕过。
    with consumed_path.open("x", encoding="utf-8") as stream:
        json.dump({"phase": args.phase, "manifest_sha256": manifest_sha,
                   "output": str(args.output.resolve()), "calibration": None if args.calibration is None else str(args.calibration.resolve()),
                   "covariance": args.covariance, "solver_settings": args.solver_settings,
                   "score_kind": args.score_kind,
                   "execution_lock_sha256": execution_lock_sha,
                   "code_sha256": code_hashes}, stream, ensure_ascii=False, indent=2)
args.output.mkdir(parents=True)
metadata = {"phase": args.phase, "manifest": str(args.manifest.resolve()),
    "manifest_sha256": manifest_sha,
    "seed_root": args.seed_root, "case_ids": [case["case_id"] for case in cases],
    "code_sha256": code_hashes, "shared_fingerprint": shared_hash, "environment": environment,
    "solver_settings": args.solver_settings, "covariance": args.covariance, "alpha": .05,
    "score_kind": args.score_kind,
    "calibration_path": None if calibration_path is None else str(calibration_path),
    "calibration_sha256": calibration_sha256,
    "execution_lock_sha256": execution_lock_sha,
    "comparison_data": "v6 H0/H1 fitted to train-20 mean, confirmation-20 used only for prediction score; seven comparators reported separately on train-20 and combined-40 means, not treated as identical data usage",
    "scope": "engineering development/calibration/stress validation; no universal or clinical FPR guarantee",
    "calibration_use": "one-use empirical null calibration, not a reusable independent test or a parameter-tuning set; raw half-means are independent but share training-derived preprocessing"}
(args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
rows, evidence_rows = [], []
started = time.perf_counter()

# %% 每例都先拟合纯表层与完整模型，再用独立观测评分。真值仅进入最后指标计算。
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    print(f"{args.phase}: {case['case_id']} starting", flush=True)
    observation = prepare(shared, case, seed_root=args.seed_root)
    null, full, fitting = fit_predictive_models(observation["training"], observation["gain"], shared["n_surf"],
        adjacency=shared["adjacency"], baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"], solver_settings=args.solver_settings)
    noise_score, evidence = score_predictive_models(observation["confirmation"], observation["gain"], null, full,
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"],
        modality_sizes=observation["metadata"]["retained_channels"])
    score = {"noise": noise_score, "excess": evidence["excess_fraction_score"],
             "conjunctive": evidence["conjunctive_modality_score"]}[args.score_kind]
    pair_key = f"{int(case['eeg_snr_db'])},{int(case['meg_snr_db'])}"
    model_convergence = {name: bool(fitting[name + "_model"]["windows"][0]["solver"]["converged"]) for name in ("null", "full")}
    converged = all(model_convergence.values())
    decision = None
    if calibration is not None:
        # 57例一次 pooled 排名；决策器不读取测试病例的真实 SNR。
        decision = conformal_decision(score, calibration["null_scores"], alpha=.05)
        decision["rule"] = "single pooled frozen null calibration; no test-SNR input"
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
    print(f"  {args.score_kind}={score:.5g} converged={model_convergence} deep={None if decision is None else decision['deep_present']} seconds={elapsed:.1f}", flush=True)

# %% 校准使用所有纯表层分数（包含拟合残差），不借用验证真值调整阈值。
assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
completion = {"complete": len(evidence_rows) == len(cases), "case_count": len(cases),
              "all_converged": all(row["all_converged"] for row in evidence_rows), "wall_seconds": time.perf_counter() - started}
if args.phase == "calibration":
    pooled_scores = [row["score"] for row in evidence_rows]
    null_scores = {}
    for row in evidence_rows:
        key = f"{int(row['eeg_snr_db'])},{int(row['meg_snr_db'])}"
        null_scores.setdefault(key, []).append(row["score"])
    (args.output / "frozen_calibration.json").write_text(json.dumps(
        {**metadata, **completion, "null_scores": pooled_scores,
         "null_scores_by_snr_diagnostic_only": null_scores}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if args.phase == "validation":
    from benchmark.deep_acceptance import evaluate_acceptance
    penalty = float(np.linalg.norm(np.ptp(shared["vertices"], axis=0)) * 1000)
    acceptance = evaluate_acceptance(cases, rows, evidence_rows, penalty)
    (args.output / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print("硬性验收：", acceptance["gates"], flush=True)
(args.output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
print(completion, flush=True)
