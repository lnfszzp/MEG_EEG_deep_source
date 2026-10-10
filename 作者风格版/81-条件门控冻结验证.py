"""# %% 冻结 residual 门控：同一校准库比较 global/observable-conditional rank decisions。"""

# %% 1. 唯一 pilot 规则、输入路径与作者风格。
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import msvcrt
import os
from pathlib import Path
import platform
import sys
import time

if not __debug__:
    raise RuntimeError("冻结验证禁止 python -O")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
from scipy import sparse

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from benchmark import protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_balanced import _smooth_temporal_basis
from candidates.oaster_conditional import observable_quality
from candidates.oaster_partial_confirmation import score_low_rank_deep_scan
from candidates.oaster_predictive import conformal_decision, fit_predictive_models
from refined_grid import load_refined_forward
import run_erp_whole_head_matrix as original

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--preflight", action="store_true", help="仅模拟60个calibration H0的质量；不做inverse、不创建输出")
args = parser.parse_args()
manifest_path = (args.manifest if args.manifest.is_absolute() else root / args.manifest).resolve()
output = (args.output if args.output.is_absolute() else root / args.output).resolve()
script_path = Path(__file__).resolve()
script_payload = script_path.read_bytes()
tree = ast.parse(script_payload.decode("utf-8"))
if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in ast.walk(tree)):
    raise RuntimeError("81必须是顺序式脚本，不定义def/class")
if script_payload.decode("utf-8").count("# %%") < 8:
    raise RuntimeError("81缺少作者风格分段")
alpha, minimum_per_bin = .05, 19
bins = ("LL", "LH", "HL", "HH")

# %% 2. manifest、旁车、冻结参数选择和原算法身份。
manifest_payload = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
if Path(str(manifest_path) + ".sha256").read_text(encoding="utf-8") != f"{manifest_sha}  {manifest_path.name}\n":
    raise ValueError("80 manifest哈希旁车不一致")
manifest = json.loads(manifest_payload)
cases = manifest["cases"]
inputs, rule = manifest["inputs"], manifest["threshold_rule"]
if manifest.get("schema_version") != 1 or manifest.get("protocol") != "erp-conditional-gate-pilot-manifest-v1" or \
        manifest.get("pilot") is not True or \
        manifest.get("formal_or_blind_validation") is not False or len(cases) != 117 or \
        [case["case_number"] for case in cases] != list(range(117)) or \
        len({case["case_id"] for case in cases}) != 117:
    raise ValueError("仅接受80冻结的117例pilot manifest")
if rule.get("alpha") != alpha or rule.get("minimum_calibration_per_bin") != minimum_per_bin or \
        rule.get("quality_threshold_linear") != 10. or rule.get("quality_threshold_db") != 10. or \
        rule.get("observable_bins") != list(bins) or rule.get("positive_gain_required") is not True:
    raise ValueError("条件bin/alpha/最小校准量/positive gain规则已变化")
if Counter(case["h0_role"] for case in cases) != {"gate_calibration": 60, "gate_audit": 30, None: 27}:
    raise ValueError("角色必须为60calibration H0、30audit H0、27H1")
for case in cases:
    configuration = int(case["configuration_number"])
    expected_role = "gate_calibration" if configuration < 20 else "gate_audit" if configuration < 30 else None
    if case["h0_role"] != expected_role or case["case_number"] != 39 * case["pair_index"] + configuration or \
            not 0 <= configuration < 39 or case["pair_index"] not in (0, 1, 2) or \
            case["replica_seed_roots"] != {key: manifest["seed_roots"][key] for key in ("fit", "check")} or \
            case["seed"] != [manifest["seed_roots"]["fit"], case["pair_index"], configuration] or \
            case["check_seed"] != [manifest["seed_roots"]["check"], case["pair_index"], configuration]:
        raise ValueError("逐例role/configuration/seed身份非法")
    if expected_role is not None and (case["deep_index"] is not None or case["scenario"] != "surface_only"):
        raise ValueError("H0校准或audit必须是预声明纯表层病例")
calibration_cases = [case for case in cases if case["h0_role"] == "gate_calibration"]
audit_cases = [case for case in cases if case["h0_role"] != "gate_calibration"]
for pair_index, pair in enumerate(((-10, -10), (-10, 20), (20, -10))):
    cell = [case for case in cases if case["pair_index"] == pair_index]
    if {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cell} != {pair} or \
            Counter(case["h0_role"] for case in cell) != {"gate_calibration": 20, "gate_audit": 10, None: 9}:
        raise ValueError("三个预声明SNR及每SNR角色数量变化")
generator_path = Path(manifest["generator"]["path"])
generator_sha = hashlib.sha256(generator_path.read_bytes()).hexdigest()
if generator_sha != manifest["generator"]["sha256"]:
    raise ValueError("80生成脚本与manifest的source hash不一致")
base_path = (root / "results/erp_whole_head/adaptive_v6/development_diagnosis"
             / "hierarchical_v3_lowrank_studentized_gate_development_mrf05/run_identity.json")
base_payload = base_path.read_bytes()
base_identity = json.loads(base_payload)
selection_path = root / base_identity["parameter_selection"]["path"]
selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
if selection_sha != "a3d669073325665786691a838643d8a955dc0744ce6985e79c0dd79a857a2ffa" or \
        base_identity["parameter_selection"]["sha256"] != selection_sha or \
        base_identity.get("parameter_profile") != "mrf05" or \
        base_identity.get("gate_mode") != "lowrank_studentized" or \
        base_identity["solver_settings"]["mrf_strength"] != .5:
    raise ValueError("必须沿用已冻结的MRF0.5 residual算法及参数来源")
settings = dict(base_identity["solver_settings"])
for relative, digest in base_identity["code_sha256"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
        raise ValueError(f"原冻结逆解/scorer/input代码已变化：{relative}")
for relative, digest in inputs["code_sha256"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
        raise ValueError(f"80冻结模拟/预处理源码已变化：{relative}")

# %% 3. 同一inverse、refined仿真forward及HEAD坐标；只复用已有模拟/预处理。
data_root = Path(inputs["inverse_root"])
sample_path = Path(inputs["sample_path"])
refined_path = Path(inputs["refined_forward"]["path"])
shared = protocol.load_shared(data_root, sample_path)
if original._shared_fingerprint(shared) != inputs["inverse_forward"]["fingerprint"] or \
        hashlib.sha256(refined_path.read_bytes()).hexdigest() != inputs["refined_forward"]["file_sha256"]:
    raise ValueError("80冻结forward身份不一致")
refined = load_refined_forward(refined_path)
n_surf, n_deep = int(shared["n_surf"]), int(shared["n_deep"])
n_truth_surf = int(np.asarray(refined["n_surf"]).ravel()[0])
n_truth_deep = int(np.asarray(refined["n_deep"]).ravel()[0])
n_truth = n_truth_surf + n_truth_deep
edges = np.asarray(refined["surface_edges"], int)
truth_graph = sparse.coo_matrix((np.ones(len(edges) * 2, np.uint8),
    (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
    shape=(n_truth, n_truth)).tocsr()
truth_shared = dict(shared)
truth_shared.update(gain_eeg=np.asarray(refined["gain_eeg"], float),
    gain_meg=np.asarray(refined["gain_meg"], float), vertices=np.asarray(refined["vertices"], float),
    adjacency=truth_graph, n_surf=n_truth_surf, n_deep=n_truth_deep)
if (n_surf, n_deep) != (inputs["inverse_forward"]["n_surface"], inputs["inverse_forward"]["n_deep"]) or \
        (n_truth_surf, n_truth_deep) != (inputs["refined_forward"]["n_surface"], inputs["refined_forward"]["n_deep"]):
    raise ValueError("forward source数量与manifest不一致")
if shared["info_eeg"]["ch_names"] != inputs["channel_order"]["eeg"] or \
        shared["info_meg"]["ch_names"] != inputs["channel_order"]["meg"]:
    raise ValueError("EEG/MEG通道顺序不一致")
code_files = (*base_identity["code_sha256"], "candidates/oaster_conditional.py")
identity = {
    "schema_version": 1, "protocol": "frozen-residual-conditional-gate-pilot",
    "pilot": True, "formal_or_blind_validation": False, "stage2_ran": False,
    "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
    "manifest_generator_path": str(generator_path), "manifest_generator_sha256": generator_sha,
    "base_identity_sha256": hashlib.sha256(base_payload).hexdigest(),
    "parameter_selection_sha256": selection_sha, "solver_settings": settings,
    "seed_roots": manifest["seed_roots"], "input_identity": inputs,
    "scorer": "residual self-normalized lowrank scan; same point directional min then masked max",
    "quality_threshold_linear": 10., "quality_threshold_db": 10., "observable_bins": list(bins),
    "alpha": alpha, "minimum_calibration_per_bin": minimum_per_bin,
    "p_rule": "(1+count(T_cal>=T))/(n_cal+1); ties conservative",
    "decision_rule": "p<=0.05 and T>0; unsupported conditional bin abstains with p=1",
    "script_sha256": hashlib.sha256(script_payload).hexdigest(),
    "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in code_files},
    "runtime": {"python": platform.python_version(), "executable": sys.executable,
                **{package: importlib.metadata.version(package) for package in ("numpy", "scipy", "mne")}},
    "limitations": ["pilot; no formal error guarantee", "geometry repeated across SNR creates dependent cases",
                    "at least 19 per observable bin gives rank resolution, not a verified 5% population FPR",
                    "unsupported bins must abstain; no source-map An_auc/SD/DLE without Stage2"],
}
identity_payload = (json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
identity_sha = hashlib.sha256(identity_payload).hexdigest()

# %% 4. 便宜calibration预检先于inverse；不看57例audit/H1质量或标签。
if output.exists() and any(output.rglob("*.json.tmp")):
    raise RuntimeError("输出残留未封存.json.tmp；先核验并移走临时文件，再resume，禁止覆盖已冻结JSON")
preview = {}
for case in calibration_cases:
    observation = prepare_trial_covariance_case(shared, case,
        seed_root=manifest["seed_roots"]["fit"], truth_shared=truth_shared)
    if observation["metadata"]["replica_seed_roots"] != case["replica_seed_roots"] or \
            observation["metadata"]["n_trials_per_half"] != 20 or \
            observation["metadata"]["truth_source_count"] != n_truth or \
            observation["metadata"]["inverse_source_count"] != n_surf + n_deep:
        raise ValueError("preflight模拟输入/独立半均值身份非法")
    data_a, data_b = observation["training"], observation["confirmation"]
    baseline, active = observation["baseline"], observation["active_windows"][0]
    basis = _smooth_temporal_basis(data_a, baseline, active)[0]
    quality = observable_quality(data_a, data_b, baseline, basis,
                                 tuple(map(int, observation["metadata"]["retained_channels"])))
    preview[int(case["case_number"])] = quality
    del observation, data_a, data_b
preview_counts = Counter(item["observable_bin"] for item in preview.values())
if any(count < minimum_per_bin for count in preview_counts.values()):
    raise ValueError(f"occupied observable bin校准不足19；禁止inverse或自动改bin：{dict(preview_counts)}")
if args.preflight:
    assert conformal_decision(1., np.zeros(19), alpha)["deep_present"]
    assert not conformal_decision(1., np.ones(19), alpha)["deep_present"]
    assert not conformal_decision(-1., np.full(19, -2.), alpha)["deep_present"]
    assert int(np.argmax(np.minimum([9., 3.], [-1., 3.]))) == 1
    print(json.dumps({"preflight": "PASS", "pilot": True, "formal_or_blind_validation": False,
        "manifest_sha256": manifest_sha, "run_identity_sha256": identity_sha,
        "solver_settings": settings, "calibration_sensor_simulations": 60, "inverse_calls": 0,
        "occupied_calibration_bin_counts": dict(preview_counts), "minimum_per_occupied_bin": minimum_per_bin,
        "unsupported_bins": [name for name in bins if name not in preview_counts],
        "inverse_calls_planned": 234, "case_count": 117, "output_created": False,
        "parameter_selection_sha256": selection_sha, "calibration_quality": preview},
        ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    raise SystemExit(0)

# %% 5. 唯一进程文件锁、身份断点与原子小型JSON；校准先全部完成。
if output.exists():
    if not (output / "run_identity.json").is_file() or (output / "run_identity.json").read_bytes() != identity_payload:
        raise ValueError("输出已有其他身份或不完整初始化；拒绝覆盖")
else:
    output.mkdir(parents=True)
    with (output / "run_identity.json").open("xb") as stream:
        stream.write(identity_payload)
with (output / ".run.lock").open("a+b") as lock:
    if lock.tell() == 0:
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise RuntimeError("该输出已有运行进程；拒绝重复实验") from error
    for name in ("calibration_scores", "audit_scores", "decisions"):
        (output / name).mkdir(exist_ok=True)
    score_hashes = {}
    records = {}
    bank = None
    bank_sha = None
    for phase, phase_cases in (("calibration", calibration_cases), ("audit", audit_cases)):
        for simulation_case in phase_cases:
            number, name = int(simulation_case["case_number"]), str(simulation_case["case_id"])
            path = output / f"{phase}_scores" / f"case_{number:03d}.json"
            new_score = not path.is_file()
            if not new_score:
                payload = path.read_bytes()
                record = json.loads(payload)
                if record.get("complete") is not True or record.get("identity_sha256") != identity_sha or \
                        record.get("case_number") != number or record.get("case_id") != name or \
                        record.get("truth_used_by_score") is not False or record.get("fit_count") != 2 or \
                        len(record.get("directions", [])) != 2:
                    raise ValueError(f"{name}: score checkpoint身份不一致")
                for direction in record["directions"]:
                    solver = direction["H0_fit"]["windows"][0]["solver"]
                    if not (solver["converged"] and solver["outer_converged"] and solver["final_inner_converged"]
                            and solver["final_stationarity_gap_relative"] <= settings["tolerance"] + 1e-12):
                        raise ValueError(f"{name}: checkpoint H0未严格收敛")
            else:
                tick = time.perf_counter()
                observation = prepare_trial_covariance_case(shared, simulation_case,
                    seed_root=manifest["seed_roots"]["fit"], truth_shared=truth_shared)
                if observation["metadata"]["replica_seed_roots"] != simulation_case["replica_seed_roots"] or \
                        observation["metadata"]["n_trials_per_half"] != 20 or \
                        observation["metadata"]["truth_source_count"] != n_truth or \
                        observation["metadata"]["inverse_source_count"] != n_surf + n_deep:
                    raise ValueError(f"{name}: 模拟输入/独立半均值身份非法")
                blind = {"A": np.asarray(observation["training"], float),
                         "B": np.asarray(observation["confirmation"], float),
                         "gain": np.asarray(observation["gain"], float),
                         "baseline": np.asarray(observation["baseline"], bool),
                         "active": np.asarray(observation["active_windows"][0], bool),
                         "modality_sizes": tuple(map(int, observation["metadata"]["retained_channels"]))}
                del observation
                basis = _smooth_temporal_basis(blind["A"], blind["baseline"], blind["active"])[0]
                quality = observable_quality(blind["A"], blind["B"], blind["baseline"], basis, blind["modality_sizes"])
                weights = np.ones(blind["gain"].shape[0], float)
                directions = []
                for fit_half, heldout_half in (("A", "B"), ("B", "A")):
                    null, full, fitting = fit_predictive_models(blind[fit_half], blind["gain"], n_surf,
                        adjacency=shared["adjacency"], baseline=blind["baseline"], active=blind["active"],
                        channel_weights=weights, solver_settings=settings, fit_full=False)
                    if full is not None or null.shape != (n_surf + n_deep, blind[fit_half].shape[1]) or \
                            np.any(null[n_surf:] != 0) or fitting["fitted_families"] != ["H0"]:
                        raise RuntimeError(f"{name}: H0-only输出source轴或family非法")
                    solver = fitting["null_model"]["windows"][0]["solver"]
                    if not (solver["converged"] and solver["outer_converged"] and solver["final_inner_converged"]
                            and solver["final_stationarity_gap_relative"] <= settings["tolerance"] + 1e-12):
                        raise RuntimeError(f"{name} {fit_half}_to_{heldout_half}: H0未严格收敛")
                    scores, info = score_low_rank_deep_scan(blind[heldout_half], blind["gain"], null, n_surf,
                        baseline=blind["baseline"], active=blind["active"], channel_weights=weights,
                        score_mode="studentized")
                    # 不重复保存7498维固定floor向量；完整solver history保留，设置从run identity重建。
                    fit_summary = {key: value for key, value in fitting["null_model"].items()
                                   if key != "solver_settings"}
                    fit_summary.update(solver_settings=settings, fitted_source_count=n_surf)
                    directions.append({"fit_half": fit_half, "heldout_half": heldout_half,
                        "deep_scores": scores.tolist(), "stage1_samepoint_residual": info,
                        "H0_fit": fit_summary})
                same_scores = np.minimum(directions[0]["deep_scores"], directions[1]["deep_scores"])
                joint = np.logical_and(directions[0]["stage1_samepoint_residual"]["identifiable"],
                                       directions[1]["stage1_samepoint_residual"]["identifiable"])
                valid = np.flatnonzero(joint)
                selected = int(valid[np.argmax(same_scores[valid])]) if valid.size else None
                record = {"complete": True, "identity_sha256": identity_sha,
                    "case_number": number, "case_id": name, "truth_used_by_score": False,
                    "fit_count": 2, "quality": quality, "directions": directions,
                    "same_deep_bidirectional_scores": same_scores.tolist(), "jointly_identifiable": joint.tolist(),
                    "selected_deep_local_index": selected,
                    "gate_statistic": -1. if selected is None else float(same_scores[selected]),
                    "elapsed_seconds": float(time.perf_counter() - tick),
                    "observation_shape": list(blind["A"].shape),
                    "modality_sizes": list(blind["modality_sizes"]),
                    "baseline_sample_count": int(blind["baseline"].sum()),
                    "active_sample_count": int(blind["active"].sum()),
                    "weighted_deep_gain_norm": np.linalg.norm(blind["gain"][:, n_surf:], axis=0).tolist(),
                    "sensor_sha256": {half: hashlib.sha256(np.ascontiguousarray(blind[half]).tobytes()).hexdigest()
                                      for half in ("A", "B")}}
                del blind, null, full, fitting
                payload = (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
                    default=lambda value: value.item() if isinstance(value, np.generic) else value.tolist()) + "\n").encode()
            quality = record["quality"]
            if quality.get("truth_used") is not False or quality.get("threshold_q") != 10. or \
                    quality.get("observable_bin") != "".join("H" if quality[key] >= 10. else "L" for key in ("q_eeg", "q_meg")) or \
                    not np.isfinite([quality["q_eeg"], quality["q_meg"]]).all() or \
                    (phase == "calibration" and quality != preview[number]):
                raise ValueError(f"{name}: quality checkpoint与冻结sensor规则不一致")
            sensor_count, time_count = record["observation_shape"]
            baseline_count = record["baseline_sample_count"]
            deep_norm = np.asarray(record["weighted_deep_gain_norm"], float)
            if sum(record["modality_sizes"]) != sensor_count or baseline_count <= 3 or \
                    deep_norm.shape != (n_deep,) or not np.isfinite(deep_norm).all() or np.any(deep_norm < 0):
                raise ValueError(f"{name}: checkpoint sensor/gain轴非法")
            for index, direction in enumerate(record["directions"]):
                info = direction["stage1_samepoint_residual"]
                fit = direction["H0_fit"]
                rank, nuisance_rank = int(info["temporal_rank"]), int(info["nuisance_rank"])
                fraction = np.asarray(info["conditional_gain_fraction"], float)
                tolerance = np.finfo(float).eps * max(sensor_count, rank)
                identifiable = (deep_norm > 0) & (fraction > tolerance)
                arrays = {key: np.asarray(info[key], float) for key in (
                    "loss_improvement", "expected_null_improvement", "projected_baseline_variance",
                    "residual_after_deep", "residual_expected_null_improvement", "studentized_denominator")}
                if direction["fit_half"] != ("A", "B")[index] or direction["heldout_half"] != ("B", "A")[index] or \
                        fit["fitted_source_count"] != n_surf or fit["solver_settings"] != settings or \
                        fit["mrf_strength"] != .5 or fit["solver_kind"] != "admm" or \
                        fit["windows"][0]["active_samples"] != record["active_sample_count"] or \
                        fraction.shape != (n_deep,) or not np.isfinite(fraction).all() or \
                        np.any((fraction < 0) | (fraction > 1 + 1e-12)) or not 0 <= nuisance_rank <= rank or \
                        any(value.shape != (n_deep,) or not np.isfinite(value).all() for value in arrays.values()) or \
                        info["identifiable"] != identifiable.tolist() or \
                        info["conditional_gain_fraction_threshold"] != tolerance or \
                        info["effective_weighted_sensor_count"] != sensor_count:
                    raise ValueError(f"{name}: per-direction mask/H0 identity非法")
                improvement, expected = arrays["loss_improvement"], arrays["expected_null_improvement"]
                total = float(info["total_null_residual"])
                dof = max(sensor_count - nuisance_rank - 1, 0)
                residual = np.maximum(total - improvement, 0.)
                residual_expectation = residual / dof if dof > 0 else np.zeros(n_deep)
                denominator = np.maximum(expected, residual_expectation)
                correction = (baseline_count - 3) / (baseline_count - 1)
                predicted = np.full(n_deep, -1., float)
                if not np.isfinite(total) or total < 0 or np.any(improvement < 0) or \
                        np.any(improvement > total + 64 * np.finfo(float).eps * max(total, 1.)) or \
                        np.any(denominator[identifiable] <= 0):
                    raise ValueError(f"{name}: conditional improvement/residual非法")
                predicted[identifiable] = correction * improvement[identifiable] / denominator[identifiable] - 1.
                if info["score_mode"] != "studentized" or info["truth_used"] is not False or \
                        info["residual_sensor_dof"] != dof or info["finite_baseline_correction"] != correction or \
                        not np.array_equal(expected, arrays["projected_baseline_variance"] *
                            (rank + info["baseline_mean_mode_correction"])) or \
                        not np.array_equal(arrays["residual_after_deep"], residual) or \
                        not np.array_equal(arrays["residual_expected_null_improvement"], residual_expectation) or \
                        not np.array_equal(arrays["studentized_denominator"], denominator) or \
                        not np.array_equal(np.asarray(direction["deep_scores"], float), predicted):
                    raise ValueError(f"{name}: per-direction residual score公式不一致")
            deep_scores = np.asarray([item["deep_scores"] for item in record["directions"]], float)
            joint = np.logical_and(record["directions"][0]["stage1_samepoint_residual"]["identifiable"],
                                   record["directions"][1]["stage1_samepoint_residual"]["identifiable"])
            same_scores = np.minimum(deep_scores[0], deep_scores[1])
            valid = np.flatnonzero(joint)
            selected = int(valid[np.argmax(same_scores[valid])]) if valid.size else None
            expected_statistic = -1. if selected is None else float(same_scores[selected])
            if deep_scores.shape != (2, n_deep) or not np.isfinite(deep_scores).all() or \
                    record["same_deep_bidirectional_scores"] != same_scores.tolist() or \
                    record["jointly_identifiable"] != joint.tolist() or \
                    record["selected_deep_local_index"] != selected or record["gate_statistic"] != expected_statistic or \
                    any(item["deep_scores"] != item["stage1_samepoint_residual"]["studentized_deep_scores"]
                        for item in record["directions"]):
                raise ValueError(f"{name}: 同点双向min/masked max checkpoint错误")
            if new_score:
                temporary = path.with_suffix(".json.tmp")
                with temporary.open("xb") as stream:
                    stream.write(payload)
                os.replace(temporary, path)
            score_hashes[number] = hashlib.sha256(payload).hexdigest()
            records[number] = record
            print(phase, name, f"bin={quality['observable_bin']} T={record['gate_statistic']:.6g}", flush=True)

# %% 6. 同一60cal库冻结后才为57audit计算两种p；不足19的bin必须abstain。
            if phase == "audit":
                statistic = record["gate_statistic"]
                observed_bin = quality["observable_bin"]
                global_decision = conformal_decision(statistic, bank["global_scores"], alpha)
                global_decision.update(unsupported=False)
                conditional_scores = bank["conditional_scores"].get(observed_bin, [])
                conditional_decision = (conformal_decision(statistic, conditional_scores, alpha)
                    if len(conditional_scores) >= minimum_per_bin else
                    {"p_value": 1., "deep_present": False, "unsupported": True, "alpha": alpha,
                     "n_calibration": len(conditional_scores), "score": statistic,
                     "positive_gain_required": True, "reason": "unsupported observable bin; abstain"})
                conditional_decision.setdefault("unsupported", False)
                decision = {"complete": True, "identity_sha256": identity_sha,
                    "bank_sha256": bank_sha, "score_sha256": score_hashes[number],
                    "case_number": number, "case_id": name, "observable_bin": observed_bin,
                    "gate_statistic": statistic, "selected_deep_local_index": selected,
                    "global": global_decision, "conditional": conditional_decision,
                    "truth_used_for_decision": False}
                decision_path = output / "decisions" / f"case_{number:03d}.json"
                decision_payload = (json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                if decision_path.is_file():
                    if decision_path.read_bytes() != decision_payload:
                        raise ValueError(f"{name}: 已冻结decision被修改或bank不一致")
                else:
                    temporary = decision_path.with_suffix(".json.tmp")
                    with temporary.open("xb") as stream:
                        stream.write(decision_payload)
                    os.replace(temporary, decision_path)
        if phase == "calibration":
            bank = {"complete": True, "identity_sha256": identity_sha, "pilot": True,
                "alpha": alpha, "minimum_per_bin": minimum_per_bin,
                "global_scores": [records[case["case_number"]]["gate_statistic"] for case in calibration_cases],
                "conditional_scores": {name: [records[case["case_number"]]["gate_statistic"]
                    for case in calibration_cases if records[case["case_number"]]["quality"]["observable_bin"] == name]
                    for name in bins},
                "calibration_score_sha256": {str(case["case_number"]): score_hashes[case["case_number"]]
                                             for case in calibration_cases},
                "same_calibration_library_for_both_modes": True,
                "audit_or_H1_used": False, "unsupported_rule": "p=1; no deep report; abstain, not supported absence"}
            if len(bank["global_scores"]) != 60 or any(0 < len(scores) < minimum_per_bin
                    for scores in bank["conditional_scores"].values()):
                raise ValueError("calibration bank incomplete或occupied bin不足19")
            # 同一个helper固定rank/并列值/positive-gain约束，不另写分类公式。
            conformal_decision(0., bank["global_scores"], alpha)
            for scores in bank["conditional_scores"].values():
                if scores:
                    conformal_decision(0., scores, alpha)
            bank_payload = (json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            bank_path = output / "calibration_bank.json"
            if bank_path.is_file():
                if bank_path.read_bytes() != bank_payload:
                    raise ValueError("已冻结calibration bank与当前60个score不一致")
            else:
                temporary = bank_path.with_suffix(".json.tmp")
                with temporary.open("xb") as stream:
                    stream.write(bank_payload)
                os.replace(temporary, bank_path)
            bank_sha = hashlib.sha256(bank_payload).hexdigest()
            print("60 calibration scores and bank frozen", bank_sha, flush=True)

# %% 7. 57个decision全部冻结后解封真值，只报告presence与粗深点诊断。
    if len(records) != 117 or len(list((output / "decisions").glob("case_*.json"))) != 57:
        raise RuntimeError("必须先完整冻结117个score和57个decision才能解封诊断")
    fine_head, coarse_deep_head = np.asarray(refined["vertices"], float), np.asarray(shared["deep_rr"], float)
    rows = []
    for case in audit_cases:
        number = int(case["case_number"])
        decision_path = output / "decisions" / f"case_{number:03d}.json"
        decision = json.loads(decision_path.read_bytes())
        record = records[number]
        if decision["bank_sha256"] != bank_sha or decision["score_sha256"] != score_hashes[number]:
            raise ValueError("truth解封前decision/bank/score哈希不一致")
        local = decision["selected_deep_local_index"]
        has_deep = case["deep_index"] is not None
        distance = (None if local is None or not has_deep else float(
            np.linalg.norm(coarse_deep_head[local] - fine_head[int(case["deep_index"])]) * 1000.))
        rows.append({"case_number": number, "case_id": case["case_id"], "scenario": case["scenario"],
            "h0_role": case["h0_role"], "has_deep_true": has_deep,
            "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
            "observable_bin": decision["observable_bin"], "gate_statistic": decision["gate_statistic"],
            "global": decision["global"], "conditional": decision["conditional"],
            "selected_deep_local_index": local, "coarse_deep_candidate_error_mm_HEAD": distance,
            "elapsed_seconds": record["elapsed_seconds"], "fit_count": 2,
            "score_sha256": score_hashes[number],
            "decision_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest()})
    groups = {"overall": rows}
    for pair in sorted({(row["eeg_snr_db"], row["meg_snr_db"]) for row in rows}):
        groups[f"eeg{pair[0]:+g}_meg{pair[1]:+g}"] = [row for row in rows
            if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    for name in bins:
        groups[f"observable_{name}"] = [row for row in rows if row["observable_bin"] == name]
    for scenario in ("deep_only", "deep_plus_surface", "deep_plus_two_surface"):
        groups[scenario] = [row for row in rows if row["scenario"] == scenario]
    comparison = {}
    for group_name, group in groups.items():
        h0 = [row for row in group if not row["has_deep_true"]]
        h1 = [row for row in group if row["has_deep_true"]]
        comparison[group_name] = {}
        for mode in ("global", "conditional"):
            supported_h0 = [row for row in h0 if not row[mode]["unsupported"]]
            supported_h1 = [row for row in h1 if not row[mode]["unsupported"]]
            auroc = None
            if supported_h0 and supported_h1:
                delta = np.asarray([1 - row[mode]["p_value"] for row in supported_h1])[:, None] - \
                        np.asarray([1 - row[mode]["p_value"] for row in supported_h0])[None, :]
                auroc = float(np.mean((delta > 0) + .5 * (delta == 0)))
            detected_distance = [row["coarse_deep_candidate_error_mm_HEAD"] for row in h1
                if row[mode]["deep_present"] and row["coarse_deep_candidate_error_mm_HEAD"] is not None]
            comparison[group_name][mode] = {
                "audit_H0_count": len(h0), "H1_count": len(h1),
                "audit_H0_false_positive_count": sum(row[mode]["deep_present"] for row in h0),
                "H1_detection_count": sum(row[mode]["deep_present"] for row in h1),
                "unsupported_abstention_count": sum(row[mode]["unsupported"] for row in group),
                "supported_H0_count": len(supported_h0), "supported_H1_count": len(supported_h1),
                "unsupported_H0_count": len(h0) - len(supported_h0),
                "unsupported_H1_count": len(h1) - len(supported_h1),
                "H0_negative_accepted_count": sum(not row[mode]["deep_present"] for row in supported_h0),
                "H1_negative_accepted_count": sum(not row[mode]["deep_present"] for row in supported_h1),
                "supported_H0_false_positive_rate": (sum(row[mode]["deep_present"] for row in supported_h0)
                    / len(supported_h0) if supported_h0 else None),
                "supported_H1_detection_rate": (sum(row[mode]["deep_present"] for row in supported_h1)
                    / len(supported_h1) if supported_h1 else None),
                "supported_only_gate_AUROC_from_1_minus_p": auroc,
                "detected_and_coarse_candidate_within10mm_count": sum(row[mode]["deep_present"]
                    and row["coarse_deep_candidate_error_mm_HEAD"] is not None
                    and row["coarse_deep_candidate_error_mm_HEAD"] <= 10. + 1e-6 for row in h1),
                "median_detected_coarse_candidate_error_mm": (
                    float(np.median(detected_distance)) if detected_distance else None)}

# %% 8. 原子冻结可审计汇总；两种决定共享所有拟合和校准病例。
    summary = {"complete": True, "pilot": True, "formal_or_blind_validation": False, "stage2_ran": False,
        "identity_sha256": identity_sha, "calibration_bank_sha256": bank_sha,
        "case_count": 117, "calibration_H0_count": 60, "audit_H0_count": 30, "H1_count": 27,
        "alpha": alpha, "same_calibration_library": True,
        "calibration_bin_counts": {name: len(scores) for name, scores in bank["conditional_scores"].items()},
        "all_234_H0_fits_strictly_converged": True,
        "inverse_fit_count": 234, "sum_case_elapsed_seconds": sum(r["elapsed_seconds"] for r in records.values()),
        "median_case_seconds": float(np.median([r["elapsed_seconds"] for r in records.values()])),
        "maximum_case_seconds": max(r["elapsed_seconds"] for r in records.values()),
        "comparison": comparison, "rows": rows,
        "truth_unsealed_after_all_audit_decisions_frozen": True,
        "metrics_scope": "presence rank AUROC and coarse candidate diagnostic; no final source-map An_auc/SD/DLE",
        "limitations": identity["limitations"]}
    summary_payload = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    summary_path = output / "conditional_gate_summary.json"
    if summary_path.is_file():
        if summary_path.read_bytes() != summary_payload:
            raise ValueError("已完成summary与当前冻结记录不一致；拒绝覆盖")
    else:
        temporary = summary_path.with_suffix(".json.tmp")
        with temporary.open("xb") as stream:
            stream.write(summary_payload)
        os.replace(temporary, summary_path)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
