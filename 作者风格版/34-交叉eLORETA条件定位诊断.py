"""开发诊断：正确 H0/H1 家族给定后，用交叉 eLORETA 替换皮层定位图。"""

# %% 固定开发病例、路径和唯一算法。
import csv
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
from scipy import sparse
from scipy.sparse.linalg import splu
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from benchmark.methods import minimum_norm_family
import run_erp_whole_head_matrix as original

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_sensor_balanced_m10_m10.json"
source_surface = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases00_01"
source_deep = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/cross_eloreta_conditional_sensor_balanced_m10_m10"
seed_root = 2026092206
mrf_strength = .8
deep_neighbors = 3
maximum_dle_worsening_mm = 5.
if output.exists():
    raise FileExistsError(f"不覆盖已有诊断：{output}")
cases = json.loads(manifest.read_text(encoding="utf-8"))
assert len(cases) == 5 and [case["configuration_number"] for case in cases] == list(range(5))
assert all(case["panel"] == "erp_v6_development" for case in cases)
assert all(case["eeg_snr_db"] == case["meg_snr_db"] == -10 for case in cases)

# %% 冻结代码与源结果指纹；没有 calibration/validation 输入。
paths = [Path(__file__), *[root / name for name in (
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py", "benchmark/erp_protocol.py",
    "benchmark/protocol.py", "benchmark/methods.py", "benchmark/metrics.py",
    "run_erp_whole_head_matrix.py", "run_strict_oaster.py", "protected_multilayer.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
source_files = {}
for case in cases:
    source = source_surface if case["configuration_number"] < 2 else source_deep
    source_files[case["case_id"]] = source
    for suffix in (".json", ".npz"):
        assert (source / (case["case_id"] + suffix)).is_file()
for source in (source_surface, source_deep):
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["phase"] == "development" and metadata["covariance"] == "trial"
    assert metadata["solver_settings"] == {"solver_kind": "admm", "mrf_strength": mrf_strength}

# %% 固定表层解剖扩散与深层 3NN 扩散，均为 MRF=0.8。
graph = sparse.csr_matrix(shared["adjacency"])
surface_graph = graph[:n_surf, :n_surf].maximum(graph[:n_surf, :n_surf].T)
surface_degree = np.asarray(surface_graph.sum(axis=1)).ravel()
surface_transition = sparse.diags(1 / np.maximum(surface_degree, 1)) @ surface_graph
surface_factor = splu((sparse.eye(n_surf) - mrf_strength * surface_transition).tocsc())
deep_vertices = np.asarray(shared["vertices"], float)[n_surf:]
deep_distance = np.linalg.norm(deep_vertices[:, None] - deep_vertices[None, :], axis=2)
deep_graph = np.zeros(deep_distance.shape, float)
for index in range(len(deep_vertices)):
    deep_graph[index, np.argsort(deep_distance[index])[1:deep_neighbors + 1]] = 1
deep_graph = np.maximum(deep_graph, deep_graph.T)
deep_degree = deep_graph.sum(axis=1)
deep_transition = sparse.diags(1 / np.maximum(deep_degree, 1)) @ sparse.csr_matrix(deep_graph)
deep_factor = splu((sparse.eye(len(deep_vertices)) - mrf_strength * deep_transition).tocsc())

output.mkdir(parents=True)
metadata = {
    "complete": False, "phase": "development", "diagnosis": "conditional_cross_eloreta_localization",
    "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "case_ids": [case["case_id"] for case in cases], "seed_root": seed_root,
    "conditional_family_rule": "case00/01 use saved H0; case02/03/04 use saved H1",
    "blind_detection_result": False,
    "surface_rule": "positive train/check eLORETA active-window cross-energy, then anatomical random-walk resolvent MRF",
    "surface_peak_normalization": "preserve selected OASTER map maximum active-window L2 amplitude",
    "deep_rule": "saved OASTER H1 deep block followed by fixed symmetric Euclidean 3NN random-walk resolvent MRF",
    "mrf_strength": mrf_strength, "deep_neighbors": deep_neighbors,
    "minimum_An_cal_AUC_each_case": .9, "maximum_DLE_worsening_mm": maximum_dle_worsening_mm,
    "candidate_grid_or_fusion_weight_search": False, "calibration_or_validation_read": False,
    "truth_used_for_family_selection": True, "truth_used_for_spatial_map_or_weight_selection": False,
    "code_sha256": code_hashes, "shared_fingerprint": original._shared_fingerprint(shared),
    "source_sha256": {case["case_id"]: {
        suffix: hashlib.sha256((source_files[case["case_id"]] / (case["case_id"] + suffix)).read_bytes()).hexdigest()
        for suffix in (".json", ".npz")} for case in cases}}
(output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 一次性计算五例；真值只决定题目指定的 H0/H1 家族并用于事后指标。
rows = []
started = time.perf_counter()
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    case_id = case["case_id"]
    family = "H0_surface_only" if case["deep_index"] is None else "H1_surface_deep"
    source = source_files[case_id]
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    with np.load(source / (case_id + ".npz")) as saved:
        baseline_estimate = np.asarray(saved["null" if family.startswith("H0") else "full"], float)
        assert np.allclose(saved["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    weights = np.asarray(observation["channel_weights"], float)
    width = observation["training"].shape[1]
    surface_gain = weights[:, None] * observation["gain"][:, :n_surf]
    both = np.hstack((weights[:, None] * observation["training"],
                      weights[:, None] * observation["confirmation"]))
    elo = minimum_norm_family(both, surface_gain)["eLORETA"]
    train_elo, check_elo = elo[:, :width], elo[:, width:]
    active = observation["active"]
    cross_energy = np.maximum(np.sum(train_elo[:, active] * check_elo[:, active], axis=1), 0)
    surface_map = np.maximum(surface_factor.solve(cross_energy), 0)

    diagnostic = baseline_estimate.copy()
    old_surface_peak = float(np.sqrt(np.sum(baseline_estimate[:n_surf, active] ** 2, axis=1)).max(initial=0))
    diagnostic[:n_surf] = 0
    if old_surface_peak > 0 and surface_map.max(initial=0) > 0:
        diagnostic[:n_surf, int(active[0])] = surface_map / surface_map.max() * old_surface_peak
    if family.startswith("H0"):
        diagnostic[n_surf:] = 0
    else:
        diagnostic[n_surf:] = deep_factor.solve(baseline_estimate[n_surf:])

    identity = {key: case[key] for key in
                ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    for method, estimate in (("selected_OASTER_baseline", baseline_estimate),
                             ("conditional_cross_eLORETA", diagnostic)):
        values = metrics.evaluate_estimate(
            estimate, observation["truth"], shared["vertices"], observation["groups"],
            n_surf, observation["active"], shared["auc_cortex"], baseline=observation["baseline"])
        rows.append({**identity, "conditional_family": family, "method": method,
            "An_cal_AUC": values["auc"], "An_auc": values["auc_tie_corrected"],
            "surface_An_auc": values["surface_auc_tie_corrected"],
            "deep_An_auc": values["deep_auc_tie_corrected"],
            "surface_SD_mm": values["surface_sd_mm"], "deep_SD_mm": values["deep_sd_mm"],
            "surface_DLE_mm": values["surface_dle_mm"], "deep_DLE_mm": values["deep_dle_mm"],
            "deep_amplitude_ratio": values["deep_score"],
            "deep_peak_distance_mm": values["deep_peak_distance_mm"]})
    np.savez_compressed(output / (case_id + ".npz"),
        truth=observation["truth"].astype(np.float32),
        conditional_estimate=diagnostic.astype(np.float32),
        surface_cross_mrf=surface_map.astype(np.float32))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = rows[-1]
    print(f"{case_id}: family={family}, An_cal={result['An_cal_AUC']:.6f}, "
          f"surface_DLE={result['surface_DLE_mm']}, deep_DLE={result['deep_DLE_mm']}, "
          f"seconds={time.perf_counter() - tick:.1f}", flush=True)

# %% 预先固定验收：五例 AUC 均达 0.9，且有真值的分层 DLE 有限并不恶化超过 5 mm。
diagnostic_rows = [row for row in rows if row["method"] == "conditional_cross_eLORETA"]
all_auc_pass = all(np.isfinite(row["An_cal_AUC"]) and row["An_cal_AUC"] >= .9 for row in diagnostic_rows)
dle_checks = []
for case in cases:
    baseline_row = next(row for row in rows if row["case_id"] == case["case_id"] and row["method"] == "selected_OASTER_baseline")
    diagnostic_row = next(row for row in rows if row["case_id"] == case["case_id"] and row["method"] == "conditional_cross_eLORETA")
    for layer, required in (("surface", bool(case["surface_centers"])), ("deep", case["deep_index"] is not None)):
        if not required:
            continue
        before, after = baseline_row[layer + "_DLE_mm"], diagnostic_row[layer + "_DLE_mm"]
        finite = bool(np.isfinite(after))
        worsening = float(after - before) if finite and np.isfinite(before) else None
        passed = finite and (worsening is None or worsening <= maximum_dle_worsening_mm)
        dle_checks.append({"case_id": case["case_id"], "layer": layer,
                           "baseline_DLE_mm": None if not np.isfinite(before) else before,
                           "diagnostic_DLE_mm": None if not finite else after,
                           "worsening_mm": worsening, "passed": passed})
all_dle_pass = all(item["passed"] for item in dle_checks)
accepted = all_auc_pass and all_dle_pass
completion = {"complete": True, "case_count": len(cases), "conditional_not_blind": True,
    "all_An_cal_AUC_at_least_0.9": all_auc_pass, "all_required_DLE_checks_pass": all_dle_pass,
    "accepted": accepted, "decision": "accept" if accepted else "reject",
    "dle_checks": dle_checks, "wall_seconds": time.perf_counter() - started}
(output / "completion.json").write_text(json.dumps(completion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(output / "metadata.json").write_text(json.dumps({**metadata, **completion}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = ["# 交叉 eLORETA 条件定位开发诊断", "",
    "这不是盲检测结果：预先给定正确模型家族，case00/01 使用 H0，case02/03/04 使用 H1。算法没有读取 calibration/validation，没有候选网格，也没有按真值选择融合权重。",
    "", "| 病例 | 家族 | 方法 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) |",
    "|---|---|---|---:|---:|---:|---:|---:|"]
for row in rows:
    show = lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
    lines.append(f"| {row['configuration_id']} | {row['conditional_family']} | {row['method']} | "
        f"{show(row['An_cal_AUC'])} | {show(row['An_auc'])} | {show(row['surface_An_auc'])}/{show(row['deep_An_auc'])} | "
        f"{show(row['surface_SD_mm'])}/{show(row['deep_SD_mm'])} | {show(row['surface_DLE_mm'])}/{show(row['deep_DLE_mm'])} |")
lines.extend(["", "## 固定验收", "", f"**{completion['decision'].upper()}**。要求五例 An_cal AUC 均不低于 0.90；每个有真值的层 DLE 必须有限，且相对原 OASTER 不恶化超过 {maximum_dle_worsening_mm:.1f} mm。",
    "", f"- 五例 AUC 门槛：{'通过' if all_auc_pass else '未通过'}",
    f"- 分层 DLE 门槛：{'通过' if all_dle_pass else '未通过'}", ""])
for item in dle_checks:
    lines.append(f"- {item['case_id']} {item['layer']}: baseline={item['baseline_DLE_mm']}, "
                 f"diagnostic={item['diagnostic_DLE_mm']}, worsening={item['worsening_mm']}, passed={item['passed']}")
lines.extend(["", "即使通过，这也只说明给定正确 H0/H1 家族后的条件定位能力，不能作为深源盲检成功率或假阳性率。"])
(output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
