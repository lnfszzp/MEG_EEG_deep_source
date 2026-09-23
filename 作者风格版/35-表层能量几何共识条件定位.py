"""开发诊断：正确 H0/H1 家族给定后，固定融合交叉 eLORETA 与 OASTER 表层能量。"""

# %% 固定路径、病例和唯一共识公式。
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
import run_erp_whole_head_matrix as original

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_sensor_balanced_m10_m10.json"
source_surface = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases00_01"
source_deep = root / "results/erp_whole_head/adaptive_v6/dev_sensor_balanced_admm_mrf080_m10_m10_cases02_04"
cross_source = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/cross_eloreta_conditional_sensor_balanced_m10_m10"
output = root / "results/erp_whole_head/adaptive_v6/development_diagnosis/geometric_consensus_conditional_sensor_balanced_m10_m10"
seed_root = 2026092206
maximum_dle_worsening_mm = 5.
if output.exists():
    raise FileExistsError(f"不覆盖已有诊断：{output}")
cases = json.loads(manifest.read_text(encoding="utf-8"))
assert len(cases) == 5 and [case["configuration_number"] for case in cases] == list(range(5))
assert all(case["panel"] == "erp_v6_development" for case in cases)
assert all(case["eeg_snr_db"] == case["meg_snr_db"] == -10 for case in cases)
cross_metadata = json.loads((cross_source / "metadata.json").read_text(encoding="utf-8"))
assert cross_metadata["complete"] and cross_metadata["phase"] == "development"
assert cross_metadata["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
assert not cross_metadata["calibration_or_validation_read"]

# %% 冻结输入和代码指纹；没有候选、权重或深层扩散参数。
paths = [Path(__file__), *[root / name for name in (
    "benchmark/erp_trial_covariance.py", "benchmark/erp_replicates.py", "benchmark/erp_protocol.py",
    "benchmark/protocol.py", "benchmark/metrics.py", "run_erp_whole_head_matrix.py",
    "run_strict_oaster.py", "protected_multilayer.py")],
    *sorted((root / "metrics/user_metrics").glob("*.py"))]
code_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
source_files, input_hashes = {}, {}
for case in cases:
    case_id = case["case_id"]
    source = source_surface if case["configuration_number"] < 2 else source_deep
    source_files[case_id] = source
    paths_to_hash = [source / (case_id + ".npz"), cross_source / (case_id + ".npz")]
    assert all(path.is_file() for path in paths_to_hash)
    input_hashes[case_id] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in paths_to_hash}

output.mkdir(parents=True)
metadata = {
    "complete": False, "phase": "development", "diagnosis": "conditional_geometric_surface_energy_consensus",
    "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "case_ids": [case["case_id"] for case in cases], "seed_root": seed_root,
    "conditional_family_rule": "case00/01 use saved H0; case02/03/04 use saved H1",
    "blind_detection_result": False,
    "surface_rule": "sqrt(normalized cross-eLORETA+MRF energy * normalized selected-OASTER surface energy)",
    "surface_source_amplitude": "sqrt(consensus energy), scaled to preserve selected OASTER maximum active-window L2 amplitude",
    "deep_rule": "saved OASTER H1 deep block unchanged; H0 deep block zero",
    "minimum_An_cal_AUC_each_case": .9, "maximum_DLE_worsening_mm": maximum_dle_worsening_mm,
    "candidate_grid_or_weight_search": False, "calibration_or_validation_read": False,
    "truth_used_for_family_selection": True, "truth_used_for_spatial_map_or_weight_selection": False,
    "code_sha256": code_hashes, "shared_fingerprint": original._shared_fingerprint(shared),
    "input_sha256": input_hashes}
(output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# %% 五例只执行同一个无权重对称共识，深层块不变。
rows = []
started = time.perf_counter()
for case in cases:
    assert code_hashes == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    tick = time.perf_counter()
    case_id = case["case_id"]
    family = "H0_surface_only" if case["deep_index"] is None else "H1_surface_deep"
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    with np.load(source_files[case_id] / (case_id + ".npz")) as saved:
        baseline_estimate = np.asarray(saved["null" if family.startswith("H0") else "full"], float)
        assert np.allclose(saved["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    with np.load(cross_source / (case_id + ".npz")) as saved_cross:
        cross_energy = np.asarray(saved_cross["surface_cross_mrf"], float)
        assert np.allclose(saved_cross["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    active = observation["active"]
    oaster_energy = np.sum(baseline_estimate[:n_surf, active] ** 2, axis=1)
    cross_normalized = cross_energy / max(float(cross_energy.max(initial=0)), np.finfo(float).eps)
    oaster_normalized = oaster_energy / max(float(oaster_energy.max(initial=0)), np.finfo(float).eps)
    consensus_energy = np.sqrt(cross_normalized * oaster_normalized)

    diagnostic = baseline_estimate.copy()
    old_surface_peak = float(np.sqrt(oaster_energy.max(initial=0)))
    diagnostic[:n_surf] = 0
    if old_surface_peak > 0 and consensus_energy.max(initial=0) > 0:
        surface_amplitude = np.sqrt(consensus_energy / consensus_energy.max()) * old_surface_peak
        diagnostic[:n_surf, int(active[0])] = surface_amplitude
    if family.startswith("H0"):
        diagnostic[n_surf:] = 0
    else:
        assert np.array_equal(diagnostic[n_surf:], baseline_estimate[n_surf:])

    identity = {key: case[key] for key in
                ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    for method, estimate in (("selected_OASTER_baseline", baseline_estimate),
                             ("conditional_geometric_consensus", diagnostic)):
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
        consensus_surface_energy=consensus_energy.astype(np.float32))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = rows[-1]
    print(f"{case_id}: family={family}, An_cal={result['An_cal_AUC']:.6f}, "
          f"surface_DLE={result['surface_DLE_mm']}, deep_DLE={result['deep_DLE_mm']}, "
          f"seconds={time.perf_counter() - tick:.1f}", flush=True)

# %% 固定验收：五例 AUC 均达 0.9，且有真值的分层 DLE 有限并不恶化超过 5 mm。
diagnostic_rows = [row for row in rows if row["method"] == "conditional_geometric_consensus"]
all_auc_pass = all(np.isfinite(row["An_cal_AUC"]) and row["An_cal_AUC"] >= .9 for row in diagnostic_rows)
dle_checks = []
for case in cases:
    baseline_row = next(row for row in rows if row["case_id"] == case["case_id"] and row["method"] == "selected_OASTER_baseline")
    diagnostic_row = next(row for row in rows if row["case_id"] == case["case_id"] and row["method"] == "conditional_geometric_consensus")
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

lines = ["# 表层能量几何共识条件定位开发诊断", "",
    "这不是盲检测结果：预先给定正确模型家族，case00/01 使用 H0，case02/03/04 使用 H1。表层只使用一个固定的无权重对称共识；深层 H1 完全保留原 OASTER，未做 3NN 扩散。",
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
lines.extend(["", "算法未读取 calibration/validation、未搜索候选或权重；无论结果成败，本诊断到此停止。"])
(output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(completion, ensure_ascii=False, indent=2), flush=True)
