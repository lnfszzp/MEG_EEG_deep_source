"""只读 5/5 dB 开发集 ADMM 定位；用独立确认数据计算浅层条件下的深源增量证据。"""

# %% 路径与固定输入。脚本不重新反演、不读取校准或验证病例。
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import metrics, protocol
from benchmark.erp_trial_covariance import prepare_trial_covariance_case
from candidates.oaster_partial_confirmation import score_partial_deep_presence
import run_erp_whole_head_matrix as original
import run_strict_oaster as archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path,
    default=root / "results/erp_whole_head/adaptive_v6/dev_trial_covariance_admm_5_5")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"不覆盖旧开发结果：{args.output}")
completion = json.loads((args.source / "completion.json").read_text(encoding="utf-8"))
source_metadata = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
assert completion["complete"] and completion["all_converged"] and completion["case_count"] == 5
assert source_metadata["phase"] == "development" and source_metadata["covariance"] == "trial"
assert source_metadata["solver_settings"]["solver_kind"] == "admm"
for relative, expected_hash in source_metadata["code_sha256"].items():
    path = root / relative
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash, \
        f"源定位结果的代码指纹已变化：{relative}"

manifest = root / "results/erp_whole_head/adaptive_v6/protocol/development_manifest.json"
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
all_cases = json.loads(manifest.read_text(encoding="utf-8"))
case_by_id = {case["case_id"]: case for case in all_cases}
assert len(case_by_id) == len(all_cases)
assert len(source_metadata["case_ids"]) == 5 and set(source_metadata["case_ids"]) <= case_by_id.keys()
cases = [case_by_id[case_id] for case_id in source_metadata["case_ids"]]
assert len(cases) == 5 and {case["configuration_number"] for case in cases} == set(range(5))
snr_pairs = {(case["eeg_snr_db"], case["meg_snr_db"]) for case in cases}
assert len(snr_pairs) == 1, "每个输入目录必须是一个完整的单 SNR 五例分片"
snr_pair = list(snr_pairs)[0]
assert source_metadata["manifest_sha256"] == manifest_hash
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert source_metadata["shared_fingerprint"] == original._shared_fingerprint(shared)
seed_root = int(source_metadata["seed_root"])
args.output.mkdir(parents=True)

# %% 训练源解只负责选浅层 nuisance 支持和一个深点；确认数据只形成嵌套回归统计量。
rows = []
for case in cases:
    observation = prepare_trial_covariance_case(shared, case, seed_root=seed_root)
    saved = args.source / (case["case_id"] + ".npz")
    original_case = json.loads((args.source / (case["case_id"] + ".json")).read_text(encoding="utf-8"))
    assert original_case["case_id"] == case["case_id"]
    assert all(original_case["fitting"][name + "_model"]["windows"][0]["solver"]["converged"]
               for name in ("null", "full"))
    if "raw_mean_sha256" in original_case:
        assert original_case.get("complete") and original_case.get("raw_means_unchanged")
        for key in ("eeg_train", "meg_train", "eeg_confirmation", "meg_confirmation"):
            actual_hash = hashlib.sha256(np.ascontiguousarray(observation[key]).tobytes()).hexdigest()
            assert original_case["raw_mean_sha256"][key] == actual_hash, f"原始均值不一致：{case['case_id']} {key}"
        producer_verification = "source code SHA + four regenerated raw-mean SHA + NPZ truth + solver convergence"
    else:
        assert original_case["simulation"]["replica_seed_roots"] == case["replica_seed_roots"]
        assert observation["metadata"]["replica_seed_roots"] == case["replica_seed_roots"]
        producer_verification = "source code SHA + simulation replica roots + NPZ truth + solver convergence"
    with np.load(saved) as arrays:
        null = arrays["null"].astype(float)
        full = arrays["full"].astype(float)
        assert np.allclose(arrays["truth"], observation["truth"], atol=1e-8, rtol=1e-6)
    partial_score, evidence = score_partial_deep_presence(
        observation["confirmation"], observation["gain"], null, full, shared["n_surf"],
        baseline=observation["baseline"], active=observation["active_windows"][0],
        channel_weights=observation["channel_weights"])
    values = metrics.evaluate_estimate(
        full, observation["truth"], shared["vertices"], observation["groups"], shared["n_surf"],
        observation["active"], shared["auc_cortex"], baseline=observation["baseline"])
    amplitude = metrics.source_amplitude(full, observation["active"], observation["baseline"])
    true_deep = case.get("deep_index")
    selected = evidence["selected_deep_index"]
    distance = (None if true_deep is None or selected is None else
        float(np.linalg.norm(shared["vertices"][selected] - shared["vertices"][true_deep]) * 1000))
    original_score = (original_case["evidence"]["loss_improvement"] /
                      original_case["evidence"]["expected_response_noise_energy"])
    identity = {key: case[key] for key in
                ("case_id", "configuration_id", "scenario", "eeg_snr_db", "meg_snr_db")}
    rows.append({**identity, "has_deep_true": int(true_deep is not None),
        "partial_score": partial_score, "old_fixed_prediction_score": original_score,
        "selected_deep_local_index": evidence["selected_deep_local_index"],
        "selected_deep_index": selected,
        "candidate_available": int(evidence["candidate_available"]),
        "candidate_identifiable": int(evidence["identifiable"]),
        "selected_deep_distance_mm": "" if distance is None else distance,
        "localized_within_10mm_if_present": ("" if true_deep is None else
            int(distance is not None and distance <= 10 + 1e-6)),
        "surface_support_count": evidence["surface_support_count"],
        "conditional_gain_fraction": evidence["conditional_gain_fraction"],
        "deep_peak_amplitude": float(amplitude[shared["n_surf"]:].max()),
        "deep_to_global_amplitude_ratio": values["deep_score"],
        "An_auc": values["auc_tie_corrected"],
        "surface_An_auc": values["surface_auc_tie_corrected"],
        "deep_An_auc": values["deep_auc_tie_corrected"],
        "surface_DLE_mm": values["surface_dle_mm"], "deep_DLE_mm": values["deep_dle_mm"],
        "surface_SD_mm": values["surface_sd_mm"], "deep_SD_mm": values["deep_sd_mm"],
        "legacy_amplitude_false_positive": values["deep_false_positive"],
        "source_npz_sha256": hashlib.sha256(saved.read_bytes()).hexdigest()})
    detail = {"case_id": case["case_id"], "source_npz": str(saved),
        "source_npz_sha256": rows[-1]["source_npz_sha256"], "partial_score": partial_score,
        "evidence": evidence, "source_producer_verification": producer_verification,
        "truth_used_by_score": False,
        "truth_used_only_for_posthoc_metrics": True, "selected_deep_distance_mm": distance}
    (args.output / (case["case_id"] + ".json")).write_text(
        json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{case['case_id']}: partial={partial_score:.6g}, deep={evidence['selected_deep_local_index']}, "
          f"An_auc={values['auc_tie_corrected']:.6g}, deep_DLE={distance}", flush=True)

archive._atomic_csv(args.output / "rows.csv", rows, rows[0].keys())
positives = [row for row in rows if row["has_deep_true"]]
negatives = [row for row in rows if not row["has_deep_true"]]
summary = {"complete": True, "phase": "development", "case_count": len(rows),
    "source": str(args.source), "source_completion_sha256": hashlib.sha256(
        (args.source / "completion.json").read_bytes()).hexdigest(),
    "manifest_sha256": manifest_hash, "eeg_snr_db": snr_pair[0], "meg_snr_db": snr_pair[1],
    "candidate_sha256": hashlib.sha256((root / "candidates/oaster_partial_confirmation.py").read_bytes()).hexdigest(),
    "decision_threshold": None, "calibration_or_validation_read": False,
    "source_producer_verification": sorted({json.loads(
        (args.output / (case["case_id"] + ".json")).read_text(encoding="utf-8"))[
            "source_producer_verification"] for case in cases}),
    "mean_An_auc": float(np.mean([row["An_auc"] for row in rows])),
    "mean_surface_An_auc": float(np.nanmean([row["surface_An_auc"] for row in rows])),
    "mean_deep_An_auc_positive": float(np.mean([row["deep_An_auc"] for row in positives])),
    "localized_positive_count_at_10mm": sum(row["localized_within_10mm_if_present"] for row in positives),
    "positive_count": len(positives),
    "partial_scores_negative": [row["partial_score"] for row in negatives],
    "partial_scores_positive": [row["partial_score"] for row in positives],
    "pure_surface_deep_peak_amplitude": [row["deep_peak_amplitude"] for row in negatives],
    "pure_surface_deep_to_global_ratio": [row["deep_to_global_amplitude_ratio"] for row in negatives],
    "interpretation": "定位指标来自训练 ADMM H1；partial_score 仅是未校准的独立确认统计量，不能据此宣称假阳性率。"}
(args.output / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
