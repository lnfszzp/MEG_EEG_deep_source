"""Development-only score audit: use frozen source fits, never rerun an inverse."""
# %% 读取已完成的五例开发拟合，重建原始独立确认观测。
from pathlib import Path
import hashlib
import json
import os
import sys
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
output = Path(__file__).resolve().parent
root = output.parents[3]
sys.path.insert(0, str(root))
import numpy as np
from benchmark import protocol
from benchmark.erp_replicates import prepare_replicated_case
from candidates.oaster_balanced import _smooth_temporal_basis
import run_erp_whole_head_matrix as original

saved_dir = root / "results/erp_whole_head/adaptive_v6/dev_predictive_wide_5_5"
metadata = json.loads((saved_dir / "metadata.json").read_text(encoding="utf-8"))
for name in ("benchmark/erp_replicates.py", "benchmark/erp_protocol.py", "benchmark/protocol.py", "benchmark/methods.py", "candidates/oaster_balanced.py"):
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == metadata["code_sha256"][name.replace("/", "\\")]
manifest = Path(metadata["manifest"])
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == metadata["manifest_sha256"]
cases = {case["case_id"]: case for case in json.loads(manifest.read_text(encoding="utf-8"))}
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
assert original._shared_fingerprint(shared) == metadata["shared_fingerprint"]
rows = []
for case_id in metadata["case_ids"]:
    case = cases[case_id]
    saved = np.load(saved_dir / (case_id + ".npz"))
    record = json.loads((saved_dir / (case_id + ".json")).read_text(encoding="utf-8"))
    observation = prepare_replicated_case(shared, case, metadata["seed_root"])
    assert observation["metadata"]["actual_snr_db"] == record["simulation"]["actual_snr_db"]
    baseline, active = observation["baseline"], observation["active_windows"][0]
    weights = observation["channel_weights"]
    confirm = observation["confirmation"]
    basis, _ = _smooth_temporal_basis(confirm, baseline, active)
    centered = confirm - confirm[:, baseline].mean(axis=1, keepdims=True)
    weighted = weights[:, None] * centered
    response = weighted @ basis.T
    gain = weights[:, None] * observation["gain"]
    null_prediction = gain @ (saved["null"].astype(float) @ basis.T)
    full_prediction = gain @ (saved["full"].astype(float) @ basis.T)
    delta = full_prediction - null_prediction
    residual = response - null_prediction
    cross = float(np.sum(delta * residual))
    squared_delta = float(np.sum(delta ** 2))
    loss_gain = float(np.sum(residual ** 2) - np.sum((response - full_prediction) ** 2))
    assert np.isclose(loss_gain, 2 * cross - squared_delta, atol=1e-8, rtol=1e-10)
    # 确认baseline只估计噪声协方差；固定训练方向Delta没有重新拟合。
    covariance = weighted[:, baseline] @ weighted[:, baseline].T / (baseline.sum() - 1)
    qsum = basis[:, active].sum(axis=1)
    mean_direction = delta @ qsum
    variance = float(np.sum(delta * (covariance @ delta)) + mean_direction @ covariance @ mean_direction / baseline.sum())
    expected_noise = float(np.trace(covariance) * (len(basis) + np.sum(qsum ** 2) / baseline.sum()))
    score = loss_gain / expected_noise
    archived_score = record["evidence"]["loss_improvement"] / record["evidence"]["expected_response_noise_energy"]
    assert np.isclose(score, archived_score, rtol=1e-5, atol=1e-7), "saved float32 fits must reproduce original evidence"
    assert variance > 0
    row = dict(case_id=case_id, configuration_kind=case["configuration_kind"],
        has_deep_true=case.get("deep_index") is not None,
        archived_loss_score=archived_score, reproduced_loss_score=score,
        direction_score=cross / np.sqrt(variance), cross_evidence=cross,
        squared_prediction_difference=squared_delta,
        loss_improvement=loss_gain, directional_variance=variance,
        positive_loss_requires_cross_greater_than=squared_delta / 2,
        null_converged=record["fitting"]["null_model"]["windows"][0]["solver"]["converged"],
        full_converged=record["fitting"]["full_model"]["windows"][0]["solver"]["converged"])
    rows.append(row)
    print(case["configuration_kind"], "T", round(score, 6), "Z", round(row["direction_score"], 6),
          "cross", round(cross, 3), "half-delta-squared", round(squared_delta / 2, 3), flush=True)
assert len(rows) == 5
(output / "directional_saved_predictions.json").write_text(json.dumps(dict(
    scope="Five frozen development fits only, 5/5 dB. No inverse refit or calibration/validation data used.",
    formula="Z=<DeltaF,B_confirmation-F0>/sqrt(tr(DeltaF.T Cbaseline DeltaF)+(DeltaF qsum).T Cbaseline(DeltaF qsum)/n_baseline)",
    warning="Not a Gaussian Z test or p-value. Repeatable cortical fitting bias can also raise this score. Requires new independent cortical-only null calibration after score is frozen.",
    saved_metadata=metadata, rows=rows), indent=2) + "\n", encoding="utf-8")
