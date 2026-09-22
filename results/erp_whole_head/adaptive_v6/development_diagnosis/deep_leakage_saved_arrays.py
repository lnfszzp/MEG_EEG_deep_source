"""Read-only v5 diagnosis; no inverse fit, threshold calibration, or result edits."""
# %% 仅读取上轮保存的源时间序列，分解既有活动能量。
from pathlib import Path
import csv
import json
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from scipy.fft import dct

output = Path(__file__).resolve().parent
root = output.parents[3]
source = root / "results/erp_whole_head/adaptive_v5/shape_final"
with (source / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
    recorded_scores = {row["case_id"]: float(row["deep_score"])
                       for row in csv.DictReader(handle) if row["method"] == "OASTER-ERP-v5"}
rows = []
for path in sorted(source.glob("shape_*.npz")):
    saved = np.load(path)
    record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    active, baseline = saved["active"], saved["baseline"]
    estimate, truth = saved["estimate_v5"].astype(float), saved["truth"].astype(float)
    n_surf = 7498
    width = int(active.sum())
    modes = dct(np.eye(width), type=2, norm="ortho", axis=0)
    modes = modes[np.exp(-.5 * (np.pi * np.arange(width) * .1) ** 2) >= .01]
    active_x = estimate[:, active]
    in_basis = (active_x @ modes.T) @ modes
    outside = active_x - in_basis
    power = np.mean(active_x ** 2, axis=1)
    null_power = np.mean(estimate[:, baseline] ** 2, axis=1)
    inside_power = np.mean(in_basis ** 2, axis=1)
    outside_power = np.mean(outside ** 2, axis=1)
    assert np.allclose(power, inside_power + outside_power, atol=1e-12, rtol=1e-9)
    amplitude = np.sqrt(np.maximum(power - null_power, 0))
    peak_deep = n_surf + int(np.argmax(amplitude[n_surf:]))
    peak_surface = int(np.argmax(amplitude[:n_surf]))
    cortical_wave = truth[saved["surface_indices"]][:, active].mean(axis=0)
    deep_wave = active_x[peak_deep]
    solver = record["diagnostics"]["v5"]["windows"][0]
    row = dict(case_id=path.stem, has_deep_true=record["deep_index"] is not None,
        eeg_snr_db=record["eeg_snr_db"], meg_snr_db=record["meg_snr_db"],
        deep_score=float(amplitude[peak_deep] / amplitude.max()),
        active_only_rms_deep_score=float(np.sqrt(power[n_surf:].max() / power.max())),
        in_basis_only_rms_deep_score=float(np.sqrt(inside_power[n_surf:].max() / inside_power.max())),
        peak_deep_index=peak_deep, peak_surface_index=peak_surface,
        deep_peak_baseline_over_active_power=float(null_power[peak_deep] / power[peak_deep]),
        surface_peak_baseline_over_active_power=float(null_power[peak_surface] / power[peak_surface]),
        deep_peak_outside_basis_energy_fraction=float(outside_power[peak_deep] / power[peak_deep]),
        deep_layer_outside_basis_energy_fraction=float(outside_power[n_surf:].sum() / power[n_surf:].sum()),
        surface_layer_outside_basis_energy_fraction=float(outside_power[:n_surf].sum() / power[:n_surf].sum()),
        peak_deep_abs_correlation_with_true_cortical_wave=float(abs(np.corrcoef(deep_wave, cortical_wave)[0, 1])),
        peak_deep_in_basis_abs_correlation_with_true_cortical_wave=float(abs(np.corrcoef(in_basis[peak_deep], cortical_wave)[0, 1])),
        response_condition=solver["response_condition"],
        source_lambda_surface=solver["source_lambda_surface"],
        source_lambda_deep=solver["source_lambda_deep"],
        converged=solver["solver"]["converged"])
    if row["has_deep_true"]:
        true_deep_wave = truth[record["deep_index"], active]
        row["peak_deep_abs_correlation_with_true_deep_wave"] = float(abs(np.corrcoef(deep_wave, true_deep_wave)[0, 1]))
    assert np.isclose(row["deep_score"], recorded_scores[path.stem], atol=1e-7, rtol=1e-7)
    rows.append(row)
assert len(rows) == 12 and sum(not row["has_deep_true"] for row in rows) == 9

# %% 真值只用于诊断输出，从未输入反演或选择阈值。
payload = dict(scope="All 12 frozen v5 non-Gaussian cases; saved float32 source arrays only. Final pilot did not save source arrays.",
    decomposition="X_active = X_active Q.T Q + X_active (I-Q.T Q), using the same 10-mode DCT as frozen v5. Energies add by orthogonality.",
    warning="The projected RMS score is a diagnosis, not a new estimator or validated detection threshold; baseline must not be zeroed in an actual method.",
    rows=rows)
(output / "deep_leakage_saved_arrays.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
for row in rows:
    print(row["case_id"], "deep:", round(row["deep_score"], 3),
          "active:", round(row["active_only_rms_deep_score"], 3),
          "basis:", round(row["in_basis_only_rms_deep_score"], 3),
          "baseline/active:", round(row["deep_peak_baseline_over_active_power"], 3),
          "out:", round(row["deep_peak_outside_basis_energy_fraction"], 3),
          "corr_cortex:", round(row["peak_deep_abs_correlation_with_true_cortical_wave"], 3))
