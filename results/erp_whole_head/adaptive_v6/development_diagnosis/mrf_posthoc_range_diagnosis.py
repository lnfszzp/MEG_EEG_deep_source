"""只读机制诊断：改变已保存 ADMM innovation 的 MRF 展开，不作为正式算法结果。"""

# %% 仅使用两个已见开发分片；不读取校准或验证数据。
import csv
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from benchmark import metrics, protocol
import run_erp_whole_head_matrix as original

result_root = root / "results/erp_whole_head/adaptive_v6"
output = Path(__file__).with_suffix(".json")
inputs = {
    "eeg+05_meg+05": result_root / "dev_trial_covariance_admm_5_5",
    "eeg-10_meg-10": result_root / "dev_trial_admm_m10_m10",
}
strengths = (0.5, 0.85, 0.9, 0.95)
shared = protocol.load_shared(root / "corrected_v2/generated", original.DEFAULT_SAMPLE_PATH)
n_surf = int(shared["n_surf"])
graph = sparse.csr_matrix(shared["adjacency"])
upper = sparse.triu(graph[:n_surf, :n_surf].maximum(graph[:n_surf, :n_surf].T), k=1).tocoo()
neighbors = sparse.csr_matrix((np.ones(2 * upper.nnz),
    (np.r_[upper.row, upper.col], np.r_[upper.col, upper.row])), shape=graph.shape)
degree = np.asarray(neighbors.sum(axis=1)).ravel()
transition = sparse.diags(1 / np.maximum(degree, 1)) @ neighbors
operators = {strength: (sparse.eye(graph.shape[0]) - strength * transition).tocsc()
             for strength in strengths}
solvers = {strength: splu(operator) for strength, operator in operators.items()}

# %% 从原 r=.5 物理解恢复 innovation，再以更强 r 重映射；不重新拟合传感器数据。
rows = []
source_rows = []
for dataset, directory in inputs.items():
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    assert completion["complete"] and completion["case_count"] == 5 and completion["all_converged"]
    with (directory / "rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        old_rows = {row["case_id"]: row for row in csv.DictReader(handle)
                    if row["method"] == "v6-full-ungated"}
    assert len(old_rows) == 5
    for case_id, old_row in old_rows.items():
        with np.load(directory / (case_id + ".npz")) as arrays:
            full = arrays["full"].astype(float)
            truth = arrays["truth"].astype(float)
            active = arrays["active"].astype(int)
            baseline = arrays["baseline"].astype(bool)
        detail = json.loads((directory / (case_id + ".json")).read_text(encoding="utf-8"))
        groups = [np.asarray(group, dtype=int) for group in detail["simulation"]["groups"]]
        innovation = operators[0.5] @ full
        deep_before = innovation[n_surf:].copy()
        for strength in strengths:
            remapped = solvers[strength].solve(innovation)
            assert np.allclose(remapped[n_surf:], deep_before, atol=1e-12, rtol=1e-12)
            values = metrics.evaluate_estimate(
                remapped, truth, shared["vertices"], groups, n_surf, active,
                shared["auc_cortex"], baseline=baseline)
            rows.append({"dataset": dataset, "case_id": case_id,
                "scenario": old_row["scenario"], "mrf_strength": strength, **values})
        source_rows.append({"dataset": dataset, "case_id": case_id,
            "source_file": str((directory / (case_id + ".npz")).relative_to(root)),
            "original_mrf_strength": 0.5})

# %% 汇总趋势；deep 变换严格为恒等，但仍保留重算指标供审计。
summary = []
for dataset in inputs:
    for strength in strengths:
        selected = [row for row in rows
                    if row["dataset"] == dataset and row["mrf_strength"] == strength]
        positives = [row for row in selected if row["has_deep_true"]]
        surface = [row for row in selected if np.isfinite(row["surface_dle_mm"])]
        detected = [row for row in positives if row["deep_detected"]]
        summary.append({"dataset": dataset, "mrf_strength": strength,
            "case_count": len(selected),
            "mean_historical_An_cal_AUC": float(np.mean([row["auc"] for row in selected])),
            "mean_An_auc": float(np.mean([row["auc_tie_corrected"] for row in selected])),
            "mean_surface_An_auc": float(np.mean([row["surface_auc_tie_corrected"] for row in surface])),
            "mean_surface_DLE_mm": float(np.mean([row["surface_dle_mm"] for row in surface])),
            "mean_surface_SD_mm": float(np.mean([row["surface_sd_mm"] for row in surface])),
            "mean_positive_deep_An_auc": float(np.mean([row["deep_auc_tie_corrected"] for row in positives])),
            "deep_detected_count": len(detected),
            "mean_detected_deep_DLE_mm": (float(np.mean([row["deep_dle_mm"] for row in detected]))
                                           if detected else None)})

payload = {
    "scope": "posthoc mechanism diagnostic on two previously seen five-case development slices only; no calibration or validation files read",
    "operation": "recover Z=(I-0.5P)J from saved physical estimates, then display J_r=(I-rP)^-1 Z",
    "not_a_candidate_result": True,
    "warning": "Changing r after fitting does not recompute the sensor fit, baseline penalties, ADMM support, H0/H1 evidence, or convergence. The apparent metric gain must not be reported as a final algorithm result.",
    "required_next_check": "Freeze one r (development suggests 0.95), rerun the actual trial-covariance ADMM H0/H1 from sensor data, and recheck convergence, held-out presence, An_cal_AUC, layer An_auc, SD and DLE before calibration or validation.",
    "mrf_is_not_fixed_radius_template": "P is the anatomical adjacency transition; stronger r propagates each data-selected innovation until the geometric series decays, without selecting among fixed 0/4/7 mm Gaussian templates.",
    "strengths": list(strengths), "summary": summary,
    "case_metrics": [{key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                      for key, value in row.items()} for row in rows],
    "sources": source_rows,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                  encoding="utf-8")
print(output)
for row in summary:
    print(row["dataset"], row["mrf_strength"],
          f"An_cal={row['mean_historical_An_cal_AUC']:.6f}",
          f"An_auc={row['mean_An_auc']:.6f}",
          f"surface_DLE={row['mean_surface_DLE_mm']:.3f}")
