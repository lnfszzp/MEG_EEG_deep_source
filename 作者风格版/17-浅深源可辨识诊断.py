"""Read-only v4 pilot diagnosis; truth is used only to measure identifiable signal.

Eight existing cases, no inverse optimization and no parameter fitting.
Run with the meg Python environment from any working directory.
"""

# %% 与20例pilot完全相同的观测、模态权重及ERP时间基。
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmark import erp_protocol, methods, protocol
from candidates.oaster_rebuilt import _evoked_temporal_basis

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / "results/erp_whole_head/adaptive_v5/identifiability")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest_path = root / "results/erp_whole_head/development_full_v3/manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
selected = {}
for case in manifest:
    pair = (case["eeg_snr_db"], case["meg_snr_db"])
    if pair in ((-10, -10), (5, 5)):
        selected.setdefault((pair, case["scenario"]), case)
assert len(selected) == 8
pilot_dir = root / "results/erp_whole_head/adaptive_v4/pilot_five_snr_all_methods"
with (pilot_dir / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    pilot_rows = {row["case_id"]: row for row in csv.DictReader(stream) if row["method"] == "OASTER-ERP-v4"}
pilot_diagnostics = {json.loads(path.read_text(encoding="utf-8"))["case_id"]: path
                     for path in pilot_dir.glob("parts_*/diagnostics/*.json")}
shared = protocol.load_shared(root / "corrected_v2/generated", Path(r"D:\mne_data\MNE-sample-data"))
n_surf = int(shared["n_surf"])
n_sources = n_surf + int(shared["n_deep"])
upper = sparse.triu(sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf]), k=1).tocoo()
neighbors = sparse.csr_matrix((np.ones(2 * upper.nnz),
    (np.r_[upper.row, upper.col], np.r_[upper.col, upper.row])), shape=(n_sources, n_sources))
degree = np.asarray(neighbors.sum(axis=1)).ravel()
transition = sparse.diags(1 / np.maximum(degree, 1)) @ neighbors
mrf_factor = splu((sparse.eye(n_sources) - 0.5 * transition).tocsc())
component_rows, layer_rows, kkt_rows, case_records = [], [], [], []

for case in selected.values():
    eeg, meg, truth, groups, baseline, active_windows, active_indices, simulation = erp_protocol.simulate_case(shared, case, seed_root=20260921)
    active = active_windows[0]
    eeg_white, eeg_gain = methods.whiten(eeg, shared["gain_eeg"])
    meg_white, meg_gain = methods.whiten(meg, shared["gain_meg"])
    data, gain = np.vstack((eeg_white, meg_white)), np.vstack((eeg_gain, meg_gain))
    data -= data[:, baseline].mean(axis=1, keepdims=True)
    modality_ratios = np.asarray([np.mean(block[:, active] ** 2) / np.mean(block[:, baseline] ** 2)
                                 for block in (data[:len(eeg_white)], data[len(eeg_white):])])
    excess = np.maximum(modality_ratios - 1, 0)
    modality_weights = np.sqrt(excess / excess.max()) if excess.max() > 0 else np.ones(2)
    channel_weights = np.repeat(modality_weights, (len(eeg_white), len(meg_white)))
    weighted_data, weighted_gain = channel_weights[:, None] * data, channel_weights[:, None] * gain
    basis, temporal_info = _evoked_temporal_basis(weighted_data, baseline, active)
    assert basis.shape[0] and np.allclose(basis @ basis.T, np.eye(len(basis)))
    # 幅度无关的平滑时间先验：固定0.1窗口宽度高斯频响，保留响应>=0.05的DCT模式。
    width = int(active.sum())
    frequencies = np.arange(width)[np.exp(-0.5 * (np.pi * np.arange(width) * .1) ** 2) >= .05]
    local_smooth = np.cos(np.pi * frequencies[:, None] * (np.arange(width)[None, :] + .5) / width) * np.sqrt(2 / width)
    local_smooth[0] /= np.sqrt(2)
    smooth_basis = np.zeros((len(frequencies), data.shape[1]))
    smooth_basis[:, active] = local_smooth
    assert np.allclose(smooth_basis @ smooth_basis.T, np.eye(len(frequencies)))
    full_dct = np.cos(np.pi * np.arange(width)[:, None] * (np.arange(width)[None, :] + .5) / width) * np.sqrt(2 / width)
    full_dct[0] /= np.sqrt(2)
    response = weighted_data @ basis.T
    clean = weighted_gain @ truth
    noise = weighted_data - clean
    projected_noise = noise @ basis.T
    noise_energy = np.sum(noise[:, active] ** 2)
    projected_noise_energy = np.sum(projected_noise ** 2)
    smooth_noise_energy = np.sum((noise @ smooth_basis.T) ** 2)
    clean_energy = np.sum(clean[:, active] ** 2)
    projected_clean_energy = np.sum((clean @ basis.T) ** 2)
    raw_eeg_noise = eeg - shared["gain_eeg"] @ truth
    raw_meg_noise = meg - shared["gain_meg"] @ truth
    components = []
    common = {"case_id": case["case_id"], "scenario": case["scenario"],
              "eeg_snr_db": case["eeg_snr_db"], "meg_snr_db": case["meg_snr_db"],
              "temporal_rank": len(basis)}
    for group_number, group in enumerate(groups):
        group = np.asarray(group, dtype=int)
        component = weighted_gain[:, group] @ truth[group]
        waveform = truth[group[np.argmax(np.linalg.norm(truth[group], axis=1))]]
        projected = component @ basis.T
        layer = "surface" if np.all(group < n_surf) else "deep"
        components.append((group, layer, projected))
        component_rows.append({**common, "group": group_number, "layer": layer,
            "group_size": len(group), "temporal_retained_energy": float(np.sum((waveform @ basis.T) ** 2) / np.sum(waveform ** 2)),
            "smooth_temporal_rank": len(frequencies),
            "smooth_temporal_retained_energy": float(np.sum((waveform @ smooth_basis.T) ** 2) / np.sum(waveform ** 2)),
            "smooth_projected_snr_db": float(10 * np.log10(np.sum((component @ smooth_basis.T) ** 2) / smooth_noise_energy)),
            "smooth10_temporal_retained_energy": float(np.sum((waveform[active] @ full_dct[:10].T) ** 2) / np.sum(waveform ** 2)),
            "smooth12_temporal_retained_energy": float(np.sum((waveform[active] @ full_dct[:12].T) ** 2) / np.sum(waveform ** 2)),
            "raw_eeg_component_snr_db": float(10 * np.log10(np.sum((shared["gain_eeg"][:, group] @ truth[group][:, active]) ** 2) / np.sum(raw_eeg_noise[:, active] ** 2))),
            "raw_meg_component_snr_db": float(10 * np.log10(np.sum((shared["gain_meg"][:, group] @ truth[group][:, active]) ** 2) / np.sum(raw_meg_noise[:, active] ** 2))),
            "sensor_active_snr_db": float(10 * np.log10(np.sum(component[:, active] ** 2) / noise_energy)),
            "projected_snr_db": float(10 * np.log10(np.sum(projected ** 2) / projected_noise_energy)),
            "sensor_fraction_of_joint_clean_energy": float(np.sum(component[:, active] ** 2) / clean_energy),
            "projected_fraction_of_joint_clean_energy": float(np.sum(projected ** 2) / projected_clean_energy)})

    sensitivity = np.linalg.norm(weighted_gain, axis=0)
    gain_scale = np.median(sensitivity[sensitivity > 0])
    weights = np.maximum(sensitivity / gain_scale, 0.1) ** 0.8
    physical_design = weighted_gain / gain_scale
    starts = np.unique(np.linspace(0, int(baseline.sum() - active.sum()), 16).astype(int))
    local_basis = basis[:, active].T
    stored = {"basis": basis, "smooth_basis": smooth_basis, "dct_local_12_modes": full_dct[:12], "source_penalty_weights": weights,
              "physical_gain_norm": np.linalg.norm(physical_design, axis=0)}
    for mrf_strength in (0.0, 0.5):
        design = physical_design if mrf_strength == 0 else mrf_factor.solve(physical_design.T, trans="T").T
        design_norm = np.linalg.norm(design, axis=0)
        baseline_projection = design.T @ weighted_data[:, baseline]
        null_scores = np.column_stack([np.linalg.norm(baseline_projection[:, start:start + len(local_basis)] @ local_basis, axis=1) / weights
                                      for start in starts])
        global_lambda = float(np.quantile(null_scores.max(axis=0), 0.99))
        layer_lambdas = [float(np.quantile(null_scores[section].max(axis=0), .99))
                         for section in (slice(0, n_surf), slice(n_surf, n_sources))]
        # 匹配16个候选点仅用于拆解“候选数”效应，不进入反演或阈值选择。
        diagnostic_rng = np.random.default_rng(401 + case["case_number"])
        matched_surface_lambda = [float(np.quantile(null_scores[diagnostic_rng.choice(n_surf, n_sources - n_surf, replace=False)].max(axis=0), .99))
                                  for _ in range(128)]
        threshold = global_lambda * weights
        observed_score = np.linalg.norm(design.T @ response, axis=1)
        for layer, section, layer_lambda in zip(("surface", "deep"), (slice(0, n_surf), slice(n_surf, n_sources)), layer_lambdas):
            layer_rows.append({**common, "mrf_strength": mrf_strength, "layer": layer,
                "source_count": design[:, section].shape[1], "global_lambda": global_lambda,
                "layer_only_lambda": layer_lambda, "global_over_layer_lambda": global_lambda / layer_lambda,
                "global_null_winner_fraction": float(np.mean((np.argmax(null_scores, axis=0) < n_surf) == (layer == "surface"))),
                "matched_16_surface_lambda_median": float(np.median(matched_surface_lambda)),
                "gain_norm_before_median": float(np.median(np.linalg.norm(physical_design[:, section], axis=0))),
                "gain_norm_after_median": float(np.median(design_norm[section])),
                "gain_norm_amplification_median": float(np.median(design_norm[section] / np.linalg.norm(physical_design[:, section], axis=0))),
                "source_penalty_min": float(threshold[section].min()), "source_penalty_median": float(np.median(threshold[section])),
                "source_penalty_max": float(threshold[section].max()), "max_observed_kkt_ratio": float(np.max(observed_score[section] / threshold[section]))})
        for group_number, (group, layer, projected) in enumerate(components):
            # 扣除其他源为oracle诊断：不是算法残差、更不用于任何拟合。
            oracle_residual = projected + projected_noise
            own_score = np.linalg.norm(design[:, group].T @ projected, axis=1)
            oracle_score = np.linalg.norm(design[:, group].T @ oracle_residual, axis=1)
            kkt_rows.append({**common, "mrf_strength": mrf_strength, "group": group_number, "layer": layer,
                "source_count": len(group), "source_penalty_mean": float(threshold[group].mean()),
                "own_clean_max_kkt_ratio": float(np.max(own_score / threshold[group])),
                "observed_mixed_max_kkt_ratio": float(np.max(observed_score[group] / threshold[group])),
                "oracle_other_sources_removed_max_kkt_ratio": float(np.max(oracle_score / threshold[group])),
                "oracle_ratio_using_layer_only_lambda": float(np.max(oracle_score / (weights[group] * layer_lambdas[layer == "deep"]))),
                "historical_deep_detected": int(pilot_rows[case["case_id"]]["deep_detected"]),
                "historical_deep_score": float(pilot_rows[case["case_id"]]["deep_score"])})
        stored[f"mrf_{mrf_strength}_gain_norm"] = design_norm
        stored[f"mrf_{mrf_strength}_null_scores"] = null_scores
        stored[f"mrf_{mrf_strength}_source_penalty"] = threshold
        stored[f"mrf_{mrf_strength}_observed_kkt_ratio"] = observed_score / threshold
        if mrf_strength == 0.5:
            previous = json.loads(pilot_diagnostics[case["case_id"]].read_text(encoding="utf-8"))["diagnostics"]["windows"][0]
            assert np.isclose(global_lambda, previous["source_lambda"], rtol=1e-10), "diagnosis did not reproduce pilot threshold"
            assert len(basis) == previous["temporal_rank"]
    np.savez_compressed(args.output / f"case_{case['case_number']:05d}_scores.npz", **stored)
    case_records.append({**case, "simulation": simulation, "modality_weights": modality_weights.tolist(),
                         "temporal_basis": temporal_info, "pilot_threshold_reproduced": True})
    print(f"{case['case_number']:05d} {case['scenario']} {case['eeg_snr_db']}/{case['meg_snr_db']}: audited", flush=True)

# %% 数值表及诊断图。KKT比仅为开始点/真值扣除后的相关强度，不等同最后迭代的KKT。
for filename, rows in (("component_retention.csv", component_rows), ("layer_noise_penalty.csv", layer_rows), ("true_source_kkt.csv", kkt_rows)):
    with (args.output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
labels = {"surface_only": "Surface", "deep_only": "Deep", "deep_plus_surface": "Deep+surface", "deep_plus_two_surface": "Deep+2 surfaces"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
for pair, marker in (((-10, -10), "o"), ((5, 5), "s")):
    source_rows = [row for row in component_rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    for row in source_rows:
        axis = axes[0]
        axis.scatter(row["sensor_active_snr_db"], row["temporal_retained_energy"], marker=marker,
                     c="#D55E00" if row["layer"] == "deep" else "#0072B2", s=45)
    deep_rows = [row for row in layer_rows if row["layer"] == "deep" and row["mrf_strength"] == .5 and (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    axes[1].plot([labels[row["scenario"]] for row in deep_rows], [row["global_over_layer_lambda"] for row in deep_rows], marker=marker, label=f"EEG/MEG {pair[0]}/{pair[1]} dB")
    deep_kkt = [row for row in kkt_rows if row["layer"] == "deep" and row["mrf_strength"] == .5 and (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    axes[2].plot([labels[row["scenario"]] for row in deep_kkt], [row["oracle_other_sources_removed_max_kkt_ratio"] for row in deep_kkt], marker=marker, label=f"EEG/MEG {pair[0]}/{pair[1]} dB")
axes[0].set(xlabel="Component SNR before temporal projection (dB)", ylabel="Retained true waveform energy", title="Temporal basis: blue surface / orange deep", ylim=(-.03, 1.03))
axes[1].set(ylabel="Global / deep-only baseline threshold", title="Global threshold inflation at MRF=0.5")
axes[2].set(ylabel="True deep-site score / source penalty", title="After oracle removal of other sources")
axes[2].axhline(1, color="#999999", linestyle="--")
for axis in axes[1:]:
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(args.output / "identifiability_diagnostic.png", dpi=180)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
for axis, pair in zip(axes, ((-10, -10), (5, 5))):
    selected_components = [row for row in component_rows if (row["eeg_snr_db"], row["meg_snr_db"]) == pair]
    short_labels = [f"{labels[row['scenario']]}\n{row['layer']} {row['group']}" for row in selected_components]
    axis.plot(short_labels, [row["temporal_retained_energy"] * 100 for row in selected_components], "o-", color="#0072B2", label="v4 data-selected, rank 3")
    axis.plot(short_labels, [row["smooth_temporal_retained_energy"] * 100 for row in selected_components], "s-", color="#D55E00", label="Fixed smooth DCT, rank 8")
    axis.plot(short_labels, [row["smooth10_temporal_retained_energy"] * 100 for row in selected_components], "^-", color="#009E73", label="Broader fixed DCT, rank 10")
    axis.axhline(95, color="#999999", linestyle="--", linewidth=1)
    axis.set(title=f"EEG/MEG {pair[0]}/{pair[1]} dB", ylabel="Retained true waveform energy (%)", ylim=(-3, 104))
    axis.tick_params(axis="x", rotation=35)
    axis.legend(fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("Temporal representation only: no inverse result, no truth used to choose DCT modes")
fig.tight_layout()
fig.savefig(args.output / "temporal_basis_retention_comparison.png", dpi=180)
plt.close(fig)
metadata = {"scope": "8 existing pilot cases; read-only diagnosis, truth never used by inverse solver",
    "seed_root": 20260921, "mrf_strengths": [0, .5], "source_weight_rule": "max(pre-MRF sensitivity / median sensitivity, .1)^.8",
    "null_rule": "same v4 99th percentile of 16 overlapping baseline-window maxima",
    "smooth_temporal_rule": "Orthonormal DCT-II; retain f with exp(-0.5*(pi*f*0.1)^2)>=0.05, without data-amplitude selection; 8 modes at width 61",
    "smooth_alternative_rules": "Same Gaussian frequency response with cutoffs .01/.001 retains 10/12 DCT modes; development-only representation ablation, not verified inverse performance",
    "limitations": ["KKT scores are at zero or after oracle subtraction, not final-solver residuals",
                    "sensor_active_snr_db and projected_snr_db use the whitened modality-weighted sensor space; raw modality SNRs are separate columns",
                    "per-layer and matched-size thresholds are diagnostic only; not validated detection thresholds",
                    "component energies are nonadditive when signals correlate",
                    "projected SNR is descriptive: its temporal basis was selected using the same noisy data"],
    "cases": case_records, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "code_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), Path(erp_protocol.__file__), Path(methods.__file__), root / "candidates/oaster_rebuilt.py", root / "candidates/oaster_adaptive.py")}}
(args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(args.output, flush=True)
