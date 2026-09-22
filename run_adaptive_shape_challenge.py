"""Non-Gaussian shape challenge: 4 fixed configurations x 3 SNR pairs.

This small diagnostic supplements, and never replaces, the full-head benchmark.
Run --self-check to validate connected truth shapes without running inverse solvers.
"""

# %% 参数：按顺序运行，无需逐个阅读自定义函数。
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from benchmark import erp_protocol, methods, metrics, protocol
from candidates import oaster_rebuilt
import protected_multilayer as protected

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=root / "results/erp_whole_head/adaptive_v4/shape_challenge")
parser.add_argument("--data-root", type=Path, default=root / "corrected_v2/generated")
parser.add_argument("--sample-path", type=Path, default=Path(os.environ.get("MNE_SAMPLE_PATH", r"D:\mne_data\MNE-sample-data")))
parser.add_argument("--self-check", action="store_true")
parser.add_argument("--case-limit", type=int, default=12)
args = parser.parse_args()
if not 1 <= args.case_limit <= 12:
    parser.error("--case-limit must be in [1, 12]")
seed_root = 20260922
snr_pairs = ((-10, -10), (5, 5), (-10, 20))
specifications = (
    ("postcentral-lh", "uniform_1hop", 1, False),
    ("superiorfrontal-rh", "uniform_3hop", 3, False),
    ("lateraloccipital-lh", "thin_path", 5, False),
    ("superiorparietal-rh", "wide_path_plus_deep", 5, True),
)

# %% 真值仅由原始皮层邻接图决定，幅度均匀，没有高斯模板。
shared = protocol.load_shared(args.data_root, args.sample_path)
n_surf = int(shared["n_surf"])
n_sources = n_surf + int(shared["n_deep"])
vertices = np.asarray(shared["vertices"])
graph = sparse.csr_matrix(shared["adjacency"][:n_surf, :n_surf])
graph.setdiag(0)
graph.eliminate_zeros()
parcels = protocol._parcels(shared)
shapes = []
for parcel, shape_name, hops, has_deep in specifications:
    center = int(protocol._farthest_three(parcels[parcel], vertices)[0])
    distance = dijkstra(graph, indices=center, directed=False, unweighted=True, limit=hops)
    local = np.flatnonzero(np.isfinite(distance))
    patch = local.copy()
    if "path" in shape_name:
        # 沿局部主轴连接两个端点，最短路径保证连通；混合病例再加一圈宽度。
        xyz = vertices[local] - vertices[local].mean(axis=0)
        axis = np.linalg.svd(xyz, full_matrices=False)[2][0]
        projection = xyz @ axis
        start, stop = int(np.argmin(projection)), int(np.argmax(projection))
        _, predecessor = dijkstra(graph[local][:, local], indices=start, directed=False,
                                  unweighted=True, return_predecessors=True)
        path = [stop]
        while path[-1] != start:
            previous = int(predecessor[path[-1]])
            assert previous >= 0, "local source graph must connect path endpoints"
            path.append(previous)
        patch = local[np.asarray(path)]
        if has_deep:
            patch = np.union1d(patch, graph[patch].indices)
    patch = np.unique(patch)
    assert patch.size > 1 and connected_components(graph[patch][:, patch], directed=False)[0] == 1
    assert np.all(patch < n_surf)
    shapes.append({"parcel": parcel, "shape": shape_name, "center": center,
                   "surface_indices": patch.tolist(), "surface_count": int(patch.size),
                   "deep_index": n_surf + 6 if has_deep else None})

baseline, active_mask = erp_protocol._masks(shared["times"])
active = np.flatnonzero(active_mask)
print(json.dumps(shapes, ensure_ascii=False, indent=2), flush=True)
if args.self_check:
    assert shapes[0]["surface_count"] < shapes[1]["surface_count"]
    print("PASS: four connected, non-Gaussian source configurations; no inverse jobs run.")
    raise SystemExit(0)

# %% 两版本接收同一白化观测。固定每病例的随机种子，保留真值、估计和收敛记录。
from candidates.oaster_adaptive import reconstruct_evoked_oaster_v4_from_whitened

args.output.mkdir(parents=True, exist_ok=True)
code_paths = [Path(__file__), root / "candidates/oaster_adaptive.py",
              Path(oaster_rebuilt.__file__), Path(erp_protocol.__file__),
              Path(methods.__file__), Path(metrics.__file__)]
code_sha256 = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in code_paths}
kernels = protected.connected_euclidean_surface_kernels(
    vertices, shared["adjacency"], n_surf, scales_mm=oaster_rebuilt.SURFACE_SCALES_MM
)
rows = []
cases = []
display_maps = {}
for shape_number, shape in enumerate(shapes):
    patch = np.asarray(shape["surface_indices"], dtype=int)
    has_deep = shape["deep_index"] is not None
    waves = erp_protocol._waveforms(shared["times"], active_mask, 2 if has_deep else 1, 0.5 if has_deep else 0.0)
    truth = np.zeros((n_sources, len(shared["times"])))
    truth[patch] = waves[0] / np.sqrt(patch.size)
    groups = [patch]
    if has_deep:
        truth[shape["deep_index"]] = 0.5 * waves[1]
        groups.append(np.array([shape["deep_index"]]))
    assert np.allclose(np.linalg.norm(truth[patch], axis=1), np.linalg.norm(truth[patch[0]]))
    for pair_number, (eeg_snr, meg_snr) in enumerate(snr_pairs):
        if len(cases) >= args.case_limit:
            break
        case_id = f"shape_{shape_number:02d}_eeg{eeg_snr:+03d}_meg{meg_snr:+03d}"
        seeds = np.random.SeedSequence([seed_root, shape_number, pair_number]).spawn(2)
        eeg, actual_eeg = erp_protocol._noise(shared["gain_eeg"] @ truth, shared["noise_factor_eeg"], baseline, active_mask, eeg_snr, seeds[0])
        meg, actual_meg = erp_protocol._noise(shared["gain_meg"] @ truth, shared["noise_factor_meg"], baseline, active_mask, meg_snr, seeds[1])
        eeg_white, eeg_gain = methods.whiten(eeg, shared["gain_eeg"])
        meg_white, meg_gain = methods.whiten(meg, shared["gain_meg"])
        data, gain = np.vstack((eeg_white, meg_white)), np.vstack((eeg_gain, meg_gain))
        data -= data[:, baseline].mean(axis=1, keepdims=True)
        # 与既有 evidence 模态融合相同；此挑战只改变空间形状先验。
        ratios = np.array([np.mean(block[:, active_mask] ** 2) / np.mean(block[:, baseline] ** 2)
                           for block in (data[:eeg_white.shape[0]], data[eeg_white.shape[0]:])])
        excess = np.maximum(ratios - 1.0, 0.0)
        weights = np.sqrt(excess / excess.max()) if excess.max() > 0 else np.ones(2)
        channel_weights = np.repeat(weights, [eeg_white.shape[0], meg_white.shape[0]])
        arrays = {"truth": truth.astype(np.float32), "times": shared["times"],
                  "vertices": vertices, "baseline": baseline, "active": active_mask,
                  "surface_indices": patch}
        case_record = {**shape, "case_id": case_id, "seed": [seed_root, shape_number, pair_number],
                       "eeg_snr_db": eeg_snr, "meg_snr_db": meg_snr,
                       "actual_snr_db": [actual_eeg, actual_meg], "diagnostics": {}}
        for version, solver in (("v3", oaster_rebuilt.reconstruct_evoked_oaster_v3_from_whitened),
                                ("v4", reconstruct_evoked_oaster_v4_from_whitened)):
            started = time.perf_counter()
            estimate, diagnostics = solver(
                data, gain, n_surf, kernels if version == "v3" else (),
                baseline=baseline, active_windows=(active_mask,),
                window_channel_weights=(channel_weights,), require_one=False,
                **({"adjacency": shared["adjacency"]} if version == "v4" else {"deep_rescue_delta": -6.0}),
            )
            assert estimate.shape == truth.shape and np.isfinite(estimate).all()
            score = metrics.evaluate_estimate(estimate, truth, vertices, groups, n_surf,
                                              active, shared["auc_cortex"], baseline=baseline)
            energy = np.sum(estimate[:n_surf, active] ** 2, axis=1)
            selected = np.flatnonzero(energy > 0.1 * energy.max(initial=0.0))
            intersection = np.intersect1d(patch, selected).size
            rows.append({"case_id": case_id, "shape": shape["shape"], "parcel": shape["parcel"],
                         "method": f"OASTER-ERP-{version}", "eeg_snr_db": eeg_snr,
                         "meg_snr_db": meg_snr, "surface_true_count": patch.size,
                         "surface_estimated_count": selected.size,
                         "surface_dice_10pct_energy": 2 * intersection / (patch.size + selected.size),
                         "elapsed_seconds": time.perf_counter() - started, **score})
            arrays["estimate_" + version] = estimate.astype(np.float32)
            case_record["diagnostics"][version] = diagnostics
        np.savez_compressed(args.output / f"{case_id}.npz", **arrays)
        cases.append(case_record)
        if (eeg_snr, meg_snr) == (5, 5):
            display_maps[shape_number] = [np.sum(arrays[key][:, active] ** 2, axis=1)
                                          for key in ("truth", "estimate_v3", "estimate_v4")]
        with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{case_id}: v3/v4 complete", flush=True)

# %% 图为全头坐标俯视投影，显示能量大于各层峰值10%的点，不是解剖渲染。
if display_maps:
    fig, axes = plt.subplots(len(display_maps), 3, figsize=(12, 3.2 * len(display_maps)), squeeze=False)
    for row_number, (shape_number, maps) in enumerate(display_maps.items()):
        for column, (label, energy) in enumerate(zip(("Truth", "OASTER v3", "OASTER v4"), maps)):
            axis = axes[row_number, column]
            axis.scatter(vertices[:n_surf, 0] * 1000, vertices[:n_surf, 1] * 1000, s=1, c="#e0e3e8", rasterized=True)
            peak = energy[:n_surf].max(initial=0.0)
            shown = np.flatnonzero(energy[:n_surf] > 0.1 * peak)
            if shown.size:
                axis.scatter(vertices[shown, 0] * 1000, vertices[shown, 1] * 1000, s=13,
                             c=energy[shown] / peak, cmap="inferno", vmin=0, vmax=1)
            axis.set(title=f"{label}: {shapes[shape_number]['shape']}", aspect="equal", xlabel="HEAD x (mm)", ylabel="HEAD y (mm)")
    fig.suptitle("EEG/MEG 5/5 dB; cortical top projection; >10% layer peak energy\nDeep sources remain in saved arrays and metrics, not projected onto cortex")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output / "shape_top_projection_5_5.png", dpi=180)
    plt.close(fig)
assert code_sha256 == {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in code_paths}, "algorithm code changed during this challenge run"
metadata = {"scope": "12-case non-Gaussian supplementary shape challenge, not the full-head seven-comparator benchmark",
            "seed_root": seed_root, "snr_pairs": snr_pairs, "snr_level": "evoked",
            "source_amplitude": "uniform within each connected surface patch; no Gaussian spatial truth",
            "deep_surface_ratio": 0.5, "code_sha256": code_sha256,
            "gain_sha256": {name: hashlib.sha256(np.ascontiguousarray(shared[name]).tobytes()).hexdigest()
                            for name in ("gain_eeg", "gain_meg")},
            "complete": len(cases) == 12, "case_count": len(cases), "cases": cases}
(args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
    default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()) + "\n", encoding="utf-8")
print(args.output, flush=True)
