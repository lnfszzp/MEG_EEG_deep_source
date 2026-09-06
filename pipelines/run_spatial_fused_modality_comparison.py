from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_spatial_fused_fusion import (
    DATA_ROOT,
    RESULTS_ROOT,
    SISSES_ROOT,
    evaluate_spatial_result,
    run_spatial_fused_solver,
    save_spatial_result,
    write_csv,
)
from pipelines.run_whole_brain_fusion import load_mat, load_sisses


OUT_ROOT = RESULTS_ROOT / "spatial_fused_modality_comparison"
POINT_ROOT = RESULTS_ROOT / "modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        print(f"\n==== spatial fused modality: {scenario} ====")
        data_dir = DATA_ROOT / scenario
        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(eeg["n_surf"]).ravel()[0])
        times = np.asarray(truth["times"], dtype=float).ravel()
        point_dir = POINT_ROOT / scenario
        configs = {
            "Both": (
                [
                    (eeg["F"], eeg["Gain3D"], eeg["Gain"]),
                    (meg["F"], meg["Gain3D"], meg["Gain"]),
                ],
                np.load(point_dir / "both_result.npz")["S"],
                load_sisses(
                    SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
                    np.asarray(eeg["Gain"]).shape[1],
                ),
            ),
            "MEG-only": (
                [(meg["F"], meg["Gain3D"], meg["Gain"])],
                np.load(point_dir / "meg_only_result.npz")["S"],
                None,
            ),
            "EEG-only": (
                [(eeg["F"], eeg["Gain3D"], eeg["Gain"])],
                np.load(point_dir / "eeg_only_result.npz")["S"],
                None,
            ),
        }
        out_dir = OUT_ROOT / scenario
        for modality, (specs, point_source, sisses_source) in configs.items():
            result = run_spatial_fused_solver(
                specs,
                adjacency,
                n_surf,
                point_source=point_source,
                sisses_source=sisses_source,
            )
            stem = modality.lower().replace("-", "_")
            save_spatial_result(out_dir / stem, result, times)
            metrics = evaluate_spatial_result(
                result["source"],
                result["region_mask"],
                truth,
                adjacency,
            )
            selected = result["selection"].point
            row = {
                "scenario": scenario,
                "modality": modality,
                "selected_lambda_fraction": selected.lambda_fraction,
                "selected_beta_fraction": selected.beta_fraction,
                "selected_deep_fusion_scale": selected.deep_fusion_scale,
                "candidate_count": len(result["candidate_indices"]),
                "raw_region_count": int(np.sum(result["raw_region_mask"])),
                "region_count": len(result["region_indices"]),
                "relative_residual": selected.result.relative_residual,
                "admm_converged": int(selected.result.converged),
                "admm_iterations": selected.result.iterations,
                "runtime_seconds": result["runtime_seconds"],
                **metrics,
            }
            rows.append(row)
            print(
                f"{modality}: range={row['region_count']} "
                f"dice={row['cortical_dice']:.3f} "
                f"deep={row['deep_range_count']} "
                f"lambda={row['selected_lambda_fraction']:.3f} "
                f"beta={row['selected_beta_fraction']:.3f}"
            )
    write_csv(OUT_ROOT / "spatial_fused_modality_summary.csv", rows)
    print("\nSaved:", OUT_ROOT)


if __name__ == "__main__":
    main()
