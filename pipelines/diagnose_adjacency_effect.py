from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.spatial_fused_fusion import (
    graph_total_variation,
    range_metrics,
)
from pipelines.run_spatial_fused_fusion import (
    OUT_ROOT,
    POINT_ROOT,
    SISSES_ROOT,
    evaluate_spatial_result,
    run_spatial_fused_solver,
    write_csv,
)
from pipelines.run_whole_brain_fusion import (
    DATA_ROOT,
    SCENARIOS,
    load_mat,
    load_sisses,
    source_amplitude,
)


def _amplitude_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_amplitude = source_amplitude(first)
    second_amplitude = source_amplitude(second)
    if np.std(first_amplitude) == 0 or np.std(second_amplitude) == 0:
        return np.nan
    return float(np.corrcoef(first_amplitude, second_amplitude)[0, 1])


def _jump_count(
    coefficients: np.ndarray,
    adjacency: sparse.spmatrix,
) -> int:
    graph = sparse.coo_matrix(adjacency)
    keep = graph.row < graph.col
    left = graph.row[keep]
    right = graph.col[keep]
    if left.size == 0:
        return 0
    amplitude = np.linalg.norm(coefficients, axis=1)
    limit = 0.05 * amplitude.max(initial=0.0)
    jumps = np.linalg.norm(coefficients[left] - coefficients[right], axis=1)
    return int(np.sum(jumps >= limit)) if limit > 0 else 0


def main() -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        print(f"\n==== adjacency diagnostic: {scenario} ====")
        data_dir = DATA_ROOT / scenario
        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(eeg["n_surf"]).ravel()[0])
        n_sources = np.asarray(eeg["Gain"]).shape[1]
        point = np.load(
            POINT_ROOT / scenario / "whole_brain_result.npz",
            allow_pickle=True,
        )
        point_source = np.asarray(point["S"], dtype=float)
        sisses_source = load_sisses(
            SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
            n_sources,
        )
        saved = np.load(
            OUT_ROOT / scenario / "spatial_fused_result.npz",
            allow_pickle=True,
        )
        selected_lambda = float(saved["selected_lambda_fraction"])
        selected_beta = float(saved["selected_beta_fraction"])
        solutions = {}
        for label, beta in (("beta_zero", 0.0), ("graph_fused", selected_beta)):
            solutions[label] = run_spatial_fused_solver(
                [
                    (eeg["F"], eeg["Gain3D"], eeg["Gain"]),
                    (meg["F"], meg["Gain3D"], meg["Gain"]),
                ],
                adjacency,
                n_surf,
                point_source=point_source,
                sisses_source=sisses_source,
                lambda_fractions=(selected_lambda,),
                beta_fractions=(beta,),
            )

        beta_zero = solutions["beta_zero"]
        fused = solutions["graph_fused"]
        correlation = _amplitude_correlation(
            beta_zero["source"],
            fused["source"],
        )
        for label, beta, result in (
            ("beta_zero", 0.0, beta_zero),
            ("graph_fused", selected_beta, fused),
        ):
            candidates = result["candidate_indices"]
            candidate_graph = adjacency[candidates][:, candidates]
            deep_mask = candidates >= n_surf
            metrics = evaluate_spatial_result(
                result["source"],
                result["region_mask"],
                truth,
                adjacency,
            )
            all_range = range_metrics(
                result["region_mask"],
                source_amplitude(np.asarray(truth["s_true"])) > 0,
                adjacency,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "solution": label,
                    "lambda_fraction": selected_lambda,
                    "beta_fraction": beta,
                    "graph_total_variation": graph_total_variation(
                        result["candidate_coefficients"],
                        candidate_graph,
                        deep_mask=deep_mask,
                    ),
                    "graph_jump_count": _jump_count(
                        result["candidate_coefficients"],
                        candidate_graph,
                    ),
                    "region_count": len(result["region_indices"]),
                    "region_component_count": all_range[
                        "estimated_component_count"
                    ],
                    "relative_residual": result[
                        "selection"
                    ].point.result.relative_residual,
                    "map_correlation_between_beta_solutions": correlation,
                    "all_auc": metrics["all_auc"],
                    "all_rmse": metrics["all_rmse"],
                    "all_sd_mm": metrics["all_sd_mm"],
                    "cortical_dice": metrics["cortical_dice"],
                    "surface_peak_dle_mm": metrics["surface_peak_dle_mm"],
                    "deep_peak_dle_mm": metrics["deep_peak_dle_mm"],
                    "simultaneous_surface_deep_detected": metrics[
                        "simultaneous_surface_deep_detected"
                    ],
                }
            )
            print(
                f"{label}: beta={beta:.3f} "
                f"TV={rows[-1]['graph_total_variation']:.3f} "
                f"jumps={rows[-1]['graph_jump_count']} "
                f"range={rows[-1]['region_count']} "
                f"AUC={rows[-1]['all_auc']:.3f}"
            )
    write_csv(OUT_ROOT / "adjacency_effect_diagnostic.csv", rows)
    print("\nSaved:", OUT_ROOT / "adjacency_effect_diagnostic.csv")


if __name__ == "__main__":
    main()
