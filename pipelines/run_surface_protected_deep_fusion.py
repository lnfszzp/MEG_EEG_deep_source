from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.surface_protected_fusion import (
    protected_candidate_mask,
    protected_lambda_weights,
    prune_surface_components,
    surface_protection_mask,
)
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


OUT_ROOT = RESULTS_ROOT / "surface_protected_deep_fusion"
POINT_ROOT = RESULTS_ROOT / "modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")


def run_surface_protected_solver(
    eeg_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    meg_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    adjacency: np.ndarray | sparse.spmatrix,
    n_surf: int,
    *,
    meg_point_source: np.ndarray | None = None,
    joint_point_source: np.ndarray | None = None,
    joint_sisses_source: np.ndarray | None = None,
) -> dict:
    """Run MEG-protected, EEG+MEG-refit spatial source reconstruction."""
    meg_result = run_spatial_fused_solver(
        meg_specs,
        adjacency,
        n_surf,
        point_source=meg_point_source,
        sisses_source=None,
    )
    surface_protected = surface_protection_mask(
        meg_result["stability_probability"],
        adjacency,
        n_surf,
        probability_threshold=0.50,
    )
    joint_result = run_spatial_fused_solver(
        eeg_specs + meg_specs,
        adjacency,
        n_surf,
        point_source=joint_point_source,
        sisses_source=joint_sisses_source,
    )
    final_candidate_mask = protected_candidate_mask(
        surface_protected,
        joint_result["region_mask"],
        joint_result["stability_probability"],
        n_surf,
    )
    deep_candidate_mask = np.zeros_like(final_candidate_mask)
    deep_candidate_mask[int(n_surf) :] = joint_result["region_mask"][int(n_surf) :]
    lambda_weights = protected_lambda_weights(
        final_candidate_mask.size,
        n_surf,
        surface_protected,
        deep_candidate_mask,
    )
    protected_result = run_spatial_fused_solver(
        eeg_specs + meg_specs,
        adjacency,
        n_surf,
        point_source=joint_point_source,
        sisses_source=joint_sisses_source,
        forced_candidate_mask=final_candidate_mask,
        lambda_weight_vector=lambda_weights,
    )
    pruned_region = prune_surface_components(
        protected_result["source"],
        protected_result["region_mask"],
        adjacency,
        n_surf,
        min_energy_fraction=0.15,
    )
    old_region_indices = protected_result["region_indices"]
    keep_region = pruned_region[old_region_indices]
    protected_result["region_mask"] = pruned_region
    protected_result["region_indices"] = old_region_indices[keep_region]
    if protected_result["directions"].shape[0] == keep_region.size:
        protected_result["directions"] = protected_result["directions"][keep_region]
    confidence = protected_result.get("direction_confidence")
    if confidence is not None:
        protected_result["direction_confidence"] = {
            key: np.asarray(value)[keep_region]
            for key, value in confidence.items()
        }
    protected_result["surface_protected_mask"] = surface_protected
    protected_result["surface_protected_probability"] = meg_result[
        "stability_probability"
    ]
    protected_result["joint_region_mask"] = joint_result["region_mask"]
    protected_result["joint_stability_probability"] = joint_result[
        "stability_probability"
    ]
    protected_result["final_candidate_mask"] = final_candidate_mask
    protected_result["deep_candidate_mask"] = deep_candidate_mask
    protected_result["lambda_weight_vector"] = lambda_weights
    protected_result["meg_protection_source"] = meg_result["source"]
    protected_result["joint_seed_source"] = joint_result["source"]
    protected_result["meg_protection_region_mask"] = meg_result["region_mask"]
    return protected_result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        print(f"\n==== surface protected deep fusion: {scenario} ====")
        data_dir = DATA_ROOT / scenario
        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(eeg["n_surf"]).ravel()[0])
        times = np.asarray(truth["times"], dtype=float).ravel()
        point_dir = POINT_ROOT / scenario
        result = run_surface_protected_solver(
            [(eeg["F"], eeg["Gain3D"], eeg["Gain"])],
            [(meg["F"], meg["Gain3D"], meg["Gain"])],
            adjacency,
            n_surf,
            meg_point_source=np.load(point_dir / "meg_only_result.npz")["S"],
            joint_point_source=np.load(point_dir / "both_result.npz")["S"],
            joint_sisses_source=load_sisses(
                SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
                np.asarray(eeg["Gain"]).shape[1],
            ),
        )
        save_spatial_result(
            OUT_ROOT / scenario / "both",
            result,
            times,
            extra_arrays={
                key: result[key]
                for key in (
                    "surface_protected_mask",
                    "surface_protected_probability",
                    "joint_region_mask",
                    "joint_stability_probability",
                    "final_candidate_mask",
                    "deep_candidate_mask",
                    "lambda_weight_vector",
                    "meg_protection_source",
                    "joint_seed_source",
                    "meg_protection_region_mask",
                )
            },
        )
        metrics = evaluate_spatial_result(
            result["source"],
            result["region_mask"],
            truth,
            adjacency,
        )
        selected = result["selection"].point
        row = {
            "scenario": scenario,
            "modality": "EEG + MEG",
            "selected_lambda_fraction": selected.lambda_fraction,
            "selected_beta_fraction": selected.beta_fraction,
            "selected_deep_fusion_scale": selected.deep_fusion_scale,
            "candidate_count": len(result["candidate_indices"]),
            "surface_protected_count": int(np.sum(result["surface_protected_mask"])),
            "final_candidate_count": int(np.sum(result["final_candidate_mask"])),
            "raw_region_count": int(np.sum(result["raw_region_mask"])),
            "region_count": len(result["region_indices"]),
            "surface_region_count": int(np.sum(result["region_mask"][:n_surf])),
            "deep_region_count": int(np.sum(result["region_mask"][n_surf:])),
            "relative_residual": selected.result.relative_residual,
            "admm_converged": int(selected.result.converged),
            "admm_iterations": selected.result.iterations,
            "runtime_seconds": result["runtime_seconds"],
            **metrics,
        }
        rows.append(row)
        print(
            f"EEG+MEG protected: range={row['region_count']} "
            f"dice={row['cortical_dice']:.3f} "
            f"deep={row['deep_region_count']} "
            f"lambda={row['selected_lambda_fraction']:.3f} "
            f"beta={row['selected_beta_fraction']:.3f}"
        )
    write_csv(OUT_ROOT / "surface_protected_metrics.csv", rows)
    print("\nSaved:", OUT_ROOT)


if __name__ == "__main__":
    main()
