from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_spatial_fused_fusion import evaluate_spatial_result, write_csv
from pipelines.run_whole_brain_fusion import DATA_ROOT, load_mat


RESULT_ROOT = ROOT / "results" / "latest" / "spatial_fused_modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")
MODALITIES = (
    ("EEG + MEG", "both"),
    ("MEG-only", "meg_only"),
    ("EEG-only", "eeg_only"),
)


def main() -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        eeg = load_mat(DATA_ROOT / scenario / "sub_EEG.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        true_surface = np.asarray(
            truth["true_surface_indices0"],
            dtype=int,
        ).ravel()
        has_deep = bool(int(np.asarray(truth["has_deep_source"]).ravel()[0]))
        true_deep = int(np.asarray(truth["true_deep_idx0"]).ravel()[0])
        for modality, stem in MODALITIES:
            result = np.load(
                RESULT_ROOT / scenario / stem / "spatial_fused_result.npz",
                allow_pickle=True,
            )
            source = np.asarray(result["S"], dtype=float)
            region = np.asarray(result["region_mask"], dtype=bool).ravel()
            raw_region = np.asarray(
                result["raw_region_mask"],
                dtype=bool,
            ).ravel()
            candidates = np.asarray(
                result["candidate_indices0"],
                dtype=int,
            ).ravel()
            stability = np.asarray(
                result["stability_probability"],
                dtype=float,
            ).ravel()
            metrics = evaluate_spatial_result(
                source,
                region,
                truth,
                adjacency,
            )
            candidate_surface_recall = (
                float(np.mean(np.isin(true_surface, candidates)))
                if true_surface.size
                else np.nan
            )
            rows.append(
                {
                    "scenario": scenario,
                    "modality": modality,
                    "selected_lambda_fraction": float(
                        result["selected_lambda_fraction"]
                    ),
                    "selected_beta_fraction": float(
                        result["selected_beta_fraction"]
                    ),
                    "selected_deep_fusion_scale": float(
                        result["selected_deep_fusion_scale"]
                    ),
                    "candidate_count": int(candidates.size),
                    "candidate_surface_recall": candidate_surface_recall,
                    "candidate_deep_covered": (
                        int(true_deep in candidates) if has_deep else -1
                    ),
                    "raw_region_count": int(raw_region.sum()),
                    "stable_region_count": int(region.sum()),
                    "mean_selected_stability": (
                        float(np.mean(stability[region]))
                        if np.any(region)
                        else np.nan
                    ),
                    "surface_region_count": int(region[:n_surf].sum()),
                    "deep_region_count": int(region[n_surf:].sum()),
                    **metrics,
                }
            )
    path = RESULT_ROOT / "stable_modality_metrics.csv"
    write_csv(path, rows)
    print("Saved:", path)


if __name__ == "__main__":
    main()
