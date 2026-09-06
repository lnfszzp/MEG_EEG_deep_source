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


PROTECTED_ROOT = ROOT / "results" / "latest" / "surface_protected_deep_fusion"
BASELINE_ROOT = ROOT / "results" / "latest" / "spatial_fused_modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")
ROWS = (
    ("Protected EEG + MEG", PROTECTED_ROOT, "both"),
    ("MEG-only", BASELINE_ROOT, "meg_only"),
    ("EEG-only", BASELINE_ROOT, "eeg_only"),
)


def _load_result(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    result = np.load(path, allow_pickle=True)
    source = np.asarray(result["S"], dtype=float)
    region = np.asarray(result["region_mask"], dtype=bool).ravel()
    return source, region, dict(result)


def main() -> None:
    rows: list[dict] = []
    for scenario in SCENARIOS:
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        eeg = load_mat(DATA_ROOT / scenario / "sub_EEG.mat")
        adjacency = sparse.csr_matrix(eeg["VertConn"])
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        for modality, root, stem in ROWS:
            source, region, result = _load_result(
                root / scenario / stem / "spatial_fused_result.npz"
            )
            metrics = evaluate_spatial_result(
                source,
                region,
                truth,
                adjacency,
            )
            row = {
                "scenario": scenario,
                "modality": modality,
                "stable_region_count": int(region.sum()),
                "surface_region_count": int(region[:n_surf].sum()),
                "deep_region_count": int(region[n_surf:].sum()),
                **metrics,
            }
            for key in (
                "selected_lambda_fraction",
                "selected_beta_fraction",
                "selected_deep_fusion_scale",
            ):
                if key in result:
                    row[key] = float(np.asarray(result[key]).ravel()[0])
            for key in (
                "surface_protected_mask",
                "final_candidate_mask",
                "deep_candidate_mask",
            ):
                if key in result:
                    row[f"{key}_count"] = int(np.asarray(result[key], dtype=bool).sum())
            rows.append(row)
    path = PROTECTED_ROOT / "surface_protected_metrics.csv"
    write_csv(path, rows)
    print("Saved:", path)


if __name__ == "__main__":
    main()
