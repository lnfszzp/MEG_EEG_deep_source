from __future__ import annotations

from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.external_metrics import external_full_head_metrics
from pipelines.run_whole_brain_fusion import DATA_ROOT, RESULTS_ROOT, load_mat, load_sisses
from pipelines.sisses_direct_utils import thresholded_source, true_source_groups


SISSES_ROOT = RESULTS_ROOT / "sisses_warm_start"
OUT_ROOT = RESULTS_ROOT / "sisses_direct"
SCENARIOS = ("deep_only", "surface_only", "deep_plus_surface", "deep_plus_two_surface")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
        n_sources = np.asarray(truth["s_true"]).shape[0]
        sisses = load_sisses(
            SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
            n_sources,
        )
        ranged, mask = thresholded_source(sisses, relative_threshold=0.10)
        metrics = external_full_head_metrics(
            ranged,
            np.asarray(truth["s_true"], dtype=float),
            np.asarray(truth["src_vertices"], dtype=float),
            true_groups=true_source_groups(truth),
        )
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        rows.append(
            {
                "scenario": scenario,
                "method": "SISSES_direct",
                "threshold": 0.10,
                "active_count": int(mask.sum()),
                "surface_active_count": int(mask[:n_surf].sum()),
                "deep_active_count": int(mask[n_surf:].sum()),
                "auc": metrics["auc"],
                "rmse": metrics["rmse"],
                "sd_mm": metrics["sd_mm"],
                "dle_mm": metrics["dle_mm"],
            }
        )
    write_csv(OUT_ROOT / "sisses_direct_metrics.csv", rows)
    print("Saved:", OUT_ROOT / "sisses_direct_metrics.csv")


if __name__ == "__main__":
    main()
