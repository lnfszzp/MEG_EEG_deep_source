"""Oracle metric audit only; never construct or modify a localization estimate.

The simulator patch helper is used solely to measure the frozen metric's
geometric floor and must not be imported by a method implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(os.environ.get("V15_BENCHMARK_ROOT", ROOT / "benchmark"))

from benchmark.protocol import _surface_patch, load_shared
from metrics.user_metrics import DLE_an, SD  # noqa: E402


def audit(panel: str) -> dict:
    shared = load_shared()
    vertices = np.asarray(shared["vertices"], dtype=float)
    n_surf = int(shared["n_surf"])
    manifest = json.loads(
        (BENCHMARK_ROOT / "results" / "protocol" / f"{panel}_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    surface_centers = sorted({int(center) for case in manifest for center in case["surface_centers"]})
    tree = cKDTree(vertices[:n_surf])
    exact_truth_dle = []
    best_grid_dle = []
    exact_truth_sd = []
    for center in surface_centers:
        group, weights = _surface_patch(shared, center)
        source = np.zeros((n_surf, 1), dtype=float)
        source[group, 0] = weights
        centroid = vertices[group].mean(axis=0)
        exact_truth_dle.append(
            float(DLE_an(source, [group + 1], vertices[:n_surf], index_base=1) * 1000.0)
        )
        exact_truth_sd.append(float(SD(source, vertices[group], vertices[:n_surf], 0.0) * 1000.0))
        best_grid_dle.append(float(tree.query(centroid)[0] * 1000.0))

    exact_truth_dle = np.asarray(exact_truth_dle)
    best_grid_dle = np.asarray(best_grid_dle)
    exact_truth_sd = np.asarray(exact_truth_sd)
    if np.any(best_grid_dle > exact_truth_dle + 1e-9):
        raise AssertionError("nearest-grid DLE cannot exceed the truth-peak DLE")
    return {
        "panel": panel,
        "surface_centers": len(surface_centers),
        "surface_grid_points": n_surf,
        "deep_grid_points": int(shared["n_deep"]),
        "target_mm": 1.0,
        "surface_exact_truth_sd_mean_mm": float(exact_truth_sd.mean()),
        "surface_exact_truth_dle_mean_mm": float(exact_truth_dle.mean()),
        "surface_exact_truth_dle_max_mm": float(exact_truth_dle.max()),
        "surface_best_existing_grid_dle_mean_mm": float(best_grid_dle.mean()),
        "surface_best_existing_grid_dle_max_mm": float(best_grid_dle.max()),
        "surface_centers_with_best_grid_dle_below_1mm": int(np.sum(best_grid_dle < 1.0)),
        "surface_dle_target_reachable_on_grid": bool(np.all(best_grid_dle < 1.0)),
        "deep_singleton_oracle_sd_mm": 0.0,
        "deep_singleton_oracle_dle_mm": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.panel)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
