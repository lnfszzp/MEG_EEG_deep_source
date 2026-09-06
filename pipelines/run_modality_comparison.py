from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from pipelines.run_whole_brain_fusion import (
    DATA_ROOT,
    load_mat,
    metric_row,
    run_single_modality_solver,
    run_whole_brain_solver,
    source_amplitude,
    surface_patch_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results" / "latest" / "modality_comparison"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")


def count_active_regions(
    active_indices: np.ndarray,
    adjacency: np.ndarray,
    n_surf: int,
) -> tuple[int, int]:
    active_indices = np.asarray(active_indices, dtype=int).ravel()
    cortical = active_indices[active_indices < int(n_surf)]
    deep = active_indices[active_indices >= int(n_surf)]

    if cortical.size:
        cortical_graph = sparse.csr_matrix(adjacency)[cortical][:, cortical]
        cortical_regions = int(
            connected_components(cortical_graph, directed=False)[0]
        )
    else:
        cortical_regions = 0
    if deep.size:
        deep_graph = sparse.csr_matrix(adjacency)[deep][:, deep]
        deep_regions = int(connected_components(deep_graph, directed=False)[0])
    else:
        deep_regions = 0
    return cortical_regions, deep_regions


def save_result(path: Path, result: dict, times: np.ndarray) -> None:
    np.savez(
        path,
        S=result["source"],
        times=times,
        active_indices0=result["active_indices"],
        group_norms=result["group_norms"],
        directions=result["directions"],
        selected_lambda_fraction=np.array(
            result["selection"].point.lambda_fraction
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def chinese_summary_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        result.append(
            {
                "场景": row["scenario"],
                "模态": row["modality"],
                "激活网格点数": row["active_source_count"],
                "估计皮层区域数": row["estimated_cortical_region_count"],
                "估计深部区域数": row["estimated_deep_region_count"],
                "全局最强峰值编号": row["global_peak_idx1"],
                "深部最强峰值编号": row["deep_peak_idx1"],
                "深部定位误差_mm": row["deep_dle_mm"],
                "真实皮层patch数": row["true_surface_patch_count"],
                "检出皮层patch数": row["detected_surface_patch_count"],
                "patch中心平均误差_mm": row["surface_patch_center_dle_mean_mm"],
                "patch中心最大误差_mm": row["surface_patch_center_dle_max_mm"],
                "皮层能量比例": row["surface_energy_ratio"],
                "深部能量比例": row["deep_energy_ratio"],
                "BIC选择的lambda比例": row["selected_lambda_fraction"],
                "方向重拟合相对残差": row["rank_one_refit_residual"],
            }
        )
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []

    for scenario in SCENARIOS:
        print(f"\n==== modality comparison: {scenario} ====")
        data_dir = DATA_ROOT / scenario
        out_dir = OUT_ROOT / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        eeg = load_mat(data_dir / "sub_EEG.mat")
        meg = load_mat(data_dir / "sub_MEG.mat")
        truth = load_mat(data_dir / "s_true.mat")
        times = np.asarray(truth["times"], dtype=float).ravel()
        n_surf = int(np.asarray(truth["n_surf"]).ravel()[0])
        adjacency = np.asarray(eeg["VertConn"])

        results = {
            "Both": run_whole_brain_solver(
                eeg["F"],
                eeg["Gain3D"],
                meg["F"],
                meg["Gain3D"],
                None,
                eeg_localization_gain=eeg["Gain"],
                meg_localization_gain=meg["Gain"],
                run_cold_comparison=False,
            ),
            "MEG-only": run_single_modality_solver(
                meg["F"],
                meg["Gain3D"],
                meg["Gain"],
            ),
            "EEG-only": run_single_modality_solver(
                eeg["F"],
                eeg["Gain3D"],
                eeg["Gain"],
            ),
        }

        for modality, result in results.items():
            file_stem = modality.lower().replace("-", "_")
            save_result(out_dir / f"{file_stem}_result.npz", result, times)

            row = metric_row(scenario, result["source"], truth)
            row.update(surface_patch_metrics(result["source"], truth))
            cortical_regions, deep_regions = count_active_regions(
                result["active_indices"],
                adjacency,
                n_surf,
            )
            amplitude = source_amplitude(result["source"])
            row.update(
                {
                    "modality": modality,
                    "active_source_count": len(result["active_indices"]),
                    "estimated_cortical_region_count": cortical_regions,
                    "estimated_deep_region_count": deep_regions,
                    "selected_lambda_fraction": result["selection"].point.lambda_fraction,
                    "rank_one_refit_residual": result["rank_one_refit_residual"],
                    "global_peak_amplitude": float(amplitude.max()),
                    "runtime_seconds": float(
                        result.get("warm_runtime_seconds", result.get("runtime_seconds", np.nan))
                    ),
                }
            )
            rows.append(row)
            print(
                f"{modality}: active={row['active_source_count']} "
                f"cortex_regions={cortical_regions} deep_regions={deep_regions} "
                f"surfaceDLE={row['surface_dle_to_patch_mm']:.3f}"
            )

    write_csv(OUT_ROOT / "modality_summary.csv", rows)
    write_csv(
        OUT_ROOT / "modality_summary_zh.csv",
        chinese_summary_rows(rows),
    )
    print(f"\nSaved: {OUT_ROOT}")


if __name__ == "__main__":
    main()
