from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.run_whole_brain_fusion import DATA_ROOT, RESULTS_ROOT, load_mat, load_sisses
from pipelines.sisses_direct_utils import best_estimated_waveform, group_waveform, true_source_groups


SISSES_ROOT = RESULTS_ROOT / "sisses_warm_start"
OUT_ROOT = RESULTS_ROOT / "figures" / "sisses_waveforms"
SCENARIOS = ("deep_only", "surface_only", "deep_plus_surface", "deep_plus_two_surface")


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    scale = float(np.max(np.abs(x), initial=0.0))
    return x / scale if scale > 0 else x


def draw_scenario(scenario: str) -> None:
    truth = load_mat(DATA_ROOT / scenario / "s_true.mat")
    true_source = np.asarray(truth["s_true"], dtype=float)
    sisses = load_sisses(
        SISSES_ROOT / scenario / "s_wen_sisses_fusion_quick_best.mat",
        true_source.shape[0],
    )
    times = np.asarray(truth["times"], dtype=float).ravel()
    groups = true_source_groups(truth)
    fig, axes = plt.subplots(len(groups), 1, figsize=(8.5, 2.4 * len(groups)), squeeze=False)
    for row, group in enumerate(groups):
        ax = axes[row, 0]
        true_wave = group_waveform(true_source, group)
        peak, est_wave = best_estimated_waveform(sisses, group)
        ax.plot(times, _normalize(true_wave), color="black", linewidth=2.0, label="truth")
        ax.plot(times, _normalize(est_wave), color="#0072b2", linewidth=1.6, label=f"SISSES idx {peak + 1}")
        ax.axvline(times[min(200, len(times) - 1)], color="0.75", linewidth=1.0)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel(f"group {row + 1}")
        ax.spines[["top", "right"]].set_visible(False)
        if row == 0:
            ax.legend(loc="upper right", frameon=False)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"SISSES waveform comparison: {scenario}", fontsize=13)
    fig.tight_layout()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_ROOT / f"sisses_waveforms_{scenario}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    for scenario in SCENARIOS:
        draw_scenario(scenario)
        print("Rendered SISSES waveform", scenario)
    print("Saved:", OUT_ROOT)


if __name__ == "__main__":
    main()
