from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.comparison_renderers import render_cortical_set


RESULT_ROOT = ROOT / "results" / "latest" / "spatial_fused_modality_comparison"
OUT_ROOT = ROOT / "results" / "latest" / "figures" / "stable_modality_cortical"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")


def row_specs(scenario: str):
    scenario_root = RESULT_ROOT / scenario
    return (
        ("Ground truth", None),
        ("EEG + MEG", scenario_root / "both" / "spatial_fused_result.npz"),
        ("MEG-only", scenario_root / "meg_only" / "spatial_fused_result.npz"),
        ("EEG-only", scenario_root / "eeg_only" / "spatial_fused_result.npz"),
    )


def main() -> None:
    render_cortical_set(
        SCENARIOS,
        OUT_ROOT,
        row_specs,
        title_template="Stable multimodal source-range reconstruction: {scenario}",
        filename_template="stable_modality_cortical_{scenario}.png",
        print_label="stable cortical comparison",
    )


if __name__ == "__main__":
    main()
