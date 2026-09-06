from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.comparison_renderers import render_cortical_set


PROTECTED_ROOT = ROOT / "results" / "latest" / "surface_protected_deep_fusion"
BASELINE_ROOT = ROOT / "results" / "latest" / "spatial_fused_modality_comparison"
OUT_ROOT = ROOT / "results" / "latest" / "figures" / "surface_protected_cortical"
SCENARIOS = ("surface_only", "deep_plus_surface", "deep_plus_two_surface")


def row_specs(scenario: str):
    return (
        ("Ground truth", None),
        (
            "Protected EEG + MEG",
            PROTECTED_ROOT / scenario / "both" / "spatial_fused_result.npz",
        ),
        ("MEG-only", BASELINE_ROOT / scenario / "meg_only" / "spatial_fused_result.npz"),
        ("EEG-only", BASELINE_ROOT / scenario / "eeg_only" / "spatial_fused_result.npz"),
    )


def main() -> None:
    render_cortical_set(
        SCENARIOS,
        OUT_ROOT,
        row_specs,
        title_template="Surface-protected source-range reconstruction: {scenario}",
        filename_template="surface_protected_cortical_{scenario}.png",
        print_label="surface-protected cortical comparison",
    )


if __name__ == "__main__":
    main()
