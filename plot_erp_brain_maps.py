"""Render one development ERP case with the exact benchmark observation and methods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

import plot_strict_brain_maps as brain_maps
import plot_strict_case as strict_case
import protected_multilayer as protected
import run_erp_whole_head_matrix as erp_run
import run_strict_comparators as comparators
import run_strict_oaster as archive
from benchmark import erp_protocol
from candidates import oaster_rebuilt as oaster


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = (
    ROOT / "results" / "erp_whole_head" / "development_multisite_v2" / "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT / "results" / "erp_whole_head" / "development_multisite_v2" / "brain_maps"
)
OASTER_METHOD = erp_run._oaster_method("v2")
METHODS = (OASTER_METHOD, *comparators.METHODS)
METHOD_SLUGS = {OASTER_METHOD: "oaster_erp_v2", **comparators.METHOD_SLUGS}


def reconstruct_case(
    shared: dict,
    kernels: tuple,
    case: dict,
    manifest_sha256: str,
    seed_root: int,
    deep_rescue_delta: float,
) -> dict:
    """Capture the exact arrays passed through the single-case benchmark runner."""
    estimates: dict[str, np.ndarray] = {}
    captured: dict[str, tuple] = {}
    original_simulate = erp_protocol.simulate_case
    original_success = erp_run._success_row

    def capture_simulation(*args, **kwargs):
        captured["simulation"] = original_simulate(*args, **kwargs)
        return captured["simulation"]

    def capture_success(base, method, estimate, *args, **kwargs):
        estimates[method] = estimate
        return original_success(base, method, estimate, *args, **kwargs)

    runtime = {
        "shared": shared,
        "kernels": kernels,
        "algorithm_version": "v2",
        "oaster_solver": erp_run._resolve_oaster("v2"),
        "modality_weighting": "evidence",
        "seed_root": int(seed_root),
        "methods": METHODS,
        "oaster_kwargs": {"deep_rescue_delta": float(deep_rescue_delta)},
    }
    # Reuse the benchmark path itself so plotting cannot silently drift from scoring.
    with patch.object(erp_protocol, "simulate_case", capture_simulation), patch.object(
        erp_run, "_success_row", capture_success
    ):
        rows = erp_run._score_case(case, runtime, manifest_sha256)
    failures = [row["error"] for row in rows.values() if row["status"] != "ok"]
    if failures or tuple(rows) != METHODS or set(estimates) != set(METHODS):
        raise RuntimeError(f"ERP reconstruction failed: {failures or 'method mismatch'}")
    eeg, meg, truth, groups, baseline, _windows, active, metadata = captured["simulation"]
    return {
        "eeg": eeg,
        "meg": meg,
        "truth": truth,
        "groups": groups,
        "baseline": baseline,
        "active": active,
        "simulation_metadata": metadata,
        "estimates": estimates,
        "rows": [rows[method] for method in METHODS],
    }


def plot_erp_brain_maps(
    manifest_path: Path = DEFAULT_MANIFEST,
    data_root: Path = erp_run.DEFAULT_DATA_ROOT,
    sample_path: Path = erp_run.DEFAULT_SAMPLE_PATH,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    case_id: str | None = None,
    case_number: int | None = None,
    eeg_snr_db: int | None = None,
    meg_snr_db: int | None = None,
    scenario: str | None = None,
    location: int | None = None,
    seed_root: int | None = None,
    deep_rescue_delta: float = oaster.ERP_V2_DEEP_RESCUE_DELTA,
    relative_threshold: float = 0.10,
    display_percentile: float = 95.0,
    dpi: int = 160,
) -> Path:
    if case_id is None and case_number is None and all(
        value is None for value in (eeg_snr_db, meg_snr_db, scenario, location)
    ):
        eeg_snr_db, meg_snr_db, scenario, location = 0, 0, "deep_plus_two_surface", 0
    if (
        not 0 < relative_threshold < 1
        or not 0 <= display_percentile < 100
        or dpi < 72
        or not np.isfinite(deep_rescue_delta)
        or (seed_root is not None and seed_root < 0)
    ):
        raise ValueError("invalid display, seed, or deep-rescue parameter")

    cases, manifest_sha256 = archive._load_manifest(Path(manifest_path))
    case = strict_case._select_case(
        cases,
        case_id,
        case_number,
        eeg_snr_db=eeg_snr_db,
        meg_snr_db=meg_snr_db,
        scenario=scenario,
        location=location,
    )
    if seed_root is None:
        seed = case.get("seed", [erp_protocol.ERP_SEED_ROOT])
        seed_root = int(seed[0])
    shared = erp_run.protocol.load_shared(Path(data_root), Path(sample_path))
    kernels = protected.connected_euclidean_surface_kernels(
        shared["vertices"],
        shared["adjacency"],
        shared["n_surf"],
        scales_mm=oaster.SURFACE_SCALES_MM,
    )
    result = reconstruct_case(
        shared,
        kernels,
        case,
        manifest_sha256,
        seed_root,
        deep_rescue_delta,
    )
    display_case = {
        **case,
        "active_window_s": result["simulation_metadata"]["active_window_s"],
    }
    loaded = {
        "case": display_case,
        "geometry": shared,
        "truth": result["truth"],
        "groups": result["groups"],
        "active": result["active"],
        "baseline": result["baseline"],
    }
    anatomy = brain_maps.load_anatomy(Path(sample_path))
    surface = brain_maps._validate_surface_source_space(
        shared, shared["src_surface"], shared["subjects_dir"]
    )
    case_dir = Path(output_root).resolve() / f"case_{int(case['case_number']):05d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    top_view = {
        "surface_name": "pial",
        "hemi": "both",
        "views": ("dorsal",),
        "view_layout": "horizontal",
        "size": (1000, 800),
    }

    truth_mri = brain_maps.render_method(
        "Simulated ERP truth",
        result["truth"],
        None,
        loaded,
        anatomy,
        case_dir / "simulation_truth_mri.png",
        relative_threshold,
        dpi,
        display_percentile=0,
        is_truth=True,
    )
    truth_surface = brain_maps.render_surface_method(
        "Simulated ERP truth",
        result["truth"],
        loaded,
        surface,
        case_dir / "simulation_truth_surface_top.png",
        dpi,
        relative_threshold=relative_threshold,
        is_truth=True,
        **top_view,
    )
    if case.get("deep_index") is not None:
        brain_maps.render_anatomy_surface_pair(
            truth_mri,
            truth_surface,
            case_dir / "simulation_truth_mri_surface.png",
            dpi,
            f"Simulated ERP truth | MRI + complete dorsal cortical render\n"
            f"{brain_maps._case_title(display_case)}",
            crop_fraction=0.0,
        )

    metric_by_method = {row["method"]: row for row in result["rows"]}
    mri_paths = []
    surface_paths = []
    combined_paths = []
    for method in METHODS:
        slug = METHOD_SLUGS[method]
        mri_path = brain_maps.render_method(
            method,
            result["estimates"][method],
            metric_by_method[method],
            loaded,
            anatomy,
            case_dir / f"{slug}_mri.png",
            relative_threshold,
            dpi,
            display_percentile=display_percentile,
        )
        surface_path = brain_maps.render_surface_method(
            method,
            result["estimates"][method],
            loaded,
            surface,
            case_dir / f"{slug}_surface_top.png",
            dpi,
            relative_threshold=relative_threshold,
            **top_view,
        )
        mri_paths.append((method, mri_path))
        surface_paths.append((method, surface_path))
        if case.get("deep_index") is not None:
            combined = brain_maps.render_anatomy_surface_pair(
                mri_path,
                surface_path,
                case_dir / f"{slug}_mri_surface.png",
                dpi,
                f"{method} | MRI + complete dorsal cortical render\n"
                f"{brain_maps._case_title(display_case)}",
                crop_fraction=0.0,
            )
            combined_paths.append((method, combined))

    title = brain_maps._case_title(display_case)
    brain_maps.render_montage(
        mri_paths,
        case_dir / "all_algorithms_mri.png",
        dpi,
        f"Eight algorithm estimates on anatomical MRI | {title}",
    )
    brain_maps.render_montage(
        surface_paths,
        case_dir / "all_algorithms_surface_top.png",
        dpi,
        f"Eight algorithm estimates | complete dorsal pial view | {title}",
    )
    if combined_paths:
        brain_maps.render_montage(
            combined_paths,
            case_dir / "all_algorithms_mri_surface.png",
            dpi,
            f"Eight algorithm estimates | MRI + complete dorsal pial view | {title}",
        )
    brain_maps._write_metrics(case_dir / "metrics.csv", result["rows"])
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "case": display_case,
                "manifest": str(Path(manifest_path).resolve()),
                "manifest_sha256": manifest_sha256,
                "erp_seed_root": seed_root,
                "algorithm_version": "v2",
                "modality_weighting": "evidence",
                "deep_rescue_delta": deep_rescue_delta,
                "methods": list(METHODS),
                "active_samples": result["active"].tolist(),
                "observation_sha256": {
                    name: hashlib.sha256(np.ascontiguousarray(values)).hexdigest()
                    for name, values in (("EEG", result["eeg"]), ("MEG", result["meg"]))
                },
                "truth_is_separate": True,
                "algorithm_truth_overlay": "none",
                "cortical_render": "complete dorsal pial; both hemispheres; classic cortex; inferno; white background; no image cropping",
                "deep_case_render": "each truth/method has anatomical MRI plus cortical render",
                "display_rule": brain_maps._display_rule(
                    relative_threshold, display_percentile
                ),
                "simulation_metadata": result["simulation_metadata"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--case-id")
    selector.add_argument("--case-number", type=int)
    parser.add_argument("--eeg-snr-db", type=int)
    parser.add_argument("--meg-snr-db", type=int)
    parser.add_argument("--scenario", choices=erp_run.SCENARIOS)
    parser.add_argument("--location", type=int)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=erp_run.DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-path", type=Path, default=erp_run.DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-root", type=int)
    parser.add_argument(
        "--deep-rescue-delta", type=float, default=oaster.ERP_V2_DEEP_RESCUE_DELTA
    )
    parser.add_argument("--relative-threshold", type=float, default=0.10)
    parser.add_argument("--display-percentile", type=float, default=95.0)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    output = plot_erp_brain_maps(
        args.manifest,
        args.data_root,
        args.sample_path,
        args.output_root,
        case_id=args.case_id,
        case_number=args.case_number,
        eeg_snr_db=args.eeg_snr_db,
        meg_snr_db=args.meg_snr_db,
        scenario=args.scenario,
        location=args.location,
        seed_root=args.seed_root,
        deep_rescue_delta=args.deep_rescue_delta,
        relative_threshold=args.relative_threshold,
        display_percentile=args.display_percentile,
        dpi=args.dpi,
    )
    print(f"Saved ERP brain maps: {output}")


if __name__ == "__main__":
    main()
