"""Development-only EBIC deep rescue experiment for rebuilt OASTER.

The experiment simulates fresh development observations.  It does not load the
strict-blind archive and it leaves ``candidates/oaster_rebuilt.py`` untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

import protected_multilayer as protected
from benchmark import metrics as benchmark_metrics
from benchmark import protocol
from candidates import oaster_rebuilt as oaster
from run_oaster_dev_matrix import (
    BASE_MANIFEST,
    _base_row,
    _load_base_manifest,
    _runtime,
    make_matrix,
)
from run_strict_oaster import METRIC_FIELDS, _aggregate, _atomic_csv, _summaries


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "dev_experiments" / "deep_rescue_ebic"
TAUS = (0.0, 2.0, 4.0)
BASELINE = "OASTER_baseline"


def _method(tau: float) -> str:
    return f"OASTER_deep_rescue_tau{float(tau):g}"


def _methods(taus: tuple[float, ...]) -> tuple[str, ...]:
    return (BASELINE, *(_method(tau) for tau in taus))


METHODS = _methods(TAUS)
ROW_FIELDS = (
    "base_manifest_sha256",
    "matrix_manifest_sha256",
    "case_number",
    "case_id",
    "configuration_number",
    "configuration_id",
    "source_case_id",
    "pair_index",
    "eeg_snr_db",
    "meg_snr_db",
    "actual_eeg_snr_db",
    "actual_meg_snr_db",
    "scenario",
    "surface_centers",
    "deep_index",
    "deep_surface_ratio",
    "correlation",
    "seed",
    "method",
    "status",
    "error",
    "elapsed_seconds",
    "temporal_rank",
    "selected_templates",
    "rescue_tau",
    "rescue_accepted",
    "rescue_deep_local",
    "rescue_ebic_delta",
    "rescue_universe",
    "recovered_design_rank",
    "has_surface_true",
    *METRIC_FIELDS,
)


def _column_space(matrix: np.ndarray) -> np.ndarray:
    """Return a stable orthonormal basis for a fitted sensor-space design."""
    matrix = np.asarray(matrix, dtype=float)
    if not matrix.size or not np.any(matrix):
        return np.zeros((matrix.shape[0], 0))
    left, singular, _right = np.linalg.svd(matrix, full_matrices=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    return left[:, singular > tolerance]


def deep_rescue_trial(
    data: np.ndarray,
    leadfield: np.ndarray,
    primary: np.ndarray,
    n_surf: int,
    basis: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Propose one observation-only deep source using conditional EBIC.

    ``_ebic_templates`` does not expose its selected design.  Its fitted reduced
    sensor signal has the same observable column space when selected temporal
    coefficients are full rank, so that space is recovered by SVD and used as
    ``D`` below.  This is the only approximation in the experiment.
    """
    data = np.asarray(data, dtype=float)
    leadfield = np.asarray(leadfield, dtype=float)
    primary = np.asarray(primary, dtype=float)
    basis = np.asarray(basis, dtype=float)
    if data.ndim != 2 or leadfield.ndim != 2 or data.shape[0] != leadfield.shape[0]:
        raise ValueError("data and leadfield must share the channel axis")
    if primary.shape != (leadfield.shape[1], data.shape[1]):
        raise ValueError("primary must be sources x time")
    if basis.ndim != 2 or basis.shape[1] != data.shape[1]:
        raise ValueError("basis must be temporal-rank x time")
    if not 0 < n_surf < leadfield.shape[1]:
        raise ValueError("the leadfield must contain surface and deep candidates")

    rescue = np.zeros_like(primary)
    universe = int(leadfield.shape[1] - n_surf)
    diagnostics = {
        "accepted_without_tau": False,
        "deep_local": -1,
        "ebic_delta_without_tau": math.inf,
        "universe": universe,
        "recovered_design_rank": 0,
    }
    if basis.size == 0:
        return rescue, diagnostics

    reduced = data @ basis.T
    fitted = leadfield @ (primary @ basis.T)
    design = _column_space(fitted)
    diagnostics["recovered_design_rank"] = int(design.shape[1])
    residual = reduced - fitted
    deep_gain = leadfield[:, n_surf:]
    if design.shape[1]:
        residual = residual - design @ (design.T @ residual)
        residualized_gain = deep_gain - design @ (design.T @ deep_gain)
    else:
        residualized_gain = deep_gain

    norms = np.sum(residualized_gain**2, axis=0)
    valid = norms > np.finfo(float).eps
    rss_old = float(np.sum(residual**2))
    if not np.any(valid) or rss_old <= np.finfo(float).eps:
        return rescue, diagnostics
    drops = np.full(universe, -np.inf)
    drops[valid] = (
        np.sum((residualized_gain[:, valid].T @ residual) ** 2, axis=1)
        / norms[valid]
    )
    index = int(np.argmax(drops))
    drop = min(float(drops[index]), rss_old)
    n_obs = int(reduced.size)
    rank = int(basis.shape[0])
    rss_new = max(rss_old - drop, np.finfo(float).tiny)
    delta = (
        n_obs * math.log(rss_new / rss_old)
        + rank * math.log(n_obs)
        + 2.0 * math.log(universe)
    )
    coefficients = (residualized_gain[:, index].T @ residual) / norms[index]
    rescue[n_surf + index] = coefficients @ basis
    diagnostics.update(
        accepted_without_tau=bool(delta < 0.0),
        deep_local=index,
        ebic_delta_without_tau=float(delta),
    )
    return rescue, diagnostics


def reconstruct_variants(
    eeg: np.ndarray,
    meg: np.ndarray,
    gain_eeg: np.ndarray,
    gain_meg: np.ndarray,
    n_surf: int,
    kernels,
    *,
    taus: tuple[float, ...] = TAUS,
) -> tuple[dict[str, np.ndarray], dict]:
    """Share the expensive OASTER pass across baseline and all EBIC penalties."""
    kernels = tuple(kernels)
    kernel_by_scale = dict(kernels)
    data, leadfield = protected.whitened_joint_system(
        {"F": eeg, "Gain": gain_eeg}, {"F": meg, "Gain": gain_meg}
    )
    basis = oaster._temporal_basis(data)
    primary, selected = oaster._ebic_templates(
        data, leadfield, n_surf, kernels, basis
    )
    spectral_fn = getattr(
        protected,
        "multiscale_spectral_evidence_source",
        oaster._multiscale_spectral_evidence_source,
    )
    add_fn = getattr(protected, "add_scaled_evidence", oaster._add_scaled_evidence)
    spectral = spectral_fn(data, leadfield, n_surf, kernel_by_scale[4.0])
    baseline = add_fn(primary, spectral, oaster.SPECTRAL_FRACTION)
    rescue, rescue_diagnostics = deep_rescue_trial(
        data, leadfield, primary, n_surf, basis
    )
    rescued = add_fn(primary + rescue, spectral, oaster.SPECTRAL_FRACTION)
    variants = {BASELINE: baseline}
    delta = float(rescue_diagnostics["ebic_delta_without_tau"])
    for tau in taus:
        variants[_method(tau)] = (
            rescued if delta + tau < 0.0 else baseline
        )
    return variants, {
        "temporal_rank": int(basis.shape[0]),
        "selected_templates": int(selected),
        **rescue_diagnostics,
    }


def _score_case(
    case: dict,
    shared: dict,
    runtime: dict,
    base_sha: str,
    matrix_sha: str,
    taus: tuple[float, ...] = TAUS,
) -> list[dict]:
    began = time.perf_counter()
    base = _base_row(case, base_sha, matrix_sha)
    try:
        eeg, meg, truth, groups, meta = protocol.simulate_case(shared, case)
        estimates, diagnostics = reconstruct_variants(
            eeg,
            meg,
            shared["gain_eeg"],
            shared["gain_meg"],
            int(shared["n_surf"]),
            runtime["kernels"],
            taus=taus,
        )
        actual = meta["actual_snr_db"]
        baseline_metrics = benchmark_metrics.evaluate_estimate(
            estimates[BASELINE],
            truth,
            shared["vertices"],
            groups,
            int(shared["n_surf"]),
            runtime["active"],
            shared["auc_cortex"],
        )
        accepted = {
            tau: diagnostics["ebic_delta_without_tau"] + tau < 0.0 for tau in taus
        }
        rescued_metrics = None
        if any(accepted.values()):
            rescued_metrics = benchmark_metrics.evaluate_estimate(
                next(
                    estimates[_method(tau)]
                    for tau in taus
                    if accepted[tau]
                ),
                truth,
                shared["vertices"],
                groups,
                int(shared["n_surf"]),
                runtime["active"],
                shared["auc_cortex"],
            )
        elapsed = time.perf_counter() - began
        rows = []
        for method in _methods(taus):
            tau = None if method == BASELINE else float(method.rsplit("tau", 1)[1])
            use_rescue = tau is not None and accepted[tau]
            row = dict(base)
            row.update(
                actual_eeg_snr_db=float(actual["eeg"]),
                actual_meg_snr_db=float(actual["meg"]),
                method=method,
                status="ok",
                error="",
                elapsed_seconds=elapsed,
                temporal_rank=diagnostics["temporal_rank"],
                selected_templates=diagnostics["selected_templates"],
                rescue_tau="" if tau is None else tau,
                rescue_accepted=int(use_rescue),
                rescue_deep_local=(
                    diagnostics["deep_local"] if diagnostics["deep_local"] >= 0 else ""
                ),
                rescue_ebic_delta=(
                    "" if tau is None else diagnostics["ebic_delta_without_tau"] + tau
                ),
                rescue_universe=diagnostics["universe"],
                recovered_design_rank=diagnostics["recovered_design_rank"],
                **(rescued_metrics if use_rescue else baseline_metrics),
            )
            rows.append(row)
        return rows
    except Exception as exc:
        elapsed = time.perf_counter() - began
        rows = []
        for method in _methods(taus):
            row = dict(base)
            row.update(
                actual_eeg_snr_db=np.nan,
                actual_meg_snr_db=np.nan,
                method=method,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                elapsed_seconds=elapsed,
                temporal_rank="",
                selected_templates="",
                rescue_tau="",
                rescue_accepted=0,
                rescue_deep_local="",
                rescue_ebic_delta="",
                rescue_universe=int(shared["n_deep"]),
                recovered_design_rank="",
                **{field: np.nan for field in METRIC_FIELDS},
            )
            rows.append(row)
        return rows


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict, name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _method_summaries(
    rows: list[dict],
    output: Path,
    penalty_mm: float,
    methods: tuple[str, ...] = METHODS,
) -> list[dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summary_rows = []
    comparison_rows = []
    for method in methods:
        selected = by_method[method]
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        _summaries(selected, method_output, penalty_mm)
        pair_rows = _read_csv(method_output / "summary_by_snr_pair_scenario_macro.csv")
        comparison_rows.extend({"method": method, **row} for row in pair_rows)
        aggregate = _aggregate(selected, penalty_mm)
        corrected_pairs = [_float(row, "auc_tie_corrected") for row in pair_rows]
        raw_pairs = [_float(row, "auc") for row in pair_rows]
        aggregate.update(
            method=method,
            pair_macro_auc=float(np.nanmean(raw_pairs)),
            worst_pair_macro_auc=float(np.nanmin(raw_pairs)),
            pair_macro_auc_tie_corrected=float(np.nanmean(corrected_pairs)),
            worst_pair_macro_auc_tie_corrected=float(np.nanmin(corrected_pairs)),
            pairs_auc_tie_corrected_ge_0_90=int(np.sum(np.asarray(corrected_pairs) >= 0.9)),
            rescue_acceptance=float(
                np.mean([int(row["rescue_accepted"]) for row in selected])
            ),
        )
        summary_rows.append(aggregate)
    _atomic_csv(output / "summary_by_method.csv", summary_rows, summary_rows[0].keys())
    _atomic_csv(
        output / "comparison_by_snr_pair_scenario_macro.csv",
        comparison_rows,
        comparison_rows[0].keys(),
    )
    return summary_rows


def _write_report(
    output: Path,
    summaries: list[dict],
    case_count: int,
    cases_per_scenario: int,
    elapsed: float,
    methods: tuple[str, ...] = METHODS,
    taus: tuple[float, ...] = TAUS,
) -> None:
    columns = (
        "method",
        "pair_macro_auc_tie_corrected",
        "worst_pair_macro_auc_tie_corrected",
        "pairs_auc_tie_corrected_ge_0_90",
        "pair_macro_auc",
        "rmse",
        "surface_sd_mm_penalized",
        "surface_dle_mm_penalized",
        "deep_sd_mm_penalized",
        "deep_dle_mm_penalized",
        "deep_sensitivity",
        "deep_specificity",
        "deep_balanced_accuracy",
        "rescue_acceptance",
    )
    lines = [
        "# OASTER deep-rescue EBIC development experiment",
        "",
        "This run used simulated development cases only; no strict-blind input or result was read.",
        "",
        "## Rule",
        "",
        r"From the global primary reduced residual $R$, recover the observable fitted-design column space $D$ by SVD. For each deep gain $g_d$, form $h_d=(I-P_D)g_d$ and $\Delta_d=\|h_d^T R\|_2^2/\|h_d\|_2^2$. The best candidate is accepted when",
        "",
        r"$$N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log p_d+\tau<0,$$",
        "",
        rf"where $r$ is the observed temporal rank, $p_d$ is the observed deep-candidate count, and $\tau\in\{{{','.join(f'{tau:g}' for tau in taus)}\}}$. At most one deep source is added. Its residual least-squares reduced time course is expanded through the observed temporal basis, then the unchanged 5% spectral evidence is fused.",
        "",
        "The global selector does not expose its design columns, so SVD of its fitted reduced sensor signal is an explicit observational approximation to that design. No truth, SNR label, or fixed true source count enters reconstruction.",
        "",
        "## Sample and runtime",
        "",
        f"- Source cases: {case_count} = 49 EEG×MEG SNR pairs × 4 scenarios × {cases_per_scenario} fixed development configurations.",
        f"- Metric rows: {case_count * len(methods)} ({len(methods)} methods on identical observations).",
        f"- Wall time: {elapsed:.1f} s.",
        "",
        "## Key results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |",
    ]
    for row in summaries:
        values = []
        for column in columns:
            value = row[column]
            if column == "method":
                values.append(str(value))
            elif column in {"pairs_auc_tie_corrected_ge_0_90"}:
                values.append(str(int(value)))
            else:
                values.append(f"{float(value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "AUC pair columns are scenario-macro means over 49 pairs; spatial penalties replace a missing expected layer localization by the head bounding-box diagonal. Raw requested AUC and corrected tied-rank An_auc are both retained.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    base_manifest: Path = BASE_MANIFEST,
    data_root: Path = protocol.DEFAULT_DATA_ROOT,
    output: Path = DEFAULT_OUTPUT,
    *,
    cases_per_scenario: int = 2,
    workers: int = 1,
    sample_path: Path | None = None,
    taus: tuple[float, ...] = TAUS,
) -> list[dict]:
    taus = tuple(float(tau) for tau in taus)
    methods = _methods(taus)
    if (
        workers < 1
        or cases_per_scenario < 1
        or not taus
        or any(not math.isfinite(tau) or tau < 0.0 for tau in taus)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("workers, cases_per_scenario and unique non-negative taus are required")
    base, base_sha = _load_base_manifest(Path(base_manifest))
    cases = make_matrix(base, cases_per_scenario=cases_per_scenario, diagonal=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matrix_sha = protocol.save_manifest(output / "manifest.json", cases)
    kwargs = {"data_root": Path(data_root)}
    if sample_path is not None:
        kwargs["sample_path"] = Path(sample_path)
    shared = protocol.load_shared(**kwargs)
    runtime = _runtime(shared)
    score = lambda case: _score_case(
        case, shared, runtime, base_sha, matrix_sha, taus
    )
    began = time.perf_counter()
    if workers == 1:
        batches = [score(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            batches = list(pool.map(score, cases))
    elapsed = time.perf_counter() - began
    rows = [row for batch in batches for row in batch]
    _atomic_csv(output / "rows.csv", rows, ROW_FIELDS)
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        raise RuntimeError(f"{len(failed)} metric rows failed; first={failed[0]['error']}")
    summaries = _method_summaries(rows, output, runtime["penalty_mm"], methods)
    metadata = {
        "purpose": "development simulation only; strict archive/results were not read",
        "base_manifest": str(Path(base_manifest).resolve()),
        "base_manifest_sha256": base_sha,
        "matrix_manifest_sha256": matrix_sha,
        "data_root": str(Path(data_root).resolve()),
        "seed_root": 20260906,
        "levels_db": [-10, -5, 0, 5, 10, 15, 20],
        "cases_per_scenario": cases_per_scenario,
        "source_case_count": len(cases),
        "metric_row_count": len(rows),
        "workers": workers,
        "wall_seconds": elapsed,
        "methods": list(methods),
        "tau_grid": list(taus),
        "deep_candidate_universe": int(shared["n_deep"]),
        "maximum_rescues": 1,
        "spectral_fraction": oaster.SPECTRAL_FRACTION,
        "strict_results_used_for_tuning": False,
        "design_recovery_approximation": "SVD column space of global fitted reduced sensor signal",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(
        output,
        summaries,
        len(cases),
        cases_per_scenario,
        elapsed,
        methods,
        taus,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=BASE_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=protocol.DEFAULT_DATA_ROOT)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases-per-scenario", type=int, default=2)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--taus", default=",".join(f"{tau:g}" for tau in TAUS))
    args = parser.parse_args()
    try:
        taus = tuple(float(value) for value in args.taus.split(","))
    except ValueError:
        parser.error("taus must be comma-separated non-negative numbers")
    run(
        args.base_manifest,
        args.data_root,
        args.output,
        cases_per_scenario=args.cases_per_scenario,
        workers=args.workers,
        sample_path=args.sample_path,
        taus=taus,
    )


if __name__ == "__main__":
    main()
