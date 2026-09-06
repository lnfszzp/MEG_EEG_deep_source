"""Validate and summarize the preserved strict-blind SISSES score table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_strict_oaster import (
    DEFAULT_MANIFEST,
    EXPECTED_CASES,
    EXPECTED_MANIFEST_SHA256,
    _load_geometry,
    _load_manifest,
    _summaries,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SCORES = Path(r"D:\oaster_strict_blind_sisses\scores.csv")
DEFAULT_OUTPUT = ROOT / "results" / "strict_blind" / "sisses_preserved"
EXPECTED_SCORES_SHA256 = (
    "db37c46a2476e4b440117825ba89b181a92f0f50b1402c41fa0d3b10102940d7"
)


def _scores_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(scores_path: Path, manifest_path: Path) -> list[dict]:
    scores_path = Path(scores_path)
    digest = _scores_digest(scores_path)
    if digest != EXPECTED_SCORES_SHA256:
        raise RuntimeError(
            f"SISSES score table SHA-256 mismatch: expected "
            f"{EXPECTED_SCORES_SHA256}, got {digest}"
        )
    cases, digest = _load_manifest(manifest_path)
    if digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("unexpected strict manifest")
    with scores_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CASES:
        raise RuntimeError(f"SISSES score table must contain {EXPECTED_CASES} rows")
    if [row.get("case_id") for row in rows] != [case["case_id"] for case in cases]:
        raise RuntimeError("SISSES rows do not follow the frozen manifest")
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"SISSES score table contains {len(failed)} failed rows")
    numeric = ("auc", "auc_tie_corrected", "rmse", "deep_score", "active_count")
    for row, case in zip(rows, cases):
        expected = {
            "scenario": str(case["scenario"]),
            "eeg_snr_db": int(case["eeg_snr_db"]),
            "meg_snr_db": int(case["meg_snr_db"]),
            "source_case_id": str(case["source_case_id"]),
        }
        if row.get("method") != "SISSES" or any(
            row.get(name) != str(value) for name, value in expected.items()
        ):
            raise RuntimeError(f"SISSES row metadata disagrees with {case['case_id']}")
        try:
            values = [float(row[name]) for name in numeric]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"SISSES row has invalid metrics: {case['case_id']}") from exc
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"SISSES row has non-finite core metrics: {case['case_id']}")
        row["has_surface_true"] = int(bool(case.get("surface_centers")))
        row["has_deep_true"] = int(case.get("deep_index") is not None)
    return rows


def run(
    scores_path: Path = DEFAULT_SCORES,
    manifest_path: Path = DEFAULT_MANIFEST,
    data_root: Path = ROOT,
    output: Path = DEFAULT_OUTPUT,
) -> list[dict]:
    rows = load_rows(Path(scores_path), Path(manifest_path))
    geometry = _load_geometry(Path(data_root))
    penalty_mm = float(
        np.linalg.norm(np.ptp(geometry["vertices"], axis=0)) * 1000.0
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _summaries(rows, output, penalty_mm)
    metadata = {
        "method": "SISSES",
        "source_scores": str(Path(scores_path).resolve()),
        "source_scores_sha256": _scores_digest(Path(scores_path)),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "case_count": len(rows),
        "aggregation": "identical to run_strict_oaster.py",
        "scores_copied": False,
        "metric_provenance": (
            "summarized preserved scores.csv; metrics were not recomputed from the "
            "315 GB archived source-estimate MAT files"
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.scores, args.manifest, args.data_root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
