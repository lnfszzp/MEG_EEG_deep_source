from __future__ import annotations

import csv
from pathlib import Path

import pytest

import summarize_strict_sisses as summary


def test_load_rows_rejects_score_order(monkeypatch, tmp_path: Path) -> None:
    manifest = [{"case_id": "a", "surface_centers": [1], "deep_index": None}]
    monkeypatch.setattr(summary, "EXPECTED_CASES", 1)
    monkeypatch.setattr(
        summary,
        "_load_manifest",
        lambda _path: (manifest, summary.EXPECTED_MANIFEST_SHA256),
    )
    scores = tmp_path / "scores.csv"
    with scores.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "status"])
        writer.writeheader()
        writer.writerow({"case_id": "wrong", "status": "ok"})
    with pytest.raises(RuntimeError, match="frozen manifest"):
        summary.load_rows(scores, tmp_path / "manifest.json")
