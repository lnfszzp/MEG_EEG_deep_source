from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import plot_snr_matrix as plotting


def _write_grid(path: Path, offset: float, missing_first: bool = False) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "eeg_snr_db",
                "meg_snr_db",
                "aggregation",
                "auc_tie_corrected",
                "auc",
                "rmse",
            ),
        )
        writer.writeheader()
        for eeg in reversed(plotting.SNR_LEVELS):
            for meg in reversed(plotting.SNR_LEVELS):
                value = 0.8 + offset + (eeg + meg + 20) / 1000
                writer.writerow(
                    {
                        "eeg_snr_db": eeg,
                        "meg_snr_db": meg,
                        "aggregation": "scenario_macro",
                        "auc_tie_corrected": ""
                        if missing_first and eeg == -10 and meg == -10
                        else value,
                        "auc": value - 0.05,
                        "rmse": 1.0 - value,
                    }
                )


def test_plot_two_shuffled_49_cell_summaries(tmp_path: Path) -> None:
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write_grid(first, 0.0, missing_first=True)
    _write_grid(second, 0.02)

    loaded = plotting.load_series("first", first, "auc_tie_corrected")["first"]
    assert loaded.shape == (7, 7)
    assert np.isnan(loaded[0, 0])
    assert np.isclose(loaded[-1, -1], 0.86)

    output = plotting.main(
        [
            "--input",
            f"first={first}",
            "--input",
            f"second={second}",
            "--output",
            str(tmp_path / "snr.png"),
        ]
    )
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 20_000
