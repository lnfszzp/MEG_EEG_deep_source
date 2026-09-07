import csv
import shutil
from pathlib import Path

import plot_strict_metrics as plotting


def test_strict_metric_loader_statistics_and_figure(tmp_path: Path) -> None:
    root = Path(__file__).parents[1] / "results" / "strict_blind"
    results = plotting.load_results(root)
    assert tuple(results) == plotting.METHODS
    assert all(len(rows) == 49 for rows in results.values())

    statistics = plotting.write_statistics(results, tmp_path / "statistics.csv")
    assert len(statistics.read_text(encoding="utf-8").splitlines()) == 91
    figure = plotting.plot_metric_table(results, tmp_path / "metrics.png")
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.stat().st_size > 30_000


def test_missing_sisses_is_na_and_excluded(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "results" / "strict_blind"
    for directory in ("oaster_v19_final", "comparators_final"):
        target = tmp_path / directory
        target.mkdir()
        shutil.copy(source / directory / plotting.SUMMARY_NAME, target / plotting.SUMMARY_NAME)

    results, availability = plotting.load_results_with_availability(tmp_path)

    assert "SISSES" not in results
    sisses = next(row for row in availability if row["method"] == "SISSES")
    assert sisses["status"] == "N/A"
    assert sisses["included_in_comparison"] == "no"

    table = plotting.write_method_comparison_table(
        results, availability, tmp_path / "method_comparison_table.csv"
    )
    with table.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == list(plotting.METHODS)
    assert rows[0]["status"] == "complete"
    assert rows[0]["snr_pair_count"] == "49"
    assert float(rows[0]["auc_tie_corrected"]) > 0.0
    sisses_row = next(row for row in rows if row["method"] == "SISSES")
    assert sisses_row["status"] == "N/A"
    assert sisses_row["auc_tie_corrected"] == "N/A"


def test_oaster_v20_summary_is_preferred_over_v19(tmp_path: Path) -> None:
    for directory in ("oaster_v19_final", "oaster_v20_final"):
        path = tmp_path / directory / plotting.SUMMARY_NAME
        path.parent.mkdir()
        path.touch()

    assert plotting._find_summary(tmp_path, "OASTER").parent.name == "oaster_v20_final"
