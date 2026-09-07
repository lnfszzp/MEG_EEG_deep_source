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
