import csv
from pathlib import Path

import plot_erp_whole_head_results as plotting


ROOT = Path(__file__).parents[1]
OASTER = ROOT / "results" / "corrected_v2" / "strict_blind" / "oaster_v20_final"
COMPARATORS = ROOT / "results" / "corrected_v2" / "strict_blind" / "comparators_final"


def test_loads_exactly_oaster_and_seven_comparators() -> None:
    macro, scenarios = plotting.load_results((OASTER, COMPARATORS))

    assert tuple(macro) == plotting.METHODS
    assert all(len(rows) == 49 for rows in macro.values())
    assert tuple(scenarios) == plotting.SCENARIOS
    assert all(
        len(scenarios[scenario][method]) == 49
        for scenario in plotting.SCENARIOS
        for method in plotting.METHODS
    )


def test_combined_rows_csv_fallback(tmp_path: Path) -> None:
    fields = (
        "method",
        "eeg_snr_db",
        "meg_snr_db",
        "scenario",
        "status",
        "has_surface_true",
        "has_deep_true",
        "auc",
        "auc_tie_corrected",
        "rmse",
        "surface_sd_mm",
        "surface_dle_mm",
        "deep_sd_mm",
        "deep_dle_mm",
    )
    with (tmp_path / "rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method in ("OASTER-ERP", *plotting.METHODS[1:]):
            for scenario in plotting.SCENARIOS:
                for eeg in plotting.SNR_LEVELS:
                    for meg in plotting.SNR_LEVELS:
                        has_surface = int(scenario != "deep_only")
                        has_deep = int(scenario != "surface_only")
                        writer.writerow(
                            {
                                "method": method,
                                "eeg_snr_db": eeg,
                                "meg_snr_db": meg,
                                "scenario": scenario,
                                "status": "ok",
                                "has_surface_true": has_surface,
                                "has_deep_true": has_deep,
                                "auc": 0.8,
                                "auc_tie_corrected": 0.9,
                                "rmse": 0.2,
                                "surface_sd_mm": 5.0 if has_surface else "nan",
                                "surface_dle_mm": 6.0 if has_surface else "nan",
                                "deep_sd_mm": 7.0 if has_deep else "nan",
                                "deep_dle_mm": 8.0 if has_deep else "nan",
                            }
                        )

    macro, scenarios = plotting.load_results((tmp_path,))

    assert tuple(macro) == plotting.METHODS
    assert len(macro["OASTER-ERP"]) == 49
    assert macro["OASTER-ERP"][0]["auc_tie_corrected"] == 0.9
    assert scenarios["deep_only"]["OASTER-ERP"][0]["deep_dle_mm_penalized"] == 8.0


def test_generates_table_and_all_requested_figures(tmp_path: Path) -> None:
    outputs = plotting.generate((OASTER, COMPARATORS), tmp_path / "erp_figures")

    assert len(outputs) == 12
    assert all(path.is_file() and path.stat().st_size > 50 for path in outputs)
    assert all(path.stat().st_size > 1_000 for path in outputs[5:])
    assert all("figures_v20" not in str(path) for path in outputs)
    with outputs[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == list(plotting.METHODS)
    assert all(row["primary_metric"] == "An_auc (auc_tie_corrected)" for row in rows)
    assert all(row["snr_pair_count"] == "49" for row in rows)
    assert outputs[5].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
