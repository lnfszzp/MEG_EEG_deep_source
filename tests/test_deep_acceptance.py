from copy import deepcopy

import pytest

from benchmark.deep_acceptance import evaluate_acceptance


def _complete_pilot():
    cases, rows, evidence = [], [], []
    for eeg, meg in ((-10, -10), (-10, 20), (20, -10)):
        for number in range(8):
            positive = number >= 4
            surface = number not in (4, 5)
            scenario = "surface_only" if not positive else "deep_plus_surface" if surface else "deep_only"
            case = {"case_id": f"{number}-{eeg}-{meg}", "configuration_id": f"configuration-{number}",
                    "eeg_snr_db": eeg, "meg_snr_db": meg, "scenario": scenario,
                    "surface_centers": [number] if surface else [], "deep_index": 20 + number if positive else None}
            identity = {key: case[key] for key in ("case_id", "configuration_id", "eeg_snr_db", "meg_snr_db", "scenario")}
            cases.append(case)
            rows.append({**identity, "method": "OASTER-ERP-v6", "status": "ok", "has_deep_true": int(positive),
                         "has_surface_true": int(surface), "deep_detected": int(positive), "deep_false_positive": 0,
                         "deep_score": 0.7 if positive else 0.01, "deep_dle_mm": 4.0 if positive else float("nan"),
                         "deep_sd_mm": 5.0 if positive else float("nan"), "deep_peak_distance_mm": 2.0 if positive else float("nan"),
                         "surface_dle_mm": 5.0 if surface else float("nan"), "surface_sd_mm": 6.0 if surface else float("nan")})
            evidence.append({**identity, "deep_present_decision": int(positive), "score": 10.0 if positive else -1.0,
                             "p_value": 0.05 if positive else 0.8, "null_converged": 1, "full_converged": 1, "all_converged": 1})
    return cases, rows, evidence


def test_complete_valid_pilot_passes_without_mutation():
    inputs = _complete_pilot()
    before = deepcopy(inputs)
    # repr preserves NaN spelling, so unchanged undefined metrics can be checked.
    before_text = repr(before)
    report = evaluate_acceptance(*inputs, 234.0)
    assert report["passed"]
    assert report["summary"]["deep_sensitivity"] == 1.0
    assert report["summary"]["deep_dle_mm_penalized"] == 4.0
    assert repr(inputs) == before_text


def test_presence_false_positive_cannot_be_hidden_by_legacy_threshold():
    cases, rows, evidence = _complete_pilot()
    evidence[0]["deep_present_decision"] = 1
    assert rows[0]["deep_false_positive"] == 0
    report = evaluate_acceptance(cases, rows, evidence, 234.0)
    assert not report["passed"]
    assert not report["gates"]["false_positive_rate_each_snr_at_most_05"]
    assert report["by_snr"]["-10,-10"]["false_positive_rate"] == 0.25
    assert report["by_snr"]["-10,-10"]["legacy_metric_false_positive_rate"] == 0.0


def test_all_deep_decisions_closed_fail_even_with_legacy_true_positives():
    cases, rows, evidence = _complete_pilot()
    for item in evidence:
        item["deep_present_decision"] = 0
    report = evaluate_acceptance(cases, rows, evidence, 234.0)
    assert not report["passed"]
    assert report["summary"]["deep_sensitivity"] == 0.0
    assert report["summary"]["deep_dle_mm_penalized"] == 234.0
    assert report["summary"]["deep_sd_mm_penalized"] == 234.0
    assert report["legacy_metric_summary"]["deep_sensitivity"] == 1.0
    assert report["detected_deep_dle_mean_mm"] is None
    assert rows[4]["deep_detected"] == 1 and rows[4]["has_deep_true"] == 1


def test_nonconvergence_is_a_hard_failure():
    cases, rows, evidence = _complete_pilot()
    evidence[0].update(null_converged=0, all_converged=0)
    report = evaluate_acceptance(cases, rows, evidence, 234.0)
    assert not report["passed"]
    assert not report["gates"]["all_models_converged"]


@pytest.mark.parametrize("damage", ("empty", "negative_only", "missing_row", "duplicate_row", "missing_evidence", "duplicate_evidence", "wrong_snr", "wrong_truth", "failed_row", "nan_score", "nan_detected_dle", "infinite_missed_dle", "missing_distance", "contradictory_convergence"))
def test_invalid_pilot_raises_instead_of_vacuously_passing(damage):
    cases, rows, evidence = _complete_pilot()
    if damage == "empty":
        cases.clear()
    elif damage == "negative_only":
        for case in cases:
            case.update(deep_index=None, surface_centers=[0])
    elif damage == "missing_row":
        rows.pop()
    elif damage == "duplicate_row":
        rows.append(dict(rows[0]))
    elif damage == "missing_evidence":
        evidence.pop()
    elif damage == "duplicate_evidence":
        evidence.append(dict(evidence[0]))
    elif damage == "wrong_snr":
        cases[0]["eeg_snr_db"] = 5
    elif damage == "wrong_truth":
        rows[0]["has_deep_true"] = 1
    elif damage == "failed_row":
        rows[0]["status"] = "error"
    elif damage == "nan_score":
        evidence[0]["score"] = float("nan")
    elif damage == "nan_detected_dle":
        rows[4]["deep_dle_mm"] = float("nan")
    elif damage == "infinite_missed_dle":
        rows[0]["deep_dle_mm"] = float("inf")
    elif damage == "missing_distance":
        rows[0].pop("deep_dle_mm")
    elif damage == "contradictory_convergence":
        evidence[0]["null_converged"] = 0
    with pytest.raises(ValueError):
        evaluate_acceptance(cases, rows, evidence, 234.0)


def test_conditional_and_miss_penalized_distances_both_matter():
    cases, rows, evidence = _complete_pilot()
    for row in rows:
        if row["has_deep_true"]:
            row["deep_dle_mm"] = 11.0
    report = evaluate_acceptance(cases, rows, evidence, 234.0)
    assert not report["gates"]["detected_deep_dle_at_most_10mm"]
    for row in rows:
        if row["has_deep_true"]:
            row["deep_dle_mm"] = 4.0
    evidence[4]["deep_present_decision"] = 0
    evidence[5]["deep_present_decision"] = 0
    report = evaluate_acceptance(cases, rows, evidence, 400.0)
    assert report["gates"]["deep_detection_at_least_80"]
    assert not report["gates"]["deep_miss_penalized_dle_at_most_60mm"]


def test_csv_numeric_fields_are_accepted():
    cases, rows, evidence = _complete_pilot()
    csv_rows = [{key: str(value) for key, value in row.items()} for row in rows]
    csv_evidence = [{key: str(value) for key, value in row.items()} for row in evidence]
    assert evaluate_acceptance(cases, csv_rows, csv_evidence, 234.0)["passed"]
