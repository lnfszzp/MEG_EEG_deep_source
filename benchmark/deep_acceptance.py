"""Fixed ERP-v6 detection gates; malformed/incomplete validation never passes."""

from collections import Counter
import math

import run_strict_oaster as archive


def _number(row, key, *, binary=False):
    value = archive._number(row, key)
    if not math.isfinite(value) or (binary and value not in (0, 1)):
        raise ValueError(f"{row.get('case_id', 'case')}: invalid {key}")
    return int(value) if binary else value


def evaluate_acceptance(cases, rows, evidence_rows, penalty_mm):
    """Evaluate the locked 24-case pilot without mutating cases or metric rows.

    Presence decisions determine false positives. Localized true positives need
    both presence and the unchanged legacy localization detection. Undefined
    metrics on absent/missed layers remain valid and retain the archive penalty.
    """
    if not isinstance(cases, (list, tuple)) or len(cases) != 24:
        raise ValueError("validation requires exactly 24 manifest cases")
    if not isinstance(rows, (list, tuple)) or not isinstance(evidence_rows, (list, tuple)):
        raise ValueError("metric and evidence rows must be lists")
    try:
        penalty_mm = float(penalty_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("penalty_mm must be finite and positive") from error
    if not math.isfinite(penalty_mm) or penalty_mm <= 0:
        raise ValueError("penalty_mm must be finite and positive")

    required_pairs = {(-10, -10), (-10, 20), (20, -10)}
    case_by_id, configuration_pairs, configuration_truth = {}, {}, {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each manifest case must be a dictionary")
        case_id = case.get("case_id")
        configuration = case.get("configuration_id")
        if not isinstance(case_id, str) or not case_id or case_id in case_by_id:
            raise ValueError("manifest case IDs must be nonempty and unique")
        if not isinstance(configuration, str) or not configuration:
            raise ValueError("manifest configuration IDs must be nonempty")
        pair = (_number(case, "eeg_snr_db"), _number(case, "meg_snr_db"))
        centers = case.get("surface_centers")
        if not isinstance(centers, (list, tuple)) or any(type(center) is not int or center < 0 for center in centers):
            raise ValueError("surface_centers must be a list of source indices")
        deep_index = case.get("deep_index")
        if "deep_index" not in case or (deep_index is not None and (type(deep_index) is not int or deep_index < 0)):
            raise ValueError("deep_index must be an integer source index or None")
        if len(centers) > 2 or len(set(centers)) != len(centers) or deep_index in centers:
            raise ValueError("invalid or repeated source indices in validation truth")
        if not centers and deep_index is None:
            raise ValueError("validation case has no true source")
        expected_scenario = "surface_only" if deep_index is None else "deep_only" if not centers else "deep_plus_surface" if len(centers) == 1 else "deep_plus_two_surface"
        if case.get("scenario") != expected_scenario:
            raise ValueError("manifest scenario disagrees with its source truth")
        if pair not in required_pairs:
            raise ValueError("unexpected validation SNR pair")
        truth = (tuple(centers), deep_index, case.get("scenario"))
        if configuration in configuration_truth and configuration_truth[configuration] != truth:
            raise ValueError("a configuration changes its truth across SNRs")
        configuration_truth[configuration] = truth
        configuration_pairs.setdefault(configuration, []).append(pair)
        case_by_id[case_id] = case
    if len(configuration_pairs) != 8 or any(len(pairs) != 3 or set(pairs) != required_pairs for pairs in configuration_pairs.values()):
        raise ValueError("validation requires the same eight configurations at all three SNRs")
    for pair in required_pairs:
        cell = [case for case in cases if (case["eeg_snr_db"], case["meg_snr_db"]) == pair]
        if Counter(case["deep_index"] is not None for case in cell) != {False: 4, True: 4}:
            raise ValueError("each SNR requires four negative and four positive configurations")

    selected, evidence = {}, {}
    for collection, target, method_filter in ((rows, selected, True), (evidence_rows, evidence, False)):
        for row in collection:
            if not isinstance(row, dict):
                raise ValueError("each result row must be a dictionary")
            if method_filter and row.get("method") != "OASTER-ERP-v6":
                continue
            case_id = row.get("case_id")
            if case_id not in case_by_id or case_id in target:
                raise ValueError("unknown or duplicated validation result case")
            if (method_filter and row.get("status") != "ok") or row.get("status", "ok") != "ok" or row.get("error"):
                raise ValueError(f"{case_id}: failed result cannot be used for acceptance")
            case = case_by_id[case_id]
            if any(row.get(key) != case.get(key) for key in ("configuration_id", "scenario")):
                raise ValueError(f"{case_id}: result identity disagrees with manifest")
            if any(_number(row, key) != _number(case, key) for key in ("eeg_snr_db", "meg_snr_db")):
                raise ValueError(f"{case_id}: result SNR disagrees with manifest")
            target[case_id] = row
        if set(target) != set(case_by_id):
            raise ValueError("one OASTER-ERP-v6 row and one evidence row are required per case")

    effective_rows, detected_dle, all_converged = [], [], True
    by_snr = {f"{eeg},{meg}": {"negative_count": 0, "positive_count": 0, "false_positives": 0,
                               "legacy_metric_false_positives": 0, "localized_true_positives": 0}
              for eeg, meg in sorted(required_pairs)}
    for case_id, case in case_by_id.items():
        row, decision = selected[case_id], evidence[case_id]
        positive = int(case["deep_index"] is not None)
        if _number(row, "has_deep_true", binary=True) != positive or _number(row, "has_surface_true", binary=True) != bool(case["surface_centers"]):
            raise ValueError(f"{case_id}: metric truth flags disagree with manifest")
        present = _number(decision, "deep_present_decision", binary=True)
        legacy_detected = _number(row, "deep_detected", binary=True)
        legacy_fp = _number(row, "deep_false_positive", binary=True)
        if (legacy_detected and not positive) or (legacy_fp and positive):
            raise ValueError(f"{case_id}: impossible legacy detection flag")
        if not 0 <= _number(row, "deep_score") <= 1 or not 0 <= _number(decision, "p_value") <= 1:
            raise ValueError(f"{case_id}: score or p_value is outside its range")
        _number(decision, "score")
        null_converged = _number(decision, "null_converged", binary=True)
        full_converged = _number(decision, "full_converged", binary=True)
        converged = _number(decision, "all_converged", binary=True)
        if converged != (null_converged and full_converged):
            raise ValueError(f"{case_id}: contradictory convergence flags")
        all_converged = all_converged and bool(converged)
        for key in ("deep_sd_mm", "deep_dle_mm", "deep_peak_distance_mm"):
            try:
                distance = float(row[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{case_id}: missing/invalid {key}") from error
            if math.isinf(distance) or distance < 0 or (legacy_detected and math.isnan(distance)):
                raise ValueError(f"{case_id}: invalid localization distance {key}")
        if legacy_detected and _number(row, "deep_peak_distance_mm") > 10.0 + 1e-6:
            raise ValueError(f"{case_id}: detected peak lies outside the frozen radius")
        localized = int(bool(present and legacy_detected))
        if localized:
            detected_dle.append(_number(row, "deep_dle_mm"))
        # Work on copies: a closed presence gate counts as a miss in the existing
        # archive aggregate, but never alters saved metrics or evaluation truth.
        effective_rows.append({**row, "deep_detected": localized, "deep_false_positive": int(present and not positive)})
        cell = by_snr[f"{int(case['eeg_snr_db'])},{int(case['meg_snr_db'])}"]
        cell["positive_count" if positive else "negative_count"] += 1
        cell["false_positives"] += int(present and not positive)
        cell["legacy_metric_false_positives"] += legacy_fp
        cell["localized_true_positives"] += localized

    for cell in by_snr.values():
        cell["false_positive_rate"] = cell["false_positives"] / cell["negative_count"]
        cell["legacy_metric_false_positive_rate"] = cell["legacy_metric_false_positives"] / cell["negative_count"]
        cell["localized_deep_sensitivity"] = cell["localized_true_positives"] / cell["positive_count"]
    summary = archive._aggregate(effective_rows, penalty_mm)
    conditional_dle = sum(detected_dle) / len(detected_dle) if detected_dle else None
    # Match evaluate_estimate's 10 mm radius rounding tolerance, not a new
    # scientific threshold. Keep unrounded measured distances in the report.
    distance_tolerance_mm = 1e-6
    gates = {
        "false_positive_rate_each_snr_at_most_05": all(cell["false_positive_rate"] <= 0.05 for cell in by_snr.values()),
        "deep_detection_at_least_80": summary["deep_sensitivity"] >= 0.8,
        "detected_deep_dle_at_most_10mm": conditional_dle is not None and conditional_dle <= 10.0 + distance_tolerance_mm,
        "deep_miss_penalized_dle_at_most_60mm": summary["deep_dle_mm_penalized"] <= 60.0 + distance_tolerance_mm,
        "all_models_converged": all_converged,
    }
    return {"passed": all(gates.values()), "gates": gates, "summary": summary,
            "legacy_metric_summary": archive._aggregate(list(selected.values()), penalty_mm),
            "by_snr": by_snr, "detected_deep_dle_mean_mm": conditional_dle, "penalty_mm": penalty_mm,
            "distance_tolerance_mm": distance_tolerance_mm,
            "false_positive_definition": "deep_present_decision on a true cortex-only case; legacy 0.14 threshold cannot mask a presence-gate false positive",
            "localized_true_positive_definition": "deep_present_decision AND original deep_detected",
            "warning": "24-case engineering pilot; repeated SNRs are not independent locations and passing does not prove population FPR <= 5%"}
