from plot_strict_case import _select_case


def test_select_case_by_snr_scenario_and_location() -> None:
    cases = [
        {
            "case_id": "a",
            "case_number": 0,
            "eeg_snr_db": -5,
            "meg_snr_db": 10,
            "scenario": "deep_only",
            "location": 3,
        },
        {
            "case_id": "b",
            "case_number": 1,
            "eeg_snr_db": 0,
            "meg_snr_db": 0,
            "scenario": "surface_only",
            "location": 3,
        },
    ]
    selected = _select_case(
        cases,
        None,
        None,
        eeg_snr_db=-5,
        meg_snr_db=10,
        scenario="deep_only",
        location=3,
    )
    assert selected["case_id"] == "a"
