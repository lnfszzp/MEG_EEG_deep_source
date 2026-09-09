from pathlib import Path

import numpy as np

from pipelines.run_ds006035_somatomotor import N20, read_somatosensory_events, source_metrics
from pipelines.summarize_ds006035 import METRICS, METHODS, SUBJECTS, negative_transfer, paired_tests


def test_events_use_fif_first_sample_and_only_somatosensory(tmp_path: Path) -> None:
    path = tmp_path / "events.tsv"
    path.write_text(
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0\t0\tsomatosensory\t32\t100\n"
        "1\t0\tFinger\t16\t200\n"
        "2\t0\tn/a\t0\t300\n",
        encoding="utf-8",
    )

    events = read_somatosensory_events(path, first_samp=188000)

    assert np.array_equal(events, [[188100, 0, 1]])


def test_laterality_uses_roi_density_not_vertex_count() -> None:
    source = np.zeros((4, N20.size))
    source[:3, N20] = 1.0
    geometry = {
        "left": np.array([True, True, False, False]),
        "right": np.array([False, False, True, False]),
        "xyz": np.zeros((4, 3)),
        "names": np.array(["postcentral-lh", "postcentral-lh", "postcentral-rh", "unknown"]),
        "vertices": [np.array([0, 1]), np.array([0, 1])],
    }

    metrics, _, _ = source_metrics(source, geometry)
    tiny_metrics, _, _ = source_metrics(source * 1e-12, geometry)

    assert np.isclose(metrics["n20_postcentral_laterality"], 0.0)
    assert np.isclose(metrics["n20_left_s1_enrichment"], 4.0 / 3.0)
    for field in ("n20_left_s1_mass_pct", "n20_left_s1_enrichment",
                  "n20_postcentral_laterality", "n20_hoyer_sparsity"):
        assert np.isclose(tiny_metrics[field], metrics[field])


def test_group_tests_use_paired_subject_values() -> None:
    rows = []
    for subject in SUBJECTS:
        for method in METHODS:
            value = {"OASTER Joint": 0.5, "OASTER EEG": 0.4,
                     "OASTER MAG": 0.8, "dSPM Joint": 1.0,
                     "eLORETA Joint": 0.6}[method]
            rows.append({"subject": subject, "method": method,
                         **{field: value for field in METRICS}})

    test = next(row for row in paired_tests(rows)
                if row["metric"] == "n20_left_s1_enrichment_median")

    assert negative_transfer(rows, "n20_left_s1_enrichment_median") == list(SUBJECTS)
    assert test["oaster_wins"] == 0
    assert np.isclose(test["exact_two_sided_sign_p"], 0.0625)
