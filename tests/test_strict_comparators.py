import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io as sio
from scipy import sparse

import run_strict_comparators as runner


def _metric_result(groups, n_surf):
    has_deep = int(any(np.all(np.asarray(group) >= n_surf) for group in groups))
    return {
        "auc": 0.8,
        "auc_tie_corrected": 0.9,
        "rmse": 0.2,
        "surface_sd_mm": 1.0,
        "surface_dle_mm": 2.0,
        "deep_sd_mm": 3.0 if has_deep else np.nan,
        "deep_dle_mm": 4.0 if has_deep else np.nan,
        "has_deep_true": has_deep,
        "deep_score": 0.8 if has_deep else 0.0,
        "deep_peak_distance_mm": 1.0 if has_deep else np.nan,
        "deep_detected": has_deep,
        "deep_false_positive": 0,
        "active_count": 1,
    }


def _check_dispatch_checkpoint_resume_and_method_grouping():
    cases = [
        {
            "case_number": 0,
            "case_id": "tiny-surface",
            "configuration_number": 0,
            "configuration_id": "tiny-source-surface",
            "pair_index": 0,
            "eeg_snr_db": 0,
            "meg_snr_db": 0,
            "snr_db": 0,
            "scenario": "surface_only",
            "surface_centers": [0],
            "deep_index": None,
            "deep_surface_ratio": None,
            "correlation": None,
            "seed": [1],
        },
        {
            "case_number": 1,
            "case_id": "tiny-deep",
            "configuration_number": 1,
            "configuration_id": "tiny-source-deep",
            "pair_index": 0,
            "eeg_snr_db": 0,
            "meg_snr_db": 0,
            "snr_db": 0,
            "scenario": "deep_only",
            "surface_centers": [],
            "deep_index": 2,
            "deep_surface_ratio": None,
            "correlation": None,
            "seed": [2],
        },
    ]
    digest = "a" * 64
    calls = defaultdict(int)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        input_root, output = root / "matlab_input", root / "output"
        generated = root / "generated" / "deep_plus_two_surface"
        input_root.mkdir()
        generated.mkdir(parents=True)
        vertices = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, 0.02]])
        times = np.linspace(0.0, 0.219, 220)
        adjacency = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8))
        gain_eeg = np.array([[1.0, 0.5, 0.2], [0.2, 0.8, 0.4]])
        gain_meg = np.array([[0.3, 0.7, 0.6], [0.9, 0.1, 0.5]])
        sio.savemat(
            generated / "sub_EEG.mat",
            {"src_vertices": vertices, "times": times, "n_surf": 2, "n_deep": 1},
        )
        sio.savemat(
            input_root / "strict_00000_00002.mat",
            {
                "F_EEG": np.ones((2, 220, 2)),
                "F_MEG": np.ones((2, 220, 2)) * 2.0,
                "Gain_EEG": gain_eeg,
                "Gain_MEG": gain_meg,
                "VertConn": adjacency,
                "case_ids": np.asarray([case["case_id"] for case in cases], dtype=object)[:, None],
                "manifest_sha256": digest,
                "format_version": runner.archive.FORMAT_VERSION,
            },
        )

        def joint(eeg, meg, eeg_gain, meg_gain):
            calls["joint"] += 1
            return np.vstack([eeg, meg]), np.vstack([eeg_gain, meg_gain])

        def family(data, gain):
            calls["family"] += 1
            estimate = np.ones((gain.shape[1], data.shape[1]))
            return {name: estimate * (index + 1) for index, name in enumerate(runner.MINIMUM_NORM_METHODS)}

        def single(name):
            def solve(data, gain, active):
                calls[name] += 1
                return np.ones((gain.shape[1], data.shape[1]))

            return solve

        def evaluate(estimate, truth, vertices, groups, n_surf, active, cortex):
            calls["metrics"] += 1
            return _metric_result(groups, n_surf)

        patches = (
            patch.object(runner.archive, "_load_manifest", return_value=(cases, digest)),
            patch.object(runner.comparator_methods, "joint_whiten", side_effect=joint),
            patch.object(runner.comparator_methods, "minimum_norm_family", side_effect=family),
            patch.object(runner.comparator_methods, "lcmv", side_effect=single("lcmv")),
            patch.object(runner.comparator_methods, "dipole_fit", side_effect=single("dipole")),
            patch.object(runner.comparator_methods, "rap_music", side_effect=single("rap")),
            patch.object(runner.benchmark_metrics, "evaluate_estimate", side_effect=evaluate),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            runner.run(root / "manifest.json", input_root, root, output, workers=1)
            first_counts = dict(calls)
            runner.run(root / "manifest.json", input_root, root, output, workers=1)
            assert dict(calls) == first_counts

            lcmv_part = next((output / "parts").glob("*__lcmv.csv"))
            lcmv_part.unlink()
            runner.run(root / "manifest.json", input_root, root, output, workers=1)

        assert first_counts == {
            "joint": 2,
            "family": 2,
            "metrics": 14,
            "lcmv": 2,
            "dipole": 2,
            "rap": 2,
        }
        assert calls["joint"] == 4 and calls["lcmv"] == 4
        assert calls["family"] == 2 and calls["dipole"] == 2 and calls["rap"] == 2
        assert len(list((output / "parts").glob("*.csv"))) == len(runner.METHODS)

        with (output / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(cases) * len(runner.METHODS)
        assert {row["method"] for row in rows} == set(runner.METHODS)
        assert {row["auc"] for row in rows} == {"0.8"}
        assert {row["auc_tie_corrected"] for row in rows} == {"0.9"}

        with (output / "summary_by_snr_scenario.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            summaries = list(csv.DictReader(stream))
        assert len(summaries) == 2 * len(runner.METHODS)
        assert {row["method"] for row in summaries} == set(runner.METHODS)
        assert {row["deep_balanced_accuracy"] for row in summaries} == {"nan"}
        with (output / "summary_by_snr_pair_scenario_macro.csv").open(
            encoding="utf-8-sig", newline=""
        ) as stream:
            macro = list(csv.DictReader(stream))
        assert len(macro) == len(runner.METHODS)
        assert {row["deep_balanced_accuracy"] for row in macro} == {"1.0"}
        completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
        assert completion == {
            "status": "complete",
            "row_count": len(cases) * len(runner.METHODS),
            "expected_row_count": len(cases) * len(runner.METHODS),
            "error_count": 0,
            "manifest_sha256": digest,
            "chunk_count": 1,
            "method_count": len(runner.METHODS),
            "methods": list(runner.METHODS),
        }


class StrictComparatorsTest(unittest.TestCase):
    def test_dispatch_checkpoint_resume_and_method_grouping(self):
        _check_dispatch_checkpoint_resume_and_method_grouping()


if __name__ == "__main__":
    unittest.main()
