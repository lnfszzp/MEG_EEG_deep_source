import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from benchmark import metrics, protocol
from metrics.user_metrics import An_cal_AUC
import generate_strict_blind_manifest as strict_manifest


class BenchmarkRecoveryTest(unittest.TestCase):
    def test_simulate_case_uses_per_modality_snr_and_legacy_fallback(self):
        shared = {
            "times": np.linspace(0.0, 1.0, 203),
            "active_start": protocol.ACTIVE_START,
            "n_surf": 2,
            "n_deep": 1,
            "gain_eeg": np.ones((1, 3)),
            "gain_meg": np.ones((1, 3)),
            "noise_factor_eeg": np.ones((1, 1)),
            "noise_factor_meg": np.ones((1, 1)),
        }
        base_case = {
            "case_id": "snr-test",
            "surface_centers": [],
            "deep_index": 2,
            "seed": [7],
            "snr_db": 5,
        }
        observed = []

        def fake_noise(clean, _factor, snr_db, _seed):
            observed.append(float(snr_db))
            return clean, float(snr_db)

        separate = dict(base_case, eeg_snr_db=-10, meg_snr_db=20)
        with patch.object(protocol, "_add_noise", side_effect=fake_noise):
            _eeg, _meg, simulated_truth, simulated_groups, _meta = protocol.simulate_case(
                shared, separate
            )
            protocol.simulate_case(shared, base_case)
        self.assertEqual(observed, [-10.0, 20.0, 5.0, 5.0])
        truth, groups, meta = protocol.truth_for_case(shared, separate)
        np.testing.assert_array_equal(truth, simulated_truth)
        self.assertEqual([group.tolist() for group in groups], [group.tolist() for group in simulated_groups])
        self.assertNotIn("actual_snr_db", meta)

    def test_benchmark_metrics_use_vendored_auc_and_layer_metrics(self):
        count = 40
        n_surf = 20
        graph = sparse.block_diag(
            [
                sparse.diags([np.ones(19), np.ones(20), np.ones(19)], [-1, 0, 1]),
                sparse.diags([np.ones(19), np.ones(20), np.ones(19)], [-1, 0, 1]),
            ],
            format="csr",
        )
        positions = np.c_[np.arange(count) / 1000.0, np.zeros((count, 2))]
        truth = np.zeros((count, 3))
        truth[[0, 20], 2] = 1.0

        result = metrics.evaluate_estimate(
            truth,
            truth,
            positions,
            [np.array([0]), np.array([20])],
            n_surf,
            np.array([2]),
            {"Vertices": positions, "Faces": graph},
        )

        self.assertIs(metrics.An_cal_AUC, An_cal_AUC)
        self.assertAlmostEqual(result["auc"], 1.0)
        self.assertAlmostEqual(result["auc_tie_corrected"], 1.0)
        self.assertAlmostEqual(result["surface_sd_mm"], 0.0)
        self.assertAlmostEqual(result["deep_sd_mm"], 0.0)
        self.assertAlmostEqual(result["surface_dle_mm"], 0.0)
        self.assertAlmostEqual(result["deep_dle_mm"], 0.0)

    def test_repository_relative_defaults_and_frozen_digest(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(protocol.DEFAULT_DATA_ROOT, root / "generated")
        self.assertEqual(strict_manifest.BENCHMARK, root / "benchmark")
        self.assertEqual(
            strict_manifest.EXPECTED_SHA256,
            "3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76",
        )


if __name__ == "__main__":
    unittest.main()
