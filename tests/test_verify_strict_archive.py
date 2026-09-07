import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio

from benchmark import protocol
from verify_strict_archive import verify_archive


class VerifyStrictArchiveTest(unittest.TestCase):
    def test_tiny_archive_including_truth_and_observed_snr(self):
        expected_format = "strict-test-format-v3"
        shared = {
            "times": np.linspace(0.0, 0.219, 220),
            "active_start": protocol.ACTIVE_START,
            "n_surf": 3,
            "n_deep": 1,
            "vertices": np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.0, 0.02]]),
            "adjacency": np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8),
            "gain_eeg": np.array([[1.0, 0.2, 0.5, 0.7], [0.1, 0.8, 0.3, 0.4]]),
            "gain_meg": np.array([[0.4, 0.6, 0.2, 1.0], [0.7, 0.1, 0.9, 0.3]]),
            "noise_factor_eeg": np.eye(2),
            "noise_factor_meg": np.eye(2),
        }
        cases = []
        for number, (eeg_snr, meg_snr) in enumerate(((-10, 5), (20, -5))):
            cases.append(
                {
                    "case_number": number,
                    "case_id": f"tiny-{number}",
                    "configuration_id": "tiny-source-0",
                    "surface_centers": [1],
                    "deep_index": 3,
                    "deep_surface_ratio": 0.5,
                    "correlation": 0.5,
                    "snr_db": eeg_snr,
                    "eeg_snr_db": eeg_snr,
                    "meg_snr_db": meg_snr,
                    "seed": [20260901, number],
                }
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, input_dir = root / "manifest.json", root / "matlab_input"
            input_dir.mkdir()
            payload = (
                json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
            manifest_path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            for case in cases:
                eeg, meg, _truth, _groups, _meta = protocol.simulate_case(shared, case)
                number = case["case_number"]
                sio.savemat(
                    input_dir / f"strict_{number:05d}_{number + 1:05d}.mat",
                    {
                        "F_EEG": eeg[:, :, None],
                        "F_MEG": meg[:, :, None],
                        "Gain_EEG": shared["gain_eeg"],
                        "Gain_MEG": shared["gain_meg"],
                        "case_ids": np.asarray([case["case_id"]], dtype=object)[:, None],
                        "manifest_sha256": digest,
                        "format_version": expected_format,
                    },
                )

            report = verify_archive(
                manifest_path,
                input_dir,
                expected_sha256=digest,
                expected_format=expected_format,
                chunk_size=1,
                sample_cases=(0, 1),
                shared=shared,
            )

        self.assertEqual(report["status"], "ok")
        self.assertEqual((report["chunks"], report["cases"], report["archived_case_ids"]), (2, 2, 2))
        self.assertFalse(report["output_tree_scanned"])
        self.assertEqual(report["format_version"], expected_format)
        np.testing.assert_allclose(
            [row["eeg_snr_db"] for row in report["sampled_cases"]], [-10.0, 20.0], atol=1e-12
        )
        np.testing.assert_allclose(
            [row["meg_snr_db"] for row in report["sampled_cases"]], [5.0, -5.0], atol=1e-12
        )
        self.assertEqual(len({row["truth_sha256"] for row in report["sampled_cases"]}), 1)
        self.assertEqual(len({row["groups_sha256"] for row in report["sampled_cases"]}), 1)


if __name__ == "__main__":
    unittest.main()
