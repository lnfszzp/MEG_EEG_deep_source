from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import run_corrected_v2_sisses as runner
import run_strict_oaster as strict


def test_full_work_requires_explicit_scope() -> None:
    chunks = [strict.Chunk(0, Path("strict_00000_00002.mat"), 0, 2)]
    with pytest.raises(ValueError, match="requires"):
        runner.select_chunks(chunks, "run", 0, None, False)
    assert runner.select_chunks(chunks, "smoke", 0, None, False) == chunks


def test_load_estimate_enforces_case_and_solver_identity(tmp_path: Path) -> None:
    path = tmp_path / "case_0001.mat"
    estimate = np.arange(12, dtype=float).reshape(3, 4)
    sio.savemat(
        path,
        {
            "case_id": "case-a",
            "method_names": np.asarray(["SISSES"], dtype=object),
            "source_estimates": estimate,
            "source_shape": np.asarray([3, 4, 1]),
            "metadata": {"temporal_rank": 2, "elapsed_sec": 1.0},
            "success": True,
            "errors": np.asarray([""], dtype=object),
            "preprocessing_elapsed_sec": 0.1,
            "format_version": runner.OUTPUT_FORMAT,
            "manifest_sha256": "manifest",
            "solver_fingerprint": "solver",
        },
    )
    actual, _ = runner.load_estimate(path, "case-a", "manifest", "solver", (3, 4))
    np.testing.assert_array_equal(actual, estimate)
    with pytest.raises(ValueError, match="stale"):
        runner.load_estimate(path, "case-a", "manifest", "other", (3, 4))
