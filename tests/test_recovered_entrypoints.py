from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_recovered_entrypoints_have_working_help() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "run_oaster_benchmark.py",
        "run_snr_comparators_matrix.py",
        "run_strict_blind_sisses.py",
    ):
        result = subprocess.run(
            [sys.executable, str(root / name), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"
