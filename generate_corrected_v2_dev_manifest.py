"""Generate corrected-v2 development source configurations outside legacy data."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from benchmark import protocol
import generate_corrected_v2_manifest as corrected


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "corrected_v2" / "dev" / "base_manifest.json"
LEGACY_ROOT = (ROOT / "benchmark" / "results" / "protocol").resolve()
EXPECTED_COUNTS = {
    "surface_only": 68,
    "deep_only": 18,
    "deep_plus_surface": 68,
    "deep_plus_two_surface": 34,
}


def generate(
    path: Path = OUTPUT,
    *,
    data_root: Path,
    sample_path: Path | None = None,
) -> str:
    path = Path(path)
    resolved = path.resolve()
    if resolved == LEGACY_ROOT or LEGACY_ROOT in resolved.parents:
        raise ValueError("refusing to overwrite the legacy development protocol")
    shared = corrected._load_shared(Path(data_root), sample_path)
    manifest = protocol.make_manifest(protocol.build_split(shared), panel="dev")
    if Counter(case["scenario"] for case in manifest) != Counter(EXPECTED_COUNTS):
        raise RuntimeError("corrected-v2 development scenario counts are invalid")
    return protocol.save_manifest(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(generate(args.output, data_root=args.data_root, sample_path=args.sample_path))


if __name__ == "__main__":
    main()
