"""Generate the frozen development and confirmation manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.protocol import build_split, load_shared, make_manifest, save_manifest


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "dev": "bc1771924276eb574616b5f23e2835689beecbc557d5d7c02305c523d60d9ff2",
    "test": "5624d8f90ac5215b482791dffbe1c17f2586a41eb71d085002c4cd775eb1991b",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmark" / "results" / "protocol"
    )
    args = parser.parse_args()
    kwargs = {
        key: value
        for key, value in (("data_root", args.data_root), ("sample_path", args.sample_path))
        if value is not None
    }
    split = build_split(load_shared(**kwargs))
    for panel in ("dev", "test"):
        digest = save_manifest(
            args.output / f"{panel}_manifest.json",
            make_manifest(split, panel),
            expected_digest=EXPECTED[panel],
        )
        print(f"{panel}: {digest}")


if __name__ == "__main__":
    main()
