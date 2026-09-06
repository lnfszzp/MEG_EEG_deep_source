"""Verify or summarize the preserved strict-blind SISSES archive.

This compatibility entry point is deliberately read-only with respect to the
archive: it never regenerates observations, invokes MATLAB, or writes SISSES
outputs. Third-party SISSES execution requires an external licensed checkout
and a separately reviewed adapter, neither of which is bundled here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import summarize_strict_sisses as summary
import verify_strict_archive as verification


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = Path(
    os.environ.get("STRICT_SISSES_ARCHIVE", r"D:\oaster_strict_blind_sisses")
)
DEFAULT_DATA_ROOT = Path(os.environ.get("SOURCE_DATA_ROOT", ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    verify = commands.add_parser("verify", help="validate immutable MATLAB input chunks")
    verify.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    verify.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results" / "strict_blind" / "manifest.json",
    )
    verify.add_argument("--chunk-size", type=int, default=20)
    verify.add_argument("--sample-cases", nargs="+", type=int, metavar="CASE_NUMBER")
    verify.add_argument("--data-root", type=Path)
    verify.add_argument("--mne-sample-path", type=Path)

    summarize = commands.add_parser("summarize", help="validate and aggregate scores.csv")
    summarize.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    summarize.add_argument("--manifest", type=Path, default=summary.DEFAULT_MANIFEST)
    summarize.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    summarize.add_argument("--output", type=Path, default=summary.DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    if args.command == "verify":
        try:
            report = verification.verify_archive(
                args.manifest,
                archive / "matlab_input",
                chunk_size=args.chunk_size,
                sample_cases=args.sample_cases or (),
                data_root=args.data_root,
                mne_sample_path=args.mne_sample_path,
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.output.resolve().is_relative_to(archive):
        parser.error("summary output must be outside the immutable SISSES archive")
    try:
        rows = summary.run(
            archive / "scores.csv", args.manifest, args.data_root, args.output
        )
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"validated {len(rows)} SISSES rows; summaries: {args.output}")


if __name__ == "__main__":
    main()
