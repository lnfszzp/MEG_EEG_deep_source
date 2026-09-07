"""Generate the corrected-v2 bilateral-thalamus strict-blind manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark import protocol
import generate_strict_blind_manifest as strict


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
ID_PREFIX = "corrected-v2-strict-blind"
PANEL = "corrected_v2_strict_blind"
SEED_ROOT = 20260907
EXPECTED_N_DEEP = 16
EXPECTED_SHA256 = "8147726ec34b548fba6f25fe6c05dd55933e4cf458c7c2cec94c31a9d7c5e60c"


def _load_shared(data_root: Path, sample_path: Path | None) -> dict:
    shared = strict._load_shared(data_root, sample_path)
    if int(shared["n_deep"]) != EXPECTED_N_DEEP:
        raise ValueError(
            f"corrected-v2 requires {EXPECTED_N_DEEP} thalamic points, "
            f"got {shared['n_deep']}"
        )
    if shared.get("deep_rr_mri") is None or shared.get("deep_aseg_labels") is None:
        raise ValueError("corrected-v2 geometry metadata is missing")
    return shared


def _guard_output(path: Path) -> None:
    legacy = (ROOT / "results" / "strict_blind").resolve()
    if path.resolve() == legacy or legacy in path.resolve().parents:
        raise ValueError("refusing to write corrected-v2 data under the legacy result tree")


def generate(path: Path, *, data_root: Path, sample_path: Path | None = None) -> str:
    _guard_output(path)
    shared = _load_shared(data_root, sample_path)
    manifest = strict.make_manifest(
        shared, id_prefix=ID_PREFIX, panel=PANEL, seed_root=SEED_ROOT
    )
    digest = strict._manifest_digest(manifest)
    strict._check(
        shared,
        manifest,
        digest,
        expected_digest=EXPECTED_SHA256,
        id_prefix=ID_PREFIX,
        panel=PANEL,
        seed_root=SEED_ROOT,
    )
    return protocol.save_manifest(path, manifest, expected_digest=EXPECTED_SHA256)


def self_check(path: Path, *, data_root: Path, sample_path: Path | None = None) -> str:
    _guard_output(path)
    shared = _load_shared(data_root, sample_path)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    digest = hashlib.sha256(payload).hexdigest()
    strict._check(
        shared,
        manifest,
        digest,
        expected_digest=EXPECTED_SHA256,
        id_prefix=ID_PREFIX,
        panel=PANEL,
        seed_root=SEED_ROOT,
    )
    sidecar = path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii")
    assert sidecar == f"{digest}  {path.name}\n"
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "self-check"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-path", type=Path)
    args = parser.parse_args()
    action = generate if args.stage == "generate" else self_check
    print(action(args.output, data_root=args.data_root, sample_path=args.sample_path))


if __name__ == "__main__":
    main()
