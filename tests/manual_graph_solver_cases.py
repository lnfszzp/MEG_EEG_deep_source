"""Development-only real-forward solver check; preserves all v4 result files."""
# %% Run with the meg environment and OMP/MKL/OPENBLAS_NUM_THREADS=1.
from pathlib import Path
import argparse
import json
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import numpy as np
from scipy.fft import dct
import run_erp_whole_head_matrix as simulation
from candidates import graph_reweight_solver as numerical
from candidates import oaster_adaptive as v4

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="results/erp_whole_head/adaptive_v5/solver_diagnostic_02")
parser.add_argument("--ranks", choices=("both", "3", "8"), default="both")
parser.add_argument("--outer-iterations", type=int, default=4)
parser.add_argument("--max-iter", type=int, default=1000)
args = parser.parse_args()
output = root / args.output
assert not output.exists(), "Use a new output directory for another solver version."
manifest = root / "results/erp_whole_head/development_full_v3/manifest.json"
cases, manifest_sha = simulation.archive._load_manifest(manifest)
case = next(case for case in cases if int(case["case_number"]) == 9110)
shared = simulation.protocol.load_shared(simulation.DEFAULT_DATA_ROOT, simulation.DEFAULT_SAMPLE_PATH)
original_basis = v4._evoked_temporal_basis
original_certificate = numerical._certificate
counter = [0]


def traced_certificate(*args):
    result = original_certificate(*args)
    counter[0] += 1
    if counter[0] % 50 == 0:
        print("certificate", counter[0], "relative_gap", result["gap_relative"], flush=True)
    return result


def dct_basis(data, baseline, active):
    width = int(np.count_nonzero(active))
    basis = np.zeros((8, data.shape[1]))
    basis[:, active] = dct(np.eye(width), axis=0, norm="ortho")[:8]
    return basis, {"active_samples": width, "temporal_rank": 8, "basis": "fixed_DCT8_performance_check"}


numerical._certificate = traced_certificate
v4.solve_reweighted_graph = numerical.solve_reweighted_graph_v5
output.mkdir(parents=True)
metadata = dict(case_id=case["case_id"], manifest_sha256=manifest_sha,
    purpose="Numerical solver comparison on the unchanged v4 spatial model; not a v5 efficacy claim.",
    command_settings=vars(args),
    code_fingerprint=simulation._code_fingerprint((numerical.__file__, v4.__file__, __file__)),
    environment=simulation._environment_versions())
(output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
rows = []
rank_choices = (("original_rank", original_basis), ("dct8", dct_basis))
if args.ranks != "both":
    rank_choices = (rank_choices[0 if args.ranks == "3" else 1],)
for rank_label, basis_function in rank_choices:
    v4._evoked_temporal_basis = basis_function
    counter[0] = 0
    diagnostics = output / rank_label
    diagnostics.mkdir()
    runtime = dict(shared=shared, kernels=(), algorithm_version="v4", oaster_solver=v4.reconstruct_evoked_oaster_v4_from_whitened,
        modality_weighting="evidence", seed_root=20260921, methods=("OASTER-ERP-v4",),
        oaster_kwargs={"mrf_strength": .5, "outer_iterations": args.outer_iterations,
                       "max_iter": args.max_iter, "tolerance": 1e-3},
        diagnostics_dir=diagnostics)
    started = time.perf_counter()
    row = simulation._score_case(case, runtime, manifest_sha)["OASTER-ERP-v4"]
    row["temporal_check"] = rank_label
    rows.append(row)
    simulation.archive._atomic_csv(output / "rows.csv", rows, (*simulation.archive.ROW_FIELDS, "temporal_check"))
    print(rank_label, row["status"], "seconds", time.perf_counter() - started, flush=True)
    assert row["status"] == "ok", row["error"]
