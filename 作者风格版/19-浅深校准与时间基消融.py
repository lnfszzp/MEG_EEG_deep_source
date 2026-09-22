# %% 同一观测分离检验：数值求解、分层噪声阈值、弱幅度时间成分保留。
from pathlib import Path
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import run_erp_whole_head_matrix as simulation

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=root / "results/erp_whole_head/adaptive_v5/ablation_development")
parser.add_argument("--solver", choices=("admm", "irls"), default="admm")
parser.add_argument("--calibration", choices=("global", "layer"))
parser.add_argument("--temporal", choices=("v4", "smooth"))
parser.add_argument("--snr-pair", nargs=2, type=int, default=(5, 5))
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--mrf-strength", type=float, default=.5)
parser.add_argument("--admm-outer", type=int, default=6)
parser.add_argument("--admm-max-iter", type=int, default=1000)
parser.add_argument("--admm-tolerance", type=float, default=.002)
args = parser.parse_args()
assert args.workers > 0
manifest = root / "results/erp_whole_head/development_full_v3/manifest.json"
cases, manifest_sha = simulation.archive._load_manifest(manifest)
selected = simulation._select_pairs(cases, [tuple(args.snr_pair)], 1)[0][1]
shared = simulation.protocol.load_shared(simulation.DEFAULT_DATA_ROOT, simulation.DEFAULT_SAMPLE_PATH)
solver = simulation._resolve_oaster("v5")
settings = [{"calibration": calibration, "temporal_mode": temporal, "solver_kind": args.solver,
             "mrf_strength": args.mrf_strength, "edge_fraction": .5, "noise_multiplier": 1.}
            for calibration in ([args.calibration] if args.calibration else ("global", "layer"))
            for temporal in ([args.temporal] if args.temporal else ("v4", "smooth"))]
if args.solver == "admm":
    for parameters in settings:
        parameters.update(outer_iterations=args.admm_outer, max_iter=args.admm_max_iter,
                          tolerance=args.admm_tolerance)
assert not (args.output / "rows.csv").exists(), "请改输出目录，不能覆盖已有实验。"
args.output.mkdir(parents=True, exist_ok=True)
paths = [root / path for path in ("candidates/oaster_balanced.py", "candidates/graph_reweight_solver.py")]
if args.solver == "irls":
    paths.append(root / "candidates/graph_irls.py")
fingerprint = simulation._code_fingerprint(paths)
metadata = dict(purpose="development-only factorial ablation; not independent validation",
                manifest_sha256=manifest_sha, seed_root=20260921, snr_pair=args.snr_pair,
                case_ids=[case["case_id"] for case in selected], settings=settings,
                code_fingerprint=fingerprint, shared_fingerprint=simulation._shared_fingerprint(shared),
                environment=simulation._environment_versions())
(args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
rows = []
started = time.perf_counter()

# %% 每个设置保留全部病例与完整收敛诊断；没有从结果中丢弃失败病例。
for parameters in settings:
    assert simulation._code_fingerprint(paths) == fingerprint, "计算期间算法改变，必须单独重跑。"
    tag = f"{parameters['solver_kind']}_{parameters['calibration']}_{parameters['temporal_mode']}"
    directory = args.output / tag
    (directory / "diagnostics").mkdir(parents=True, exist_ok=True)
    runtime = dict(shared=shared, kernels=(), algorithm_version="v5", oaster_solver=solver,
                   modality_weighting="evidence", seed_root=20260921, methods=("OASTER-ERP-v5",),
                   oaster_kwargs=parameters, diagnostics_dir=directory / "diagnostics")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(simulation._score_case, case, runtime, manifest_sha) for case in selected]
        for case, future in zip(selected, futures):
            row = future.result()["OASTER-ERP-v5"]
            row.update(setting=tag)
            rows.append(row)
            print(tag, case["scenario"], row["status"], row["auc_tie_corrected"], flush=True)
    simulation.archive._atomic_csv(args.output / "rows.csv", rows, (*simulation.archive.ROW_FIELDS, "setting"))
assert simulation._code_fingerprint(paths) == fingerprint
completion = dict(cases=len(rows), error_count=sum(row["status"] != "ok" for row in rows),
                  wall_seconds=time.perf_counter() - started, code_fingerprint=fingerprint)
(args.output / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
assert completion["error_count"] == 0
print(completion, flush=True)
