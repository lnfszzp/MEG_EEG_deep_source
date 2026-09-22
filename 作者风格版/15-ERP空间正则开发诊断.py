# %%
# 开发诊断：同四个开发 case，检查 SISSES 式 MRF 空间创新参数。
# 只在开发集做参数探索，不能将其中最好结果作为独立确认结论。

from pathlib import Path
import inspect
import json
import os
import sys
import time

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 多 case 并行时，每 case 的 BLAS 只用 1 个线程；本诊断顺序运行也沿用该限制。
for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import run_erp_whole_head_matrix as simulation


# %% 1. 固定数据和探索参数。
manifest_path = project_root / "results/erp_whole_head/development_full_v3/manifest.json"
save_root = project_root / "results/erp_whole_head/adaptive_v4/spatial_mrf_screen_01"
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
parameter_sets = [
    {"edge_fraction": 0.5, "noise_multiplier": 1.0, "mrf_strength": value}
    for value in (0.0, 0.5, 0.8)
]
seed_root = 20260921
snr_pair = (5, 5)

assert not (save_root / "rows.csv").exists(), "此实验已存在，请使用新的输出目录保存下一版。"
cases, manifest_sha256 = simulation.archive._load_manifest(manifest_path)
selected = simulation._select_pairs(cases, [snr_pair], cases_per_scenario=1)[0][1]
assert len(selected) == 4
assert {case["scenario"] for case in selected} == set(simulation.SCENARIOS)
shared = simulation.protocol.load_shared(simulation.DEFAULT_DATA_ROOT, sample_data_path)
solver = simulation._resolve_oaster("v4")
fingerprint_paths = (
    __file__, simulation.__file__, simulation.oaster.__file__,
    simulation.erp_protocol.__file__, simulation.protocol.__file__,
    simulation.comparator_methods.__file__, simulation.benchmark_metrics.__file__,
    project_root / "candidates/oaster_adaptive.py",
    project_root / "algorithms/spatial_fused_fusion.py",
)
code_fingerprint = simulation._code_fingerprint(fingerprint_paths)
save_root.mkdir(parents=True, exist_ok=True)
metadata = {
    "purpose": "development-only spatial regularization diagnosis",
    "manifest_sha256": manifest_sha256,
    "case_ids": [case["case_id"] for case in selected],
    "snr_pair": snr_pair,
    "seed_root": seed_root,
    "parameter_sets": parameter_sets,
    "solver_signature": str(inspect.signature(solver)),
    "code_fingerprint": code_fingerprint,
    "environment": simulation._environment_versions(),
    "convergence_note": "status=ok means computed output; read each complete solver diagnostic",
}
(save_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
all_rows = []
fieldnames = (*simulation.archive.ROW_FIELDS, *parameter_sets[0])
expected_cases = len(selected) * len(parameter_sets)
started = time.perf_counter()


# %% 2. 顺序运行 3 个参数组合 × 4 个固定 case；每个参数组合独立保存。
for parameters in parameter_sets:
    assert simulation._code_fingerprint(fingerprint_paths) == code_fingerprint, "运行期间代码发生变化"
    tag = "_".join(f"{name}_{value:g}" for name, value in parameters.items()).replace(".", "p")
    setting_root = save_root / tag
    diagnostics_root = setting_root / "diagnostics"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    runtime = {
        "shared": shared,
        "kernels": (),
        "algorithm_version": "v4",
        "oaster_solver": solver,
        "modality_weighting": "evidence",
        "seed_root": seed_root,
        "methods": ("OASTER-ERP-v4",),
        "oaster_kwargs": parameters,
        "diagnostics_dir": diagnostics_root,
    }
    setting_rows = []
    for case in selected:
        row = simulation._score_case(case, runtime, manifest_sha256)["OASTER-ERP-v4"]
        row.update(parameters)
        setting_rows.append(row)
        print(tag, case["scenario"], row["status"], "AUC=", row["auc_tie_corrected"], flush=True)
    simulation.archive._atomic_csv(setting_root / "rows.csv", setting_rows, fieldnames)
    all_rows.extend(setting_rows)
    simulation.archive._atomic_csv(save_root / "rows.csv", all_rows, fieldnames)


# %% 3. 保存完整性和时间；收敛状态按求解诊断单独计数。
assert simulation._code_fingerprint(fingerprint_paths) == code_fingerprint, "运行期间代码发生变化"
diagnostic_files = sorted(save_root.glob("*/diagnostics/case_*.json"))
solver_windows = []
for path in diagnostic_files:
    diagnostics = json.loads(path.read_text(encoding="utf-8"))["diagnostics"]
    solver_windows.extend(window["solver"] for window in diagnostics["windows"] if window.get("solver") is not None)
completion = {
    "expected_cases": expected_cases,
    "computed_cases": len(all_rows),
    "error_count": sum(row["status"] != "ok" for row in all_rows),
    "diagnostic_files": len(diagnostic_files),
    "solver_windows": len(solver_windows),
    "solver_windows_converged": sum(bool(window["converged"]) for window in solver_windows),
    "wall_seconds": time.perf_counter() - started,
    "code_fingerprint": code_fingerprint,
}
(save_root / "completion.json").write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")
assert len(all_rows) == expected_cases
assert completion["error_count"] == 0
assert len(diagnostic_files) == expected_cases
print(json.dumps(completion, indent=2), flush=True)
