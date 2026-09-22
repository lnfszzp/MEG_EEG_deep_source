# %%
# OASTER-ERP-v5：浅层与深层联合重加权，和七个原对比方法使用同一份仿真观测。
# 默认复测旧 v4 的五个 SNR、四个开发配置；这是开发小样本，不是全头独立确认。

from pathlib import Path
from collections import Counter
import csv
import json
import os
import sys
import time


# %% 1. 参数集中在这里。None 扩展为全部 188 个开发配置 × 49 个 SNR。
project_root = Path(__file__).resolve().parents[1]
manifest_path = project_root / "results/erp_whole_head/development_full_v3/manifest.json"
geometry_root = project_root / "corrected_v2/generated"
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
algorithm_version = "v5"
cases_per_scenario = 1
snr_pairs = [(-10, -10), (-10, 20), (5, 5), (20, -10), (20, 20)] if cases_per_scenario is not None else None
seed_root = 20260921
modality_weighting = "evidence"
mrf_strength = 0.5
edge_fraction = 0.5
noise_multiplier = 1.0
workers = 2
force = False
save_root = project_root / "results/erp_whole_head/adaptive_v5" / (
    "pilot_five_snr_all_methods" if cases_per_scenario == 1
    else "full_development_all_methods" if cases_per_scenario is None
    else f"development_{cases_per_scenario}_per_scenario_five_snr"
)

# 多 case 并行，每 case 的 BLAS 只用 1 个线程，避免占满电脑。
for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[thread_variable] = "1"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
import run_erp_whole_head_matrix as simulation

cases, manifest_sha256 = simulation.archive._load_manifest(manifest_path)
selected = simulation._select_pairs(cases, snr_pairs, cases_per_scenario)
expected_methods = ("OASTER-ERP-v5",) + simulation.comparators.METHODS
expected_cases = sum(len(pair_cases) for _pair, pair_cases in selected)
expected_rows = expected_cases * len(expected_methods)
scenario_counts = Counter(case["scenario"] for _pair, pair_cases in selected for case in pair_cases)
assert manifest_sha256 == "0989782ea769bc0a698e8e6f250795ee829398afc5984a904eeb9544a32432b5"
assert set(scenario_counts) == set(simulation.SCENARIOS)
print("结果目录：", save_root)
print("开发 case 数：", expected_cases, "；方法结果行：", expected_rows)
print("场景：", dict(scenario_counts), "；SNR 组合数：", len(selected))
print("本次沿用固定 20–120 ms ERP 活动期、原噪声抽样和 An_auc/SD/DLE 评价。")


# %% 2. 运行：版本与参数具有独立指纹，完整检查点自动续跑。
started = time.perf_counter()
simulation.run(
    manifest_path=manifest_path, data_root=geometry_root, sample_path=sample_data_path,
    output=save_root, snr_pairs=snr_pairs, cases_per_scenario=cases_per_scenario,
    workers=workers, force=force, algorithm_version=algorithm_version,
    modality_weighting=modality_weighting, seed_root=seed_root,
    # 为兼容旧调用，这三个 runner 参数名称保留 v4_ 前缀，实际供 v4/v5 共用。
    v4_mrf_strength=mrf_strength, v4_edge_fraction=edge_fraction,
    v4_noise_multiplier=noise_multiplier,
)
print("运行时间（分钟）：", (time.perf_counter() - started) / 60)


# %% 3. 检查计算完整性，并单独读取求解器收敛情况。
metadata = json.loads((save_root / "metadata.json").read_text(encoding="utf-8"))
completion = json.loads((save_root / "completion.json").read_text(encoding="utf-8"))
with (save_root / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))
assert completion["status"] == "complete"
assert completion["expected_row_count"] == len(rows) == expected_rows
assert all(row["status"] == "ok" for row in rows)
assert {row["method"] for row in rows} == set(expected_methods)
assert len({row["case_id"] for row in rows}) == expected_cases
assert metadata["oaster_algorithm_version"] == "v5"
assert metadata["manifest_sha256"] == manifest_sha256
assert metadata["oaster_kwargs"] == {
    "mrf_strength": mrf_strength, "edge_fraction": edge_fraction,
    "noise_multiplier": noise_multiplier,
    "calibration": "layer", "temporal_mode": "smooth", "solver_kind": "admm",
}
parts = list(save_root.glob(f"parts_v5_*_run_{metadata['checkpoint_fingerprint'][:12]}"))
assert len(parts) == 1
solver_windows = []
for path in sorted((parts[0] / "diagnostics").glob("case_*.json")):
    diagnostics = json.loads(path.read_text(encoding="utf-8"))["diagnostics"]
    solver_windows.extend(window["solver"] for window in diagnostics["windows"] if window.get("solver") is not None)
print("计算完成：", len(rows), "行；v5 求解窗口：", len(solver_windows))
print("报告收敛的窗口：", sum(bool(window["converged"]) for window in solver_windows))
print("计算完成不代表收敛、深源检出或指标达标，请一起查看完整诊断与结果表。")
