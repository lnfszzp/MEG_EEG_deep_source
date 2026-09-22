# %%
# OASTER-ERP-v4：源幅度和相邻源差异迭代重加权，全头四种情况、多 SNR。
# 直接复用已冻结的 ERP 观测和七个对比方法，便于只检查空间算法的变化。
# 本文件按顺序运行，不定义函数；求解器的数学计算在 candidates 中。

from pathlib import Path
from collections import Counter
import csv
import json
import os
import sys
import time


# %% 1. 参数：使用全部开发位置；原 v3 的结果保存在原目录。
project_root = Path(__file__).resolve().parents[1]
manifest_path = project_root / "results/erp_whole_head/development_full_v3/manifest.json"
geometry_root = project_root / "corrected_v2/generated"
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")

algorithm_version = "v4"
modality_weighting = "evidence"
seed_root = 20260921  # 与 development_full_v3/v3_locked 完全相同的噪声抽样。
snr_levels = (-10, -5, 0, 5, 10, 15, 20)
snr_pairs = [(eeg, meg) for eeg in snr_levels for meg in snr_levels]
cases_per_scenario = 1  # 首先 49 格各做四种情况，共 196 例；改成 None 做完整 9,212 例。
workers = max(1, min(4, os.cpu_count() or 1))
force = False  # False 自动校验并继续已有检查点。
save_root = project_root / "results/erp_whole_head/adaptive_v4" / (
    "development_all_methods" if cases_per_scenario is None
    else f"development_{cases_per_scenario}_per_scenario_all_methods"
)

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import run_erp_whole_head_matrix as simulation
import plot_erp_whole_head_results as result_plot

cases, manifest_sha256 = simulation.archive._load_manifest(manifest_path)
selected = simulation._select_pairs(cases, snr_pairs, cases_per_scenario)
expected_methods = ("OASTER-ERP-v4",) + simulation.comparators.METHODS
expected_cases = sum(len(pair_cases) for _pair, pair_cases in selected)
expected_rows = expected_cases * len(expected_methods)
scenario_counts = Counter(case["scenario"] for _pair, pair_cases in selected for case in pair_cases)

assert len(cases) == 9212
assert len(snr_pairs) == 49
assert set(scenario_counts) == set(simulation.SCENARIOS)
assert manifest_sha256 == "0989782ea769bc0a698e8e6f250795ee829398afc5984a904eeb9544a32432b5"
print("结果目录：", save_root)
print("方法：", expected_methods)
print("场景 case 数：", dict(scenario_counts))
print("case 总数：", expected_cases, "；方法结果行：", expected_rows)
print("该数据是开发集；本次结果不作为新的独立确认集。")
print("本次仍使用固定的 20–120 ms 仿真活动期，以单独检验空间重加权。")


# %% 2. 运行仿真：每种 EEG/MEG SNR 组合完成后原子保存，支持断点续跑。
started = time.perf_counter()
simulation.run(
    manifest_path=manifest_path,
    data_root=geometry_root,
    sample_path=sample_data_path,
    output=save_root,
    snr_pairs=snr_pairs,
    cases_per_scenario=cases_per_scenario,
    workers=workers,
    force=force,
    algorithm_version=algorithm_version,
    modality_weighting=modality_weighting,
    seed_root=seed_root,
)
print("运行时间（小时）：", (time.perf_counter() - started) / 3600.0)


# %% 3. 检查结果完整性：An_auc、SD、DLE 仍按原评价程序分表层/深层计算。
completion = json.loads((save_root / "completion.json").read_text(encoding="utf-8"))
metadata = json.loads((save_root / "metadata.json").read_text(encoding="utf-8"))
with (save_root / "rows.csv").open(encoding="utf-8-sig", newline="") as stream:
    rows = list(csv.DictReader(stream))

assert completion["status"] == "complete"
assert completion["expected_row_count"] == expected_rows == len(rows)
assert all(row["status"] == "ok" for row in rows)
assert {row["method"] for row in rows} == set(expected_methods)
assert len({row["case_id"] for row in rows}) == expected_cases
assert metadata["oaster_algorithm_version"] == "v4"
assert metadata["manifest_sha256"] == manifest_sha256
print("成功结果：", len(rows), "行，错误 0 行。")


# %% 4. 输出新版本与七个对比方法的指标图、分场景表和统计表。
# 这里仅传入 v4 目录。旧绘图程序会统一 OASTER 名称，不能混入 v3 目录。
figure_root = save_root / "figures_erp_whole_head"
products = result_plot.generate([save_root], output=figure_root)
print("v4 的图和表：", figure_root)
for product in products:
    print(product)
