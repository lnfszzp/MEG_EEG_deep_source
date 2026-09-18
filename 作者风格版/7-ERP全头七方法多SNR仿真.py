# %%
# 当前 OASTER-ERP 的全头、多 SNR 正式仿真入口。
# 本文件不复制算法，也不定义函数：参数集中放在前面，正式计算交给批处理入口。
# 每个 case 的同一份 ERP 数据会同时交给 OASTER-ERP 和七个对比方法。

from pathlib import Path
import csv
import json
import os
import sys
import time


# %%
# ==================== 1. 参数区：平时主要改这里 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
manifest_path = project_root / "results" / "corrected_v2" / "strict_blind" / "manifest.json"
geometry_root = project_root / "corrected_v2" / "generated"
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
save_root = project_root / "results" / "erp_whole_head" / "current_method_v1"

# EEG 和 MEG 分别取 7 个 SNR，共 7 x 7 = 49 种组合。
snr_levels = (-10, -5, 0, 5, 10, 15, 20)
snr_pairs = [(eeg_snr, meg_snr) for eeg_snr in snr_levels for meg_snr in snr_levels]

# None 表示每个 SNR 都使用 manifest 中全部 186 个全头位置配置，不抽样。
# 当前“全头”覆盖双侧 68 个皮层分区代表位置和 16 个双侧丘脑位置。
cases_per_scenario = None
workers = max(1, min(4, os.cpu_count() or 1))

# False：保留并校验已经完成的 SNR 检查点，适合断点续跑。
# True：即使已有结果也重新计算全部 49 个 SNR 组合。
force = False

assert len(snr_pairs) == 49
assert len(set(snr_pairs)) == 49
assert manifest_path.is_file(), f"找不到 manifest：{manifest_path}"
assert geometry_root.is_dir(), f"找不到几何和 forward：{geometry_root}"
assert sample_data_path.is_dir(), f"找不到 MNE sample 数据：{sample_data_path}"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import plot_erp_whole_head_results as result_plot
import run_erp_whole_head_matrix as simulation

print("结果保存到：", save_root)
print("SNR：", snr_levels)
print("SNR 组合数：", len(snr_pairs))
print("方法：", simulation.METHODS)
print("并行 worker：", workers)


# %%
# ==================== 2. 运行全头 ERP 仿真 ====================
# 四种情况都会运行：纯表层、纯深层、深层+单表层、深层+双表层。
# 每个 SNR 组合完成后保存一个检查点；程序中断后重新运行本块即可继续。

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
)

print("总运行时间（小时）：", (time.perf_counter() - started) / 3600.0)


# %%
# ==================== 3. 检查完整性并读取结果表 ====================

completion_path = save_root / "completion.json"
metadata_path = save_root / "metadata.json"
rows_path = save_root / "rows.csv"

assert completion_path.is_file(), "没有 completion.json，说明完整任务还没有结束"
assert metadata_path.is_file(), "没有 metadata.json"
assert rows_path.is_file(), "没有 rows.csv"

completion = json.loads(completion_path.read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

row_count = 0
error_count = 0
finished_methods = set()
finished_snr_pairs = set()
finished_scenarios = set()

with rows_path.open(encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
        row_count += 1
        error_count += row["status"] != "ok"
        finished_methods.add(row["method"])
        finished_snr_pairs.add((int(row["eeg_snr_db"]), int(row["meg_snr_db"])))
        finished_scenarios.add(row["scenario"])

assert completion["status"] == "complete"
assert row_count == completion["expected_row_count"]
assert error_count == 0
assert finished_snr_pairs == set(snr_pairs)
assert finished_methods == set(simulation.METHODS)
assert finished_scenarios == set(simulation.SCENARIOS)

print("完成状态：", completion["status"])
print("成功结果行：", row_count, "/", completion["expected_row_count"])
print("错误行：", error_count)
print("独立仿真 case：", metadata["selected_case_count"])
print("完成 SNR 组合：", len(finished_snr_pairs), "/ 49")
print("完成方法数：", len(finished_methods), "/ 8（OASTER-ERP + 七个对比方法）")
print("完成场景：", sorted(finished_scenarios))


# %%
# ==================== 4. 画指标对比图并生成八方法对比表 ====================
# 图片包括：总体/分场景 An_auc 热图、SNR 鲁棒性曲线、表深层 SD/DLE。

figure_root = save_root / "figures_erp_whole_head"
products = result_plot.generate([save_root], output=figure_root)

print("图和表保存到：", figure_root)
for product in products:
    print("  ", product)

