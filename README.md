# 联合 EEG–MEG 深浅层源定位恢复工程

本目录恢复了 PPT 对应的 V18 多层 SISSES 后处理、原始仿真、用户指标、经典对比方法和冻结留出 EEG×MEG 信噪比协议（目录名保留为 `strict_blind`）。冻结候选 OASTER V19（Observation-Adaptive Spatiotemporal Evidence Reconstruction）及七种对比方法的 9,065 例留出矩阵均已完整运行并通过复核。

> **源空间审计更正：** 皮层源覆盖双侧 68 个 `aparc` 分区；原计划的 15 个“丘脑点”因旧生成脚本遗漏 HEAD→MRI 变换，实际位于脑干、小脑和第四脑室附近，丘脑点为 0。现有 deep 指标不能解释为丘脑性能，详见 `SOURCE_SPACE_AUDIT.md`。

## 已确认的恢复边界

- V4–V18 来自 Git 仓库历史；除 PPT 中 AUC `0.909` 与仓库精确值 `0.900910` 不一致外，PPT 最终表格的其余数值与 V18 结果吻合。V18 流程为：模态内基线白化、联合候选、深层残差救援、0/4/7 mm 表层模板重拟合，以及 25% 弱表层范围校正。
- `metrics/user_metrics/` 保留 `An_auc`、SD、DLE、RMSE 接口。历史 `An_roc` 对并列分数顺序敏感；现在 `auc_tie_corrected` 使用并列秩修正，`auc` 仍保留历史 parcel-AUC 口径，便于复核旧表。
- OASTER 的投影证据、谱滤波、谱证据和缩放融合来自恢复出的精确代码片段；EBIC 选择循环由精确的 GCV 实验版本还原。只有 `_temporal_basis` 的原函数正文未保存，它按同一实验中留下的基线谱边缘公式重建，代码中已明确标注。
- 原频谱 OASTER 保留给振荡/诱发功率定位；`reconstruct_evoked_oaster_from_whitened` 将多尺度时间基、0/4/7-mm 空间模板、EBIC稀疏选择和条件深源补救迁移到相位锁定ERP/ERF。每个预注册潜伏期窗口独立选择并进行完整有符号时域回归。旧 `reconstruct_evoked_from_whitened` 保留为可审计的 dSPM-like 基线，不再作为OASTER核心结果。
- ERP迁移只用 `sm09` 开发，随后在 `sm04/sm06/sm07/sm12` 锁定验证。结果没有支持“已稳定解决N20”：4名验证受试者中只有1名的三-run中位N20左S1富集超过1；P30为3/4。原频谱OASTER在这组数据上更稳定，但N20/P30几乎总是同一峰，因此也不能冒充时间分辨的ERP定位。完整表格见 `results/real_data/ds006035_oaster_erp_v2/group_final/GROUP_REPORT.md`。
- 当前 ERP 实测只验证了皮层表面源；诊断信息会报告深度先验动态范围，体积/深部 ERP 在完成个体解剖和深度权重上限验证前不作性能结论。
- 冻结留出清单为 49 个 `(EEG SNR, MEG SNR)` 组合 × 185 个源配置，共 9,065 例；冻结 SHA-256 为 `3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76`。

## 目录

- `pipelines/generate_datasets.py`：MNE sample 数据上的四类深浅层仿真。
- `benchmark/protocol.py`：开发集、确认集和逐例仿真；`truth_for_case` 可在不重生噪声的情况下恢复冻结真值。
- `generate_protocol_manifests.py`、`generate_strict_blind_manifest.py`：带哈希校验的清单生成。
- `protected_multilayer.py`、`run_frozen_v15.py`–`run_frozen_v18.py`：PPT 算法演进和冻结评估。
- `candidates/oaster_rebuilt.py`：恢复的 observation-only OASTER 核心。
- `作者风格版/`：按 `#%%` 从上到下展开、没有自定义函数的真实 ERP、仿真和四范式验证入口。
- `benchmark/methods.py`：MNE、dSPM、sLORETA、eLORETA、LCMV、网格偶极子拟合和 RAP-MUSIC 的统一数值实现。
- `run_oaster_dev_matrix.py`：开发集 EEG×MEG 信噪比矩阵；`run_oaster_benchmark.py` 是旧命令名的兼容入口。
- `run_strict_oaster.py`、`run_strict_comparators.py`：直接读取冻结观测的 OASTER 与七种 Python 对比方法；`run_snr_comparators_matrix.py` 是后者的旧命令名兼容入口。
- `run_strict_blind_sisses.py`：只读核验与汇总旧 SISSES 归档。
- `run_corrected_v2_sisses.py`、`corrected_v2_sisses_adapter.m`：从外部 `SISSES_ROOT` 调用已恢复的作者 MATLAB 核心，在 corrected-v2 上分片运行和统一评分；不复制第三方源码。
- `visualization/`、`plot_results.py`：皮层、MRI、波形和指标图。
- `plot_strict_metrics.py`：九方法十指标的分布、森林图、鲁棒性曲线和 7×7 SNR 热图。
- `analyze_strict_statistics.py`：49 个 SNR 配对区组的描述统计、bootstrap、Friedman/Wilcoxon、Holm 校正与效应量。
- `plot_strict_brain_maps.py`：在同一 MNE sample 空间绘制各方法的 MRI 切片图；加 `--surface-maps` 可同时生成 MNE/PyVista 膨胀皮层外侧/内侧四视图。
- `FINAL_RESULTS.md`：恢复证据、开发实验、严格矩阵、复现命令和解释边界的总报告。
- `VISUAL_STATISTICAL_REPORT.md`：新增指标图、统计结论、比较表和脑空间投影的集中索引。
- `SOURCE_SPACE_AUDIT.md`：四类场景、完整 SNR 矩阵、表层覆盖及非皮层坐标问题的审计。

历史 V15–V18 入口仍依赖当时 `benchmark/results/scores.csv` 所索引的逐例 SISSES 源估计；这些输入没有保存在 Git 历史或现存严格盲测归档中。因此仓库可以核对已提交的历史汇总、审查并在输入补齐后运行后处理链，但目前不能从零重跑 V15–V18。无需历史 SISSES 输入的 OASTER、七种 Python 对比方法、严格矩阵与绘图入口均可直接运行。

## 环境与数据

```powershell
python -m pip install -r requirements.txt
python pipelines\generate_datasets.py
```

若复用已有仿真数据：

```powershell
$env:SOURCE_DATA_ROOT='D:\博士\工作＆汇报\源定位\codex\roi_deep_multimethod_comparison\generated'
```

按确定性协议重生成并核验开发/确认清单，再只读核验严格清单：

```powershell
python generate_protocol_manifests.py --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data'
python generate_strict_blind_manifest.py self-check --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data'
```

两个生成器都会在写入前校验预期哈希，漂移时拒绝覆盖；审计严格清单生成确定性时，应给 `generate` 指定一个新的 `--output` 路径，不要覆盖冻结入口。

开发矩阵、最终冻结留出评估和七种对比方法：

```powershell
python run_oaster_dev_matrix.py --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data' --cases-per-scenario 5 --workers 4 --output results\dev_matrix\v19_oaster_rebuilt_baseline_5
python run_strict_oaster.py --input-root 'D:\oaster_strict_blind_sisses\matlab_input' --data-root $env:SOURCE_DATA_ROOT --workers 4 --output results\strict_blind\oaster_v19_final
python run_strict_comparators.py --input-root 'D:\oaster_strict_blind_sisses\matlab_input' --data-root $env:SOURCE_DATA_ROOT --workers 4 --output results\strict_blind\comparators_final
```

两个严格入口逐块原子保存 checkpoint，可安全重复同一命令续跑；只有全量行数、零错误和清单哈希验证成功后才生成 `completion.json`。每个算法版本必须使用新的输出目录；代码改变后若故意复用旧目录，必须加 `--force`。最终 `metadata.json` 记录 Git commit、算法/指标文件 SHA-256 和本次数值环境，但不承诺跨线性代数环境逐位一致。信噪比热图和单例源定位图：

```powershell
python plot_snr_matrix.py --input 'OASTER=results\strict_blind\oaster_v19_final\summary_by_snr_pair_scenario_macro.csv' --input 'SISSES=results\strict_blind\sisses_preserved\summary_by_snr_pair_scenario_macro.csv' --input 'Python=results\strict_blind\comparators_final\summary_by_snr_pair_scenario_macro.csv' --metric auc_tie_corrected --output results\strict_blind\figures\auc_matrix.png
python plot_strict_case.py --case-number 0 --input-root 'D:\oaster_strict_blind_sisses\matlab_input' --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\figures\strict_case_00000.png
```

九方法完整指标图、统计分析和脑空间投影：

```powershell
python plot_strict_metrics.py
python analyze_strict_statistics.py
python plot_strict_brain_maps.py --case-number 3124
python plot_strict_brain_maps.py --case-number 92
python plot_strict_brain_maps.py --manifest results\corrected_v2\strict_blind\manifest.json --input-root 'D:\oaster_corrected_v2_sisses\matlab_input' --data-root corrected_v2\generated --results-root results\corrected_v2\strict_blind --sisses-mode skip --case-number 6489 --output-root results\corrected_v2\strict_blind\brain_maps_v20 --surface-maps --relative-threshold 0.10 --display-percentile 95
```

每个病例分别生成 `simulation_truth_mri.png` / `simulation_truth_surface.png` 仿真真值图，以及完全不叠加真值的算法 MRI / 皮层图；真值只用连续能量热图表达，不使用星号、圆圈、绿色 label 或 foci。算法 MRI 与 surface 默认显示 `cutoff = max(10% 峰值, P95)`；surface 色标使用 P95/P97/P99 对应的归一化能量值，稀疏结果的分位点退化时回退到归一化能量的 10%/55%/100%，真值图只使用 10% 峰值 cutoff。

所有含深层源的代表病例还生成 `simulation_truth_mri_surface.png`、每算法 `*_mri_surface.png` 和 `all_methods_brain_combined.png`。联合图左侧解剖 MRI 显示完整深浅层源，右侧渲染皮层只显示表层分量；两面板独立归一化，颜色不能用于跨面板比较幅值。完整情形与联合图入口见 [`brain_maps_v20/INDEX.md`](results/corrected_v2/strict_blind/brain_maps_v20/INDEX.md)。表层图使用白底、`classic` 沟回底色、`inferno` 激活色和 10 步平滑；深部源不会被错误投影到皮层。表层批量渲染还需要 `pyvistaqt` 与 `PyQt6`（已列入 `requirements.txt`）。

结果索引见 `VISUAL_STATISTICAL_REPORT.md`；完整方法比较工作簿位于 `outputs/01a0747c-f942-7dc2-b982-f01e067136c6/strict_benchmark_method_comparison.xlsx`。

现存 SISSES 归档只读核验与汇总：

```powershell
$env:STRICT_SISSES_ARCHIVE='D:\oaster_strict_blind_sisses'
python run_strict_blind_sisses.py verify
python run_strict_blind_sisses.py summarize --data-root $env:SOURCE_DATA_ROOT
```

作者 SISSES 核心已在本机恢复并通过 MATLAB 冒烟测试；旧入口仍保持只读。corrected-v2 使用单独的 `run_corrected_v2_sisses.py` 调用外部源码目录，运行器会记录核心文件 SHA-256、适配器哈希和清单哈希。上游仓库没有显式许可证，因此第三方源码本身不复制到本仓库；只有适配器和调用说明进入 Git。

## 检查

```powershell
python -m pytest -q
```

V18 的 1,110 例确认集场景宏平均 AUC 为 `0.900910`，但 0 dB 混合源仍低于 0.90，因此不能把总体均值解释为所有条件达标。严格矩阵中，OASTER 的场景宏主 `An_auc` 平均 `0.968997`、最差 `0.955137`，49/49 个 SNR 对达到 0.90；SISSES 为 `0.965320`、最差 `0.834602`，46/49 达标。OASTER 的 196 个 SNR×场景单元有 191 个达标，逐例有 8,302/9,065 达标，所以同样不能把聚合结论写成“所有病例均超过 0.90”。完整九方法对比见 `FINAL_RESULTS.md` 和 `results/strict_blind/comparators_final/REPORT.md`。
