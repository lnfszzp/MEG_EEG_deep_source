# 联合 EEG–MEG 深浅层源定位恢复工程

本目录恢复了 PPT 对应的 V18 多层 SISSES 后处理、原始仿真、用户指标、经典对比方法和严格 EEG×MEG 信噪比协议。当前开发中的无 SISSES 主算法为 OASTER（Observation-Adaptive Spatiotemporal Evidence Reconstruction）。

## 已确认的恢复边界

- V4–V18 来自 Git 仓库历史；除 PPT 中 AUC `0.909` 与仓库精确值 `0.900910` 不一致外，PPT 最终表格的其余数值与 V18 结果吻合。V18 流程为：模态内基线白化、联合候选、深层残差救援、0/4/7 mm 表层模板重拟合，以及 25% 弱表层范围校正。
- `metrics/user_metrics/` 保留 `An_auc`、SD、DLE、RMSE 接口。历史 `An_roc` 对并列分数顺序敏感；现在 `auc_tie_corrected` 使用并列秩修正，`auc` 仍保留历史 parcel-AUC 口径，便于复核旧表。
- OASTER 的投影证据、谱滤波、谱证据和缩放融合来自恢复出的精确代码片段；EBIC 选择循环由精确的 GCV 实验版本还原。只有 `_temporal_basis` 的原函数正文未保存，它按同一实验中留下的基线谱边缘公式重建，代码中已明确标注。
- 严格盲测清单为 49 个 `(EEG SNR, MEG SNR)` 组合 × 185 个源配置，共 9,065 例；冻结 SHA-256 为 `3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76`。

## 目录

- `pipelines/generate_datasets.py`：MNE sample 数据上的四类深浅层仿真。
- `benchmark/protocol.py`：开发集、确认集和逐例仿真；`truth_for_case` 可在不重生噪声的情况下恢复冻结真值。
- `generate_protocol_manifests.py`、`generate_strict_blind_manifest.py`：带哈希校验的清单生成。
- `protected_multilayer.py`、`run_frozen_v15.py`–`run_frozen_v18.py`：PPT 算法演进和冻结评估。
- `candidates/oaster_rebuilt.py`：恢复的 observation-only OASTER 核心。
- `benchmark/methods.py`：MNE、dSPM、sLORETA、eLORETA、LCMV、网格偶极子拟合和 RAP-MUSIC 的统一数值实现。
- `run_oaster_dev_matrix.py`：开发集 EEG×MEG 信噪比矩阵；`run_oaster_benchmark.py` 是旧命令名的兼容入口。
- `run_strict_oaster.py`、`run_strict_comparators.py`：直接读取冻结观测的 OASTER 与七种 Python 对比方法；`run_snr_comparators_matrix.py` 是后者的旧命令名兼容入口。
- `run_strict_blind_sisses.py`：只读核验与汇总现存 SISSES 归档，不再生成观测或调用 MATLAB。
- `visualization/`、`plot_results.py`：皮层、MRI、波形和指标图。

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

开发矩阵、最终严格盲测和七种对比方法：

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

现存 SISSES 归档只读核验与汇总：

```powershell
$env:STRICT_SISSES_ARCHIVE='D:\oaster_strict_blind_sisses'
python run_strict_blind_sisses.py verify
python run_strict_blind_sisses.py summarize --data-root $env:SOURCE_DATA_ROOT
```

第三方 SISSES 源码和可复现的 MATLAB 执行适配器均未随本仓库保存，因此该入口不会重新运行 SISSES，也不会写入或覆盖冻结归档。

## 检查

```powershell
python -m pytest -q
```

V18 的 1,110 例确认集场景宏平均 AUC 为 `0.900910`，但 0 dB 混合源仍低于 0.90，因此不能把总体均值解释为所有条件达标。严格 SISSES 旧结果的 corrected AUC 为 `0.960941`，仍有 4/49 个低信噪比组合低于 0.90。最终 OASTER 严格矩阵结果会以同一清单、同一指标同时报告场景宏平均和病例加权平均；不会以改指标口径来“达到”目标。
