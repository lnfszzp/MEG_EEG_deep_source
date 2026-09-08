# corrected-v2：源空间修复与严格盲测说明

> 状态（2026-09-07）：源空间、严格盲测清单和输入归档已经冻结并通过完整性检查；OASTER V20 的 9,114 个病例与七种 Python 对比方法的 63,798 行结果均已完成且零错误。完整结论见[最终报告](../results/corrected_v2/strict_blind/FINAL_REPORT.md)。

## 1. 为什么必须建立 corrected-v2

旧生成流程把体积 forward 中的 HEAD 坐标直接当作 MRI RAS 坐标查询 FreeSurfer `aseg`。因此，旧数据里标作“丘脑”的 15 个点实际落在脑干、小脑、第四脑室等区域，不能用来说明丘脑定位性能。

corrected-v2 在 ROI 查询前执行 `HEAD -> MRI` 变换，按 `aseg` 标签筛选丘脑，再保留同一批点对应的 HEAD 坐标和 forward 列用于仿真与 DLE 计算。生成器还检查：

- 变换方向必须是 HEAD 到 MRI；
- HEAD→MRI→HEAD 往返误差绝对容差为 `1e-10 m`；
- 深部点标签必须严格为 `{10: 9, 49: 7}`；
- corrected-v2 不得写入旧的 `generated` 或 `D:\oaster_strict_blind_sisses` 路径。

旧几何、旧观测和旧结果保持原样，只能作为历史记录，不能与 corrected-v2 混排或改名复用。

## 2. 源空间与仿真位置

### 2.1 皮层覆盖

- MNE sample 双侧皮层源空间共有 `7,498` 个点。
- 严格盲测在双侧 FreeSurfer `aparc` 的 `68` 个分区中各选一个此前未用于旧开发/测试清单的 maximin 中心，即左右半球各 `34` 个中心。
- 每个中心是一个皮层活动 patch 的中心；真值同时保存 patch 内全部活动顶点和精确中心。
- 因此，本协议按解剖分区覆盖了整个双侧皮层，但没有把 7,498 个皮层顶点逐个作为独立中心穷举。

### 2.2 深部覆盖

10 mm 体积网格共有 `1,306` 个候选点；经正确坐标变换和 `aseg` 查询后，选出 `16` 个双侧丘脑点。混合源空间总计 `7,514 = 7,498 + 16` 个点。

标签定义：`10 = Left-Thalamus-Proper`，`49 = Right-Thalamus-Proper`。下表中的 global index 为 Python/MAT 元数据采用的 0-based 索引；MRI RAS 与 HEAD 坐标单位均为 mm。

| local | global | aseg | 解剖标签 | MRI RAS (x, y, z) | HEAD (x, y, z) |
|---:|---:|---:|---|---:|---:|
| 0 | 7498 | 10 | Left-Thalamus-Proper | (-20, -20, 20) | (-24.069, 2.011, 56.081) |
| 1 | 7499 | 10 | Left-Thalamus-Proper | (-20, -20, 30) | (-24.427, 7.840, 64.198) |
| 2 | 7500 | 10 | Left-Thalamus-Proper | (-10, -20, 30) | (-14.434, 7.968, 64.547) |
| 3 | 7501 | 49 | Right-Thalamus-Proper | (10, -20, 30) | (5.552, 8.223, 65.245) |
| 4 | 7502 | 49 | Right-Thalamus-Proper | (20, -20, 30) | (15.545, 8.351, 65.594) |
| 5 | 7503 | 10 | Left-Thalamus-Proper | (-20, -10, 30) | (-24.327, 15.965, 58.368) |
| 6 | 7504 | 10 | Left-Thalamus-Proper | (-10, -10, 30) | (-14.334, 16.092, 58.717) |
| 7 | 7505 | 49 | Right-Thalamus-Proper | (10, -10, 30) | (5.652, 16.347, 59.415) |
| 8 | 7506 | 49 | Right-Thalamus-Proper | (10, 0, 30) | (5.752, 24.471, 53.584) |
| 9 | 7507 | 10 | Left-Thalamus-Proper | (-20, -20, 40) | (-24.785, 13.670, 72.315) |
| 10 | 7508 | 49 | Right-Thalamus-Proper | (10, -20, 40) | (5.194, 14.053, 73.362) |
| 11 | 7509 | 10 | Left-Thalamus-Proper | (-20, -10, 40) | (-24.685, 21.794, 66.485) |
| 12 | 7510 | 10 | Left-Thalamus-Proper | (-10, -10, 40) | (-14.692, 21.922, 66.834) |
| 13 | 7511 | 49 | Right-Thalamus-Proper | (10, -10, 40) | (5.294, 22.177, 67.532) |
| 14 | 7512 | 10 | Left-Thalamus-Proper | (-10, 0, 40) | (-14.592, 30.046, 61.004) |
| 15 | 7513 | 49 | Right-Thalamus-Proper | (10, 0, 40) | (5.394, 30.301, 61.702) |

深部范围仅代表双侧丘脑，不代表对所有皮层下结构进行全脑穷举。

## 3. 四种协议情形与样本数

科学问题可归为三类：纯表层、纯深层、表层加深层。第三类又按一个或两个皮层 patch 分成两个协议场景，因此清单里共有四个 `scenario` 名称。

| 情形家族 | 协议名称 | 每个 SNR 单元的配置数 | 49 个 SNR 单元的病例数 | 真值组成 |
|---|---|---:|---:|---|
| 纯表层 | `surface_only` | 68 | 3,332 | 1 个皮层 patch |
| 纯深层 | `deep_only` | 16 | 784 | 1 个丘脑点；16 点各一次 |
| 表层 + 深层 | `deep_plus_surface` | 68 | 3,332 | 1 个丘脑点 + 1 个皮层 patch |
| 表层 + 深层 | `deep_plus_two_surface` | 34 | 1,666 | 1 个丘脑点 + 2 个左右半球皮层 patch |
| **合计** | 4 个协议名称 | **186** | **9,114** | — |

每个 SNR 单元复用同一组 186 个源配置。混合场景通过 `location % 16` 轮换丘脑点，并非穷举“每个皮层中心 × 每个丘脑点”的笛卡尔积；双皮层场景的 34 组中心为左右半球配对。

## 4. EEG×MEG 信噪比矩阵

EEG 与 MEG 的信噪比分别独立取值：

```text
{-10, -5, 0, 5, 10, 15, 20} dB
```

因此共有 `7 × 7 = 49` 个有序 `(EEG SNR, MEG SNR)` 组合。四个协议场景全部运行这 49 种组合：不存在只给某类源或某个模态测试部分 SNR 的情况。每个病例有唯一 `case_id` 与随机种子，两个模态分别按其目标 SNR 加噪。

## 5. 冻结清单与不可变输入归档

### 5.1 路径和规模

- 几何与基础数据：`D:\博士\工作＆汇报\源定位\新建文件夹\corrected_v2\generated`
- 严格盲测清单：`D:\博士\工作＆汇报\源定位\新建文件夹\results\corrected_v2\strict_blind\manifest.json`
- MATLAB/Python 共用输入：`D:\oaster_corrected_v2_sisses\matlab_input`
- 输入格式：`strict_blind_sisses_input_v2`
- 归档规模：`456` 个 MAT chunk，合计 `9,234,435,864` bytes（约 `8.60 GiB`）；标准 chunk 为 20 个病例，最后一个为 14 个病例。

### 5.2 身份与完整性

严格清单 SHA-256：

```text
8147726ec34b548fba6f25fe6c05dd55933e4cf458c7c2cec94c31a9d7c5e60c
```

所有 chunk 嵌入同一清单 SHA、格式版本、连续 `case_id`，并必须具有下列跨 chunk 一致的数组指纹：

| 数组 | SHA-256 |
|---|---|
| `Gain_EEG` | `36108ed1e6ce24f2a30027e2cd11884527be086ace54bb930ffe2f9069021f51` |
| `Gain_MEG` | `74bb0117947bee5431a74b78728724b852dd3f845021fad65b863901219026fe` |
| `VertConn` | `f5d524269d56d9e466543fb5c2430af0bbddec5f3bf3743b843937eef4760955` |

这里的归档身份是“清单 SHA + 格式版本 + 全部 chunk 边界/`case_id` + 三个跨 chunk 数组指纹”的组合约束；当前没有另行声明一个单文件式的全目录 SHA。运行器直接读取归档的 `F_EEG`/`F_MEG`，不重新生成观测；真值只在评分阶段由清单确定性重建。

旧清单仍保持字节级不变，其 SHA-256 为：

```text
3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76
```

旧归档 `D:\oaster_strict_blind_sisses` 与 corrected-v2 不可互换。

## 6. SISSES 状态

corrected-v2 输入归档保留了供 MATLAB 适配器读取的结构，但当前机器上没有恢复出完整、可执行的 SISSES 核心和完整适配流程；能找到的作者公开代码包也没有清晰可见的许可证，不能据此把第三方源码复制进本仓库或宣称已合法完整复现。

因此：

- corrected-v2 的 SISSES 结果标记为 **N/A（未重跑）**；
- 旧 SISSES 输出基于错误的 legacy 深部几何，标记为 **legacy geometry; not comparable**；
- SISSES 不进入 corrected-v2 排名、显著性检验或“最佳方法”结论；
- 以后只有在取得完整上游实现、确认许可证并完成独立适配审计后，才能在同一冻结归档上补跑。

## 7. 复现命令

以下命令在仓库根目录的 PowerShell 中运行。重新生成几何或归档会产生大量数据，应只在有意建立新版本时执行；已冻结的 corrected-v2 chunk 不应原地修改。

### 7.1 生成并核验几何、清单和输入

```powershell
$repo = 'D:\博士\工作＆汇报\源定位\新建文件夹'
$dataRoot = Join-Path $repo 'corrected_v2\generated'
$manifest = Join-Path $repo 'results\corrected_v2\strict_blind\manifest.json'
$archive = 'D:\oaster_corrected_v2_sisses\matlab_input'
Set-Location -LiteralPath $repo

$env:CORRECTED_DATA_ROOT = $dataRoot
python .\pipelines\generate_datasets.py
python .\generate_corrected_v2_manifest.py generate --data-root $dataRoot --output $manifest
python .\generate_corrected_v2_manifest.py self-check --data-root $dataRoot --output $manifest
python .\generate_strict_blind_inputs.py --manifest $manifest --data-root $dataRoot --output $archive --chunk-size 20
```

对 456 个 chunk 全部检查元数据/边界，并对指定病例重算真值、观测与实际 SNR：

```powershell
python .\verify_strict_archive.py `
  --manifest $manifest `
  --input-dir $archive `
  --data-root $dataRoot `
  --expected-sha256 8147726ec34b548fba6f25fe6c05dd55933e4cf458c7c2cec94c31a9d7c5e60c `
  --sample-cases 0 68 93 152 1116 3141 7812 8928 9113
```

### 7.2 运行 OASTER 与七种 Python 对比方法

```powershell
$strictRoot = Join-Path $repo 'results\corrected_v2\strict_blind'

python .\run_strict_oaster.py `
  --manifest $manifest --input-root $archive --data-root $dataRoot `
  --output (Join-Path $strictRoot 'oaster_v20_final') --workers 4

python .\run_strict_comparators.py `
  --manifest $manifest --input-root $archive --data-root $dataRoot `
  --output (Join-Path $strictRoot 'comparators_final') --workers 4
```

对比运行器默认包含 `MNE`、`dSPM`、`sLORETA`、`eLORETA`、`LCMV`、`Dipole fitting (grid)` 和 `RAP-MUSIC`。分片运行可使用 `--start-chunk` 与 `--limit-chunks`；所有分片结束后必须再执行一次不带这两个参数的命令，以验证全部 checkpoint 并汇总。

### 7.3 指标图、统计分析与脑空间图

```powershell
python .\plot_strict_metrics.py `
  --root $strictRoot --output (Join-Path $strictRoot 'figures_v20')

python .\analyze_strict_statistics.py `
  --root $strictRoot --output (Join-Path $strictRoot 'statistics_v20')

python .\plot_strict_brain_maps.py `
  --manifest $manifest --input-root $archive --data-root $dataRoot `
  --results-root $strictRoot --sisses-mode skip `
  --eeg-snr-db 0 --meg-snr-db 0 `
  --scenario deep_plus_two_surface --location 13 `
  --output-root (Join-Path $strictRoot 'brain_maps_v20') --surface-maps `
  --relative-threshold 0.10 --display-percentile 95
```

修正几何的脑空间图位于 `results\corrected_v2\strict_blind\brain_maps_v20`，分类入口见其中的 [`INDEX.md`](../results/corrected_v2/strict_blind/brain_maps_v20/INDEX.md)。每个病例单独生成 `simulation_truth_mri.png`；`--surface-maps` 还会生成 `simulation_truth_surface.png` 以及 MNE/PyVista 膨胀皮层的左右半球外侧/内侧四视图。真值只用连续能量热图表达，不使用星号、圆圈、绿色 label 或 foci；所有算法图也完全不叠加仿真真值。

算法 MRI 与 surface 默认显示 `cutoff = max(10% 峰值, P95)`；surface 色标使用 P95/P97/P99 对应的归一化能量值，稀疏结果的分位点退化时回退到归一化能量的 10%/55%/100%，真值图只使用 10% 峰值 cutoff。所有含深层源的病例还生成 `simulation_truth_mri_surface.png`、每算法 `*_mri_surface.png` 和 `all_methods_brain_combined.png`：左侧解剖 MRI 包含深浅层源，右侧渲染皮层只含表层分量，两面板独立归一化。丘脑真值不投影到皮层；不要改用旧的 `results\strict_blind\brain_maps_v2`，后者属于错误的 legacy 深部几何。

## 8. 指标口径

- AUC：优先报告原请求的 `An_cal_AUC`，并同时保留 tied-rank 修正版 `An_auc`（结果字段 `auc_tie_corrected`）。
- 空间指标：分别计算表层与深层的 SD、DLE，不用一个总体值掩盖两类源的差异。
- 深部检测：同时报告敏感度、纯表层特异度及其 balanced accuracy。
- 统计比较：只对完整、可比的方法做配对 bootstrap、Wilcoxon + Holm 校正、Friedman 检验与排名；N/A 方法不参与。

## 9. 最终结果

严格盲测已完成：OASTER V20 为 `9,114/9,114`、零错误；七种 Python 对比方法合计 `63,798/63,798` 行、零错误。所有结果对应同一 corrected-v2 manifest SHA、456 个不可变输入 chunk 和修正后的双侧丘脑几何。

以每个 EEG×MEG SNR 区组内四场景等权宏平均为主口径，OASTER V20 的结果为：

| 指标 | 结果 |
|---|---:|
| An_auc（49 区组均值） | 0.967424 |
| An_auc（49 区组最小值） | 0.951656 |
| raw / 原始 AUC | 0.924775 |
| RMSE 历史字段（实际为平方相对误差） | 0.677792 |
| 深部 balanced accuracy | 0.896459 |

OASTER 的 An_auc 在 49 个 SNR 区组中相对每一种对比方法均为 `49/49` 胜出；次优 dSPM 的均值为 0.936026，OASTER 的配对优势为 +0.031398。需要保留的限制是：细分到单独场景×SNR 后，复杂的 `deep_plus_two_surface` 最差单元为 0.880359，因此不能声称 196 个细分单元全部超过 0.90。

SISSES 仍为 N/A，不参与排名或统计检验。各方法完整指标表、置信区间、显著性检验、结果图和独立真值连续热图见[最终报告](../results/corrected_v2/strict_blind/FINAL_REPORT.md)。
