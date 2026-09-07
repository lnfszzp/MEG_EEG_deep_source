# 联合 EEG–MEG 深浅层源定位恢复与验证报告

## 1. 结论先行

本仓库已恢复可审查的 PPT/V18 后处理链、原始仿真协议、用户指标、七种统一 Python 对比方法、EEG×MEG 独立信噪比开发矩阵、冻结留出矩阵入口和源定位绘图工具。当前冻结候选是 **OASTER V19**（Observation-Adaptive Spatiotemporal Evidence Reconstruction）。它只接收 EEG/MEG 观测、前向矩阵和源空间几何，不接收真实源、SNR 标签、真实源数量或仿真时间过程。

> **解剖学更正：** 旧生成脚本在 `aseg` ROI 查询前漏做 HEAD→MRI 变换。冻结的 15 个所谓“丘脑点”实际为脑干/小脑/第四脑室附近的非皮层点，丘脑点为 0；所以本报告的 deep 结果只适用于该冻结非皮层网格。详见 `SOURCE_SPACE_AUDIT.md`。

开发矩阵共有 980 例：7 个 EEG SNR × 7 个 MEG SNR × 4 个场景 × 每场景 5 个固定源配置。以用户优先指定的直接 `An_auc` 为主指标，即结果列 `auc_tie_corrected`：

- 49/49 个 SNR 组合的场景宏平均均不低于 0.90，平均 `0.974974`，最差 `0.956545`。
- 196/196 个“SNR 组合 × 场景”单元均不低于 0.90，最差 `0.907528`。
- 这不等于每个随机实现都达到 0.90：逐例仍有 **40/980** 个主 AUC 低于 0.90，逐例最低 `0.763910`。
- 次要的历史 parcel-AUC（列 `auc`）平均 `0.934927`，最差 SNR 组合 `0.883494`，只有 47/49 个组合达到 0.90。

冻结留出矩阵现已完整结束：OASTER 为 9,065/9,065 例，七种 Python 对比方法为 63,455/63,455 个“病例×方法”结果，均为零错误。OASTER 的冻结场景宏主 `An_auc` 平均 `0.968997`、最差 SNR 对 `0.955137`，49/49 个 SNR 对达到 0.90；在 OASTER、SISSES 和七种 Python 对比共九种方法中排名第一。这个结论仍不能扩大为“每个场景和每个病例都达到 0.90”：196 个 SNR×场景单元中有 5 个、9,065 个逐例结果中有 763 个低于 0.90。完整比较见第 8 节。

## 2. PPT 与 V18 的恢复边界

取证文件为 `C:\Users\zzp\Desktop\新建 Microsoft PowerPoint 演示文稿.pptx`，共 8 页，SHA-256 为：

```text
D89F4B7B9931DAB291362691A32E327331DFF85DD1CB127629A73FF82A0C791B
```

PPT 明示的算法骨架包括：

1. EEG、MEG 分别用刺激前基线协方差白化，再沿通道轴堆叠。
2. 以数据拟合、图差分稀疏和源稀疏组成初始联合逆问题，并用双分裂 ADMM 求解。
3. 将源分成表层与深层，用观测 SVD 做时间降维，再进行分层候选重拟合。
4. 用活动期相对基线期的残差投影功率增量寻找深层候选。
5. 固定深层贡献后，在 `0/4/7 mm` 高斯表层模板上做残差下降证据选择。
6. 用前向模板相关和时间过程相关去重，联合重拟合，再加入较弱的初始表层范围；深层结果保持不变。

Git 中可核对的 V18 入口是 `run_frozen_v18.py`，对应历史提交 `5ab46b8`。可执行后处理链从历史 SISSES 逐例估计出发，包含模态内白化、分层 ADMM 候选重拟合、深层证据救援、`0/4/7 mm` 表层模板重拟合，以及 25% 弱表层范围校正。PPT 未给出的若干参数可由该冻结代码核对，但不能据此声称 PPT 初始求解器已经逐字恢复。

V18 确认集为 1,110 例，只覆盖 EEG 与 MEG 使用共同 `0/10/20 dB` 的对角线条件。仓库精确场景宏平均 AUC 为 `0.9009097458761279`；PPT 写作 `0.909`，而其余表格指标与仓库三位小数吻合。现有证据不足以判断这一个 AUC 差异的来源，所以同时保留，不把 PPT 数字改写为仓库数字。

V18 不能在现有文件上从零复跑：`benchmark/results/scores.csv` 及其索引的历史逐例 SISSES 源估计未保存在 Git 历史或现存严格归档中。当前可以审查代码、核对 `results/v18_no_oracle/` 的机器可读历史汇总，并在原输入补齐后重新运行；不能把这些已提交汇总称为当前环境重新计算的结果。更完整的逐公式证据边界见 `PPT_ALGORITHM_RECOVERY.md`。

## 3. 冻结候选 OASTER V19

实现位于 `candidates/oaster_rebuilt.py`。处理流程为：

1. 对 EEG 和 MEG 分别做基线噪声白化并堆叠观测与 lead field。
2. 由活动期/基线期功率比构造自适应频谱滤波器，不使用预先给定的源频率。
3. 对滤波后的活动窗做 SVD，以基线最大奇异值经窗口尺寸校正后的噪声边缘选择时间秩。
4. 构造 `0/4/7 mm` 表层模板和逐点深层模板，以扩展 BIC（EBIC）逐步选择；相关性达到 `0.98` 的重复模板不再进入。
5. 对最终模板做比例为 `0.03` 的 ridge 联合重拟合。
6. 将 4 mm 多尺度谱证据按主估计活动期峰值的 `0.05` 比例融合到同一张源图。

投影证据、谱滤波、谱证据、证据缩放融合以及 EBIC 选择循环来自恢复出的精确代码片段。原 `_temporal_basis` 函数正文没有保存；当前正文依据残存调用关系和基线谱边缘公式重建，并已在源码中标注。这是 OASTER 恢复证据中唯一明确的函数级重建边界。

重建函数的输入只有：

```text
EEG observation, MEG observation,
EEG gain, MEG gain,
surface/deep boundary, geometric surface kernels
```

真实源位置、真实分组、真实 SNR、场景名、真实源数量、仿真时间基和真实频率只在仿真生成或事后评分中出现，不进入 `reconstruct()`。源数量由观测驱动的 EBIC 停止规则决定。

## 4. 指标口径

统一评分入口为 `benchmark/metrics.py::evaluate_estimate`，用户原指标保存在 `metrics/user_metrics/`。同一病例的所有指标都来自同一张 `J_est`，没有为 AUC、SD 或 DLE 分别选择输出。

### 4.1 主 AUC：`auc_tie_corrected`

主指标直接调用用户的 `An_auc`。对活动窗内每个源的能量排序，真实活动源为正类。历史 `An_roc` 在分数并列时受行顺序影响；当前实现把每个相等分数块整体推进 ROC，使并列样本获得标准的半权并保证次序不变。为保留结果列兼容性，该主指标字段仍命名为 `auc_tie_corrected`。零估计没有排序信息，固定记为中性值 `0.5`。

### 4.2 次要 AUC：`auc`

`auc` 调用 `An_cal_AUC`，阈值为 `0.01`。多源病例先按最近真实源组划分 parcel，再对各源组的 parcel-AUC 取平均。该列用于复核旧表和保留用户历史口径，不取代直接 `An_auc`，也不用于把未达标条件改写成达标。

### 4.3 RMSE、SD 与 DLE

- `rmse` 沿用旧函数名称，实际定义是两个 Frobenius 归一化源矩阵之间的**平方相对误差**，不是通常意义上的逐元素均方根误差。
- 支持集在表层和深层分别按该层活动窗能量峰值的 10% 阈值确定，避免强表层峰值直接吞掉深层支持。
- `surface_sd_mm` / `surface_dle_mm` 与 `deep_sd_mm` / `deep_dle_mm` 分层计算。DLE 显式指定 Python 的零基索引，不再根据内容猜测 MATLAB/Python 索引基。
- 条件 SD/DLE 在真实层存在但该层没有任何估计支持时为 `NaN`。只对有限值求平均会隐藏漏检，因此正式比较优先同时报告 `*_penalized`：缺失的期望层定位以头部坐标包围盒对角线 `234.029112 mm` 计罚。
- 深层检出固定使用 `deep_score >= 0.1`，且预测深层峰到真实深层源距离不超过 10 mm；平衡准确率 `BA=(sensitivity+specificity)/2`。冻结留出入口不重新选择阈值。

每个 SNR 组合同时输出两类汇总：`scenario_macro` 对四个场景等权，`case_weighted` 按病例数加权。主文比较应优先场景宏平均，同时提供病例加权表供复核。

## 5. OASTER V19 开发矩阵结果

开发输入来自 `benchmark/results/protocol/dev_manifest.json`，其 185 个基础源配置按场景等距取 5 个，再固定源配置、独立改变两个模态的 SNR。SNR 水平为 `-10/-5/0/5/10/15/20 dB`，共 49 个有序组合和 980 例。矩阵种子根为 `20260906`；本轮只调用 `benchmark.protocol.simulate_case`，没有读取冻结留出归档。

下表为 49 个 SNR 组合的场景宏平均再取平均；“最差组合”也在同一聚合层级上计算：

| 指标 | 平均 | 最差组合/说明 |
|---|---:|---:|
| 主 `An_auc` (`auc_tie_corrected`) | 0.974974 | 0.956545；49/49 组合 ≥ 0.90 |
| 次要 parcel-AUC (`auc`) | 0.934927 | 0.883494；47/49 组合 ≥ 0.90 |
| 历史命名 RMSE | 0.783931 | 越低越好 |
| 表层 penalized SD | 3.105882 mm | 缺失层已计罚 |
| 表层 penalized DLE | 7.898508 mm | 缺失层已计罚 |
| 深层 penalized SD | 3.587094 mm | 缺失层已计罚 |
| 深层 penalized DLE | 3.038223 mm | 缺失层已计罚 |
| 深层 sensitivity | 0.823129 | 固定 0.1 阈值与 10 mm 半径 |
| 深层 specificity | 0.995918 | 固定 0.1 阈值 |
| 深层 balanced accuracy | 0.909524 | 最差 SNR 组合为 0.733333 |

机器可读结果位于 `results/dev_matrix/v19_oaster_rebuilt_baseline_5/`：

- `rows.csv`：980 个逐例结果；
- `summary_by_snr_scenario.csv`：196 个 SNR×场景单元；
- `summary_by_snr_pair_scenario_macro.csv`：49 个场景宏平均组合；
- `summary_by_snr_pair_case_weighted.csv`：49 个病例加权组合；
- `manifest.json`、`metadata.json`：本轮配置、种子和输入来源。

开发结果的主要局限是：聚合达标不能覆盖逐例失败；40/980 个逐例主 AUC 低于 0.90，最低 `0.763910`。此外，深层 BA 的最差组合仍只有 `0.733333`，表明某些不对称 SNR 条件下深层是否存在的判断仍比全局排序 AUC 更困难。

## 6. deep-rescue 实验及拒绝理由

为改善低 SNR 深层漏检，开发集上试验了一个最小的残差 EBIC rescue：从 OASTER 主估计的拟合传感器子空间中残差化 15 个深层 gain，每例最多追加一个能显著降低残差的深层点。接收规则为

\[
N\log(\mathrm{RSS}_{new}/\mathrm{RSS}_{old})+r\log N+2\log 15+\tau<0.
\]

该重建规则仍不读取真值、SNR 标签或真实源数量。`tau=0/2/4` 先在每场景 2 个配置的开发筛查上测试，再在每场景 5 个配置、共 980 例的完整开发扩展上验证。为降低配置复用带来的乐观偏差，又排除筛查使用的首尾配置，形成 588 例未见子集。预先约定的接收门槛包括：未见子集 specificity 下降不超过 1 个百分点、BA 不下降、最差主 AUC 至少不差于基线（目标 0.95），且其余均值不劣化。

`tau=4` 在 980 例上将主 AUC 从 `0.974974` 提高到 `0.976323`、parcel-AUC 从 `0.934927` 提高到 `0.945049`、RMSE 从 `0.783931` 降到 `0.766258`、深层 penalized SD/DLE 从 `3.587/3.038 mm` 降到 `2.490/1.894 mm`，BA 从 `0.909524` 提高到 `0.929932`；但 specificity 从 `0.995918` 降到 `0.967347`，下降 2.86 个百分点。

未见 588 例上，`tau=4` 的 specificity 从 `0.993197` 降到 `0.952381`。对已计算开发行做更保守的 `tau=6/8/10/12/16` 事后重分类后，最保守的 `tau=16` 仍降到 `0.979592`，下降 1.36 个百分点，超过预注册的 1 个百分点上限。其余主要均值虽改善或持平，仍不能抵消这一明确失败项。因此 rescue **未合并**，`candidates/oaster_rebuilt.py` 保持 V19 不变；完整负结果保存在 `results/dev_experiments/deep_rescue_ebic_validation5/`，避免只报告有利试验。

该实验和阈值复核均未读取冻结留出观测或结果。

## 7. 冻结留出协议与 SISSES 归档

冻结清单位于 `results/strict_blind/manifest.json`，SHA-256 为：

```text
3eda43e22ce70a17b4659658742aade66053ff7943140638868281e166a0bd76
```

协议包含 49 个有序 EEG/MEG SNR 组合 × 185 个源配置，共 9,065 个唯一病例。每个组合使用同一组 185 个配置，便于算法间和 SNR 间配对比较。输入归档位于 `D:\oaster_strict_blind_sisses\matlab_input`，共有 454 个连续、无重叠的 chunk；case ID、区间、格式版本、嵌入清单哈希、gain 和邻接图指纹均由只读验证器检查。

`D:\oaster_strict_blind_sisses` 并未整体损坏。盘点得到 10,435 个文件、总计 324,261,054,550 字节；其中输入约 9.186 GB，现存 9,065 个 SISSES 输出 MAT 文件约 315 GB，`scores.csv` 的 9,065 行状态均为 `ok`。一份旧日志和隔离文件名曾造成“仍损坏”的表象；当前使用文件已修复并通过清单/输入/分数映射核验。快速核验默认不重新遍历 315 GB 输出内容，因此“归档完整”指现存文件计数、冻结输入、映射和分数表一致，不应表述成每个大型 MAT 数值载荷均重新计算过。

重新生成观测时，特征分解向量可能因 LAPACK 符号选择造成逐位差异，但代回重构的相对误差约为 `1e-15`。严格 runner 直接读取归档的 `F_EEG/F_MEG`，不重新生成观测，从而避免把这种等价但非逐位一致的差异混入比较。

这里的“strict-blind”准确含义是：代码与参数在完整 9,065 例评估前冻结，开发实验不读取该矩阵，最终 runner 只消费冻结观测。它仍是本项目内部生成并保存的**冻结留出仿真**，不是由独立第三方保管、密封和揭盲的外部盲测。首个 20 例 chunk 曾用于输入格式和 checkpoint 的 smoke check；没有据此修改 V19 参数，但这也应在论文中披露。

## 8. 冻结留出最终结果

最终汇总只在所有 checkpoint 完成后生成。根代理再次独立读取逐行 CSV 与所有 part 文件复核，结果如下：

| 检查 | OASTER | 七种 Python 对比 |
|---|---:|---:|
| 完成状态 | `complete` | `complete` |
| 结果行 / 预期行 | 9,065 / 9,065 | 63,455 / 63,455 |
| 唯一病例或 `(case_id, method)` | 9,065 | 63,455 |
| 错误行 | 0 | 0 |
| 输入 chunk | 454 / 454 | 454 / 454 |
| 方法 checkpoint | 454 | 3,178 / 3,178 |
| manifest SHA-256 | `3eda43e2…760d76` | `3eda43e2…760d76` |

现存 SISSES `scores.csv` 另有 9,065/9,065 个 `status=ok` 的保存结果。它通过病例、SNR、场景和关键指标映射核验，但不是本轮重新运行第三方 MATLAB 算法所得。

### 8.1 49 个 SNR 对的场景宏比较

“场景宏”先在每个 SNR×场景内按病例平均，再让四类场景等权。`均值/最差/≥.90` 都以 49 个 SNR 对为单位；RMSE 沿用第 4 节说明的历史定义。分层 SD/DLE 只对真实存在该层的场景有定义，因此表层和深层各自在三个适用场景间取宏平均。

| 排名 | 方法 | 主 An_auc 均值/最差/≥.90 | parcel AUC 均值/最差/≥.90 | RMSE |
|---:|---|---:|---:|---:|
| 1 | OASTER | 0.969 / 0.955 / 49 | 0.928 / 0.902 / 49 | 0.673 |
| 2 | SISSES | 0.965 / 0.835 / 46 | 0.859 / 0.686 / 10 | 1.252 |
| 3 | dSPM | 0.936 / 0.867 / 47 | 0.897 / 0.797 / 28 | 1.828 |
| 4 | eLORETA | 0.932 / 0.794 / 43 | 0.917 / 0.815 / 42 | 1.813 |
| 5 | sLORETA | 0.919 / 0.787 / 40 | 0.901 / 0.813 / 36 | 1.824 |
| 6 | MNE | 0.919 / 0.780 / 38 | 0.897 / 0.776 / 30 | 1.805 |
| 7 | LCMV | 0.758 / 0.731 / 0 | 0.805 / 0.762 / 0 | 1.837 |
| 8 | RAP-MUSIC | 0.649 / 0.631 / 0 | 0.692 / 0.628 / 0 | 1.162 |
| 9 | Dipole fitting (grid) | 0.643 / 0.624 / 0 | 0.663 / 0.621 / 0 | 1.249 |

正式空间误差使用漏检惩罚后的分层值，单位均为 mm：

| 方法 | 表层 SD | 表层 DLE | 深层 SD | 深层 DLE | 深层 BA |
|---|---:|---:|---:|---:|---:|
| OASTER | 3.518 | 8.544 | 4.406 | 4.126 | 0.891 |
| SISSES | 33.543 | 38.675 | 15.086 | 11.536 | 0.495 |
| dSPM | 26.680 | 16.204 | 18.689 | 9.731 | 0.453 |
| eLORETA | 29.887 | 13.094 | 18.802 | 9.379 | 0.409 |
| sLORETA | 32.381 | 20.523 | 18.467 | 11.066 | 0.335 |
| MNE | 34.821 | 30.335 | 19.409 | 16.128 | 0.268 |
| LCMV | 51.763 | 47.993 | 11.030 | 3.823 | 0.699 |
| RAP-MUSIC | 4.632 | 8.913 | 76.464 | 76.462 | 0.835 |
| Dipole fitting (grid) | 6.215 | 10.258 | 109.327 | 109.323 | 0.756 |

深层 BA 的两个组成量不能省略；三者场景宏平均为：

| 方法 | sensitivity | specificity | BA |
|---|---:|---:|---:|
| OASTER | 0.795 | 0.987 | 0.891 |
| SISSES | 0.565 | 0.424 | 0.495 |
| dSPM | 0.602 | 0.305 | 0.453 |
| eLORETA | 0.676 | 0.143 | 0.409 |
| sLORETA | 0.607 | 0.062 | 0.335 |
| MNE | 0.461 | 0.075 | 0.268 |
| LCMV | 0.800 | 0.598 | 0.699 |
| RAP-MUSIC | 0.671 | 0.999 | 0.835 |
| Dipole fitting (grid) | 0.518 | 0.993 | 0.756 |

### 8.2 49 个 SNR 对的病例加权比较

病例加权结果让每个 SNR 对的 185 例直接等权，因此病例/源配置数量较多的场景权重更高。空间列仍是 miss-penalized 值。

| 方法 | 主 An_auc 均值/最差/≥.90 | parcel AUC 均值/最差/≥.90 | RMSE | 表层 SD/DLE | 深层 SD/DLE | 深层 BA |
|---|---:|---:|---:|---:|---:|---:|
| OASTER | 0.968 / 0.953 / 49 | 0.919 / 0.886 / 40 | 0.742 | 3.392 / 7.487 | 5.272 / 4.927 | 0.872 |
| SISSES | 0.961 / 0.811 / 45 | 0.873 / 0.696 / 18 | 1.243 | 34.874 / 39.266 | 15.905 / 12.335 | 0.478 |
| dSPM | 0.929 / 0.856 / 46 | 0.876 / 0.777 / 4 | 1.805 | 27.166 / 15.432 | 19.472 / 11.571 | 0.407 |
| eLORETA | 0.925 / 0.761 / 42 | 0.897 / 0.774 / 33 | 1.793 | 30.599 / 12.035 | 19.634 / 11.732 | 0.367 |
| MNE | 0.909 / 0.743 / 36 | 0.875 / 0.738 / 18 | 1.795 | 35.788 / 29.921 | 20.343 / 17.848 | 0.238 |
| sLORETA | 0.908 / 0.746 / 37 | 0.877 / 0.765 / 6 | 1.816 | 32.839 / 19.931 | 19.296 / 12.820 | 0.303 |
| LCMV | 0.705 / 0.674 / 0 | 0.749 / 0.703 / 0 | 1.914 | 54.164 / 49.285 | 13.088 / 4.716 | 0.673 |
| RAP-MUSIC | 0.572 / 0.556 / 0 | 0.625 / 0.553 / 0 | 1.409 | 4.821 / 8.552 | 93.568 / 93.565 | 0.798 |
| Dipole fitting (grid) | 0.564 / 0.555 / 0 | 0.590 / 0.551 / 0 | 1.504 | 6.243 / 9.681 | 134.912 / 134.906 | 0.698 |

### 8.3 结论和未达标部分

OASTER 在场景宏与病例加权的主 `An_auc` 均值、最差 SNR 对和达标对数上均为第一；它也是场景宏主 AUC 和 parcel-AUC 唯一同时 49/49 达标的方法。其表层/深层 penalized 空间误差、RMSE 和深层 BA 的组合也最均衡。LCMV 的深层 DLE 较小、RAP-MUSIC 和网格偶极子拟合的深层特异度很高，但前者表层误差与 AUC 较弱，后两者因深层漏检而出现很大的惩罚后深层 SD/DLE，因此不能只选一个有利指标判断优劣。

严格矩阵的剩余短板必须原样保留：

- 196 个 SNR×场景单元中，OASTER 有 191 个主 `An_auc≥0.90`；5 个未达标单元均为 `deep_plus_two_surface`，最差为 EEG `-10 dB`、MEG `+5 dB` 的 `0.892252`。
- 逐例有 8,302/9,065 个主 AUC 达标，763 个未达标，最低 `0.548685`。
- 深层 BA 场景宏平均为 `0.891357`、最差 SNR 对 `0.740196`。高特异度 `0.987395` 与较低灵敏度 `0.795318` 表明主要短板仍是低 SNR 复合场景中的深层漏检。
- 最终严格结果只用于一次性评估；拒绝的 deep-rescue 没有因看到这些结果而重新启用，V19 参数也没有回调。

OASTER 分片计算约 56 分 47 秒；七种对比方法计算约 118.9 分钟，连同完整合并复核约 123.1 分钟。两批任务曾并发且机器有前台负载，因此这些墙钟时间不能作为方法速度排名。

机器可读汇总和完成标记位于 `results/strict_blind/oaster_v19_final/`、`results/strict_blind/comparators_final/` 与 `results/strict_blind/sisses_preserved/`。完整逐例 `rows.csv` 和 checkpoint 留在本机但因体积与可再生性不提交 Git；代码、汇总、元数据、完成标记、报告和图均纳入版本控制。详细九方法最差值、病例加权表和耗时见 `results/strict_blind/comparators_final/REPORT.md`，OASTER 的场景级失败与代表病例见 `results/strict_blind/oaster_v19_final/REPORT.md`。

## 9. 对比方法与未恢复项

`benchmark/methods.py` 和 `run_strict_comparators.py` 可在同一冻结观测、白化和评分口径下直接运行七种方法：

| 可运行方法 | 当前实现边界 |
|---|---|
| MNE、dSPM、sLORETA、eLORETA | 统一最小范数族数值实现；一次分解输出四种估计 |
| LCMV | 联合 EEG–MEG 白化数据上的固定方向波束形成 |
| Dipole fitting | 观测时间基压缩，BIC 选择 1–3 个网格点 |
| RAP-MUSIC | 联合白化观测上的递归子空间定位 |

这些是仓库内统一、可运行的数值实现，不应冒充各上游工具箱逐行相同的官方代码。

现存 SISSES 只提供冻结 `scores.csv` 和大型逐例 MAT 归档的只读核验/汇总。第三方 SISSES 源码及可复现 MATLAB 适配器未随项目保存，因此当前入口不会重新运行 SISSES，也绝不覆盖归档。SI-DCB、TS-Champagne、Champagne、FAST-IRES 与 SISSY 目前只有已确认上游或论文线索，没有经过审查的本仓库适配器；ConvDip 是 EEG-only、皮层-only 且需重新训练，不能进入联合 EEG–MEG 深层主排名。详细许可证与接入边界见 `COMPARATORS.md`。

## 10. 完整复现命令

以下命令均在 PowerShell 中运行。当前机器验证过的组合是用 Anaconda Python 作为入口，并把现有 MNE 环境的 site-packages 放入 `PYTHONPATH`；在新机器上更推荐创建干净环境后直接安装 `requirements.txt`。

```powershell
Set-Location 'D:\博士\工作＆汇报\源定位\新建文件夹'
$env:PYTHONPATH='C:\Users\zzp\.conda\envs\meg\Lib\site-packages'
$PY='D:\app\anaconda\python.exe'
$env:SOURCE_DATA_ROOT='D:\博士\工作＆汇报\源定位\codex\roi_deep_multimethod_comparison\generated'
$env:STRICT_SISSES_ARCHIVE='D:\oaster_strict_blind_sisses'

& $PY -m pip install -r requirements.txt
& $PY -m pytest -q
```

若仿真数据不存在，先生成；已有数据则不要重复生成：

```powershell
& $PY pipelines\generate_datasets.py
& $PY generate_protocol_manifests.py --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data'
& $PY generate_strict_blind_manifest.py self-check --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data'
```

只读核验 SISSES 冻结归档：

```powershell
& $PY run_strict_blind_sisses.py verify --archive $env:STRICT_SISSES_ARCHIVE
& $PY run_strict_blind_sisses.py summarize --archive $env:STRICT_SISSES_ARCHIVE --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\sisses_preserved
```

重跑 OASTER V19 开发矩阵：

```powershell
& $PY run_oaster_dev_matrix.py --data-root $env:SOURCE_DATA_ROOT --sample-path 'D:\mne_data\MNE-sample-data' --cases-per-scenario 5 --workers 4 --output results\dev_matrix\v19_oaster_rebuilt_baseline_5
```

冻结留出全量 OASTER 与七方法比较；重复同一命令会验证并续用已完成的原子 chunk checkpoint：

```powershell
& $PY run_strict_oaster.py --input-root "$env:STRICT_SISSES_ARCHIVE\matlab_input" --data-root $env:SOURCE_DATA_ROOT --workers 4 --output results\strict_blind\oaster_v19_final
& $PY run_strict_comparators.py --input-root "$env:STRICT_SISSES_ARCHIVE\matlab_input" --data-root $env:SOURCE_DATA_ROOT --workers 4 --output results\strict_blind\comparators_final
```

分片运行时可给两个严格入口添加 `--start-chunk` 和 `--limit-chunks`，每个分片必须使用不同输出目录。全部分片复制到同一最终目录后，必须再执行一次不带范围限制的命令；只有它会验证全部 454 个 chunk 并生成最终 `rows.csv`、汇总表和 `completion.json`。算法或评分代码改变后必须使用新输出目录；只有明确审查旧 checkpoint 后才允许 `--force`。

生成主 AUC 热图和单例源定位图：

```powershell
& $PY plot_snr_matrix.py --input 'OASTER=results\strict_blind\oaster_v19_final\summary_by_snr_pair_scenario_macro.csv' --input 'SISSES=results\strict_blind\sisses_preserved\summary_by_snr_pair_scenario_macro.csv' --input 'Python=results\strict_blind\comparators_final\summary_by_snr_pair_scenario_macro.csv' --metric auc_tie_corrected --output results\strict_blind\figures\auc_matrix.png
& $PY plot_strict_case.py --case-number 0 --input-root "$env:STRICT_SISSES_ARCHIVE\matlab_input" --data-root $env:SOURCE_DATA_ROOT --output results\strict_blind\figures\strict_case_00000.png
```

V18 的历史入口只有在缺失的 `benchmark/results/scores.csv` 及其逐例 SISSES 源估计恢复后才能执行；在此之前请直接查看 `results/v18_no_oracle/`，不要把失败的入口包装成“从零复现”。

## 11. 目录导航

| 内容 | 路径 |
|---|---|
| PPT/V18 取证说明 | `PPT_ALGORITHM_RECOVERY.md` |
| OASTER V19 核心 | `candidates/oaster_rebuilt.py` |
| 仿真协议与真值恢复 | `benchmark/protocol.py`、`pipelines/generate_datasets.py` |
| 用户指标与统一评分 | `metrics/user_metrics/`、`benchmark/metrics.py` |
| 七种 Python 对比 | `benchmark/methods.py`、`run_strict_comparators.py` |
| 开发矩阵入口/结果 | `run_oaster_dev_matrix.py`、`results/dev_matrix/v19_oaster_rebuilt_baseline_5/` |
| deep-rescue 负结果 | `experiment_oaster_deep_rescue.py`、`results/dev_experiments/deep_rescue_ebic_validation5/` |
| 冻结清单与严格入口 | `results/strict_blind/manifest.json`、`run_strict_oaster.py` |
| OASTER 严格结果 | `results/strict_blind/oaster_v19_final/REPORT.md`、`completion.json`、`summary*.csv` |
| 九方法严格比较 | `results/strict_blind/comparators_final/REPORT.md`、`completion.json`、`summary*.csv` |
| SISSES 只读核验 | `verify_strict_archive.py`、`run_strict_blind_sisses.py` |
| 绘图与正式图片 | `plot_snr_matrix.py`、`plot_strict_case.py`、`results/strict_blind/figures/` |
| 仿真数据 | `D:\博士\工作＆汇报\源定位\codex\roi_deep_multimethod_comparison\generated` |
| 冻结观测/SISSES 归档 | `D:\oaster_strict_blind_sisses`（只读） |

## 12. 已验证环境

本次恢复和开发矩阵所在机器：

| 项目 | 版本/位置 |
|---|---|
| OS | Windows 11 `10.0.26200` |
| 开发/测试 Python 入口 | `D:\app\anaconda\python.exe`, Python `3.13.5` |
| 最终 strict runner | `C:\Users\zzp\.conda\envs\meg\python.exe`, Python `3.13.15` |
| Python 扩展路径 | `C:\Users\zzp\.conda\envs\meg\Lib\site-packages` |
| NumPy | `2.5.2` |
| SciPy | `1.18.1` |
| h5py | `3.16.0` |
| MNE-Python | `1.12.1` |
| Matplotlib | `3.11.1` |
| nibabel | `5.4.2` |
| pytest | `8.3.4` |
| MNE sample 数据 | `D:\mne_data\MNE-sample-data` |
| MATLAB | `D:\app\matlab2020a\bin\matlab.exe`；维护中的 OASTER/七方法严格入口不调用 MATLAB |

`requirements.txt` 给出可安装下限，不是锁文件。严格 runner 的 `metadata.json` 会记录 Git commit、算法/指标 SHA-256 和数值环境；不同 BLAS/LAPACK 环境不承诺逐位一致，应以容差内指标复现和清单/代码哈希一致为准。

## 13. 不可越过的解释边界

- PPT 描述、Git V18、OASTER V19 是相关但不同的证据层；不能把 OASTER 的开发结果写成 PPT/V18 的原始结果。
- V18 总体 AUC `0.900910` 不能代表其每个 SNR/场景达标；其 0 dB 混合源条件明显较弱，且历史逐例输入缺失。
- OASTER V19 在冻结留出的 49/49 个 SNR 对上达到主 `An_auc≥0.90`，但 196 个 SNR×场景单元中有 5 个、9,065 个逐例结果中有 763 个未达标；病例加权 parcel-AUC 也只有 40/49 个 SNR 对达标。
- SD/DLE 的有限值均值会忽略漏检；正式表格必须保留 penalized 列，并将表层与深层分开。
- deep-rescue 改善多数均值仍被拒绝，因为它违反了预注册 specificity 安全门槛；不能因结果看起来更好而事后放宽门槛。
- 冻结留出结果已经在完整性验证后一次性汇总；它不能继续用于选择 V19 参数，未来算法迭代必须另设预声明开发/留出协议。
- 本项目的严格矩阵是冻结留出仿真，不是独立第三方密封盲测。论文中的用语应保持这一限定。
