# 旧版高 AUC 与 v6 的只读审计

范围：只检查历史/PPT、V18、corrected-v2 V20、ERP v3-v5 和 v6 **development** 结果；未读取或运行 v6 冻结校准/验证病例。

## 结论

当前问题不是“AUC 与不报深源只能二选一”。同一批 5/5 dB 开发病例中，`trial covariance + ADMM` 的 v6 完整模型 `An_auc` 分别为
`0.988883, 0.976519, 1.000000, 0.992009, 0.984050`，均值 `0.988292`，而且 H0/H1 全部收敛。空间排序已经回到旧 v5 的水平。

但它的旧式幅度判定为深源 `TP=3/3, FP=2/2`，独立预测分数又把两个混合深源病例判成负增益（`T=-0.0790, -0.0600`）。所以剩余故障是“深源存在性证据”，不能再用高全局 AUC 或幅度阈值掩盖。

## 数值不能混为一个 AUC

| 版本 | 报告的主数值 | 病例/协议 | 同时暴露的问题 |
|---|---:|---|---|
| PPT / V18 | `auc=0.900910` | 1,110 例；EEG=MEG，仅 0/10/20 dB | 这是 `An_cal_AUC`，不是现在优先的 tied-rank `An_auc`；0 dB 深+单浅/双浅仅 `0.5818/0.6520` |
| ERP v3 独立确认 | `An_auc=0.960057` | 18,081 例；369 配置×49 SNR | 深层 AUC `0.8905`，深源敏感度 `0.651`、特异度 `0.893`、漏检惩罚深层 DLE `82.20 mm` |
| corrected-v2 V20 | `An_auc=0.967424` | 9,114 例；186 配置×49 SNR | 单次观测、频谱型仿真；检测阈值历史记录为 `0.1`，BA `0.8965` |
| ERP v5 小型复测 | `An_auc=0.988010` | 4 配置×5 SNR，共 20 例 | 纯表层 `FP=5/5`；高 AUC 明确没有解决“无深源时不报” |
| v6 IRLS-300 开发 | `An_auc=0.871091` | 5 配置×4 SNR，共 20 例 | `TP=6/12, FP=5/8`；IRLS 空间解本身退化 |
| v6 trial-cov ADMM | `An_auc=0.988292` | 当前仅 5 配置×5/5 dB | `TP=3/3, FP=2/2`；只证明空间图恢复，不是验收通过 |

字段含义容易误读：结果中的 `auc_tie_corrected` 才是用户要求优先使用的 `An_auc`；字段 `auc` 是历史 parcel-based `An_cal_AUC(threshold=0.01)`。同一批 v6 trial-cov ADMM 五例，前者均值 `0.988292`，后者只有 `0.748631`，不能把两列直接当作算法前后退步。

## 已核实的四类差异

### 1. 仿真与病例难度

- PPT/V18 使用旧频谱协议：活动期是带包络的 `11, 14, 17 Hz` 正弦波；算法同时包含自适应谱滤波和 4 mm 多尺度谱证据。它与当前瞬态 ERP 问题不是同一任务。
- PPT/V18 只测共同 SNR `0/10/20 dB`，没有 `-10 dB` 或 EEG/MEG 不对称组合；聚合均值掩盖了 0 dB 混合源失败。
- v3 的每例逆解使用一个“40 次平均等效”观测；V20/PPT 旧频谱协议只是单个合成 epoch，并没有保存或拆分 trial。两者都按该次活动窗的实际噪声能量缩放，因此每例实际 SNR 恰好等于目标。
- v6 把同一 40 次预算拆成独立的 train-20/check-20；每半期望 SNR 比目标低 `3.0103 dB`，并按期望噪声能量一次定标，不按抽到的活动噪声重新归一化。H1 空间 AUC目前评估的是更吵的 train-20 解，check-20 只负责存在性证据。
- 所有这些仿真仍用同一固定 MNE sample forward 生成和反演，不能当成跨被试或 forward 失配验证。

### 2. 空间模型和求解器

- 旧 V20/v3 在每个皮层点预建 `0/4/7 mm` 高斯模板，再做 EBIC 稀疏选择；仿真真值恰好也是二跳邻域内的 `sigma=10 mm` 高斯 patch。虽非逐例真值输入，但模型族高度匹配，天然有利于排序 AUC。
- v4-v6 按用户要求改成 7,514 点联合图重加权，不再从三个固定模板挑选。这个改变提高了任务自由度，也使数值收敛和浅深惩罚更难。
- v5 与 v6 H1 使用同一个 `reconstruct_evoked_oaster_v5_from_whitened` 核心。历史和当前配对结果都说明 ADMM 可保留高空间 AUC；低 AUC 主要来自当前 IRLS 解及其扩散，并非独立检出验收本身必然造成。

### 3. 预处理与“同一观测”

- v3/V20 用同一 noisy 观测完成白化、模态权重、空间选择和输出；随后 AUC 与真值比较。它们没有 held-out 观测检验“加入深层块是否可重复”。
- v6 的 H0/H1 只拟合 train-20，check-20 不参与拟合；模态权重也只由训练观测得出。trial-covariance 版本仅在确有训练试次基线时增加协方差信息，不从 ERP 均值伪造真实试次。
- 因此不能把旧版在训练观测上的空间 AUC与 v6 的 held-out presence 分数视作同一指标。

### 4. 评价口径、阈值和真值

- `An_auc` 是所有活动真值顶点与其余顶点的全局排序。源空间有 7,498 个皮层点、仅 16 个深点；少量深层误报对全局 AUC影响很小。
- 纯表层病例没有深层正类，`deep_auc_tie_corrected` 被定义为 `NaN`，所以“深层 AUC”从定义上不能检验纯表层假阳性。v5 的 `An_auc=0.9880` 与 `FP=5/5` 是现成反例。
- `0.1/0.125/0.14` 是深源相对幅度判定阈值，不参与 tied-rank `An_auc`。旧开发集曾用标签选择阈值；正式确认后虽冻结，但它仍是幅度规则，不是独立存在性证据。
- 代码路径核查未发现正式逆解读取逐例真值：V18、ERP v3-v5、corrected-v2 V20 都只在最后 `evaluate_estimate/_row` 评分时传入 truth。历史的 oracle 代码出现在明确标注的开发诊断，不在正式逆解入口。

## 最小可复用结论

1. 保留 v5/v6 的 `smooth DCT + layer calibration + ADMM` 空间解；有真实 epochs 时保留 trial-baseline covariance。这是当前恢复 AUC 的最小路径。
2. 不恢复 0/4/7 mm 固定模板作为最终算法；它会违背“活动区形状随数据自适应”的要求，并重新引入仿真模型匹配优势。
3. 不用全局 AUC、deep AUC 或 0.14 幅度阈值替代 presence gate。下一步只需修独立浅/深模型证据，同时继续以 FPR、localized sensitivity 和 DLE 作硬门。

## 证据文件

- `PPT_ALGORITHM_RECOVERY.md`
- `results/v18_no_oracle/test_summary.csv`, `test_by_cell.csv`
- `ERP_V3_CONFIRMATION_RESULTS.md`
- `results/erp_whole_head/adaptive_v5/paired_v3_v4_v5/version_summary.csv`
- `results/erp_whole_head/adaptive_v6/development_summary/method_summary.csv`
- `results/erp_whole_head/adaptive_v6/dev_trial_covariance_admm_5_5/{rows,evidence}.csv`
- `benchmark/{protocol,erp_protocol,erp_replicates,metrics}.py`
- `candidates/{oaster_rebuilt,oaster_balanced,oaster_predictive}.py`
