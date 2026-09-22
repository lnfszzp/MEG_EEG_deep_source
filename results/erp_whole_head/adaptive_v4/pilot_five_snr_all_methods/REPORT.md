# ERP-v4 自适应空间重加权：开发复测

## 结论

已实现不选择高斯模板的逐点/逐边迭代重加权，并完成本轮五 SNR、四场景、八方法复测。**新版表层 AUC 提高，但深层明显退化，尚不能替代 v3，更不能宣称各种情形均达到 90%。**

本轮是开发诊断：20 个病例 = 4 个固定源配置 × 5 组 SNR；不是全 68 分区 × 全 49 SNR 的完整复测，也不是独立盲测。每次反演仍搜索整个 7,514 点源网格，未用真值缩小候选集。

## 已完成规模

- 场景：纯表层、纯深层、深+单表层、深+双表层，每类 5 例。
- EEG/MEG SNR（dB）：−10/−10、−10/20、5/5、20/−10、20/20；为40次平均等效的 evoked-level SNR。
- 皮层配置中心 1317；双表层增加 7204；深源全局索引 7498。索引为0起始。这轮没有重新扫描所有真实源位置。
- 方法：v4、MNE、dSPM、sLORETA、eLORETA、LCMV、Dipole fitting (grid)、RAP-MUSIC，共160行，计算错误0行。
- 旧 v3 同病例结果按 case_id 配对；manifest、噪声根种子 20260921、共享前向/噪声指纹均核对一致。
- 全部方法接收同一模拟观测、白化、基线和活动窗；OASTER 保留旧版的模态证据权重。

## 同病例新旧版本

| 指标 | v3固定模板 | v4自适应图 |
|---|---:|---:|
| 总体 An_auc | 0.9831 | 0.9208 |
| 表层 An_auc | 0.9773 | 0.9893 |
| 深层 An_auc | 0.9689 | 0.6311 |

表层/深层只在存在对应真值的病例中平均。20例是同4个源配置跨SNR重复，不是20个独立受试者。

| 场景 | v3总体 An_auc | v4总体 An_auc | v4表层 An_auc | v4深层 An_auc |
|---|---:|---:|---:|---:|
| 纯表层 | 0.9812 | 0.9933 | 0.9933 | 不适用 |
| 纯深层 | 0.9996 | 0.8002 | 不适用 | 0.9000 |
| 深+单表层 | 0.9846 | 0.9410 | 0.9933 | 0.4933 |
| 深+双表层 | 0.9671 | 0.9487 | 0.9814 | 0.5000 |

混合源总体 AUC 仍有0.94左右，但深源排序接近随机、实际没有检出。因此总体 AUC 高不等于深浅源都定位成功。

## 与七个对比方法

下表为四场景等权、五SNR格等权的描述均值。

| 方法 | 总体 An_auc | 表层 An_auc | 深层 An_auc | 深源敏感度 |
|---|---:|---:|---:|---:|
| OASTER-ERP-v4 | 0.9208 | 0.9893 | 0.6311 | 0.2667 |
| MNE | 0.8901 | 0.8619 | 0.8222 | 0.3333 |
| dSPM | 0.9046 | 0.8771 | 0.5467 | 0.2667 |
| sLORETA | 0.8635 | 0.8192 | 0.7200 | 0.3333 |
| eLORETA | 0.9094 | 0.8925 | 0.7733 | 0.4667 |
| LCMV | 0.6938 | 0.5827 | 0.8756 | 0.5333 |
| Dipole grid | 0.6306 | 0.5077 | 0.6667 | 0.3333 |
| RAP-MUSIC | 0.6292 | 0.5058 | 0.6667 | 0.3333 |

v4 深源检出4/15，纯浅病例深源假阳性0/5，表层惩罚SD/DLE=13.91/19.04mm，深层惩罚SD/DLE=171.62/171.62mm。深层的大距离是漏检惩罚，不是说检出的点实际偏离171mm。逐例原始距离、检测及罚分均保留，没有只统计成功病例。

## 数值状态：不能把“程序运行完”当“算法已收敛”

- 48项数值/集成回归测试通过。
- 20病例中，8例满足本版内外层停止标准。
- 16例所有已接受内层迭代达到容差；10例达到外层变化阈值；7例触发目标上升保护。这些是重叠计数。
- 其余12例是带明确诊断的近似结果，不作为成熟算法优势证据。图表保留所有病例，没有删除未收敛或低分病例。

本版固定参数：MRF强度0.5、边缘系数0.5、噪声系数1；ADMM每子问题最多2000步、相对容差3e−4，外层最多12轮、相对变化阈值1%。目标下降检查允许与内层容差一致的数值误差，超出则回退并标记。每个病例的完整诊断在 `parts_*/diagnostics/`。

## 为什么不能直接宣布成功

直接对物理电流作逐点稀疏会压缩连片源，单纯增强TV又会改变浅/深源的相对惩罚。加入MRF邻域创新先验后，皮层恢复改善，但混合中弱深源被吸收到皮层解释。**下一步关键是浅/深源的联合可辨识性与惩罚平衡，以及数值收敛，不是继续在三个模板中挑一个。**

本轮没有改真实数据的 GFP/选时流程，也没有证明真实 ERP 已定位成功。旧 v3 保留，v4是实验分支。

## 图和表

- `figures_pilot/eight_method_pilot_comparison.csv`：八方法全部指标。
- `figures_pilot/eight_method_pilot_metrics.png`：分层指标、漏检惩罚距离、深检率。
- `figures_pilot/four_scenarios_five_snr_pilot.png`：四场景五SNR逐格结果。
- `paired_v3_v4/paired_cases.csv`：20例逐一配对。
- `paired_v3_v4/paired_summary.csv`：配置聚类的描述性区间，只有4配置，不能作为论文级统计推断。
- `paired_v3_v4/paired_auc_scatter.png`、`paired_auc_snr_change.png`：新旧版配对图，未运行的SNR格明确灰色。

代码和完整公式见项目根目录 `ERP_V4_ADAPTIVE_EXPERIMENT.md` 与 `candidates/oaster_adaptive.py`。非高斯12病例补充验证另存于 `../shape_mrf_final/`。

## 复现这20例

在项目根目录、meg Python环境、BLAS/OMP线程为1时运行：

```powershell
python run_erp_whole_head_matrix.py --method OASTER-ERP-v4 --manifest results/erp_whole_head/development_full_v3/manifest.json --output results/erp_whole_head/adaptive_v4/pilot_five_snr_all_methods --modality-weighting evidence --seed-root 20260921 --cases-per-scenario 1 --workers 4 --snr-pair -10 -10 --snr-pair -10 20 --snr-pair 5 5 --snr-pair 20 -10 --snr-pair 20 20
python plot_adaptive_pilot.py
python 作者风格版/16-ERP新版旧版配对结果.py results/erp_whole_head/adaptive_v4/pilot_five_snr_all_methods
```

移除 `--cases-per-scenario` 才是每格全部188配置；移除所有 `--snr-pair` 才是完整49格。本报告不把这些未完成的大规模实验计入结果。
