# 公开真实数据多方法对照 v3

此前只展示 ds004998 beta-ERD 的 DICS 与 LCMV，是方法筛选，不是原定的完整对照。仿真中的 OASTER 与七个基线已完成；本轮另外对两种真实信号按各自可比较的物理量补跑，不能把时域 ERP 与频域 beta 功率混成一个排行榜。

| 数据与目标 | 本轮可比较的方法 | 完成范围 | 主要结果 |
|---|---|---|---|
| ds006035 右腕刺激 N20/P30 ERP | OASTER-ERP-v3 + MNE、dSPM、sLORETA、eLORETA、LCMV、网格偶极拟合、RAP-MUSIC | sm09 开发检查 1 run；4 名留出受试者各 3 run | OASTER-v3 N20 左 S1 富集高于均匀基线 2/4 人、P30 3/4；四种最小范数法 N20 均 4/4。LCMV 在本短窗正向功率口径下未检出，不排名。 |
| ds004998 HoldL beta-ERD | DICS、LCMV、MNE、dSPM、sLORETA、eLORETA | 1 名受试者、1 个完整 run；rest/Hold 各 238 个等量 epoch | DICS 全局峰位于预设右感觉运动 ROI；LCMV 与最小范数家族峰在 ROI 外。MNE/dSPM/sLORETA 的 ERD 图逐点相同，不算三项独立验证。 |
| ds006035 左指反应锁定 beta-ERD | DICS、LCMV、MNE、dSPM、sLORETA、eLORETA | sm09 一名受试者、3 run，共 113 个可用反应试次 | 右 precentral 富集 DICS 17.49、LCMV 15.17；LCMV 全局峰在该 ROI，DICS 峰距 ROI 边界 14.8 mm。MNE/dSPM/sLORETA 的 beta 功率比图数值相同。 |

## 结果入口

- [ds006035 完整八方法报告](ds006035_erp_eight_methods/REPORT.md)、[四名受试者三 run 的对比表](ds006035_erp_eight_methods/heldout_method_comparison.csv)、[富集热图](ds006035_erp_eight_methods/heldout_eight_method_enrichment_heatmaps.png)、[N20 脑表面图](ds006035_erp_eight_methods/sub-sm09_run-1/n20_eight_methods_pial_dorsal.png)、[P30 脑表面图](ds006035_erp_eight_methods/sub-sm09_run-1/p30_eight_methods_pial_dorsal.png)。
- [ds004998 六方法报告](ds004998_dbs_v3_multimethod/sub-0cGdk9_HoldL_MedOn_run-1/MULTIMETHOD_REPORT.md)、[方法状态与指标表](ds004998_dbs_v3_multimethod/sub-0cGdk9_HoldL_MedOn_run-1/beta_multimethod_comparison.csv)、[相同 MRI 切面图](ds004998_dbs_v3_multimethod/sub-0cGdk9_HoldL_MedOn_run-1/beta_multimethod_same_slices_mri.png)。
- [左指 beta 六方法简报](ds006035/sub-sm09_finger_beta_multimethod/REPORT.md)、[指标表](ds006035/sub-sm09_finger_beta_multimethod/beta_multimethod_comparison.csv)、[右运动区富集与峰距图](ds006035/sub-sm09_finger_beta_multimethod/beta_multimethod_roi_comparison.png)、[六方法脑表面图](ds006035/sub-sm09_finger_beta_multimethod/beta_six_methods_pial_dorsal_montage.png)。
- 对应可逐块阅读的代码为 `作者风格版/9-ds006035八方法ERP对比.py`、`作者风格版/9-DBS多方法beta对照.py`、`作者风格版/10-左指运动多方法beta对照.py` 和只读源图的 `作者风格版/11-左指运动beta六方法脑图.py`；Finger beta 与刺激锁定 N20/P30 分表呈现。

## 不能跨越的解释边界

真实数据没有已知源真值，不计算仿真意义的 AUC、SD、DLE；左 S1 富集与峰到 ROI 距离仅检查解剖合理性。ds006035 使用 sample 模板 ico4 皮层源网格，没有深部体积源；它不验证深层定位。OASTER-v3 在 N20 的跨受试者命中率和跨 run 稳定性都不够好，P30 约 36 的高富集受稀疏模板与 ROI 面积分母饱和影响，不能解读为定位精度提高几十倍。Finger beta 只有一名受试者，而且其 −400 至 +200 ms 宽运动窗与腕部电刺激重叠，不能单独归因于运动。ds004998 只有一条完整记录，MRI 仅用于模板显示，又无 DBS 触点真值，不能宣称定位到了 STN 或 DBS 电极。

网格偶极拟合与 RAP-MUSIC 的现有实现是时域波形逆解；旧频谱 OASTER 的当前证据针对活动功率高于基线，不直接估计 Hold 期间的 beta 功率下降。它们在 beta 表中标为“需适配”，没有填写伪造数值。真正检验 OASTER 在非锁相运动 beta 上的优势，需要先定义并冻结方向性 ERD 版本，然后在独立数据上验证。
