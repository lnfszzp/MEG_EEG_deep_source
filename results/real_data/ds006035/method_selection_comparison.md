# ds006035：按信号成分选择源定位方法（现有结果复核）

本报告只整理已经算出的结果，不重新运行逆解。数据是右腕正中神经电刺激（event 32）后抬起左食指（event 16）的同步 EEG–MEG；本地有 `sm04/sm06/sm07/sm09/sm12` 各 3 个 run。听觉、视觉图来自 **MNE sample**，不属于 ds006035，不能合并为本数据集的验证样本。

| 成分与锁定事件 | 使用的时间/频段 | 适合的现有方法及对比结论 |
|---|---|---|
| 右腕刺激 N20，event 32 | 刺激后 18–24 ms | 时域 ERP：dSPM/eLORETA 作为当前稳健参照；OASTER-ERP 保留对比，但锁定验证中 N20 左 S1 富集 >1 仅 1/4 名受试者，dSPM 为 4/4。旧频谱 OASTER 的高 ROI 富集不能证明它分辨出 N20。 |
| 右腕刺激 P30，event 32 | 刺激后 28–40 ms | 时域 ERP：OASTER-ERP、dSPM、eLORETA 都可展示；OASTER-ERP 的 P30 左 S1 富集受试者中位数 37，dSPM 7.08，eLORETA 8.47，但单模板造成离散饱和值 37，不能视为连续精度优势；相对 dSPM 的符号检验 p=0.625。 |
| 左指动作 MF，event 16 | 反应前 −80 至 −20 ms | 反应锁定时域：eLORETA 右 precentral 富集 1.66、峰距 5.3 mm；dSPM 为 1.49、15.4 mm（共同成功的 runs 1、3）。OASTER-ERP 的右 precentral 质量为 0，且仅 2/3 run 通过 EBIC；不宜作主要定位结果。 |
| 左指动作 MEFI，event 16 | 反应后 20–60 ms | 反应锁定时域：dSPM 右 precentral 富集 2.98、峰位于 ROI；eLORETA 右 postcentral 富集 3.43、峰位于 ROI。可解释为感觉运动响应，但不能把 MEFI 单独认定为纯运动或纯感觉。 |
| 左指动作 beta-ERD | 15–30 Hz，反应前 −400 至反应后 200 ms 相对远端基线 | 单试次共同滤波器 DICS：3 run 汇总的右 precentral 富集 17.49、质量 54.7%；右 postcentral 富集 4.25。是目前最明确的非锁相感觉运动网络证据。 |
| 左指动作 beta-PMBR | 15–30 Hz，反应后 500–1000 ms 相对远端基线 | DICS：右 precentral 富集 1.92、峰距 5.3 mm，但 run 2 的全局峰不在预期区域，可信度低于 beta-ERD。 |

刺激锁定 N20/P30 的正式对照采用 `sm09` 开发、其余 **4 名受试者/12 个 run** 锁定验证，保留 474/480 个触觉试次。每名受试者先取自身 3 个 run 的指标中位数，再以受试者为统计单位；完整表与检验见 [锁定验证报告](../ds006035_oaster_erp_v2/group_final/GROUP_REPORT.md)。旧频谱联合 OASTER 在 N20 与 P30 的峰顶点 12/12 个 run 完全相同，因此虽然 ROI 富集高，也不能充当两个潜伏期的独立定位。OASTER-ERP 与 dSPM 的 N20 富集差异同样不显著（符号检验 p=0.625）。

动作结果只针对开发受试者 `sm09` 的 3 个 run：118 对严格匹配的刺激—反应事件、113 个可用反应锁定 epoch。时域同窗比较在每窗使用相同 run、相同试次权重：MF 和 MEFII 为 runs 1、3（75 次），MEFI 为 runs 1–3（113 次）；OASTER 未定位的 run 另列成功率，不能只看条件成功后的脑图。MEFII pooled 全局峰不在预期右侧中央区，mu-DICS pooled 峰也偏离中央区，这两项不作为主要结论。详细证据见 [Finger 双链报告](sub-sm09_finger_dual_chain/RESULTS.md)、[同窗时域指标](sub-sm09_finger_dual_chain/time_domain_matched_method_comparison.csv)、[OASTER 定位成功率](sub-sm09_finger_dual_chain/oaster_detection_rate.csv)和 [DICS 指标](sub-sm09_finger_dual_chain/dics_mu_beta_erd_pmbr_metrics.csv)。

图像：[MF/MEFI/MEFII 同窗脑图](sub-sm09_finger_dual_chain/time_domain_same_window_pial_dorsal_montage.png)、[beta DICS 脑图](sub-sm09_finger_dual_chain/dics_beta_pial_dorsal_montage.png)。图中各方法分别峰值归一化，P95 仅控制显示，**不能跨方法比较振幅或激活范围**。时域 ERP 与频域 DICS 测量的物理量不同，应并列呈现，不能按同一数值指标直接排名。当前尚无同试次、同频段、同基线的旧频谱 OASTER 对 DICS 的 Finger beta 对照，不应声称其中一方获胜。

解释边界：这里的真实数据 OASTER-ERP 结果来自历史迁移版，**不是**最新全头仿真使用的 ERP-v3；不能把仿真 AUC 直接转作真实数据表现。真实数据没有已知源位置，不计算仿真意义的 AUC/DLE；ROI 富集与“峰到 ROI 距离”只检验解剖合理性，峰距 0 不是零定位误差。现有逆解采用刚性配准的 MNE sample 皮层模板而非个体 MRI/BEM/trans，尚无深部体积源；毫米距离仅供算法排错，不能用作论文或临床精度。Finger 的 beta-ERD 宽时间窗还覆盖多数试次的腕部刺激，不能将全部功率变化归因于运动。N20/P30 锁定验证仅 4 名受试者，Finger 仅 1 名开发受试者；不把 run 当独立受试者，也不把这些结果外推为算法普适性。后续真正的外部验证应先完成个体头模，在其他受试者上按固定的事件、时间窗、通道和评分规则重复。

代码入口：刺激锁定批处理为 [`pipelines/run_ds006035_somatomotor.py`](../../../pipelines/run_ds006035_somatomotor.py) 与 [`pipelines/summarize_ds006035.py`](../../../pipelines/summarize_ds006035.py)；Finger 逐块脚本为 [`作者风格版/5-左指运动双链定位.py`](../../../作者风格版/5-左指运动双链定位.py)，已有源图的渲染脚本为 [`作者风格版/6-左指运动结果俯视图.py`](../../../作者风格版/6-左指运动结果俯视图.py)。后两者的参数和窗口固定在脚本开头。锁定验证属于历史 v2 输出，精确重跑还须固定当时的代码版本与参数；本报告不覆盖已有结果。
