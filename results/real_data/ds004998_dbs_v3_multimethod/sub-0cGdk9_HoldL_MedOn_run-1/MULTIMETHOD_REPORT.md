# ds004998 HoldL：六种可比 beta-ERD 方法

DICS、LCMV、MNE、dSPM、sLORETA、eLORETA 使用同一条记录、等量 rest/Hold epoch、202 个 grad、同一体积源网格与 13–30 Hz，比较 `10 log10(rest power / Hold power)`。
所有逆算子在两条件间固定。MNE 家族使用 pooled beta 协方差作白化参考，它不是独立空室噪声，故此结果只能用于探索性方法比较。

- [指标及方法状态](beta_multimethod_comparison.csv)
- [逐源 ERD 图数据](beta_multimethod_source_maps.csv)
- [相同 MRI 切面图](beta_multimethod_same_slices_mri.png)

固定方向时 MNE/dSPM/sLORETA 的逐源常数缩放会在功率比中抵消；本次自由方向逆解是否数值相同，以 metadata 的逐点最大差值为准，不能把相同图重复当独立证据。
Dipole fitting、RAP-MUSIC 与旧频谱 OASTER 尚无同目标 beta 功率对比实现，表中保留未运行状态，不能以 ERP/波形定位替代。

单个患者的一条真实记录没有源真值，不计算 AUC/DLE；峰到右感觉运动 ROI 的距离只是解剖合理性检查。各方法独立 P95 显示阈值仅用于看空间形状，不能通过颜色亮度或面积判断胜负。 MRI 是 fsaverage 模板而非患者个体解剖；STN-LFP 未进入 MEG 逆解，不能据此定位 DBS 电极。
