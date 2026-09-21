# ds004998 DBS-MEG：DICS 与 LCMV 探索性 beta-ERD 对照

- 数据：sub-0cGdk9 / HoldL / MedOn / run-1；本地只有这一条完整可分析记录。
- 共同输入：238 个 rest 和 238 个 Hold epoch，202 个 planar grad，4231 个个体体积源点。
- DICS：13–30 Hz multitaper CSD；LCMV：13–30 Hz 带通协方差。两者均由合并条件建立共同滤波器，再计算 rest/Hold 的功率比。
- 峰位与 ROI 数值见 `dics_lcmv_beta_method_comparison.csv`；相同模板 MRI 切面图见 `dics_lcmv_beta_same_slices_mri.png`。
- 两方法逐源数据分别见 `dics_beta_source_table.csv` 和 `lcmv_beta_source_table.csv`。
- DICS 峰 MNI [46, -16, 56] mm，距右感觉运动 ROI 中心 15.62 mm；LCMV 峰 [-34, -16, 64] mm，距中心 74.32 mm。
- ROI laterality index：DICS +0.0412；LCMV +0.0256；两者均有明显双侧 ERD。
- 图中的 P95 是各方法独立的显示阈值，不可凭面积/颜色亮度比较绝对源功率。

这里只比较一条真实记录的 beta 去同步空间模式；没有仿真真值，不能计算 AUC/DLE，也不能以峰到预注册感觉运动 ROI 的距离冒充定位误差。
STN-LFP 被排除在 MEG 逆解外，且没有 DBS 触点坐标；这些结果不证明 STN 或 DBS 电极定位。
fsaverage MRI/pial 仅供显示，不是患者个体解剖。
MNE 一层 BEM 桥接不等于原 FieldTrip Nolte 单壳数值解。

- 总耗时：148.7 秒。
