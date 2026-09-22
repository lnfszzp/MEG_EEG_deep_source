# 非高斯形状补充验证

已完成4种形状 × 3组EEG/MEG SNR（−10/−10、5/5、−10/20）=12病例，对比v3/v4。v4使用与主复测相同的固定参数：MRF=.5、edge=.5、noise=1。真值为均匀幅度的连通图区域，不再是高斯片。

| 形状 | 分区 | 真值表层点数 |
|---|---|---:|
| 1-hop小片 | postcentral-lh | 7 |
| 3-hop大片 | superiorfrontal-rh | 37 |
| 细长路径 | lateraloccipital-lh | 10 |
| 宽路径+丘脑 | superiorparietal-rh | 37 |

| 12例描述均值 | v3 | v4 |
|---|---:|---:|
| 总体 An_auc | 0.9559 | 0.9667 |
| 表层支持 Dice | 0.3088 | 0.2519 |

**总体AUC略升，但形状重叠Dice下降，不能宣称已经恢复真实活动区形状。** 例如5/5 dB细长片，v4总体AUC=0.9864，而以层内峰能量10%为显示/支持阈值时Dice=0。该例很好地说明高AUC并不保证活动范围画得正确。

v4的12例中7例达到当前数值停止标准，另5例保留未收敛诊断。每例真值、完整有符号源时程、各方法估计保存在对应NPZ；完整参数、随机种子、代码哈希和迭代历史在metadata.json。

`shape_top_projection_5_5.png` 是源网格坐标的诊断性俯视投影，真值与算法分列，**不是MNE皮层解剖渲染**；深源不投影到皮层表面，数据保留在NPZ与指标表。这张图不作为论文最终脑图。

12个NPZ合计266,222,428字节，保存在本机。GitHub保存代码、指标、诊断与PNG，不上传这些可重建的大数组。

复现命令（meg环境，BLAS/OMP线程为1）：

```powershell
python run_adaptive_shape_challenge.py --output results/erp_whole_head/adaptive_v4/shape_mrf_final --mrf-strength 0.5
```

这是4个位置的补充挑战，不是全头位置覆盖验证。主五SNR八方法结果在相邻 `pilot_five_snr_all_methods` 目录。
