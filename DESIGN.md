# SISSES protected multilayer design

目标：在不修改原始 SISSES 代码的前提下，基于 SISSES 输出做一个薄的多层保护后处理。

核心思路：

1. `MEG-only SISSES` 只用于表层保护。MEG 对表层更稳定，先从它的表层高能候选里保留连通、能量足够的 patch，避免 EEG+MEG 把一个表层源抹成过大的范围。
2. `EEG+MEG SISSES` 作为主重建。它保留联合信息，并提供深层候选。
3. 深层候选只从 `EEG+MEG SISSES` 的深部点中取少量高能点，避免深部图像显示成一整团体积。
4. 最终候选集合为：`MEG 表层保护候选 + EEG+MEG 深层候选`。
5. 在最终候选集合上，用 EEG+MEG 白化后的前向模型做一次 ridge 最小二乘波形重拟合，用于改善源波形和 RMSE。

评价原则：

- 指标始终按全头源点计算，不只算单一 ROI。
- 指标只用于结果评价，不参与参数选择或定位优化。
- 对比方法包括 `SISSES_EEG_MEG_direct`、`SISSES_protected_multilayer`、`SISSES_MEG_only`、`SISSES_EEG_only`。

需要的输入：

- `generated/<scenario>/sub_EEG.mat`
- `generated/<scenario>/sub_MEG.mat`
- `generated/<scenario>/s_true.mat`

输出：

- `sisses_protected_multilayer/results/sisses_runs/...`: 单模态和联合 SISSES 结果。
- `sisses_protected_multilayer/results/protected/...`: protected multilayer 源结果。
- `sisses_protected_multilayer/results/metrics.csv`: 全头指标。
- `sisses_protected_multilayer/results/figures/...`: 定位图和波形图。
