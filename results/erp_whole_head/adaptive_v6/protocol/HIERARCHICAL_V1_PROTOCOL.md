# Hierarchical v1 正式清单

状态：只冻结病例，尚未运行仿真、反演、校准或 validation。

## 三个新面板

- 一级 deep-presence gate 校准：19 个固定纯表层几何跨三个 EEG/MEG SNR，共 57 个独立噪声病例；每档 10 单表层、9 双表层。这是 pooled fixed-design 工程校准，不声称 i.i.d. 总体 conformal 保证。
- 二级 surface-presence 校准：57 个纯深层病例；三个 SNR 各 19 例。病例 ID 和噪声身份唯一，均衡重复 validation 排除后的 7 个深点；其中只有 1 个点相对列明的当前 protocol 未见，另 6 个在历史清单出现过。真值深点禁止进入反演、候选选择或分数。
- untouched validation：每个 SNR 7 个独立几何，共 21 例，跨 SNR 不复用位置。每档含 2 个单表层、2 个双表层、1 个纯深层、1 个深层+单表层、1 个深层+双表层，即 4 个 H0 和 3 个 H1。

SNR 固定为 `(-10,-10)`、`(-10,+20)`、`(+20,-10)` dB，不做参数网格。新 roots 分别为 gate `2026093001/2026093002`、surface `2026093003/2026093004`、validation `2026100101/2026100102`。

## 空间隔离

- Validation 先选并保留 27 个表层中心和 9 个相对当前 protocol 未见的深点；后续两套校准显式排除这些位置。
- Gate 校准再选 28 个唯一表层中心并固定跨 SNR。三个新面板的 55 个表层二阶 patch 内部互斥；每个新中心及其一阶邻域与列明的全部历史 development、calibration、已消费 validation 及旧 `lock.json` 的 272 个中心二阶 patch 零交集。由于当前源空间的历史覆盖过密，新二阶 patch 的最外圈允许与历史二阶 patch 边界重叠，精确数量写入 audit，不声称完整 patch 全历史未见。
- 所有双表层中心相距至少 50 mm，并启用传感器分量平衡；validation 混合病例的深浅传感器幅度比固定为 0.5。
- 二级校准均衡使用 validation 排除后的 7 个深点，其中 1 个相对当前 protocol 未见；validation 使用 9 个唯一未见深点，左右丘脑按 5/4 分配，并留下 1 个未见点给校准。

## 诚实限制

更早的 pre-v6 仿真已使用过当前全部 16 个深层网格点。因此这里的深点只能称为相对列明的当前 development、calibration 和已消费 validation 未见；若要求全项目历史绝对新深点，必须重建深源空间与 forward。

21 例 validation 是 21 个独立几何，每个 SNR 各 7 个；这是一次性工程验收，不证明总体误报率。必须先冻结最终算法和两个校准决策，再一次性消费 validation；不得按 validation 结果修改算法、阈值或病例。
