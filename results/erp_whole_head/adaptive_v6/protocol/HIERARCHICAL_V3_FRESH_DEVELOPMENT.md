# v3 全新开发面板

状态：只冻结全新开发病例，不是正式校准或 blind validation，也不含性能结果。

- 13 套固定几何原样跨三个 `(EEG, MEG) SNR`：`(-10,-10)`、`(-10,+20)`、`(+20,-10)`，共 39 例；每档固定 4 个 H0 与 9 个 H1。
- H0 为纯表层单源 2 套、双源 2 套；单/双各 1 套用于 gate calibration，另各 1 套只用于 gate audit。
- H1 为纯深层、深层+单表层、深层+双表层各 3 套；每种情形分别覆盖 low/mid/high forward-alias。
- truth forward 固定为 oct7 表层 + 5 mm 双侧丘脑，缓存 SHA256 为 `f7417a1bbabd6ac863da8811470c5c273be3f950b028fff5efae9ac9a899b266`；inverse 仍是 coarse oct6/16点深网格，指纹为 `a97e6991fde1cedfac66857d5c07683c43af62f16ad19bfa31ea7199a9083556`。
- 15 个表层 patch 两两不重叠，中心距所有历史中心至少 `10.308` mm；5 个双源对均至少 50 mm。truth 到最近 inverse 表层点固定为 2–5 mm。
- 9 个深层 truth 均距最近 inverse 深点 3–8 mm，truth 点和最近 inverse 点都不复用；左右丘脑为 `Counter({10: 5, 49: 4})`。
- alias 只由 forward 与固定 noise covariance whitening 计算；候选按 rho 排序后三等分，再在 tier、左右标签和最近 coarse 点唯一约束下匹配。
- fit/check roots 固定为 `2026100301/2026100302`，与全部 `30` 个历史病例清单的 ID、seed 和 source positions 均无复用。
