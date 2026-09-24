# ERP-v6 component-balanced v4 独立校准协议

状态：`algorithm_not_yet_frozen`。本生成器只冻结病例和 pooled 决策规则，不运行校准、validation 或反演。

## 病例与几何

- 57 个纯表层病例对应 57 个不同空间配置；每个配置只分配一个 SNR。
- EEG/MEG SNR 为 `(-10,-10)`、`(-10,20)`、`(20,-10)` dB，每层 19 例：10 单表层、9 双表层。
- 每层模板固定为 5 S-L、5 S-R、3 D-LL、3 D-RR、3 D-LR；27 个双表层病例均启用 sensor balance，双源中心距离至少 50 mm。
- 84 个源中心左右半球各 42 个。新中心不落入任何历史二阶 patch；v4 patch 在内部互不相交，并与当前 development、已消费 v2/v3 校准、受保护 validation 的 patch 交集为 0。
- 由于全隔离候选不足，v4 只允许 patch 边缘与更早的历史支持重叠：重叠点 519 个；新中心本身仍全部在历史支持外。该放宽在运行前固定，不能按分数选择。
- fit/check roots 为 `2026092601/2026092602`；几何/SNR seeds 为 `2176220568/577955680`，由七个冻结输入哈希分域派生。

## 一个 pooled 决策

`p(T)=(1+count(T_cal>=T))/58`。只有 `T>0` 且 `p<=0.05` 才报深源；tie 保守计数。这等价于 T 严格超过 57 个校准分数的第二大值，完全不使用测试真实 SNR。

该规则只是三个 SNR 等权混合分布下的边际工程 FPR 校准，不宣称严格 iid/conformal 或每 SNR 总体保证。最终 validation 仍必须每个 SNR 对 4 个 H0 均 `0/4` 假阳性。

## v3 退役与下一步

v3 校准完成 57 例，但仅 56 例全部模型收敛；病例 `018` 的 H1/full model 未收敛。因此 v3 全部分数禁止用于正式阈值，只能用于求解器诊断，原结果和一次性 marker 保留不变。

v4 计划使用 ADMM、MRF 0.8、1 次 outer、最多 20 次 inner retry；这只是预登记计划，尚未形成 execution lock。下一步必须冻结包含最终代码哈希的 v4 execution lock，才能一次性消费 `calibration_component_balanced_v4_manifest_consumed.json`。任一分数缺失、非有限或求解器未收敛，整个 v4 校准作废，不得删除病例后继续。

受保护 validation 继续直接引用原 v2 manifest SHA256 `68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7`；生成器仅读取该 manifest，不读取 validation 结果，并在开始和结束都要求全局 `validation_manifest_consumed.json` 不存在。
