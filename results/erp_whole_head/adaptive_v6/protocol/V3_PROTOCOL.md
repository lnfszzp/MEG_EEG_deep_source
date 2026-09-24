# ERP-v6 component-balanced v3 独立校准协议

状态：`algorithm_not_yet_frozen`。本生成器只冻结病例和决策规则，不运行校准、validation 或反演。

## 病例与几何

- 57 个纯表层病例对应 57 个不同空间配置；每个配置只分配一个 SNR。
- EEG/MEG SNR 为 `(-10,-10)`、`(-10,20)`、`(20,-10)` dB，每层 19 例：10 单表层、9 双表层。
- 27 个双表层病例均启用 sensor balance；双源中心距离至少 50 mm。
- 84 个源中心左右半球各 42 个。全部二阶邻接 patch 在 v3 内互不相交，且与历史开发、已消费 v2 校准和受保护 v2 validation 的 patch 交集为 0。
- fit/check roots 为 `2026092501/2026092502`；几何和 SNR 分配 seed 由四个冻结输入哈希分域派生，不由结果挑选。

## 一个 pooled 决策

`p(T)=(1+count(T_cal>=T))/58`。只有 `T>0` 且 `p<=0.05` 才报深源；tie 保守计数。这等价于 T 严格超过 57 个校准分数的第二大值，完全不使用测试真实 SNR。

该规则只是三个 SNR 等权混合分布下的边际工程 FPR 校准。几何受保护隔离且共享同一前向模型，不宣称严格 iid/conformal 或每 SNR 总体保证。最终 validation 仍必须每个 SNR 对 4 个 H0 均 `0/4` 假阳性。

## v2 退役与一次性消费

v2 校准虽完成 57 例，但 6 个 full model 未收敛，因此其分数禁止用于正式阈值，只能用于求解器诊断。原 marker、执行锁和结果保留不变。

v3 校准使用独立 marker `calibration_component_balanced_v3_manifest_consumed.json`，不被旧的 `calibration_manifest_consumed.json` 阻断。受保护 validation 继续直接引用原 v2 manifest SHA256 `68a405ace52d0c04a193c37198304682a0440a0007fed0433fe3a43e686d9aa7`，且仍由全局 `validation_manifest_consumed.json` 保证只消费一次。

下一步必须先完成算法修改和开发验证，再生成不可覆盖的 v3 execution lock。任一 v3 校准分数缺失、非有限或求解器未收敛，整个 v3 校准作废，不得删除病例后继续。
