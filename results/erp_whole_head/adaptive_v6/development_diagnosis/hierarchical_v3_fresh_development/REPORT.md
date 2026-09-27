# v3 全新开发结果

本目录仅是 fresh development，不是正式校准或 blind validation。

- gate statistic：`min(partial_A_to_B, partial_B_to_A)`。
- 阈值：6个预声明 calibration H0 的最大值 `0.252505832`；严格 `>` 才报H1。
- calibration H0误报：0/6；audit H0误报：1/6。
- H1 family检出：13/27；同时满足10 mm深源定位：10/27。
- 全局An_auc最小/中位：0.001/0.888。
- AUC使用78冻结的fine→coarse映射；SD、DLE和深源距离直接使用真实fine HEAD坐标。
