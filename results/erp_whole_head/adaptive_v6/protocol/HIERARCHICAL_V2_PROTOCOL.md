# Hierarchical v2 正式清单

状态：只冻结两套新校准病例；未运行仿真、逆解、校准或 validation。

## 冻结输入和分数

- 71 号清单、计划中的 72 号正式 runner、70 号盲可靠性选择 summary、predictive helper、共享 forward 指纹和所有列明历史清单都以 SHA256 绑定。
- 一级门控固定为 70 号预选的 `adaptive_balance_p1`：`h_m=g_m/sqrt(1+q_m)`，`b=min(1+q)/max(1+q)`，`w_EEG=0.5+0.25*b`，最终为 `w_EEG*h_EEG+(1-w_EEG)*h_MEG`。逐白化后的 primary H0/H1 拟合与门控固定 `W=1`；deep LOO 候选选择和深源幅度估计使用训练阶段得到的原 observation weights；conditional surface 的 train20 与 combined40 均固定 `W=1`。
- 这些开发选择不构成独立性能结果；正式功效只能由全新校准和一次性 untouched validation 检验。

## 两套新校准

- Gate：19 个固定纯表层配置跨 `(-10,-10)`、`(-10,+20)`、`(+20,-10)` 三个 EEG/MEG SNR，共 57 例；每档 10 单源、9 双源。roots 为 `2026100201/2026100202`。
- Surface：19 个固定纯深层配置槽跨三个 SNR，共 57 例，只均衡复用 validation 未占的 7 个 deep_local `[0, 1, 5, 7, 9, 10, 13]`。roots 为 `2026100203/2026100204`；case ID 和噪声身份全新。

## 空间隔离和诚实边界

- Gate 共 28 个唯一中心。中心落入任一列明历史 patch support 的数量为 0；新 gate patch 内部重叠为 0；与受保护 validation patch 重叠为 0；所有双源中心距离至少 `77.394498` mm。
- Gate patch 允许与旧非 validation 历史 patch 的最外圈重叠，本次精确重叠顶点数为 `354`。因此不声称完整 patch 相对全历史绝对未见。
- 当前 16 点深网格已被历史耗尽。Surface 校准的 7 点只保证完全排除 validation 使用的 9 点，不声称相对全项目历史为新位置。

## Untouched validation 只读引用

- 继续引用原 `hierarchical_v1_validation_manifest.json`，SHA256 固定为 `6ef97341434bca99bc7c75d205c8dc6d3815b3260ca3c62c10c615ce2efb0eff`；71 号不复制、不改写、也不生成第二份 validation manifest。
- 原 validation roots 必须保持 `2026100101/2026100102`。共享 consumed marker、v1/v2 validation 输出和对应 run-lock 必须全部不存在，否则本脚本立即拒绝生成或核对。
- 21 例 validation 是一次性固定工程面板，不证明总体或临床误报率。
