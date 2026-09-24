# v5 深浅共线惩罚 + 多轮 MM 开发探针

本探针复用已经冻结的 `development_localization_v5_mm_floor_probe.json` 四病例，只做有标签根因验证，不用于正式校准或 validation。

- 唯一新增规则：对第 `j` 个深层列计算其与所有皮层列的最大绝对余弦 `rho_j`，令深层惩罚乘以 `1 / sqrt(max(1-rho_j^2, machine_epsilon))`。
- 计算坐标：训练数据决定的 EEG/MEG 权重、深度归一化和 MRF innovation design；不读取源真值。
- 其余固定设置：ADMM、MRF `0.8`、outer `20`、inner retries `20`、source/deep/edge floor 均 `0.5`、`epsilon_fraction=0.05`、`tolerance=0.001`、`outer_tolerance=0.01`。
- 不运行倍率或阈值网格。
- 主条件：`max(T00, T10) < min(T02, T04)`，且八个 H0/H1 拟合都达到多轮固定点。
- 早停：若首例 case 00 的 H1 未收敛，或其 excess 分数不低于旧值 `1.444032`，立即否决，不继续其余病例。
