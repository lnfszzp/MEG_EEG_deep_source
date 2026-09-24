# v5 EEG/MEG 合取证据 + 稳定多轮 MM 开发探针

本探针复用已冻结的 `development_localization_v5_mm_floor_probe.json` 四病例，只开发深层 presence 证据，不用于正式校准或 validation。

- H0/H1 仍只用 training-20 拟合；confirmation-20 不重拟合。
- 在同一个固定 10 维平滑 ERP 基中，对白化后、训练模态加权前的 EEG 与 MEG 分别计算
  `T_m = (loss_H0,m - loss_H1,m) / expected_baseline_noise_m`。
- 唯一新分数是 `T_and = min(T_EEG, T_MEG)`；这是无倍率的 intersection-union 证据，要求同一个训练源解能同时改善两种模态。
- 评分必须显式使用 metadata 中的 EEG/MEG retained channel counts 分块，不能从训练权重猜分块；不读取真值、位置或 SNR。
- 求解设置保持 stable bounded-MM：ADMM、MRF `0.8`、outer `20`、inner retries `20`、source/deep/edge floor 均 `0.5`，不启用 alias penalty。
- 数值条件：八个 H0/H1 拟合均达到多轮固定点；科学条件仍为 `max(T00,T10) < min(T02,T04)`。
- 不从开发病例取中点或阈值。若探针通过，必须用全新纯表层病例做上尾 conformal 校准，并继续要求分数大于零。

已知信息边界：该规则由旧 development 面板和本轮 case 00/02 诊断提出，所以整个四病例探针仍只是开发证据，不能充当独立验证。
