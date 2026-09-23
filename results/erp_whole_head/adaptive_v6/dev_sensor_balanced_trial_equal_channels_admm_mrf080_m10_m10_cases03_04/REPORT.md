# 白化后逐通道等权重开发消融

只使用 sensor-balanced 开发病例 03/04；未读取 calibration 或 validation。原权重解读取已完成 trial-covariance 结果，全 1 权重解复用同一份 `prepare_trial_covariance_case` 输出，唯一变化是 `channel_weights`。

| 病例 | 权重 | 假设 | 收敛 H0/H1 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) | prediction | partial |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| erp-v6-development-03 | training_evidence | H0_surface_only | 1/1 | 0.4857 | 0.9298 | 0.9674/0.5000 | 8.1386/NA | 11.8224/NA | -0.0599 | 0.3444 |
| erp-v6-development-03 | training_evidence | H1_surface_deep | 1/1 | 0.7382 | 0.9691 | 0.9674/1.0000 | 8.1388/27.6073 | 11.8224/0.0000 | -0.0599 | 0.3444 |
| erp-v6-development-03 | whitened_channel_equal | H0_surface_only | 1/1 | 0.5517 | 0.9501 | 0.9889/0.5000 | 10.2409/NA | 14.8885/NA | -0.0132 | 0.4715 |
| erp-v6-development-03 | whitened_channel_equal | H1_surface_deep | 1/1 | 0.8512 | 0.9960 | 0.9961/0.9333 | 7.2814/18.1204 | 9.9033/14.1421 | -0.0132 | 0.4715 |
| erp-v6-development-04 | training_evidence | H0_surface_only | 1/1 | 0.7230 | 0.9541 | 0.9869/0.5000 | 7.2814/NA | NA/NA | -0.0027 | 0.8067 |
| erp-v6-development-04 | training_evidence | H1_surface_deep | 1/1 | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 | -0.0027 | 0.8067 |
| erp-v6-development-04 | whitened_channel_equal | H0_surface_only | 1/1 | 0.5511 | 0.8395 | 0.8682/0.5000 | 7.2814/NA | NA/NA | 0.0044 | 0.0205 |
| erp-v6-development-04 | whitened_channel_equal | H1_surface_deep | 1/1 | 0.7329 | 0.7229 | 0.7136/0.7333 | 7.2814/26.0612 | NA/14.1421 | 0.0044 | 0.0205 |

## 结论

**REJECT**。预先固定的明显改善标准为：两例 H1 的 An_cal AUC 均提高至少 0.02，且等权重 H0/H1 全部收敛。

- erp-v6-development-03-eeg-10-meg-10: An_cal AUC +0.1130，An_auc +0.0270
- erp-v6-development-04-eeg-10-meg-10: An_cal AUC +0.0000，An_auc -0.0002

该结论仅是开发消融，不是验证结果；confirmation 分数未校准，不能解释为假阳性率。
