# 表层能量几何共识条件定位开发诊断

这不是盲检测结果：预先给定正确模型家族，case00/01 使用 H0，case02/03/04 使用 H1。表层只使用一个固定的无权重对称共识；深层 H1 完全保留原 OASTER，未做 3NN 扩散。

| 病例 | 家族 | 方法 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) |
|---|---|---|---:|---:|---:|---:|---:|
| erp-v6-development-00 | H0_surface_only | selected_OASTER_baseline | 0.9417 | 0.9908 | 0.9908/NA | 14.0588/NA | 31.5307/NA |
| erp-v6-development-00 | H0_surface_only | conditional_geometric_consensus | 0.9710 | 0.9922 | 0.9921/NA | 11.7826/NA | 9.5826/NA |
| erp-v6-development-01 | H0_surface_only | selected_OASTER_baseline | 0.7167 | 0.9286 | 0.9284/NA | 6.5847/NA | NA/NA |
| erp-v6-development-01 | H0_surface_only | conditional_geometric_consensus | 0.7088 | 0.9359 | 0.9358/NA | 7.0634/NA | NA/NA |
| erp-v6-development-02 | H1_surface_deep | selected_OASTER_baseline | 1.0000 | 0.9997 | NA/0.8667 | NA/11.5682 | NA/10.0000 |
| erp-v6-development-02 | H1_surface_deep | conditional_geometric_consensus | 1.0000 | 0.9997 | NA/0.8667 | NA/11.5682 | NA/10.0000 |
| erp-v6-development-03 | H1_surface_deep | selected_OASTER_baseline | 0.7382 | 0.9691 | 0.9674/1.0000 | 8.1388/27.6073 | 11.8224/0.0000 |
| erp-v6-development-03 | H1_surface_deep | conditional_geometric_consensus | 0.8228 | 0.9705 | 0.9690/1.0000 | 8.3309/27.6073 | 11.8224/0.0000 |
| erp-v6-development-04 | H1_surface_deep | selected_OASTER_baseline | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 |
| erp-v6-development-04 | H1_surface_deep | conditional_geometric_consensus | 0.8197 | 0.7234 | 0.7139/1.0000 | 7.4410/21.4847 | NA/0.0000 |

## 固定验收

**REJECT**。要求五例 An_cal AUC 均不低于 0.90；每个有真值的层 DLE 必须有限，且相对原 OASTER 不恶化超过 5.0 mm。

- 五例 AUC 门槛：未通过
- 分层 DLE 门槛：未通过

- erp-v6-development-00-eeg-10-meg-10 surface: baseline=31.530702327656556, diagnostic=9.582599556323082, worsening=-21.948102771333474, passed=True
- erp-v6-development-01-eeg-10-meg-10 surface: baseline=None, diagnostic=None, worsening=None, passed=False
- erp-v6-development-02-eeg-10-meg-10 deep: baseline=10.00000054854849, diagnostic=10.00000054854849, worsening=0.0, passed=True
- erp-v6-development-03-eeg-10-meg-10 surface: baseline=11.822374003563763, diagnostic=11.822374003563763, worsening=0.0, passed=True
- erp-v6-development-03-eeg-10-meg-10 deep: baseline=0.0, diagnostic=0.0, worsening=0.0, passed=True
- erp-v6-development-04-eeg-10-meg-10 surface: baseline=None, diagnostic=None, worsening=None, passed=False
- erp-v6-development-04-eeg-10-meg-10 deep: baseline=0.0, diagnostic=0.0, worsening=0.0, passed=True

算法未读取 calibration/validation、未搜索候选或权重；无论结果成败，本诊断到此停止。
