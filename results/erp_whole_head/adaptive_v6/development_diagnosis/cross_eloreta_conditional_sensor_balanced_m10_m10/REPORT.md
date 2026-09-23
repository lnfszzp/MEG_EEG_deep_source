# 交叉 eLORETA 条件定位开发诊断

这不是盲检测结果：预先给定正确模型家族，case00/01 使用 H0，case02/03/04 使用 H1。算法没有读取 calibration/validation，没有候选网格，也没有按真值选择融合权重。

| 病例 | 家族 | 方法 | An_cal AUC | An_auc | surface/deep AUC | surface/deep SD (mm) | surface/deep DLE (mm) |
|---|---|---|---:|---:|---:|---:|---:|
| erp-v6-development-00 | H0_surface_only | selected_OASTER_baseline | 0.9417 | 0.9908 | 0.9908/NA | 14.0588/NA | 31.5307/NA |
| erp-v6-development-00 | H0_surface_only | conditional_cross_eLORETA | 0.9556 | 0.9907 | 0.9907/NA | 13.0609/NA | 7.7551/NA |
| erp-v6-development-01 | H0_surface_only | selected_OASTER_baseline | 0.7167 | 0.9286 | 0.9284/NA | 6.5847/NA | NA/NA |
| erp-v6-development-01 | H0_surface_only | conditional_cross_eLORETA | 0.9638 | 0.9829 | 0.9829/NA | 28.7769/NA | 37.4107/NA |
| erp-v6-development-02 | H1_surface_deep | selected_OASTER_baseline | 1.0000 | 0.9997 | NA/0.8667 | NA/11.5682 | NA/10.0000 |
| erp-v6-development-02 | H1_surface_deep | conditional_cross_eLORETA | 1.0000 | 0.9997 | NA/0.8667 | NA/11.6552 | NA/14.1421 |
| erp-v6-development-03 | H1_surface_deep | selected_OASTER_baseline | 0.7382 | 0.9691 | 0.9674/1.0000 | 8.1388/27.6073 | 11.8224/0.0000 |
| erp-v6-development-03 | H1_surface_deep | conditional_cross_eLORETA | 0.8997 | 0.8435 | 0.8365/1.0000 | 45.1636/27.2203 | 27.2825/0.0000 |
| erp-v6-development-04 | H1_surface_deep | selected_OASTER_baseline | 0.7329 | 0.7231 | 0.7136/1.0000 | 7.2814/21.4847 | NA/0.0000 |
| erp-v6-development-04 | H1_surface_deep | conditional_cross_eLORETA | 0.9819 | 0.9663 | 0.9671/0.9333 | 48.0503/22.8885 | 38.6399/10.0000 |

## 固定验收

**REJECT**。要求五例 An_cal AUC 均不低于 0.90；每个有真值的层 DLE 必须有限，且相对原 OASTER 不恶化超过 5.0 mm。

- 五例 AUC 门槛：未通过
- 分层 DLE 门槛：未通过

- erp-v6-development-00-eeg-10-meg-10 surface: baseline=31.530702327656556, diagnostic=7.755141529980423, worsening=-23.775560797676132, passed=True
- erp-v6-development-01-eeg-10-meg-10 surface: baseline=None, diagnostic=37.41071561828437, worsening=None, passed=True
- erp-v6-development-02-eeg-10-meg-10 deep: baseline=10.00000054854849, diagnostic=14.142136031724077, worsening=4.142135483175586, passed=True
- erp-v6-development-03-eeg-10-meg-10 surface: baseline=11.822374003563763, diagnostic=27.28245918731176, worsening=15.460085183747996, passed=False
- erp-v6-development-03-eeg-10-meg-10 deep: baseline=0.0, diagnostic=0.0, worsening=0.0, passed=True
- erp-v6-development-04-eeg-10-meg-10 surface: baseline=None, diagnostic=38.639858768420176, worsening=None, passed=True
- erp-v6-development-04-eeg-10-meg-10 deep: baseline=0.0, diagnostic=10.00000054854849, worsening=10.00000054854849, passed=False

即使通过，这也只说明给定正确 H0/H1 家族后的条件定位能力，不能作为深源盲检成功率或假阳性率。
