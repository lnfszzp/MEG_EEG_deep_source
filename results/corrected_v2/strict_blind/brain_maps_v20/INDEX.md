# corrected-v2 源定位图索引

本目录的图已按“仿真真值”和“算法估计”严格分开：所有算法图均不叠加绿色仿真源，仿真位置只出现在 `simulation_truth_*` 文件中。每张图的标题均写明病例号、源组合、EEG SNR 和 MEG SNR。

## 各类仿真

| 情形 | 病例与真值位置 | EEG / MEG SNR | 单独仿真 MRI | 单独仿真皮层 | 八算法 MRI | 八算法皮层 |
|---|---|---:|---|---|---|---|
| 纯表层 | `case_00186`：左侧 bankssts | -10 / -5 dB | [truth MRI](case_00186/simulation_truth_mri.png) | [truth surface](case_00186/simulation_truth_surface.png) | [estimates MRI](case_00186/all_methods_brain_mri.png) | [estimates surface](case_00186/all_methods_brain_surface.png) |
| 纯深层 | `case_01385`：右丘脑 | -5 / -10 dB | [truth MRI](case_01385/simulation_truth_mri.png) | [空皮层说明](case_01385/simulation_truth_surface.png) | [estimates MRI](case_01385/all_methods_brain_mri.png) | [cortical leakage](case_01385/all_methods_brain_surface.png) |
| 深层 + 单表层 | `case_03264`：左侧 insula + 左丘脑 | 0 / +5 dB | [truth MRI](case_03264/simulation_truth_mri.png) | [truth surface](case_03264/simulation_truth_surface.png) | [estimates MRI](case_03264/all_methods_brain_mri.png) | [estimates surface](case_03264/all_methods_brain_surface.png) |
| 深层 + 双表层 | `case_06489`：左侧 lingual + 右侧 caudalanteriorcingulate + 右丘脑 | +10 / +20 dB | [truth MRI](case_06489/simulation_truth_mri.png) | [truth surface](case_06489/simulation_truth_surface.png) | [estimates MRI](case_06489/all_methods_brain_mri.png) | [estimates surface](case_06489/all_methods_brain_surface.png) |
| EEG 弱 / MEG 强的混合源 | `case_01218`：左侧 insula + 左丘脑 | -10 / +20 dB | [truth MRI](case_01218/simulation_truth_mri.png) | [truth surface](case_01218/simulation_truth_surface.png) | [estimates MRI](case_01218/all_methods_brain_mri.png) | [estimates surface](case_01218/all_methods_brain_surface.png) |
| EEG 强 / MEG 弱的混合源 | `case_07977`：左侧 lingual + 右侧 caudalanteriorcingulate + 右丘脑 | +20 / -10 dB | [truth MRI](case_07977/simulation_truth_mri.png) | [truth surface](case_07977/simulation_truth_surface.png) | [estimates MRI](case_07977/all_methods_brain_mri.png) | [estimates surface](case_07977/all_methods_brain_surface.png) |

## 文件含义

- `simulation_truth_mri.png`：单独的完整仿真真值，包含表层和丘脑源能量、patch 与精确中心。
- `simulation_truth_surface.png`：只画真实表层源；丘脑不伪投影到皮层。纯深层病例因此显示空皮层并提示查看 MRI。
- `all_methods_brain_mri.png`、`all_methods_brain_surface.png`：八种算法估计总览，不含仿真真值标记。
- `oaster_v20.png`、`dspm.png` 等及其 `*_surface.png`：单个算法结果，不含仿真真值标记。
