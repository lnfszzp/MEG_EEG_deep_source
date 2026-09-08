# corrected-v2 源定位图索引

本目录的图已按“仿真真值”和“算法估计”严格分开：所有算法图均不叠加仿真真值，`simulation_truth_*` 也只用连续能量热图表达真值，不使用星号、圆圈、绿色 label 或 foci。每张图的标题均写明病例号、源组合、EEG SNR 和 MEG SNR。

算法 MRI 与 surface 默认使用 `cutoff = max(10% 峰值, P95)`；surface 色标由 P95/P97/P99 对应的三个归一化能量值控制，稀疏结果的分位点退化时回退到归一化能量的 10%/55%/100%。truth 图只使用 10% 峰值 cutoff。旧的单独 10% 峰值规则在 6 个代表病例中使 dSPM/eLORETA/sLORETA 的可见皮层顶点比例中位数分别达到 61.46%/69.65%/45.66%，纯深层病例约为 100%/100%/99.67%；全头发亮因此是弥散解与低显示阈值的结果，不是渲染错误。

## 各类仿真

| 情形 | 病例与真值位置 | EEG / MEG SNR | 单独仿真 MRI | 单独仿真皮层 | 八算法 MRI | 八算法皮层 | MRI + surface 联合图 |
|---|---|---:|---|---|---|---|---|
| 纯表层 | `case_00186`：左侧 bankssts | -10 / -5 dB | [truth MRI](case_00186/simulation_truth_mri.png) | [truth surface](case_00186/simulation_truth_surface.png) | [estimates MRI](case_00186/all_methods_brain_mri.png) | [estimates surface](case_00186/all_methods_brain_surface.png) | — |
| 纯深层 | `case_01385`：右丘脑 | -5 / -10 dB | [truth MRI](case_01385/simulation_truth_mri.png) | [空皮层说明](case_01385/simulation_truth_surface.png) | [estimates MRI](case_01385/all_methods_brain_mri.png) | [cortical leakage](case_01385/all_methods_brain_surface.png) | [truth](case_01385/simulation_truth_mri_surface.png) / [8 methods](case_01385/all_methods_brain_combined.png) |
| 深层 + 单表层 | `case_03264`：左侧 insula + 左丘脑 | 0 / +5 dB | [truth MRI](case_03264/simulation_truth_mri.png) | [truth surface](case_03264/simulation_truth_surface.png) | [estimates MRI](case_03264/all_methods_brain_mri.png) | [estimates surface](case_03264/all_methods_brain_surface.png) | [truth](case_03264/simulation_truth_mri_surface.png) / [8 methods](case_03264/all_methods_brain_combined.png) |
| 深层 + 双表层 | `case_06489`：左侧 lingual + 右侧 caudalanteriorcingulate + 右丘脑 | +10 / +20 dB | [truth MRI](case_06489/simulation_truth_mri.png) | [truth surface](case_06489/simulation_truth_surface.png) | [estimates MRI](case_06489/all_methods_brain_mri.png) | [estimates surface](case_06489/all_methods_brain_surface.png) | [truth](case_06489/simulation_truth_mri_surface.png) / [8 methods](case_06489/all_methods_brain_combined.png) |
| EEG 弱 / MEG 强的混合源 | `case_01218`：左侧 insula + 左丘脑 | -10 / +20 dB | [truth MRI](case_01218/simulation_truth_mri.png) | [truth surface](case_01218/simulation_truth_surface.png) | [estimates MRI](case_01218/all_methods_brain_mri.png) | [estimates surface](case_01218/all_methods_brain_surface.png) | [truth](case_01218/simulation_truth_mri_surface.png) / [8 methods](case_01218/all_methods_brain_combined.png) |
| EEG 强 / MEG 弱的混合源 | `case_07977`：左侧 lingual + 右侧 caudalanteriorcingulate + 右丘脑 | +20 / -10 dB | [truth MRI](case_07977/simulation_truth_mri.png) | [truth surface](case_07977/simulation_truth_surface.png) | [estimates MRI](case_07977/all_methods_brain_mri.png) | [estimates surface](case_07977/all_methods_brain_surface.png) | [truth](case_07977/simulation_truth_mri_surface.png) / [8 methods](case_07977/all_methods_brain_combined.png) |

## 文件含义

- `simulation_truth_mri.png`、`simulation_truth_surface.png`：连续真值能量热图；后者只含真实表层分量，丘脑不伪投影到皮层，纯深层病例因此显示空皮层。
- `all_methods_brain_mri.png`、`all_methods_brain_surface.png`：八种算法估计总览，不含仿真真值。
- `oaster_v20.png`、`dspm.png` 等及其 `*_surface.png`：单个算法结果，不含仿真真值。
- `simulation_truth_mri_surface.png`：含深层病例的独立真值联合图。
- `*_mri_surface.png`：含深层病例的单算法联合图；`all_methods_brain_combined.png` 是八算法联合图总览。

联合图左侧解剖 MRI 包含完整的深层与表层源，右侧渲染皮层仅包含表层分量。两面板独立归一化，不能根据左右颜色直接比较幅值。
