# ERP 四种情形脑空间定位图

固定条件：EEG SNR = 5 dB，MEG SNR = 5 dB；代表配置在脚本参数区固定。

显示阈值：先保留全源空间峰值的 10% 以上，再用第 95 百分位设置色阶。

仿真真值与算法结果严格分开。含深层源的情形同时提供 MRI 解剖图和完整 dorsal 俯视皮层图。

## surface_only

case_id：`erp-v3-confirmation-08946-surface_only-eeg+05-meg+05`；location：`90`。
configuration_number：`90`；surface_centers：`[4125]`；deep_index：`None`。

- 仿真真值 MRI：[case_08946/simulation_truth_mri.png](case_08946/simulation_truth_mri.png)
- 仿真真值俯视皮层：[case_08946/simulation_truth_surface_top.png](case_08946/simulation_truth_surface_top.png)
- 八方法 MRI 对比：[case_08946/all_algorithms_mri.png](case_08946/all_algorithms_mri.png)
- 八方法俯视皮层对比：[case_08946/all_algorithms_surface_top.png](case_08946/all_algorithms_surface_top.png)

## deep_only

case_id：`erp-v3-confirmation-08992-deep_only-eeg+05-meg+05`；location：`1`。
configuration_number：`136`；surface_centers：`[]`；deep_index：`7499`。

- 仿真真值 MRI：[case_08992/simulation_truth_mri.png](case_08992/simulation_truth_mri.png)
- 仿真真值俯视皮层：[case_08992/simulation_truth_surface_top.png](case_08992/simulation_truth_surface_top.png)
- 八方法 MRI 对比：[case_08992/all_algorithms_mri.png](case_08992/all_algorithms_mri.png)
- 八方法俯视皮层对比：[case_08992/all_algorithms_surface_top.png](case_08992/all_algorithms_surface_top.png)
- 仿真真值 MRI + 俯视皮层：[case_08992/simulation_truth_mri_surface.png](case_08992/simulation_truth_mri_surface.png)
- 八方法 MRI + 俯视皮层：[case_08992/all_algorithms_mri_surface.png](case_08992/all_algorithms_mri_surface.png)

## deep_plus_surface

case_id：`erp-v3-confirmation-09112-deep_plus_surface-eeg+05-meg+05`；location：`90`。
configuration_number：`256`；surface_centers：`[4125]`；deep_index：`7499`。

- 仿真真值 MRI：[case_09112/simulation_truth_mri.png](case_09112/simulation_truth_mri.png)
- 仿真真值俯视皮层：[case_09112/simulation_truth_surface_top.png](case_09112/simulation_truth_surface_top.png)
- 八方法 MRI 对比：[case_09112/all_algorithms_mri.png](case_09112/all_algorithms_mri.png)
- 八方法俯视皮层对比：[case_09112/all_algorithms_surface_top.png](case_09112/all_algorithms_surface_top.png)
- 仿真真值 MRI + 俯视皮层：[case_09112/simulation_truth_mri_surface.png](case_09112/simulation_truth_mri_surface.png)
- 八方法 MRI + 俯视皮层：[case_09112/all_algorithms_mri_surface.png](case_09112/all_algorithms_mri_surface.png)

## deep_plus_two_surface

case_id：`erp-v3-confirmation-09207-deep_plus_two_surface-eeg+05-meg+05`；location：`50`。
configuration_number：`352`；surface_centers：`[626, 6435]`；deep_index：`7499`。

- 仿真真值 MRI：[case_09207/simulation_truth_mri.png](case_09207/simulation_truth_mri.png)
- 仿真真值俯视皮层：[case_09207/simulation_truth_surface_top.png](case_09207/simulation_truth_surface_top.png)
- 八方法 MRI 对比：[case_09207/all_algorithms_mri.png](case_09207/all_algorithms_mri.png)
- 八方法俯视皮层对比：[case_09207/all_algorithms_surface_top.png](case_09207/all_algorithms_surface_top.png)
- 仿真真值 MRI + 俯视皮层：[case_09207/simulation_truth_mri_surface.png](case_09207/simulation_truth_mri_surface.png)
- 八方法 MRI + 俯视皮层：[case_09207/all_algorithms_mri_surface.png](case_09207/all_algorithms_mri_surface.png)
