# %%
# 画独立确认集里四种情形的脑空间定位图。
# 本文件不定义函数：上面改参数，下面按顺序运行即可。
# 仿真真值单独成图，不和算法结果叠加；含深层源时同时输出 MRI 解剖图和完整俯视皮层图。

from pathlib import Path
import json
import sys


# %%
# ==================== 1. 参数区：平时主要改这里 ====================

project_root = Path(r"D:\博士\工作＆汇报\源定位\新建文件夹")
manifest_path = project_root / "results" / "erp_whole_head" / "confirmation_v3" / "manifest.json"
geometry_root = project_root / "corrected_v2" / "generated"
sample_data_path = Path(r"D:\mne_data\MNE-sample-data")
save_root = project_root / "results" / "erp_whole_head" / "confirmation_v3" / "brain_maps_snr_5_5"

# 四种情形使用同一 EEG/MEG SNR。
# 代表配置固定在下面：纯表层和“深层+单表层”共享一个俯视可见的高位皮层点；
# 三个含深层场景共享同一丘脑点；双表层配置的两个皮层点也从俯视可见。
eeg_snr_db = 5
meg_snr_db = 5
relative_threshold = 0.10
display_percentile = 95.0
dpi = 160

scenarios = (
    "surface_only",
    "deep_only",
    "deep_plus_surface",
    "deep_plus_two_surface",
)
representative_configuration_numbers = {
    "surface_only": 90,
    "deep_only": 136,
    "deep_plus_surface": 256,
    "deep_plus_two_surface": 352,
}

assert manifest_path.is_file(), f"找不到 manifest：{manifest_path}"
assert geometry_root.is_dir(), f"找不到几何和 forward：{geometry_root}"
assert sample_data_path.is_dir(), f"找不到 MNE sample 数据：{sample_data_path}"

manifest_cases = json.loads(manifest_path.read_text(encoding="utf-8"))
selected_cases = {}
for scenario in scenarios:
    candidates = [
        case
        for case in manifest_cases
        if int(case["eeg_snr_db"]) == eeg_snr_db
        and int(case["meg_snr_db"]) == meg_snr_db
        and case["scenario"] == scenario
        and int(case["configuration_number"])
        == representative_configuration_numbers[scenario]
    ]
    assert len(candidates) == 1, (scenario, len(candidates))
    selected_cases[scenario] = candidates[0]

assert selected_cases["surface_only"]["surface_centers"] == selected_cases[
    "deep_plus_surface"
]["surface_centers"]
assert selected_cases["deep_only"]["deep_index"] == selected_cases[
    "deep_plus_surface"
]["deep_index"]
assert selected_cases["deep_only"]["deep_index"] == selected_cases[
    "deep_plus_two_surface"
]["deep_index"]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import plot_erp_brain_maps as brain_plot


# %%
# ==================== 2. 依次画四种情形 ====================

case_directories = []

for scenario in scenarios:
    selected_case = selected_cases[scenario]
    print("正在画：", scenario, "EEG SNR =", eeg_snr_db, "MEG SNR =", meg_snr_db)
    case_directory = brain_plot.plot_erp_brain_maps(
        manifest_path=manifest_path,
        data_root=geometry_root,
        sample_path=sample_data_path,
        output_root=save_root,
        case_id=selected_case["case_id"],
        relative_threshold=relative_threshold,
        display_percentile=display_percentile,
        dpi=dpi,
    )
    case_directories.append(case_directory)


# %%
# ==================== 3. 检查输出并写索引 ====================

index_lines = [
    "# ERP 四种情形脑空间定位图",
    "",
    f"固定条件：EEG SNR = {eeg_snr_db} dB，MEG SNR = {meg_snr_db} dB；代表配置在脚本参数区固定。",
    "",
    f"显示阈值：先保留全源空间峰值的 {relative_threshold:.0%} 以上，再用第 {display_percentile:g} 百分位设置色阶。",
    "",
    "仿真真值与算法结果严格分开。含深层源的情形同时提供 MRI 解剖图和完整 dorsal 俯视皮层图。",
    "",
]

for scenario, case_directory in zip(scenarios, case_directories, strict=True):
    selected_case = selected_cases[scenario]
    assert (case_directory / "simulation_truth_mri.png").is_file()
    assert (case_directory / "simulation_truth_surface_top.png").is_file()
    assert (case_directory / "all_algorithms_mri.png").is_file()
    assert (case_directory / "all_algorithms_surface_top.png").is_file()
    if scenario != "surface_only":
        assert (case_directory / "simulation_truth_mri_surface.png").is_file()
        assert (case_directory / "all_algorithms_mri_surface.png").is_file()

    relative_case = case_directory.relative_to(save_root).as_posix()
    index_lines.extend(
        (
            f"## {scenario}",
            "",
            f"case_id：`{selected_case['case_id']}`；location：`{selected_case['location']}`。",
            f"configuration_number：`{selected_case['configuration_number']}`；surface_centers：`{selected_case['surface_centers']}`；deep_index：`{selected_case.get('deep_index')}`。",
            "",
            f"- 仿真真值 MRI：[{relative_case}/simulation_truth_mri.png]({relative_case}/simulation_truth_mri.png)",
            f"- 仿真真值俯视皮层：[{relative_case}/simulation_truth_surface_top.png]({relative_case}/simulation_truth_surface_top.png)",
            f"- 八方法 MRI 对比：[{relative_case}/all_algorithms_mri.png]({relative_case}/all_algorithms_mri.png)",
            f"- 八方法俯视皮层对比：[{relative_case}/all_algorithms_surface_top.png]({relative_case}/all_algorithms_surface_top.png)",
            "",
        )
    )
    if scenario != "surface_only":
        index_lines[-1:-1] = (
            f"- 仿真真值 MRI + 俯视皮层：[{relative_case}/simulation_truth_mri_surface.png]({relative_case}/simulation_truth_mri_surface.png)",
            f"- 八方法 MRI + 俯视皮层：[{relative_case}/all_algorithms_mri_surface.png]({relative_case}/all_algorithms_mri_surface.png)",
        )

save_root.mkdir(parents=True, exist_ok=True)
index_path = save_root / "INDEX.md"
index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")

print("四种情形全部完成：", save_root)
print("索引：", index_path)
