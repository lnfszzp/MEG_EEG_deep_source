# OASTER 作者风格版

这里的两个脚本按你的习惯写成 `# %%` 分块、从上到下执行的形式。常用参数集中放在开头，关键中间量直接保留在工作区，并在重要步骤后 `print` 或画图检查。

两个脚本互相独立：`1` 用公开真实 ERP 数据，`2` 用带真值的仿真数据。编号表示阅读顺序，不要求先运行 `1` 才能运行 `2`。

## 怎么运行

1. 在 PyCharm 中打开脚本，选择项目使用的 `meg` Python 环境。
2. 先运行 import 和“参数区”，再按 `# %%` 从上到下逐块运行。
3. 第一次运行或修改了源空间、通道、时间窗后，建议重启 Python Console，再从第一块开始。
4. 不要直接跳到后面的定位或画图块，因为变量按顺序留在全局工作区中。

## 1-OASTER真实ERP.py

用途：检查 ds006035 中一个受试者、一个 run 的 ERP/ERF 和源定位全过程。

运行顺序：

1. 设置数据路径、受试者、run 和输出位置。
2. 读取同步 EEG/MEG 和刺激事件。
3. 滤波、坏道/坏试次处理，构造 epochs。
4. 用 `epochs.average()` 得到叠加平均 ERP/ERF，并先检查传感器波形。
5. 配准并建立 EEG、MEG forward。
6. 用基线单试次数据估计噪声和白化矩阵。
7. 先运行保留 ERP 极性和逐时刻变化的 OASTER ERP 时间域分支。
8. 再展开原 OASTER 的时间基、EBIC 空间选择和频谱证据作对照；当前模板只建皮层源空间，所以不运行深层补救。
9. 同时计算 dSPM、eLORETA 作为参照。
10. 计算 N20/P30 指标并保存表格；把 `draw_brain` 改为 `True` 时再渲染脑表面图。

通常只改开头参数区：数据根目录、`subject`、`run`、滤波范围、epoch/基线/激活时间窗、源空间精度和输出目录。第一次阅读算法时，先不要改 OASTER 固定参数；否则很难判断结果变化来自数据还是参数。

`3-ERP四类任务验证.py` 用同一个固定的时间域逆解验证四类诱发反应：MNE sample 的左右听觉、左右视觉，以及 ds006035 的右腕电刺激 N20/P30 和左指运动反应。它把电刺激与运动事件分开叠加、分开估计基线噪声；目标脑区只在全部逆解完成后用于评价，不参与参数或峰时刻选择。

这个脚本只处理一个受试者的一个 run，适合逐块看变量。五名受试者和全部 run 的正式结果仍用原批处理入口。

## 2-OASTER仿真.py

用途：选取一个有真值的严格盲测 case，完整查看仿真信号怎样进入 OASTER，以及 AUC、SD、DLE 怎样得到。

运行顺序：

1. 设置项目、manifest、数据和输出路径。
2. 选择 `scenario`、EEG SNR、MEG SNR 和 `cell_case_number`。
3. 从 manifest 找到唯一 case，读取与 forward 对齐的几何信息。
4. 构造表层/深层真值和时间波形。
5. 经 forward 投影，并分别给 EEG、MEG 加一次有色噪声。
6. 白化并形成同步 EEG/MEG 联合系统。
7. 展开时间基、EBIC、深层补救、频谱证据和最终源时序合成。
8. 用同一真值计算 AUC、SD、DLE，保存结果并画图。

常改参数是 `scenario`、`eeg_snr_db`、`meg_snr_db`、`cell_case_number` 和画图开关。`scenario` 可选：

- `surface_only`：纯表层源；
- `deep_only`：纯深层源；
- `deep_plus_surface`：一个深层源加一个表层源；
- `deep_plus_two_surface`：一个深层源加两个表层源。

当前正式 SNR 网格为 `-10、-5、0、5、10、15、20 dB`。这个脚本一次只看一个 case；它不会替代完整 SNR 网格批处理。

## 原脚本对应关系

| 作者风格版 | 原来的正式入口/底层代码 | 适用范围 |
|---|---|---|
| `1-OASTER真实ERP.py` | `pipelines/run_ds006035_somatomotor.py` | 单受试者、单 run 阅读调试 |
| `1-OASTER真实ERP.py` | `pipelines/summarize_ds006035.py` | 全部受试者完成后的组级统计 |
| `2-OASTER仿真.py` | `generate_strict_blind_manifest.py`、`generate_strict_blind_inputs.py` | 建立冻结 manifest 和批量仿真输入 |
| `2-OASTER仿真.py` | `run_strict_oaster.py` | 完整场景和 SNR 网格批处理 |
| 两个脚本中的 OASTER 段 | `candidates/oaster_rebuilt.py`、`protected_multilayer.py` | 正式算法实现和公共数值检查 |
| `2-OASTER仿真.py` 的评价段 | `benchmark/metrics.py` | AUC、SD、DLE 的正式实现 |
| `2-OASTER仿真.py` 的真值/噪声段 | `benchmark/protocol.py` | 冻结真值、波形和噪声协议 |

## 为什么底层批处理仍保留 `def`

你直接阅读的两个入口不再用很多 `def` 把主流程藏起来；关键计算按执行顺序展开。

底层正式代码仍保留函数，是因为完整实验需要循环多名受试者、四种源场景、49 种 EEG/MEG SNR 组合，并支持断点续跑、多进程、输入校验和自动测试。如果把这些函数全部复制到每个循环里，代码会更长，也更容易让不同 case 使用到不一致的公式。

因此两套代码分工如下：

- 想看算法、改参数、逐步调试：运行“作者风格版”。
- 想重新生成论文级完整结果：运行原批处理入口。
- 修改核心公式后：先在作者风格版检查一个 case，再运行测试和完整批处理。
