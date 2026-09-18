# OASTER 作者风格版

这里的脚本按你的习惯写成 `# %%` 分块、从上到下执行的形式。常用参数集中放在开头，关键中间量直接保留在工作区，并在重要步骤后 `print` 或画图检查。

脚本互相独立：`1` 用公开真实 ERP 数据，`2` 用带真值的仿真数据，`3` 验证四类 ERP/ERF，`4` 用 DBS 患者的 MEG 做 beta 频谱定位，`5` 专门修正 ds006035 左指运动定位，`6` 从 `5` 的结果快速渲染完整 pial 俯视图，`7` 批量运行当前 OASTER-ERP 的全头七方法多 SNR 仿真。编号表示阅读顺序，不要求依次运行。

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
7. 用N20、P30各自的多尺度时间基运行OASTER-ERP，并通过EBIC选择0/4/7 mm空间模板。
8. 再展开原频谱 OASTER 的时间基、EBIC空间选择和频谱证据作对照；当前模板只建皮层源空间，所以不运行深层补救。
9. 同时计算 dSPM、eLORETA 作为参照。
10. 计算 N20/P30 指标并保存表格；把 `draw_brain` 改为 `True` 时再渲染脑表面图。

通常只改开头参数区：数据根目录、`subject`、`run`、滤波范围、epoch/基线/激活时间窗、源空间精度和输出目录。第一次阅读算法时，先不要改 OASTER 固定参数；否则很难判断结果变化来自数据还是参数。

`3-ERP四类任务验证.py` 用迁移后的多尺度 OASTER-ERP 核心验证四类诱发反应：MNE sample 的左右听觉、左右视觉，以及 ds006035 的右腕电刺激 N20/P30 和左指运动反应。它把电刺激与运动事件分开叠加、分开估计基线噪声；目标脑区只在全部逆解完成后用于评价，不参与参数或峰时刻选择。结果写入 `results/real_data/erp_four_paradigms_v2`，不会覆盖旧时间域基线结果。

当前实跑中，OASTER-ERP 的听觉和视觉峰落在预期ROI附近；右腕N20/P30的ROI富集未过线，左指结果虽有ROI能量但主峰位置错误。这是迁移版的真实边界，不应按ROI继续调参后再把同一数据当独立验证。

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

## 4-DBS频谱定位.py

用途：在 ds004998 的完整 `sub-0cGdk9 / HoldL / MedOn / run-1` 上，以共同滤波器 DICS 比较静息和左臂持续用力的 13–30 Hz beta 功率。FIF 中名为 `EEG001-EEG008` 的通道实际是 STN-LFP 触点，本脚本明确不把它们当头皮 EEG，也不让它们进入逆解。

脚本使用数据集自带的个体 4-mm FieldTrip 网格和单壳边界，试运行时降为约 8 mm；输出逐源表、预注册双侧感觉运动 ROI 指标、MNI 图，以及仅用于展示的 fsaverage MRI/pial 渲染。真实数据没有仿真真值，因此这里不计算 AUC 或 DLE。

## 5-左指运动双链定位.py

用途：修正 `3-ERP四类任务验证.py` 中把 Finger 只看作 `+20~80 ms` ERP 的问题，并使用 `sm09` 三个 run 做专门的运动定位。

- 每个 run 严格按 `event32 → event16` 一对一配对，保存孤立事件、无效反应时和全部有效 RT；
- `event32` 的 `-2~+8 ms` 电刺激脉冲在线性插值后才进行 notch/filter；
- 保留 EEG、102 个 magnetometer 和 204 个 gradiometer；
- 三个 run 的 `dev_head_t` 不同，因此各自重算 forward、OASTER/dSPM/eLORETA 和 DICS，最后只在共同 sample 源网格上按有效试次数聚合；
- 时域分别看 response-lock 的 MF `-80~-20 ms`、MEFI `+20~60 ms`、MEFII `+120~180 ms`；stimulus-lock 的 M1 `180~240 ms` 默认关闭，因为本数据中 78.8% 的试次与 response-MF 是同一段物理时间，不能当作独立验证；
- 频域用单试次共同滤波器 DICS 计算 mu `8~14 Hz`、beta `15~30 Hz` 的运动 ERD 和 PMBR；
- DICS 的数据副本先抗混叠降采样到 `200 Hz`（最高分析频率仅 `30 Hz`），时域 ERP 保留原始采样率；
- response 基线固定为安全的 `-1.5~-1.0 s`，stimulus 基线为 `-0.5~-0.05 s`，不再使用旧 Finger `-0.55~-0.35 s`；
- OASTER 的 EBIC 不支持某个窗时返回“未定位”，不再用 `require_one` 强制制造一个峰；
- 每个 run/窗的模板 index、尺度、时间秩和 EBIC delta 单独保存到 `oaster_window_diagnostics.csv`；
- 所有逆解完成后才读取右侧 `precentral` 与 `postcentral`，分别保存 ROI 富集和峰距；
- 默认不启动三维窗口；把 `draw_brain=True` 后可输出 pooled pial 俯视图。

主要输出目录是 `results/real_data/ds006035/sub-sm09_finger_dual_chain`。脚本计算量较大，建议先逐块运行到时域结果，确认通道和 epoch 数，再运行 DICS 块。

## 6-左指运动结果俯视图.py

用途：直接读取脚本 `5` 已保存的逐 run 源图，不重复计算逆解。每个时间窗先取三种方法共同可定位的 run，再用完全相同的试次权重聚合，渲染为 3×3 匹配比较图；DICS beta 结果单独成图，不与 ERP 方法作优劣比较。同时保存匹配后的源图、ROI 指标表、比较图和 `oaster_detection_rate.csv`。各图分别峰值归一化，P95 只控制显示，因此脑图只定性比较峰位和空间模式，不比较振幅或激活范围；共同集指标描述“成功定位后定到哪里”，全部 run 的定位成功率描述“能否稳定定位”，两者必须同时报告。

## 7-ERP全头七方法多SNR仿真.py

用途：运行当前相位锁定 OASTER-ERP 和七个固定对比方法的完整仿真。脚本集中展示 manifest、几何、MNE sample、输出路径、7×7 EEG/MEG SNR 和 worker 参数，然后调用正式批处理，不复制算法。

默认覆盖纯表层、纯深层、深层加单表层、深层加双表层四种情况；每个 SNR 使用全部 186 个位置配置，共 49 个 SNR 组合。这里的“全头”是双侧 68 个皮层分区代表位置和 16 个双侧丘脑点，不等于把 7498 个皮层顶点逐点作为真值。结果按 SNR 组合断点保存，重跑脚本会校验并跳过完整检查点；结束后自动检查结果行数并生成 An_auc、SD、DLE 图和八方法对比表。

完整实跑结果位于 `results/erp_whole_head/current_method_v1`：72,912 行全部成功。当前 OASTER-ERP 的 49 个 SNR 四场景宏平均 An_auc 为 0.823559，主要短板是深源漏检；详见该目录的 `REPORT.md`。

## 原脚本对应关系

| 作者风格版 | 原来的正式入口/底层代码 | 适用范围 |
|---|---|---|
| `1-OASTER真实ERP.py` | `pipelines/run_ds006035_somatomotor.py` | 单受试者、单 run 阅读调试 |
| `1-OASTER真实ERP.py` | `pipelines/summarize_ds006035.py` | 全部受试者完成后的组级统计 |
| `2-OASTER仿真.py` | `generate_strict_blind_manifest.py`、`generate_strict_blind_inputs.py` | 建立冻结 manifest 和批量仿真输入 |
| `2-OASTER仿真.py` | `run_strict_oaster.py` | 完整场景和 SNR 网格批处理 |
| `4-DBS频谱定位.py` | MNE common-filter DICS + ds004998 自带 FieldTrip 网格/边界 | DBS-MEG 真实 beta-ERD 基线 |
| `5-左指运动双链定位.py` | MNE/OASTER run-specific inverse + common-filter DICS | ds006035 三 run 左指运动双链定位 |
| `7-ERP全头七方法多SNR仿真.py` | `run_erp_whole_head_matrix.py`、`plot_erp_whole_head_results.py` | 当前 ERP 算法、七个对比方法和 49 组 EEG×MEG SNR |
| 作者风格脚本中的 OASTER 段 | `candidates/oaster_rebuilt.py`、`protected_multilayer.py` | 正式算法实现和公共数值检查 |
| `2-OASTER仿真.py` 的评价段 | `benchmark/metrics.py` | AUC、SD、DLE 的正式实现 |
| `2-OASTER仿真.py` 的真值/噪声段 | `benchmark/protocol.py` | 冻结真值、波形和噪声协议 |

## 为什么底层批处理仍保留 `def`

你直接阅读的两个入口不再用很多 `def` 把主流程藏起来；关键计算按执行顺序展开。

底层正式代码仍保留函数，是因为完整实验需要循环多名受试者、四种源场景、49 种 EEG/MEG SNR 组合，并支持断点续跑、多进程、输入校验和自动测试。如果把这些函数全部复制到每个循环里，代码会更长，也更容易让不同 case 使用到不一致的公式。

因此两套代码分工如下：

- 想看算法、改参数、逐步调试：运行“作者风格版”。
- 想重新生成论文级完整结果：运行原批处理入口。
- 修改核心公式后：先在作者风格版检查一个 case，再运行测试和完整批处理。
