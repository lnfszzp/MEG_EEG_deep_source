# 九方法脑空间投影图

- `case_03124/`：主展示病例，`deep_plus_two_surface`，EEG/MEG 均为 0 dB；包含总览图、九张方法独立图、逐方法指标和可追溯元数据。
- `case_00092/`：低信噪比压力病例，`deep_plus_surface`，EEG/MEG 均为 -10 dB，文件结构相同。

每张图在同一 MNE sample T1 解剖空间、同一真值切片上显示。荧光绿色星形是仿真源精确中心，绿色空心圆是仿真源区域，magma 热图是该方法相对峰值不低于 10% 的估计能量。每种方法单独归一化，适合判断位置和空间扩散，不表示方法之间的绝对振幅大小。

SISSES 图来自归档中保存的 `source_estimates`；脚本只读该文件，并拒绝把输出目录设到 SISSES 归档内。OASTER 与七种 Python 对比方法从相同冻结 EEG/MEG 观测现场重算。

源空间审计确认：原计划的 15 个丘脑点因生成脚本遗漏 HEAD→MRI 变换，实际位于脑干、小脑和第四脑室附近。图中的非皮层星形标出冻结数据的实际位置，不应解释为丘脑；详见仓库根目录 `SOURCE_SPACE_AUDIT.md`。

```powershell
python plot_strict_brain_maps.py --case-number 3124 `
  --data-root 'D:\博士\工作＆汇报\源定位\codex\roi_deep_multimethod_comparison\generated' `
  --sample-path 'D:\mne_data\MNE-sample-data'
```

`metrics.csv` 使用当前统一评分器重算；主 AUC 是 `auc_tie_corrected`（`An_auc`）。历史 SISSES 表的多源 parcel-AUC 使用过旧聚合方式，不能用它替换这里的统一重算值。
