# 对比方法与代码边界

统一 Python 对比由 `benchmark/methods.py` 提供，全部接收同一套模态内基线白化后的联合 EEG–MEG 数据和固定方向 lead field：

| 方法 | 本仓库入口 | 主要出处 |
|---|---|---|
| MNE / dSPM / sLORETA / eLORETA | `minimum_norm_family` | MNE-Python inverse 文档 |
| LCMV | `lcmv` | Van Veen et al., 1997, DOI `10.1109/10.623056` |
| 网格偶极子拟合 | `dipole_fit` | 观测时基压缩 + BIC 选 1–3 个网格点 |
| RAP-MUSIC | `rap_music` | Mosher & Leahy, 1999, DOI `10.1109/78.740118` |

扩展方法通过外部适配器调用，避免把许可证缺失或用途受限的第三方源码重新发布：

| 方法 | 上游 | 本仓库策略 |
|---|---|---|
| SISSES | `github.com/Buaawen/SISSES-code-for-MEG` | `SISSES_ROOT` 外部路径；不复制源码 |
| SI-DCB | `github.com/Buaawen/SI-DCB-for-MEG-source-imaging` | 可通过外部 MATLAB 适配器接入 |
| TS-Champagne | `github.com/Buaawen/TS-Champagne-for-MEG` | 上游未见许可证，保持外部调用 |
| Champagne | UCSF NeuroImagingTools/NUTMEG | 优先调用上游实现 |
| FAST-IRES | `github.com/YingLab/FAST-IRES` | 许可证/用途条件需由使用者确认，保持外部调用 |
| SISSY | NeuroImage 2017, DOI `10.1016/j.neuroimage.2017.05.046` | 未确认可靠官方实现，不把本地散落副本作为可复现基线 |

ConvDip 属于 EEG-only 且需要重新训练，没有深层输出，因此可作补充实验，不进入联合 EEG–MEG 深层主排名。
