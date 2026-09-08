# 对比方法与代码边界

统一 Python 对比由 `benchmark/methods.py` 提供，全部接收同一套模态内基线白化后的联合 EEG–MEG 数据和固定方向 lead field。SISSES 通过独立 MATLAB 适配器读取同一冻结观测：

| 方法 | 本仓库入口 | 主要出处 |
|---|---|---|
| MNE / dSPM / sLORETA / eLORETA | `minimum_norm_family` | MNE-Python inverse 文档 |
| LCMV | `lcmv` | Van Veen et al., 1997, DOI `10.1109/10.623056` |
| 网格偶极子拟合 | `dipole_fit` | 观测时基压缩 + BIC 选 1–3 个网格点 |
| RAP-MUSIC | `rap_music` | Mosher & Leahy, 1999, DOI `10.1109/78.740118` |
| SISSES | `run_corrected_v2_sisses.py` → `corrected_v2_sisses_adapter.m` | Li et al., 2024；外部 `SISSES_ROOT` |

SISSES 核心已在本机恢复并通过原函数小矩阵测试及 corrected-v2 单病例端到端测试。当前聚合结果仍为 N/A，直至 9,114 例全部运行、评分并生成 `completion.json`。上游未提供显式许可证，因此仓库只发布适配器、哈希和引用，不重新发布第三方源码。

下列扩展方法是已确认的上游候选，不是本仓库当前可运行的对比代码。若以后补充独立外部适配器，应从上游 checkout 调用：

| 方法 | 上游 | 本仓库策略 |
|---|---|---|
| SI-DCB | `github.com/Buaawen/SI-DCB-for-MEG-source-imaging` | 尚无适配器；可作为后续外部 MATLAB 接入 |
| TS-Champagne | `github.com/Buaawen/TS-Champagne-for-MEG` | 尚无适配器；上游未见许可证 |
| Champagne | UCSF NeuroImagingTools/NUTMEG | 尚无适配器；后续应优先调用上游实现 |
| FAST-IRES | `github.com/YingLab/FAST-IRES` | 尚无适配器；许可证/用途条件需由使用者确认 |
| SISSY | NeuroImage 2017, DOI `10.1016/j.neuroimage.2017.05.046` | 未确认可靠官方实现，不把本地散落副本作为可复现基线 |

ConvDip 属于 EEG-only 且需要重新训练，没有深层输出，因此可作补充实验，不进入联合 EEG–MEG 深层主排名。
