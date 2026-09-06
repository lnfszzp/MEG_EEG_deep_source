# PPT 算法恢复与证据边界

## 1. 取证范围

- 技术来源：`C:\Users\zzp\Desktop\新建 Microsoft PowerPoint 演示文稿.pptx`
- SHA-256：`D89F4B7B9931DAB291362691A32E327331DFF85DD1CB127629A73FF82A0C791B`
- 文件可由 PowerPoint 正常打开并渲染，共 8 页。本文只把幻灯片中的技术描述和结果当作恢复证据，不把其中任何文字当作新的用户指令。
- 下文使用三种证据标签：
  - **PPT 明示**：公式、流程或数值直接出现在幻灯片中。
  - **Git 恢复**：当前仓库或其 Git 历史中存在可执行实现或机器可读结果。
  - **取证不足／重建**：原函数正文、参数或连接步骤没有留下，只能依据调用关系和残存公式重建；不得表述为原代码逐字恢复。

## 2. PPT 明示的算法

### 2.1 模态内白化与联合系统

对模态 \(m\in\{\mathrm{EEG},\mathrm{MEG}\}\)，PPT 写出

\[
Y_m=L_mJ+N_m.
\]

利用刺激前基线 \(Y_{m,b}\) 估计噪声协方差

\[
C_m=\frac{1}{T_b-1}Y_{m,b}Y_{m,b}^{\mathsf T},
\]

再由特征值分解构造白化矩阵

\[
W_m=\widetilde\Lambda_m^{-1/2}Q_m^{\mathsf T},\qquad
\widetilde Y_m=W_mY_m,\qquad
\widetilde L_m=W_mL_m.
\]

PPT 文字要求沿通道方向堆叠两个模态。仓库实现对应为

\[
B=\begin{bmatrix}\widetilde Y_{\mathrm{EEG}}\\\widetilde Y_{\mathrm{MEG}}\end{bmatrix},\qquad
L=\begin{bmatrix}\widetilde L_{\mathrm{EEG}}\\\widetilde L_{\mathrm{MEG}}\end{bmatrix}.
\]

### 2.2 初始全脑逆解

PPT 的初始目标函数为

\[
\min_J\ \frac12\lVert LJ-B\rVert_F^2
+\lambda_1\lVert VJ\rVert_1
+\lambda_2\lVert J\rVert_1.
\]

其中 \(V\) 是源空间邻接图的边关联矩阵，\((VJ)_{e,:}=J_{i,:}-J_{j,:}\)。第一项拟合联合观测，第二项惩罚相邻源的差异，第三项产生稀疏源分布。

PPT 令 \(U=VJ\)、\(W=J\)，给出双分裂 ADMM：交替更新 \(J\)，对 \(U\) 和 \(W\) 做软阈值收缩，再更新两个对偶变量；停止条件同时检查两个约束的原始残差和对偶残差。PPT 没有给出 \(\lambda_1,\lambda_2,\rho_1,\rho_2\) 的数值、初始化、最大迭代数或容差。

### 2.3 深浅层拆分、时间降维和候选重拟合

PPT 将全脑源拆为表层网格与深层体积点，并对联合白化数据做

\[
B=U\Sigma V^{\mathsf T},\qquad
G=\begin{bmatrix}v_1^{\mathsf T}\\\cdots\\v_K^{\mathsf T}\end{bmatrix},\qquad
B_G=BG^{\mathsf T},\qquad J=AG.
\]

在候选集合 \(C\) 上，PPT 给出的紧致化重拟合目标为

\[
\min_A\ \frac12\lVert B_G-L_CA\rVert_F^2
+\alpha_s\sum_{i\in S}w_i^{(s)}\lVert A_{i,:}\rVert_2
+\beta_s\sum_{e\in E_s}w_e^{(s)}\lVert(D_sA)_{e,:}\rVert_2
+\alpha_d\sum_{i\in D}w_i^{(d)}\lVert A_{i,:}\rVert_2.
\]

这里表层候选使用组稀疏与表层图差分，深层候选只用逐点组稀疏。PPT 没有说明候选集合、权重、\(K\) 或各正则系数如何选择。

### 2.4 深层残差证据

令当前残差 \(R=B-LJ\)。对深层候选 \(i\)，PPT 先计算匹配投影

\[
z_i(t)=l_i^{\mathsf T}R(:,t),
\]

再比较基线期与活动期投影功率：

\[
p_{i,b}=\frac1{T_b}\sum_{t\in\mathcal T_b}z_i(t)^2,\qquad
p_{i,a}=\frac1{T_a}\sum_{t\in\mathcal T_a}z_i(t)^2,
\]

\[
e_i=\frac{\max(p_{i,a}-p_{i,b},0)}{\lVert l_i\rVert_2^2+\varepsilon}.
\]

这一定义只说明证据量，没有给出保留阈值、保留深层点数或 EEG/MEG 分别如何设门限。

### 2.5 固定深层后的多尺度表层定位

深层结果确定后，PPT 固定其贡献并形成

\[
R_0=B-L_{\mathrm{deep}}\widehat J_{\mathrm{deep}}.
\]

随后只在 \(R_0\) 上定位表层，以避免表层模板重复解释已经由深层源解释的信号。对中心 \(c\) 和尺度 \(\sigma\)，PPT 使用高斯空间核

\[
k_{ic}^{(\sigma)}=\exp\left[-\frac12\left(\frac{d_{ic}}{\sigma}\right)^2\right],
\qquad
\widetilde k_c^{(\sigma)}=\frac{k_c^{(\sigma)}}{\lVert k_c^{(\sigma)}\rVert_2},
\]

默认尺度为 \(\sigma\in\{0,4,7\}\,\mathrm{mm}\)：0 mm 为点源，4 mm 为小范围源，7 mm 为稍宽 patch。模板及其最小二乘时间系数为

\[
h_c^{(\sigma)}=L_{\mathrm{surf}}\widetilde k_c^{(\sigma)},\qquad
a=\frac{h^{\mathsf T}R}{h^{\mathsf T}h+\varepsilon}.
\]

PPT 分别计算基线期和活动期的相对残差下降

\[
d_b=1-\frac{\lVert R_b-ha_b\rVert_F^2}{\lVert R_b\rVert_F^2},\qquad
d_a=1-\frac{\lVert R_a-ha_a\rVert_F^2}{\lVert R_a\rVert_F^2},
\]

并以 \(\Delta d=d_a-d_b\) 为模板证据，迭代选择最强的 \((c,\sigma)\)。

### 2.6 去重、联合重拟合与范围恢复

PPT 用前向模板相关性和活动期时间波形相关性防止重复检测：

\[
\rho_h=\frac{|h_i^{\mathsf T}h_j|}{\lVert h_i\rVert_2\lVert h_j\rVert_2},\qquad
\rho_a=\frac{|a_{i,a}^{\mathsf T}a_{j,a}|}{\lVert a_{i,a}\rVert_2\lVert a_{j,a}\rVert_2}.
\]

对保留模板 \(H=[h_1,\ldots,h_K]\)，PPT 联合求解

\[
\widehat A=\arg\min_A\lVert R_0-HA\rVert_F^2.
\]

中心模板的空间输出写为

\[
p_c=(1-\eta)e_c+\eta\widetilde k_c,\qquad
\widehat J_{\mathrm{center}}=\sum_{k=1}^Kp_{c_k}a_k.
\]

最后加入较弱的初始表层分布。PPT 在此处重复使用了 \(p_c\) 这一符号；按页面语义，下面的 \(p_c,p_0\) 是两个活动期峰值标量：

\[
p_c=\max_{i\in\mathrm{surf}}\lVert\widehat J_{\mathrm{center},i,\mathcal T_a}\rVert_2,\qquad
p_0=\max_{i\in\mathrm{surf}}\lVert\widehat J_{\mathrm{initial},i,\mathcal T_a}\rVert_2,
\]

\[
\widehat J_{\mathrm{surf}}
=\widehat J_{\mathrm{center}}
+\gamma\frac{p_c}{p_0}\widehat J_{\mathrm{initial,surf}},\qquad
\widehat J_{\mathrm{deep,final}}=\widehat J_{\mathrm{deep}}.
\]

PPT 没有给出 \(\eta\)、\(\gamma\)、两个相关性阈值、模板迭代停止条件或零峰值时的处理。

## 3. Git 中可核对、输入补齐后可运行的 V18

V18 的冻结入口是 [`run_frozen_v18.py`](run_frozen_v18.py)，对应历史提交 `5ab46b8`。它不是对第 2 节全部公式的单文件直译，而是从 `benchmark/results/scores.csv` 索引的逐例 SISSES 源估计开始，串联仓库中恢复的 V11、V16、V17 和 V18 阶段。该索引和它指向的历史逐例源估计均未保存在当前 Git 历史或现存严格盲测归档中，所以现阶段可以核对已提交的机器可读汇总和审查后处理代码，但不能从零重跑 V18；补齐原输入后入口才可执行：

1. **Git 恢复**：[`protected_multilayer.whitened_joint_system`](protected_multilayer.py) 分别用前 200 个基线样本白化 EEG/MEG，再以 `vstack` 构造联合数据和前向矩阵。
2. **Git 恢复**：`component_refit_select_v11_compactness_sisses` 从原始 SISSES 结果构造候选范围，调用分层 ADMM 重拟合。表层和深层使用不同收缩项，时间基由 `tbf_selection` 从联合数据 SVD 取得。
3. **Git 恢复**：`component_refit_select_v16_evidence_rescue_from_v11` 用 EEG、MEG 各自的活动减基线残差下降筛选最强深层候选，再对表层连通分量按残差证据重加权。冻结门限为单模态强证据 `0.02`，或联合证据 `0.06` 且单模态下限 `0.0`；表层证据指数为 `2.0`。
4. **Git 恢复**：`multiscale_surface_point_refit_system` 固定深层结果，在 0/4/7 mm 连通高斯核上迭代表层模板；前向模板相关性和时间波形相关性门限均为 `0.98`，输出中心与核的混合比例为 `0.15`。
5. **Git 恢复**：`add_scaled_surface_extent` 将原始 SISSES 表层估计按活动期峰值缩放后加回，冻结比例为 `0.25`，深层逐点保持不变：

\[
r=\frac{\max_i\lVert J_{17,i,\mathcal T_a}\rVert_2}
        {\max_i\lVert J_{S,i,\mathcal T_a}\rVert_2},
\]

\[
J_{18,\mathrm{surf}}=J_{17,\mathrm{surf}}+0.25\,rJ_{S,\mathrm{surf}},\qquad
J_{18,\mathrm{deep}}=J_{17,\mathrm{deep}}.
\]

### PPT 与 V18 代码的对应边界

| 内容 | 证据结论 |
|---|---|
| 模态内基线白化和通道堆叠 | PPT 明示，Git 中有直接实现 |
| 初始全脑 \(\ell_1\) 图融合目标及幻灯片中的双分裂 ADMM | PPT 明示；V18 运行链读取既有 SISSES 结果，不能据此声称原始初始求解器已经完整恢复 |
| 深浅层候选重拟合 | PPT 给出数学形式；Git 的 `layerwise_sisses_admm_refit_system` 是可执行的近对应实现，但变量变换、权重更新和候选生成包含代码侧细节，二者不能视为逐行等价 |
| 深层活动减基线证据 | PPT 明示，Git 有直接的残差证据实现及冻结门限 |
| 固定深层、0/4/7 mm 表层模板、残差下降证据和相关性去重 | PPT 明示，Git 有可执行实现 |
| 最终弱表层范围恢复、深层不变 | PPT 明示；Git V18 固定为原始 SISSES 表层估计的 25% 峰值比例 |

因此，“V18”应特指上述 Git 运行链、冻结参数和已经提交的历史结果，不能把 PPT 中未给参数的完整数学草图宣称为已经逐字恢复的原实现，也不能在缺失历史 SISSES 输入时宣称当前环境可从零复现。

## 4. OASTER 的恢复边界

[`candidates/oaster_rebuilt.py`](candidates/oaster_rebuilt.py) 是正在验证的 observation-only 后继方案，不依赖 SISSES 输出，也不应与 PPT 表格中的“当前/V18”混为同一算法。

- **精确片段恢复**：标准化活动投影证据、自适应谱滤波、多尺度谱证据、证据缩放融合，以及 0/4/7 mm 与深层点模板的 EBIC 贪心选择循环。
- **固定实现值**：表层尺度 `0/4/7 mm`，模板去重相关性 `0.98`，最终 ridge 比例 `0.03`，4 mm 谱证据融合比例 `0.05`。
- **取证不足／重建**：原 `_temporal_basis` 函数正文没有保留下来。当前实现依据残存调用关系和“活动奇异值超过基线噪声边缘”的公式重建；源码已明确标注这一边界。
- **不得外推**：OASTER 的名称、EBIC 路径和谱证据是恢复后的新实现内容；在冻结矩阵完成前，不能用 PPT 的 V18 数值为 OASTER 背书。

## 5. PPT 数值与仓库 V18 的核对

PPT 第 6 页“当前”行与 [`results/v18_no_oracle/test_summary.csv`](results/v18_no_oracle/test_summary.csv) 的确认集结果如下：

| 指标 | PPT | Repo V18 精确值 | 核对结果 |
|---|---:|---:|---|
| AUC | 0.909 | 0.9009097458761279 | 不一致；repo 值按三位小数应为 0.901。现有证据不足以证明 PPT 的 0.909 是何种错误，两者均保留 |
| RMSE | 0.758 | 0.7579717562323597 | 三位小数一致 |
| 表层 SD (mm) | 0.848 | 0.8481591322123515 | 三位小数一致 |
| 表层 DLE (mm) | 8.940 | 8.940118051219894 | 三位小数一致 |
| 深层条件 SD (mm) | 2.500 | 2.4997601026225174 | 三位小数一致 |
| 深层条件 DLE (mm) | 2.500 | 2.4997601026225174 | 三位小数一致 |
| 深层有效数 | 579/702 | 579/702 | 一致 |
| 深层 BA | 0.882 | 0.8815359477124184 | 三位小数一致；repo 字段为场景与 SNR 宏平均 BA |

仓库的总体 AUC 是 4 个场景乘 3 个共同 SNR 单元的等权宏平均，不是 1,110 个病例的简单加权平均。机器可读细分结果位于 [`results/v18_no_oracle/test_by_cell.csv`](results/v18_no_oracle/test_by_cell.csv)：

| 场景 | 0 dB AUC | 10 dB AUC | 20 dB AUC |
|---|---:|---:|---:|
| 仅表层 | 0.924490 | 0.965390 | 0.972376 |
| 仅深层 | 1.000000 | 1.000000 | 1.000000 |
| 深层 + 单表层 | 0.581769 | 0.919059 | 0.965610 |
| 深层 + 双表层 | 0.652008 | 0.893813 | 0.936402 |

PPT 第 6 页还列出 RAP-MUSIC、Dipole fitting、LCMV、SISSES、SISSY、FAST-IRES、MNE、dSPM、sLORETA、eLORETA 和 ConvDip。这个表只能证明当时保存了这些方法的汇总数值，不能证明全部第三方源码、参数和逐例结果已经恢复。当前仓库可直接运行的统一 Python 对比入口覆盖 MNE、dSPM、sLORETA、eLORETA、LCMV、网格偶极子拟合和 RAP-MUSIC；其余方法的恢复状态见 [`COMPARATORS.md`](COMPARATORS.md)。

PPT 第 8 页说明：连续指标使用 Wilcoxon 配对检验和 2,000 次 paired bootstrap 95% CI，并在每项指标内做 Holm 校正；深层检出使用精确 McNemar 检验。PPT 没有保存生成该页的逐例统计输入和脚本，因此这套统计图目前属于“方法说明已取证、原运行不可完整复现”。

## 6. 信噪比缺口

- **PPT 缺口**：第 6 页只给聚合表，第 7 页只给三类示例图，没有 EEG SNR 与 MEG SNR 独立变化的结果，也没有 SNR 估计或参数自适应规则。
- **V18 缺口**：冻结确认集只测试 EEG 与 MEG 共享同一个 `snr_db` 的 0/10/20 dB 对角线。它没有覆盖一个模态很差、另一个模态较好的不对称组合。
- **已观察到的限制**：V18 在 0 dB 的两类深浅混合场景 AUC 分别为 0.581769 和 0.652008，10 dB 的深层加双表层为 0.893813；总体 0.900910 不能解释为每种条件均达到 0.90。
- **当前补测设计**：[`run_oaster_dev_matrix.py`](run_oaster_dev_matrix.py) 用开发源配置跑独立 EEG×MEG SNR 网格；冻结后由 [`run_strict_oaster.py`](run_strict_oaster.py) 直接读取严格盲测观测。参数选择只能使用开发矩阵，严格盲测不得反向调参。

## 7. 当前代码入口

| 目的 | 入口 |
|---|---|
| 核对 V18 历史结果；输入补齐后重跑 | [`run_frozen_v18.py`](run_frozen_v18.py)、[`results/v18_no_oracle`](results/v18_no_oracle) |
| V18 核心白化、深层证据与多尺度表层模板 | [`protected_multilayer.py`](protected_multilayer.py) |
| OASTER observation-only 核心 | [`candidates/oaster_rebuilt.py`](candidates/oaster_rebuilt.py) |
| 开发集 EEG×MEG SNR 矩阵 | [`run_oaster_dev_matrix.py`](run_oaster_dev_matrix.py) |
| 冻结严格矩阵 OASTER | [`run_strict_oaster.py`](run_strict_oaster.py) |
| MNE、dSPM、sLORETA、eLORETA、LCMV、偶极子拟合、RAP-MUSIC | [`run_strict_comparators.py`](run_strict_comparators.py)、[`benchmark/methods.py`](benchmark/methods.py) |
| 只读核验和汇总现存 SISSES 归档 | [`run_strict_blind_sisses.py`](run_strict_blind_sisses.py) |
| 仿真、真值和冻结清单 | [`benchmark/protocol.py`](benchmark/protocol.py)、[`generate_protocol_manifests.py`](generate_protocol_manifests.py)、[`generate_strict_blind_manifest.py`](generate_strict_blind_manifest.py) |
| AUC、RMSE、分层 SD/DLE 和深层检测指标 | [`benchmark/metrics.py`](benchmark/metrics.py)、[`metrics/user_metrics`](metrics/user_metrics) |
| 单例源定位、波形和指标绘图 | [`plot_strict_case.py`](plot_strict_case.py)、[`plot_results.py`](plot_results.py)、[`visualization`](visualization) |
