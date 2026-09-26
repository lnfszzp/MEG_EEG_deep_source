# hierarchical-v2 等权门控六种评分开发比较

只读取15例 development 冻结拟合；没有读取 validation，也没有重新运行逆解。标签和SNR只在盲分数全部算完后用于开发评估。

## 六种评分

1. **Raw fixed 3:1**：`0.75*g_EEG + 0.25*g_MEG`
2. **Global-energy fixed 3:1**：`(0.75*g_EEG + 0.25*g_MEG) / sqrt(1 + max(r_EEG,r_MEG))`
3. **Per-modality-energy fixed 3:1**：`0.75*g_EEG/sqrt(1+r_EEG) + 0.25*g_MEG/sqrt(1+r_MEG)`
4. **SNR-blind consensus**：`[min(g_EEG,g_MEG)+sqrt(max(g_EEG,0)*max(g_MEG,0))] / [1+max(r_EEG,r_MEG)]^0.25`
5. **Diagonal studentized 3:1**：`0.75*D_EEG/sqrt(Vdiag_EEG) + 0.25*D_MEG/sqrt(Vdiag_MEG)`
6. **Full-covariance studentized 3:1 (ablation)**：`0.75*D_EEG/sqrt(Vfull_EEG) + 0.25*D_MEG/sqrt(Vfull_MEG)`

两种能量归一化分别比较，不预先选择。`r` 是确认观测活动响应相对基线噪声期望的非负超额比。

## 总体结果

| 方法 | An_auc | H0最大值 | H1最小值 | 分离间隔 | H0正分 | H1正分 |
|---|---:|---:|---:|---:|---:|---:|
| Raw fixed 3:1 | 0.8704 | 0.0500257 | -0.0111221 | -0.0611478 | 2/6 | 8/9 |
| Global-energy fixed 3:1 | 0.9259 | 0.00529458 | -0.00140681 | -0.0067014 | 2/6 | 8/9 |
| Per-modality-energy fixed 3:1 | 0.9259 | 0.00531076 | 0.00156082 | -0.00374994 | 2/6 | 9/9 |
| SNR-blind consensus | 0.8889 | 0.00262692 | -0.0225534 | -0.0251804 | 2/6 | 6/9 |
| Diagonal studentized 3:1 | 0.8519 | 2.3283 | 0.0446381 | -2.28366 | 2/6 | 9/9 |
| Full-covariance studentized 3:1 (ablation) | 0.8333 | 2.65866 | 0.0178057 | -2.64086 | 2/6 | 9/9 |

## 分SNR结果

| 方法 | EEG/MEG SNR | An_auc | H0最大值 | H1最小值 | 分离间隔 |
|---|---|---:|---:|---:|---:|
| Raw fixed 3:1 | -10/-10 | 1.0000 | -0.0127424 | 0.00157701 | 0.0143194 |
| Raw fixed 3:1 | -10/+20 | 1.0000 | -0.0282153 | 0.114082 | 0.142297 |
| Raw fixed 3:1 | +20/-10 | 0.6667 | 0.0500257 | -0.0111221 | -0.0611478 |
| Global-energy fixed 3:1 | -10/-10 | 1.0000 | -0.0121968 | 0.00146202 | 0.0136588 |
| Global-energy fixed 3:1 | -10/+20 | 1.0000 | -0.00223629 | 0.011808 | 0.0140443 |
| Global-energy fixed 3:1 | +20/-10 | 0.6667 | 0.00529458 | -0.00140681 | -0.0067014 |
| Per-modality-energy fixed 3:1 | -10/-10 | 1.0000 | -0.0120675 | 0.00156082 | 0.0136283 |
| Per-modality-energy fixed 3:1 | -10/+20 | 1.0000 | -0.00134458 | 0.0141416 | 0.0154861 |
| Per-modality-energy fixed 3:1 | +20/-10 | 0.6667 | 0.00531076 | 0.00165943 | -0.00365133 |
| SNR-blind consensus | -10/-10 | 1.0000 | -0.0280933 | -0.0225534 | 0.00553988 |
| SNR-blind consensus | -10/+20 | 1.0000 | -0.0329218 | 0.01408 | 0.0470019 |
| SNR-blind consensus | +20/-10 | 0.6667 | 0.00262692 | -0.00695675 | -0.00958367 |
| Diagonal studentized 3:1 | -10/-10 | 1.0000 | -2.43559 | 0.0446381 | 2.48023 |
| Diagonal studentized 3:1 | -10/+20 | 1.0000 | -0.410165 | 1.86468 | 2.27484 |
| Diagonal studentized 3:1 | +20/-10 | 0.6667 | 2.3283 | 0.214618 | -2.11368 |
| Full-covariance studentized 3:1 (ablation) | -10/-10 | 1.0000 | -2.14853 | 0.0178057 | 2.16633 |
| Full-covariance studentized 3:1 (ablation) | -10/+20 | 1.0000 | -0.375061 | 1.82574 | 2.2008 |
| Full-covariance studentized 3:1 (ablation) | +20/-10 | 0.6667 | 2.65866 | 0.185522 | -2.47314 |

## 限制

- `selected_method = null`：这只是15例开发比较，不能代替新校准。
- full-covariance studentized 只作消融。基线中心化后样本协方差秩最多为 `B-1`；通道数大于 `B-1` 时必然奇异。
- studentized 值是经验排序分数，不是Gaussian Z检验，也不能直接解释成p值。
