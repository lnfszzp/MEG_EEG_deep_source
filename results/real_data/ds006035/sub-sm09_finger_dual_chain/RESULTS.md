# ds006035 sub-sm09 左指运动双链定位

## 数据与质控

- 任务：右腕正中神经电刺激（event 32）后，尽快抬起左食指（event 16）。
- 三个 run 严格一对一配对得到 118 对有效事件；反应锁定剔除后保留 113 个 epoch（38、38、37）。
- 反应时中位数分别为 252.5、273.4、285.4 ms。
- 刺激后 180~240 ms 与反应前 MF 窗在 93/118（78.8%）试次中重叠，因此不作为独立验证。
- 每个 run 保留 69 EEG、102 MAG、204 GRAD；刺激脉冲在滤波前插值。
- 三个 run 各用自己的头位建立 forward/inverse，之后才在共同的 4699 点皮层网格聚合。
- 三个 run 的传感器 GFP 峰均落在预注册窗：MF 为 -45.8/-77.7/-54.8 ms，MEFI 为 42.8/32.9/36.9 ms，MEFII 为 122.5/146.4/179.3 ms。

## 主要结果

| 链路 | 主要数值 | 判断 |
|---|---:|---|
| eLORETA，MF -80~-20 ms | 右 precentral 富集 1.86；峰距 5.3 mm | 支持运动前右侧中央区活动 |
| dSPM，MF -80~-20 ms | 右 precentral 富集 1.61；峰距 12.1 mm | 与 eLORETA 方向一致 |
| dSPM，MEFI 20~60 ms | 右 precentral 富集 2.98；峰位于 precentral | 清楚，但不把它单独解释成纯感觉反馈 |
| eLORETA，MEFI 20~60 ms | 右 postcentral 富集 3.43；峰位于 postcentral | 最符合运动后感觉反馈 |
| DICS，beta-ERD | 右 precentral 富集 17.49、质量 54.7%；右 postcentral 富集 4.25 | 强烈支持右侧感觉运动网络 |
| DICS，beta-PMBR | 右 precentral 富集 1.92；峰距 5.3 mm | 支持运动后 M1 beta rebound，但 run-2 稳定性较弱 |
| OASTER-ERP | MF/MEFII 各有 2/3 run 通过 EBIC，多数窗选中右 superiorfrontal 稀疏模板；右 precentral 质量为 0 | 当前版本不适合作为 Finger 主结论 |
| mu-DICS | pooled 峰远离中央区 | 当前数据/窗下不支持 |

MEFII 的 pooled 全局峰落到左 posterior-cingulate/superior-frontal，尽管右 precentral ROI 富集仍大于 1；因此暂不把 MEFII 当作可靠的 M1 定位证据。

跨 run 看，MF 的 eLORETA 右 precentral 富集为 1.56/2.25/1.82，MEFI 的 eLORETA 右 postcentral 富集为 3.54/3.62/3.11；这是当前最稳定的时域证据。beta-PMBR 的右侧峰在 run-1/run-3 重复，但 run-2 的全局峰落到左 superiorfrontal，因此只算中等可信。

## 使用结论

这项任务不能只用旧的刺激后 `20~80 ms` ERP 窗。当前最稳妥的组合是：

1. response-lock 的 MF/MEFI 用 dSPM、eLORETA定位锁相运动场；
2. beta-ERD 与 beta-PMBR 用共同滤波器 DICS 定位非锁相感觉运动功率；
3. OASTER-ERP 保留为对照，在解决跨窗固定模板问题前不用于宣称运动皮层定位成功。

## 限制

- 当前使用 MNE sample 模板解剖，不是 `sub-sm09` 的个体 FreeSurfer/BEM/trans；表中的毫米峰距只用于算法排错，不能当作论文或临床精度。
- 真实数据没有源位置真值，因此不能计算仿真意义上的 AUC/DLE。
- 图中 P95 阈值只控制显示；表格中的 ROI 富集、质量和峰距均由未阈值源图计算。
- run-2 MF 和 run-2 MEFII 没有通过 EBIC，当前 CSV 已明确记录为 `not_localized/NaN`，跨 run 聚合也已排除；其余重复的 OASTER 点有负 EBIC delta，不能简单视为程序卡死。
- 最终完整运行耗时约 554 秒；DICS 副本抗混叠降采样到 200 Hz，时域 ERP 保留原始采样率。

下一步应先完成五位受试者的个体 MRI 重建和配准，再按同一预注册窗口做组水平复现；如果刺激锁定与反应锁定仍相互污染，再加入双事件回归去卷积，而不是按 ROI 反复调参。
