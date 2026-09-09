# ds004998 DBS-MEG 频谱定位试运行

- 数据：sub-0cGdk9，HoldL，MedOn，run-1
- 方法：共同空间滤波器 DICS，13-30 Hz
- 通道：202 个 planar gradiometer；STN-LFP 未进入逆解
- epoch：rest 238 个，Hold 298 个；每种等量使用 238 个
- 8 mm 网格：BEM 检查前 4651 点，检查后 4231 点
- 最大 beta ERD：1.8791 dB
- 峰 MNI：[ 46.0, -16.0, 56.0 ] mm
- 峰到预注册右感觉运动中心：15.62 mm；到 25-mm ROI 边界：0.00 mm
- 右/左 ROI 平均正 ERD：1.2064 / 1.1111 dB
- ROI laterality index：+0.0412（正值表示右侧/对侧更强）
- 全脑 P95 点右/左：97 / 115；count LI=-0.0849

这里没有仿真真值，所以不能计算 AUC 或 DLE。当前结果只回答频谱定位链路是否可运行；
峰落在预注册右感觉运动 ROI 内，但双侧高值都很明显，不能宣称强侧化或优秀空间特异性。
当前本地下载也不足以比较 MedOn/MedOff 或进行组统计。
MRI 和 pial 图使用 fsaverage 模板且只用于显示，不是该患者的个体 MRI。
