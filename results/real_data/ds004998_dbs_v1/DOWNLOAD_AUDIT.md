# ds004998 本地下载检查

检查日期：2026-09-09。远端基准为 NEMAR `on004998 v1.0.0` manifest。

- 远端：20 名受试者、642 个文件、145 个 MEG FIF，总计 173.73 GB（161.8 GiB）。
- 本地：20 个受试者目录，但只有 11 个正规 `.fif`，正规 FIF 共 3.10 GB；另有 3 个随机后缀的未完成下载片段，共 1.16 GB。
- 与远端文件大小逐项核对后，只有 `sub-0cGdk9_ses-PeriOp_task-HoldL_acq-MedOn_run-1_meg.fif` 是完整、足够长且同时包含 rest 与五段 HoldL 的可分析记录（660 秒）。
- 其余 5 个 `split-02` 文件缺少相应 `split-01`，不能作为完整 run 分析；另外 5 个短 run 虽与远端文件大小一致，但时长仅 28.5–105.7 秒，不足以替代完整的约 5 分钟 rest + movement 设计。
- 个体模型：20 个 headmodel、19 个 sourcemodel；`sub-QZTsn6` 缺 sourcemodel。

当前真实定位因此只使用 `sub-0cGdk9 / HoldL / MedOn / run-1`。现有文件足够做单记录方法冒烟测试，不足以做 MedOn/MedOff 配对、跨任务重复性或组统计。
