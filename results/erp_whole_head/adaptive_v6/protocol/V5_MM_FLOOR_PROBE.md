# v5 多轮 MM 四病例开发探针

本探针在 v5 新位置开发失败后按角色选取，只用于根因验证，不能作为校准或正式验收。

- 病例：00/10 两个纯表层误报位置，02 真深源，04 深层+双表层漏报位置。
- 源清单 SHA256：`54fd2126c34b251939f854e73f11901329e7efb24e3c5669ecc28260a100190b`。
- 唯一求解设置：`{"deep_reweight_floor": 0.5, "edge_weight_floor": 0.5, "epsilon_fraction": 0.05, "max_inner_retries": 20, "mrf_strength": 0.8, "outer_iterations": 20, "outer_tolerance": 0.01, "solver_kind": "admm", "surface_reweight_floor": 0.5, "tolerance": 0.001}`。
- 不运行参数网格，不复用旧正式 validation 阈值作为验收线。
- 预声明主条件：`max(T00, T10) < min(T02, T04)`。
- 数值条件：四例 H0/H1 均完成多轮 MM，最终新权重 stationarity gap `<= 0.001`。
- 若主条件失败，停止该候选；不得逐病例选择设置。
