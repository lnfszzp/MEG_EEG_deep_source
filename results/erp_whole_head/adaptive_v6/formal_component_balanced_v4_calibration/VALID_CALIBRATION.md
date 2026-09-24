# v4 正式校准冻结记录

- 状态：有效；57/57 个校准案例完成，114/114 个 H0/H1 拟合全部收敛。
- 校准清单 SHA256：`09dde6a2c14ceeeaa4a3185d218f8017bc6141b37e4172d19f6f85c22cf73ff1`。
- 执行锁 SHA256：`05c8edd1130ef22f0af8938d94cdc8c25fb133385adcdb5c440f5cb7b594df83`。
- 冻结校准文件 SHA256：`7dc2b57e97acaab71de4ea684ca9ae48ff731a15a7e6e6b67203fff044d8caf4`。
- pooled conformal 规则：`p=(1 + # calibration scores >= test score) / 58`，`alpha=0.05`，并要求检验分数大于 0。
- 因此正式盲测的严格判定边界为 `test score > 0.1899568000886776`；等于边界不检出。
- 本记录生成时，受保护 validation 尚未运行，消费标记不存在。

校准结果只用于冻结判定规则，不能作为独立测试集，也不能在查看 validation 后重新调参。
