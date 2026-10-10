# 自主时序数据处理研究入口

阶段4当前以[v2协议](phase4/20261009_v2/PROTOCOL.md)和[进度](phase4/20261009_v2/progress.json)为准。v1因CPU PyTorch地址空间兼容问题停止，全部代码、两次基线和API费用保留；v2在首个调用前完成61项检查及真实数据预检，使用新ID与上下文。Agent自行编写处理程序，修复仅为执行环境与接口信息。

用户已授权开始阶段4：[四组独立搜索协议](phase4/20261009/PROTOCOL.md)已登记，59项组件检查及真实T处理程序预检通过。Agent自主编写诊断、窗口处理及输入处理函数，平台校验并固定评价；输入变换不限定为预写配方菜单。首批12次搜索，当前见[批次进度](phase4/20261009/progress.json)。阶段5未启动，下文“Agent暂缓”为历史状态。

直接研究及完整备份核验现已全部完成，按用户要求停止；当前无训练队列，自组Agent仍暂缓。最新[报告](../direct_processing/20261009/REPORT.md)与[下载回执](../direct_processing/20261009/download_verification.json)已保存，下文“核验后结束”为此前阶段记录。

2026-10-09直接研究已完成科学确认：[固定模型数据处理报告](../direct_processing/20261009/REPORT.md)。25次新训练，两个内部原点、三个额外种子及两种消融通过，确认NMSE平均改善4.2385%、NMAE退化0.0541%，统一步数收益仍为4.1524%。完成完整资产核验后结束本轮；当前无训练队列，自组Agent发现仍暂缓。这是现有MetroPT3的受控开发成果，不计自主Agent成功率或新数据泛化。

2026-10-09用户调整执行顺序：先由当前研究助手直接探索数据特征、预测不足及处理假设，暂缓自组Agent发现和正式对照。当前工作入口为[直接探索协议](../direct_diagnostics/20261009/PROTOCOL.md)，已完成的Agent工程与经验库保留。研究助手指导的发现不能计入自主Agent实验成功率；未来以独立任务和冻结对照检验Agent能力。

研究范围、阶段门槛与停止条件见[长期计划](ROADMAP.md)，当前执行位置见[进度账本](progress.json)。历史134次训练与冻结MLP继续保留。2026-10-09真实自主诊断→处理代码→4090训练→反馈更新→最终选择闭环工程验收通过；候选NMSE/NMAE退化3.5672%/3.0753%，Agent拒绝候选并保留基线，详见[实测报告](acceptance/20261009/REPORT.md)。正式Agent重复实验和独立评价尚未启动。

已整理10个真实历史开发案例，保留原文范围、来源哈希、条件性结论和证据限制。经验组合允许本任务经验，新任务发现排除本数据集历史答案。只核验出处和检索边界，不把Agent或整理者的判断自动标记为科学真相。

2026-10-09已新增[可质疑的历史证据经验库v2](memory/v2/README.md)：覆盖134次历史训练结果及10条条件性摘要，共144条。支持目录、中文/英文检索、原记录详情，保留实测与解释的区别；无方法族禁用或强制检索，新任务过滤按数据集家族执行。12项新增测试和真实查询工具验收通过，本次无新增API/训练。其作用是辅助定位性能不足、检验处理假设，不把查阅经验或减少重复当作研究成果。

已实现分离的证据账本：实验前观察/假设/预期/控制/否定条件；可信执行回执；结果后判断更新；人工介入。前后记录不可覆盖，工程记录不计自主实验，执行失败不能直接否定科学假设。

用户已确认所有研究代码在专用4090分区直接运行。原生执行器、独立作业目录、资产完整性、权重/采样校验、可信训练及Agent工具已接通并通过真实闭环验收，详见[执行约定](EXECUTION_CONTRACT.md)和[开发协议](acceptance/20261009/DEVELOPMENT_PROTOCOL.md)。阶段3的正确信号独立评分、正式参数范围和预算仍待冻结。

服务器代码目录为/root/data/auto_data_research/agent_system_20261008_native，33项原生执行/处理/证据检查及1项API设置回归已通过。以下为该目录下的Python入口；本地只承担编辑、同步和审阅，不再执行研究Python脚本：

```text
python -B -m unittest discover -s tests -p test_agent_research.py -v
python -B -m online_ts.agent_research prepare
python -B -m online_ts.agent_research verify
python -B -m online_ts.agent_research experience --dataset metropt3_v1 --track experience_combination --query coverage
python -B -m online_ts.agent_research status
python -B -m online_ts.agent_research run-engineering --run-id NEW_UNIQUE_RUN_ID
```

prepare只整理文本证据，不读取传感器CSV、不生成记录或训练。已有经验库不能静默覆盖；来源改变后需要新的资产版本及冻结身份。

用户已确认超参数适配包括时序参数、学习率、batch和正则化，取值范围尚未冻结；新数据已存在，位置待提供。正确信号评分口径仍是待验证草案。
