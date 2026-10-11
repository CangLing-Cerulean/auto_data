# 自主时序数据处理研究

当前验证 Agent 能否准确诊断预测不足，编写数据处理和匹配控制代码，执行固定模型训练，并根据实测反馈更新假设。[长期规划](ROADMAP.md)仍适用；当前执行阶段为4。

以[signal v2机器协议](phase4/20261010_signal_v2/protocol.json)、[最新进度账本](progress.json)和[项目接续说明](../HANDOFF.md)为入口。`full/no_memory/no_feedback`各3次独立搜索，不额外提供近期有效配方，累计资源无上限。分别评价事实、定位、局部命中、控制、整体效果与更新。

最后[本地快照](phase4/20261010_signal_v2/progress_snapshot.json)来自2026-10-10 23:09北京时间：3次启动、1次B0结束、1次平台失败，另1次搜索当时运行。尚未确认自主正确信号或处理收益。2026-10-11接续SSH连接被关闭，当前状态未知；不要重启批次或复用旧ID。

已保留的[直接数据处理成果](../direct_processing/20261009/REPORT.md)是研究助手探索的受控开发结果。[最新已知方案工程复现](capability_acceptance/20261010_v4/REPORT.md)证明执行链可用，不能当作自主发现。冻结[MLP成果](../model_exploration/REPORT.md)独立保留，不能计入数据处理收益。

[经验库v2](memory/v2/README.md)含134次历史结果和10条条件性摘要，共144条。允许质疑、重访与复现，检索不强制，不设置方法族禁用。库文件、副本来源及8份原文报告保持原身份；后续新增须另建版本。

所有研究Python、API与训练在专用4090服务器运行，本地只编辑、整理、同步和审阅。[执行约定](EXECUTION_CONTRACT.md)保留；当前代码不因清理改动。入口 `scripts/run_signal_study.py` 是冻结批次执行器，不应在核实状态前重新启动。

历史实验主体和重复归档已移至项目旁的 `auto_data_history/20261011_204809/`，仍可按清单恢复。最新成功工程复现的完整资产保留于项目内 `archives/capability_acceptance_20261010_v4/`。接续先完成[当前阶段](../NEXT_STEPS.md)，再进入独立任务与论文证据整理。
