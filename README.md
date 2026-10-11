# 时序数据诊断与处理研究

当前目标是让 Agent 定位预测性能不足，编写针对性的数据处理代码，执行受控训练，并依据实际反馈修正判断。主要研究变量是数据处理；当前批次固定原 DLinear、训练参数和预测任务。

## 已保留的成果

| 成果 | 实测结果 | 证据范围 |
| --- | --- | --- |
| [数据输入变换](research_runs/direct_processing/20261009/REPORT.md) | 三个额外种子平均 NMSE 改善 **4.2385%**，NMAE 退化 **0.0541%** | 现有 MetroPT3 上的受控开发验证，保留全部对照、消融及失败记录；研究助手直接探索 |
| [冻结残差 MLP](research_runs/model_exploration/REPORT.md) | 已冻结测试平均 NMSE 改善 **29.2698%**，NMAE 退化 **0.7691%** | 独立模型成果，检查点和测试记录保留，不能计入数据处理或 Agent 发现收益 |
| [Agent 已知方案工程复现](research_runs/agent_research/capability_acceptance/20261010_v4/REPORT.md) | 单种子 NMSE 改善 **4.082989%**，NMAE 改善 **0.233700%** | 已提供注册配方的工程验收，自主发现、机制及泛化尚未证实 |

数据处理成果的诊断入口为[数据特征与误差报告](research_runs/direct_diagnostics/20261009/REPORT.md)。完整研究路线见[长期规划](research_runs/agent_research/ROADMAP.md)。

## 当前阶段

阶段4 signal v2 比较 `full / no_memory / no_feedback`，每组3次独立搜索，分别评价事实、预测不足定位、局部命中、匹配控制、总体效果和反馈更新。

以[冻结 v2 机器协议](research_runs/agent_research/phase4/20261010_signal_v2/protocol.json)和[接续说明](research_runs/HANDOFF.md)为准。[原冻结说明](research_runs/agent_research/phase4/20261010_signal_v2/PROTOCOL.md)保留早期批次描述，其末尾记录 v2 更换；项目路径、ID和批次前计数以机器协议为准。

最后核对的[服务器快照](research_runs/agent_research/phase4/20261010_signal_v2/progress_snapshot.json)来自 **2026-10-10 23:09（北京时间）**：9次注册、3次启动、1次选择基线结束、1次平台失败、另1次搜索当时运行。尚无确认的正确信号或自主处理收益。2026-10-11 接续时 SSH 连接被关闭，当前服务器状态未核实。

累计 **200次完整训练、1079次 API 尝试、3632209 个实际已知 tokens** 仅为该快照，不能当作最终计数。38条旧未知用量和293757个预约 tokens 分列保留；没有新增预算上限，也没有因整理重置成本。[最新进度账本](research_runs/agent_research/progress.json)保存基数和接续步骤。

## 项目入口

- [下一步](research_runs/NEXT_STEPS.md)：先核实服务器状态，再收齐当前批次、确认与评分。
- [研究概览](research_runs/OVERVIEW.md)：有效成果及尚未证明的结论。
- [冻结经验库 v2](research_runs/agent_research/memory/v2/README.md)：144条可质疑的历史证据；其来源报告仍保留原路径和哈希。
- `online_ts/`、`scripts/`、`tests/`：当前运行组件、有效成果的复现依赖及测试。
- `data/metropt3_v1/`：[不可变基础数据身份](data/metropt3_v1/manifest.json)，[原任务协议](research_runs/PROTOCOL.md)。
- `archives/capability_acceptance_20261010_v4/`：最新成功工程复现的完整冻结资产。

所有研究 Python、API 调用和模型训练在专用4090服务器执行。本地负责编辑、文件整理、同步与审阅。预处理仅在训练分区拟合，验证真值及困难样本不得删除，旧测试已用过且不能再用于探索或称为未见数据。

## 本次整理

2026-10-11 将5633个历史文件约 **18.3 GB** 移至项目旁的 `../auto_data_history/20261011_204809/`，完整清单为该目录的 `archive_manifest.json`。项目主体由约20.1 GB降至约1.8 GB（不含 Git 历史），原始数据、冻结代码和保留成果哈希未改变。这里是移出工作树，磁盘上的历史证据仍可恢复。

[整理报告](research_runs/cleanup/20261011/REPORT.md)记录路径映射、核验和公开同步范围。旧下载回执中的已迁移本地路径按归档清单解释，不改写旧收据。Git 历史继续保留原有公开内容。
