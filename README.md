# 时序数据诊断与处理研究

当前目标是让 Agent 定位预测性能不足，编写针对性的数据处理代码，执行受控训练，并依据实际反馈修正判断。主要研究变量是数据处理；当前批次固定原 DLinear、训练参数和预测任务。

面向研究交流的完整[数据探索与训练结果汇总（2026-10-13）](research_runs/sharing/20261013_data_exploration/REPORT.md)包含数据身份、任务划分、方法、对照、逐种子结果、负结果、成本和复现资产；出处与SHA256随稿保存。

## 已保留的成果

| 成果 | 实测结果 | 证据范围 |
| --- | --- | --- |
| [数据输入变换](research_runs/direct_processing/20261009/REPORT.md) | 三个额外种子平均 NMSE 改善 **4.2385%**，NMAE 退化 **0.0541%** | 现有 MetroPT3 上的受控开发验证，保留全部对照、消融及失败记录；研究助手直接探索 |
| [冻结残差 MLP](research_runs/model_exploration/REPORT.md) | 已冻结测试平均 NMSE 改善 **29.2698%**，NMAE 退化 **0.7691%** | 独立模型成果，检查点和测试记录保留，不能计入数据处理或 Agent 发现收益 |
| [Agent 已知方案工程复现](research_runs/agent_research/capability_acceptance/20261010_v4/REPORT.md) | 单种子 NMSE 改善 **4.082989%**，NMAE 改善 **0.233700%** | 已提供注册配方的工程验收，自主发现、机制及泛化尚未证实 |

数据处理成果的诊断入口为[数据特征与误差报告](research_runs/direct_diagnostics/20261009/REPORT.md)。完整研究路线见[长期规划](research_runs/agent_research/ROADMAP.md)。

## 当前阶段

2026-10-13主线为“固定预测骨干下，面向数据干预的不足诊断与反馈决策学习”。最新[数值定位及独立补验](research_runs/agent_research/intervention_diagnosis/20261013_proxy_numeric_diagnosis_r1/REPORT.md)支持M1响应预测MAE为0.407517个百分点，较固定父评分更低；但二者三个任务的推荐相同，选择增量为0。原开发阶段数值审计失败及推进false保持，补验没有追改原验收身份。

累计完成**331次完整训练**；最新27次已另经补验，最近原规则整阶段验收累计304。历史外部API为2544次尝试、14389704个实际已知tokens；40条未知用量和347757个预约tokens分列，近期Sol high内部用量未知。没有重置历史成本。

下一项同M1的[训练数据选择开发协议](research_runs/agent_research/intervention_diagnosis/20261013_proxy_selection_r1/PROTOCOL.md)已另冻并进入预测阶段；注册36完整子标签须在预测锁后执行，尚未纳入上述331次完成快照。实际状态以该阶段progress为准，不从预期数量推断完成。旧signal v2九槽位已结束（6个B0、3失败），语义验收仍单列待补；不重启旧ID。接续以[HANDOFF](research_runs/HANDOFF.md)、[state](research_runs/state.json)和最新阶段报告为准。

## 项目入口

- [下一步](research_runs/NEXT_STEPS.md)：保持单一M1，另冻训练数据动作及强对照，不依已揭示D或旧测试设计方法。
- [研究概览](research_runs/OVERVIEW.md)：有效成果及尚未证明的结论。
- [冻结经验库 v2](research_runs/agent_research/memory/v2/README.md)：144条可质疑的历史证据；其来源报告仍保留原路径和哈希。
- `online_ts/`、`scripts/`、`tests/`：当前运行组件、有效成果的复现依赖及测试。
- `data/metropt3_v1/`：[不可变基础数据身份](data/metropt3_v1/manifest.json)，[原任务协议](research_runs/PROTOCOL.md)。
- `archives/capability_acceptance_20261010_v4/`：最新成功工程复现的完整冻结资产。

所有研究 Python、API 调用和模型训练在专用4090服务器执行。本地负责编辑、文件整理、同步与审阅。预处理仅在训练分区拟合，验证真值及困难样本不得删除，旧测试已用过且不能再用于探索或称为未见数据。

## 本次整理

2026-10-11 将5633个历史文件约 **18.3 GB** 移至项目旁的 `../auto_data_history/20261011_204809/`，完整清单为该目录的 `archive_manifest.json`。项目主体由约20.1 GB降至约1.8 GB（不含 Git 历史），原始数据、冻结代码和保留成果哈希未改变。这里是移出工作树，磁盘上的历史证据仍可恢复。

[整理报告](research_runs/cleanup/20261011/REPORT.md)记录路径映射、核验和公开同步范围。旧下载回执中的已迁移本地路径按归档清单解释，不改写旧收据。Git 历史继续保留原有公开内容。
