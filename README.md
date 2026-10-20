# 时序科研与滚动预测平台

当前目标是让 Agent 定位预测性能不足，编写针对性的数据处理代码，执行受控训练，并依据实际反馈修正判断。主要研究变量是数据处理；当前批次固定原 DLinear、训练参数和预测任务。

统一工程入口为 `python -m online_ts.platform`。平台已整合实验冻结、执行收据、恢复与对账，以及持续数据接入、滚动预测、后台训练、监视和模型发布。安装、合成演示、真实流接入和扩展接口见[平台使用说明](docs/PLATFORM.md)。

面向研究交流的完整[数据探索与训练结果汇总（2026-10-13）](research_runs/sharing/20261013_data_exploration/REPORT.md)包含数据身份、任务划分、方法、对照、逐种子结果、负结果、成本和复现资产；出处与SHA256随稿保存。

## 已保留的成果

| 成果 | 实测结果 | 证据范围 |
| --- | --- | --- |
| [数据输入变换](research_runs/direct_processing/20261009/REPORT.md) | 三个额外种子平均 NMSE 改善 **4.2385%**，NMAE 退化 **0.0541%** | 现有 MetroPT3 上的受控开发验证，保留全部对照、消融及失败记录；研究助手直接探索 |
| [冻结残差 MLP](research_runs/model_exploration/REPORT.md) | 已冻结测试平均 NMSE 改善 **29.2698%**，NMAE 退化 **0.7691%** | 独立模型成果，检查点和测试记录保留，不能计入数据处理或 Agent 发现收益 |
| [Agent 已知方案工程复现](research_runs/agent_research/capability_acceptance/20261010_v4/REPORT.md) | 单种子 NMSE 改善 **4.082989%**，NMAE 改善 **0.233700%** | 已提供注册配方的工程验收，自主发现、机制及泛化尚未证实 |

数据处理成果的诊断入口为[数据特征与误差报告](research_runs/direct_diagnostics/20261009/REPORT.md)。完整研究路线见[长期规划](research_runs/agent_research/ROADMAP.md)。

## 当前阶段

2026-10-20 当前工作为可扩展科研平台工程完善与滚动数据预测。研究主线仍是“固定预测骨干下，诊断信号学习与诊断—方案分离更新”。[r10 冷启动报告](research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10/REPORT.md)记录 16 次查询、32 次 Sol high 和 51 次训练；双方在 2/4/8 节点均选择父模型，尚无选择增量。

累计完成**515次完整研究训练**。[r11 开放处理](research_runs/agent_research/intervention_diagnosis/20261018_open_processing_r11/INTERRUPTION_REPORT.md)首次调用因传输异常按冻结规则失败并停止，未执行生成程序、未增加研究训练，原始记录及失败成本保持。工程合成示例与测试单列，不能计为科学效果证据。

当前没有新科研训练或测试队列。r11 恢复须另行冻结传输处理与恢复方案，平台实现不重开旧批次。D/test 与已登记存储禁区保持。研究接续以[HANDOFF](research_runs/HANDOFF.md)、[state](research_runs/state.json)和最新阶段报告为准。

## 项目入口

- [平台使用说明](docs/PLATFORM.md)：实验工作流、持续预测、监视、恢复和扩展。
- [下一步](research_runs/NEXT_STEPS.md)：研究执行边界与接续事项。
- [研究概览](research_runs/OVERVIEW.md)：有效成果及尚未证明的结论。
- [冻结经验库 v2](research_runs/agent_research/memory/v2/README.md)：144条可质疑的历史证据；其来源报告仍保留原路径和哈希。
- `online_ts/`、`scripts/`、`tests/`：当前运行组件、有效成果的复现依赖及测试。
- `data/metropt3_v1/`：[不可变基础数据身份](data/metropt3_v1/manifest.json)，[原任务协议](research_runs/PROTOCOL.md)。
- `archives/capability_acceptance_20261010_v4/`：最新成功工程复现的完整冻结资产。

真实研究 Python、API 调用和模型训练在专用4090服务器执行。本地负责编辑、文件整理、同步、审阅及合成工程测试。预处理仅在训练分区拟合，验证真值及困难样本不得删除，旧测试已用过且不能再用于探索或称为未见数据。

## 本次整理

2026-10-11 将5633个历史文件约 **18.3 GB** 移至项目旁的 `../auto_data_history/20261011_204809/`，完整清单为该目录的 `archive_manifest.json`。项目主体由约20.1 GB降至约1.8 GB（不含 Git 历史），原始数据、冻结代码和保留成果哈希未改变。这里是移出工作树，磁盘上的历史证据仍可恢复。

[整理报告](research_runs/cleanup/20261011/REPORT.md)记录路径映射、核验和公开同步范围。旧下载回执中的已迁移本地路径按归档清单解释，不改写旧收据。Git 历史继续保留原有公开内容。
