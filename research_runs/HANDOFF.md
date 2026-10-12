# 研究接续：2026-10-12

最新用户要求复用完整历史经验并由 Sol high 自行探索，已明确确认开始。新 [R2 协议](agent_research/autonomous_memory/20261012_sol_high_r2/PROTOCOL.md)取消预列候选，使用冻结144条历史库及10份后续补充证据。离线决策执行器已本地准备，历史研究子代理正在阅读证据。远程上传/准备被自动审批拒绝；提交既有开机与继续研究授权后仍拒绝，要求拒绝后的明确确认。当前等待用户答复，不得绕过；服务器新目录尚未创建、R2新增训练0、外部API0。累计训练仍217。原R1完成和同步状态保持，旧signal语义审查单独待补齐。

当前主线是数据性能诊断与处理。用户要求融合signal与E1，并先用Sol high子代理替代DeepSeek。已完成[融合先导第一轮](agent_research/fused_pilot/20261011_sol_high_r1/REPORT.md)：新增10次训练、0次API、0次失败，两个任务最终均保留B0，未达到多种子确认门槛。累计217次完整训练；本轮关闭，无后台队列，DeepSeek继续关闭。

## 当前批次

- 机器协议：[protocol.json](agent_research/phase4/20261010_signal_v2/protocol.json)。保持冻结代码、评分门槛、参数及运行ID。
- 服务器项目：`/tmp/auto_data_signal_project_20261010_v2`。
- 服务器输出：`/tmp/auto_data_signal_study_20261010_v2`。
- 前缀：`p4signal2`；启动 PID：79401。PID仅为启动收据，接续时需核实当前进程及输出。
- 三组 `full/no_memory/no_feedback`，各3次独立上下文。Agent自行编写诊断、处理和匹配控制代码，不额外注入近期有效配方。原 DLinear7920参数，3000步，batch256、lr0.001、weight_decay0、kernel25。
- [最后本地快照](agent_research/phase4/20261010_signal_v2/progress_snapshot.json)：2026-10-10T15:09:26Z；3次启动、1次B0结束、1次平台失败，另1次当时运行。没有已确认的自主正确信号或处理收益。
- 2026-10-11用户确认服务器开机后已恢复只读核验：[机器完成快照](agent_research/fused_pilot/20261011_sol_high_r1/prior_signal_status.json)。9个槽位均结束，6个保留B0、3个搜索失败，队列与进程均为空。旧批次没有已确认处理收益，语义审查仍待补齐；不重启控制器或重复旧ID。

## 计数与故障

v2批次前固定基数为198次完整训练、1053次API尝试、3464696个实际已知tokens。完整机器账本新增9次完整训练、1次失败、1491次API尝试、10925008个已知tokens；本轮前为207/2544/14389704。融合先导再增加10次训练，当前为217/2544/14389704。接续按固定基数核算，不能把完整批次再次加到已有快照上。

未知用量累计40条、347757个预约tokens（旧38条/293757，加v2的2条/54000），保持分列。融合先导训练进程665.715746秒，内部Sol high会话费用未知。累计资源无上限，单作业契约仍生效；原134次历史训练和旧计时未重置。

`full_r1`因候选处理代码 `ValueError: 12 is not in list` 后选择B0结束，不能计为科学成功。`no_memory_r1`基线遇到 `NATIVE_OUTPUT_TYPE`，文件事后正常，原因尚未复现；保留为平台失败，不能归为诊断错误或免费替换。v1平台错误的[关闭报告](agent_research/phase4/20261010_signal/CLOSURE_REPORT.md)保留，仅作故障证据。

## 保留资产与边界

[直接处理报告](direct_processing/20261009/REPORT.md)、[诊断](direct_diagnostics/20261009/REPORT.md)、[MLP冻结测试报告](model_exploration/REPORT.md)、[最新工程验收](agent_research/capability_acceptance/20261010_v4/REPORT.md)均保留完整依赖。经验库144条及其8份原文出处保持字节身份，不把旧解释当作正确信号标签。

`data/metropt3_v1/`、原 `protocol.json`、核心训练代码不变。研究Python只在4090服务器运行；实验前按数据可用范围执行校验。服务器仅T/D时不能宣称完整三分区校验。禁止利用已评估的旧测试继续选择方案。

历史训练资产、旧协议、失败追踪和原始发送包已移至 `C:/Users/31227/Desktop/auto_data_history/20261011_204809/files/`，相对路径不变；原入口文档和旧state另存该归档的 `metadata_before_cleanup/`。恢复映射与SHA256在 `archive_manifest.json`。历史ID仍被占用，不因工作树目录移走而允许重跑。

先执行[下一步](NEXT_STEPS.md)。每个阶段验收后同步现有GitHub；公开范围沿用[GITHUB_SYNC.md](GITHUB_SYNC.md)，原始调用、追踪和完整私有归档不随清理自动公开。

2026-10-11新增数据建库：用户另行要求从data发送包挑选一到两个数据集。已建立 `data/skab_v1/`（46860行、8通道）与 `data/uci_hydraulic_v1/`（132300行、8通道），详见[建库报告](dataset_build/20261011/REPORT.md)。仅建库和完整性验收，没有启动新数据实验；原有MetroPT3冻结研究和账本保持。新任务须按各自manifest中的原实验/周期边界与原生时间尺度另行冻结。同步状态见建库阶段收据。
