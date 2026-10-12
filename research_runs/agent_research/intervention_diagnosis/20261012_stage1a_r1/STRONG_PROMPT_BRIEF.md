# 共同选择池：独立强提示词事前预测

你是四条件中的独立strong_prompt比较组。固定DLinear、父状态、两种输入处理P1/CP1、相同训练和完整D。请预测S1_P1、S1_CP1、S2_P1、S2_CP1相对于各自S父B0的整体NMSE增量类别，并给每任务P1>CP1、CP1>P1或tie排序。尚无任何S候选响应。C1/C2来自MetroPT3，S1/S2是SKAB整文件隔离开发组；旧SKAB历史已知，不能称完全未见来源。

类别：相对NMSE改善>0.25%为improve，<-0.25%为harm，闭区间为unchanged；uncertain是主动弃权，不是近零变化。处理采用另需NMSE改善≥1%、NMAE改善≥-1%，不把这两个问题混同。主评分仅四格整体，不评局部推测。统一分母同时报告正确、错误、弃权和覆盖率；不要为提高条件准确率隐藏弃权。可以依据证据主动弃权。排序实测差距≤0.25个百分点时不可辨别。所有预测在任何S候选训练前锁定。

首先完整读本目录selection_common_facts.json，含相同T输入样本的通用统计、父D指标、完整C三种子响应与失败标注。可读本目录CANDIDATE_PROPOSAL.md、proposal_median3.py、CANDIDATE_PROTOCOL.md、construction_state.json、candidate_predictions.json、construction_feedback.json、construction_confirmation_recovered.json、CONFIRMATION_RECOVERY_PROTOCOL.md。不得读本目录任何其他文件，尤其SIGNAL*、signal_*、selection_*其他文件、源代码比较方法、S候选响应。不得读取原始数据、测试、SSH或运行实验。

允许的既有开发历史：research_runs/agent_research/memory/v2/library.json；research_runs/agent_research/autonomous_memory/20261012_sol_high_r2/memory_manifest.json及其memory/所列10份原文、REPORT.md与experience_entries.json；research_runs/agent_research/intervention_diagnosis/20261012_phase0/REPORT.md、20261012_stage1_readiness/PLAN.md、20261012_skab_pilot/REPORT.md。原始索引不是独立任务数。记录实际全文读过与仅检索的文件，不虚报。

你有与其他条件相同的T原始输入统计计算权限：如果确有必要，在预测前可返回至多一个纯NumPy/math标量函数statistic(raw_x,parent_x,changed_x,parent_metrics)，≤100非空非注释行，无I/O、模型、随机搜索或任务/动作/数据集/文件/通道身份识别，所有通道同规则。输入是各C/S合法T起点等间隔最多8192个32行窗口，parent_metrics只含初始父NMSE、NMAE、signed_bias、channel_NMSE、channel_NMAE。主代理计算相同C/S值后允许你一次完成预测。不得请求候选D反馈。也可以直接依据共同事实输出最终预测。

返回JSON：predictions映射四格类别；ranking映射S1/S2上述排序；reasons给逐格理由；optional_statistic_requested布尔；files_read_completely与files_searched数组；limitations数组。如请求统计先只返回代码和依据，不提前输出最终预测。不要写文件、派生子代理或发送外部消息。所有研究LLM为Sol high。
