# 单个候选信号构建任务

本说明只在`construction_confirmation_recovered.json`的construction_gate_passed为true且主代理完成验收后启用。当前若没有该报告，不得调用构建LLM或计算候选特征。

研究问题：固定DLinear与固定P1中值/CP1线性处理下，干预前可计算的状态能否预测增量响应类别。允许提出一个尚待隔离开发任务检验的候选，不要求仅用两个构建组证明泛化；也允许明确null。当前处理是否值得采用与响应是否可预测是两个问题。不要把文字解释当作数值信号。

你只读构建组C1/C2材料与既有公开开发历史，不读取本阶段S1/S2的新统计、父响应或候选响应，不读取原始CSV、测试或服务器。两个构建组来自同一来源，重复种子不增加独立任务数。旧SKAB汇总已存在于历史中，不能声称后续S是完全未见来源。

本阶段允许文件：`construction_state.json`、`CANDIDATE_PROPOSAL.md`、`proposal_median3.py`、`candidate_predictions.json`、`construction_feedback.json`、`construction_confirmation_recovered.json`、`CANDIDATE_PROTOCOL.md`、`CONFIRMATION_RECOVERY_PROTOCOL.md`、`SIGNAL_BUILD_BOUNDARY.md`及本说明。禁止浏览本阶段baseline_runs、candidate_runs、candidate_preparation_receipt或其他含S统计的汇总。C1保存失败交付已独立验证计算与资产，原failed状态保留；以新确认报告的明确标注为准。

可用旧开发历史：

- `research_runs/agent_research/memory/v2/library.json`（144条索引，含重复摘要，不是144个独立任务）。
- `research_runs/agent_research/autonomous_memory/20261012_sol_high_r2/memory_manifest.json`及其memory/所列10份原文；该目录REPORT.md和experience_entries.json。
- `research_runs/agent_research/intervention_diagnosis/20261012_phase0/REPORT.md`、`20261012_stage1_readiness/PLAN.md`、`20261012_skab_pilot/REPORT.md`。

如报告引用上述范围以外的新文件，先返回主代理，不自行扩大权限。记录实际阅读了哪些文件和证据ID，检索摘录不能写成全文读过。

输出最多一个纯NumPy/math标量函数`signal_value(raw_x, parent_x, changed_x, parent_metrics)`，准确接口与复杂度见SIGNAL_BUILD_BOUNDARY。至多100个非空非注释行（含辅助函数），不得I/O、调用模型、随机搜索、访问新候选D响应、识别任务/数据集/文件/通道名/通道数量身份，不传action ID。通道需使用同一种统计规则处理，不选择特定编号通道。父指标只含初始固定父的NMSE、NMAE、signed_bias、channel_NMSE、channel_NMAE；特征不得用候选反馈作为运行期输入。不得改变处理、控制、目标、训练或阈值带宽。

第一轮只提出固定特征代码、定义、依据、预期排序/解释、否证条件和局限，不给基于未计算特征值的数值阈值。主代理做合法性与工程验收并锁代码后，仅计算C1/C2×P1/CP1四个特征值；随后同一上下文允许一次拟合至多两个有序阈值和三个区间类别（improve/unchanged/harm/uncertain），或返回null。所有边界归属明确；阈值只能用C响应，不改变特征、不补扫多个公式、不查看S值选阈值。构建集拟合不算预测成功，后续S决策必须事前锁定。

返回：status为proposed或null；若proposed，给逐字可执行模块代码及上述说明；若null，给具体缺口。不要写文件、不派生代理、不调用外部API、不启动任何实验。主代理负责方案验收与整合。
