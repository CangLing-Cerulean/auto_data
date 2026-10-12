# 当前研究概览

最新：[阶段1a完整确认](agent_research/intervention_diagnosis/20261012_stage1a_r1/CONFIRMATION_REPORT.md)。恢复补9项成功，累计成功264，另保留1失败交付。18格审计通过，中值在C1平均改善0.356%、C2平均退化2.327%，线性控制基本不变，稳定差异门槛通过、采用门槛无通过项。下一步仅构建单个C-only新信号，不提前宣称方法优势；旧中断记录保留。

当前已停止：[阶段1a确认中断](agent_research/intervention_diagnosis/20261012_stage1a_r1/CONFIRMATION_INTERRUPTION_REPORT.md)。8项成功、1项保存结果后输出失败、9项未启动，累计成功255。C1中值三个额外种子改善约0.36%，未达1%采用门槛；完整确认与稳定差异仍未评估。无新信号、S候选或方法优势结论。九组资产通过补审计，失败状态保留，无后台队列。

最新：[阶段1a初始响应](agent_research/intervention_diagnosis/20261012_stage1a_r1/INITIAL_REPORT.md)再完成4次，累计247；中值在C1小幅改善、C2退化，线性控制均基本不变。没有达到采用门槛的操作，事前响应类别差异门槛通过，将进行三种子重复确认。该进展不等于新诊断信号或方法优势已成立。

当前：[阶段1a固定参照](agent_research/intervention_diagnosis/20261012_stage1a_r1/BASELINE_REPORT.md)已新增6次完整训练，累计243；四任务、四类别评分和配对审计通过。历史source-half在两个构建组上的单种子NMSE改善3.1506%/4.7037%，尚无新候选结果。全阶段执行已获授权，研究LLM使用Sol high，当前候选构建进行中；阶段同步见新目录progress/receipt。以下237次与暂无新训练是上一阶段快照。

最新口径补充：SKAB先导的6/7、4/4是原始符号统计；[补充审计](agent_research/intervention_diagnosis/20261012_stage1_readiness/PLAN.md)说明它与阶段0±0.25%不变带的差别。全部八个整体响应均在不变带内，当前优先登记更多合法开发任务以检验响应是否可辨别，再学习新信号。原始结果、B0选择及237次计数不变，暂无新训练队列。

主课题为“固定预测骨干下，面向数据干预的不足诊断与反馈决策学习”。近期检验反例能否帮助发现有预测价值的干预前信号，以及事前响应预测与父子配对反馈能否减少错误归因；处理效果与假设支持分别更新。

最新[SKAB共同响应先导](agent_research/intervention_diagnosis/20261012_skab_pilot/REPORT.md)完成9次固定DLinear训练，累计237次。最大整体NMSE改善0.1144%，八候选均未达推进门槛；strong_prompt与full全程保留B0。明确方向命中分别4/4和6/7，覆盖4/24与7/24；full历史规则为null，尚无方法优势、规则迁移或反馈因果收益证据。[阶段0真实运行](agent_research/intervention_diagnosis/20261012_runtime_r1/REPORT.md)已完成9项测试、9检查点及6组父子逐元素审计。旧工作复用价值见[专项审查](agent_research/intervention_diagnosis/20261012_phase0/VALUE_REVIEW.md)，早期仅本地交付保持历史快照。

## 已确认的成果

**数据处理主线**：[输入变换研究](direct_processing/20261009/REPORT.md)在固定DLinear上完成25次训练、两个内部原点、三个额外种子和对照消融，平均NMSE改善4.2385%，NMAE退化0.0541%。这是研究助手直接探索后的受控开发结果；没有证明新数据泛化或Agent自主发现。

**独立模型成果**：[残差MLP](model_exploration/REPORT.md)已有冻结测试结果，平均NMSE改善29.2698%，NMAE退化0.7691%。保留模型、检查点、基线和冻结文件，不把收益计入数据处理，不继续利用该测试调参。

**Agent工程成果**：[最新已知方案验收](agent_research/capability_acceptance/20261010_v4/REPORT.md)成功复现注册配方，单种子NMSE/NMAE改善4.082989%/0.233700%；历史判断库12/12通过，继承自v3。它证明接口和执行链可工作，科学信号和自主发现仍未建立。

## 最近完成与待验收

[Sol high自主探索R2](agent_research/autonomous_memory/20261012_sol_high_r2/REPORT.md)复用完整历史经验，自主诊断并提出一个符号处理候选。三个额外种子确认NMSE相对B0改善3.9409%、NMAE改善0.3616%，但相对已知全正号source-half对照，NMSE退化0.3842%、登记局部退化13.1094%。整体有效性E通过，增量控制C失败，未建立新增自主可操作信号。新增11次完整训练和1次保留失败，累计228次；无新测试、外部API0、无活动队列，内部LLM费用未知。监控原子重命名缺陷已通过7项回归修复，恢复仅补齐缺失对照。

[阶段4 signal v2](agent_research/phase4/20261010_signal_v2/protocol.json)比较有经验有反馈、无经验、无效果反馈，分别评分F事实、P预测不足、L局部命中、C匹配控制、E整体效果、U反馈更新。每组3次搜索只提供首批方向性证据。当前无限累计预算，不能宣称相同探索成本优势。

恢复连接后已核验[signal v2机器完成快照](agent_research/fused_pilot/20261011_sol_high_r1/prior_signal_status.json)：6次保留B0、3次搜索失败，队列关闭，未确认处理收益；语义验收仍待补齐。其完整机器账本为9次训练、1次失败，不能把程序结束当成科学成功。

用户随后授权完成[signal×E1融合先导第一轮](agent_research/fused_pilot/20261011_sol_high_r1/REPORT.md)，以Sol high子代理作LLM。两个同源内部任务、五个冻结动作，共10次训练；增权/降权均未达到整体收益门槛，最终仍为B0，选择收益0。已保存事前决策、两次反馈、冻结推荐、完整匹配控制与源行映射。没有运行独立反馈屏蔽分支，尚未证明反馈增益或迁移；DeepSeek调用0，继续关闭。当前累计217次完整训练、2544次API尝试、14389704个已知tokens，未知用量另列。

## 尚未解决

- 平台失败与科学负结果的分离、候选控制的语义有效性、全部搜索与额外种子确认的最终审查。
- 经验及效果反馈是否提高信号发现率，代价是否合理；首批重复数和不等探索量限制结论。
- 独立任务和未参与选择的数据确认。现有数据可继续开发验证，旧测试不能重新称为未见数据。
- 更广的超参数适配取值范围尚未另行冻结，本批保持固定训练参数。

接续见[HANDOFF](HANDOFF.md)与[下一步](NEXT_STEPS.md)。已停止的探索和失败主体移出工作树，经验库原文及必要基线保留；整理不改变旧结论或成本。
