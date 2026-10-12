# Sol high原始提案记录

来源：内部实验LLM `/root/stage1_candidate_solhigh`，Sol high，独立上下文，一次提案，无外部API。此文由主代理把实际输出整理成结构化叙述；proposal_median3.py的模块代码和candidate_predictions.json的预测/理由按返回内容逐字转存。提案前没有候选训练结果。未强迫输出两项，实际只提出一项。

P1为末端保留的三点中值表示。父表示z的内部30个位置取前/中/后三点中值，首尾不变；控制取0.25前+0.5中+0.25后。两者相同输入支撑、维度、端点及训练预算，比较非线性顺序统计与线性平滑；改动幅度和改动元素数不保证相同。因此它是登记的结构化表示机制控制，不称完全相同剂量的稀疏删除控制。两者全窗口应用，权重与采样均保持均匀。

不足假说是过去输入的短暂折返可能干扰固定线性预测器。提案明确承认，既有NMSE/NMAE及相邻变化汇总不能证明测量噪声，折返也可能是真实状态信号；删除它可能有害。父子增量只比较相对于固定source-half，不能重复归因父方案收益。

目标窗口定义仅用于分组评分：在父表示之前的标准化输入中，任一内部位置/通道的中心值与三点中值的绝对差，严格超过该通道T合法输入窗口一步差的RMS且RMS>0。RMS从全部合法T输入窗口拟合并存float32；不读取每个窗口的未来8条，不用D拟合。目标评分覆盖整窗口全部预测步与通道，补集为精确反集。target/control数组相同表示共同诊断切片，不宣称从非目标中抽取匹配样本。

固定参数：输入32、三点支撑、端点保留、控制系数0.25/0.5/0.25、严格大于T一步差RMS；无阈值/倍率扫描。目标或补集为空时局部指标标记不可计算，不改阈值。非有限、接口不兼容或无法配对时停止并保留失败。

主操作目标NMSE退化超过0.25%反驳本任务上去掉折返有益；不优于控制则不能支持中值额外价值；仅局部改善伴随整体退化不支持采用。基本不变与弃权分开，处理采用仍为整体NMSE≥1%且NMAE退化≤1%。重复性由主代理在响应前另冻结。

受测LLM实际完整阅读：本阶段CANDIDATE_BRIEF、PROPOSAL_API、construction_state；R2 memory/07_REPORT与08_experience_entries（FPR1_O60_counterexample、FPR1_O80_counterexample）；R2 REPORT与experience_entries（R2_known_recipe_reuse、R2_source_sign_increment）；phase0 REPORT、readiness PLAN、旧SKAB REPORT、R2 memory_manifest。对v2/library仅作关键词检索和读取部分匹配摘录，没有逐条阅读144条；不得宣称完整逐条使用或绝对首次发现。未读取新S1/S2响应、baseline_runs、CSV、测试或服务器。
