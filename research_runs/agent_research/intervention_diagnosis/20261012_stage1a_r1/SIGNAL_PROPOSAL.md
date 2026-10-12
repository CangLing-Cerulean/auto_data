# Sol high候选：相对窗口均值漂移

来源为独立内部Sol high `stage1_signal_solhigh`，一次提案，无外部API。以下文字由主代理将实际回复结构化记录；`signal_mean_shift.py`模块按实际回复逐字转存。提出时已知C构建响应、未知该特征的C/S数值，不属于对C响应的事前预测。

定义：对每个窗口与通道，计算changed_x与parent_x在32条历史输入上的均值差；对该差平方后在窗口/通道上取平均为D。信号s=D/(D+初始固定父整体NMSE)，D与NMSE同时为0时定义s=0。代码用比值分支避免加法溢出，返回[0,1]有限数。raw_x仅作形状与有限性校验；所有通道同等处理。复杂度O(NLC)、辅助空间O(256LC)，无拟合参数，模块低于100个非空非注释行。

线性控制CP1在精确实数算术下，窗口均值变化为(x0-x1-x30+x31)/(4×32)，内部变化大多相消；实际float32计算还包含舍入误差。P1中值没有同样的面积抵消性质，重复同向修正可能积累窗口均值漂移。父NMSE作为相同标准化尺度下的误差参照。这是表示扰动风险的待检验假说，没有证明折返为噪声，也没有证明均值漂移决定重新训练后的效果。

提案的固定预期顺序为：两个CP1位于低剂量区，C1_P1居中，C2_P1较高；对应候选类别顺序unchanged→improve→harm。本次不给数值阈值。中间区可能改善来自C反馈，不能从信号定义直接推出。若四个C值不符合该排序，或两个有序阈值不能分开稳定类别，应返回null，不改公式或扫描别的特征。C拟合成功也只算构建拟合，后续必须事前锁定S预测。

支持提出该候选的证据是初始与三个额外种子C1_P1小幅改善、C2_P1退化、CP1基本不变；C2多个通道一步变化更大且父NMSE更低。旧证据`R2_source_sign_increment`、`FPR1_O60_counterexample`、`FPR1_O80_counterexample`提醒相关性、覆盖和局部收益不能替代增量效果。该候选不能据此晋升采用。

局限：均值相消的时序形状变化、逐通道误差差异与未来关联被忽略；DLinear重新训练可能补偿均值改变；小漂移不保证unchanged。低剂量稳定伤害、高剂量稳定改善、或相同剂量不同响应均削弱解释。只有两个同源构建任务，重复种子不是独立任务；旧SKAB汇总已知，后续S并非完全未见来源。

LLM报告实际完整读取19份：本阶段brief、boundary、activation、construction_state、CANDIDATE_PROPOSAL、proposal_median3、candidate_predictions、construction_feedback、construction_confirmation_recovered、CANDIDATE_PROTOCOL、CONFIRMATION_RECOVERY_PROTOCOL；旧R2的memory_manifest、REPORT、experience_entries、memory/07_REPORT、memory/08_experience_entries（history_extra_7/8）；旧phase0 REPORT、readiness PLAN、SKAB先导REPORT。未读取v2/library的144条索引或其余8份memory原文，没有仅检索文件；未读取S新材料、CSV、测试、服务器或网络。可用输入清单不等于实际阅读清单。
