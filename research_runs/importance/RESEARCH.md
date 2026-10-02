# 文献调查与新假设：保持目标的数据呈现分配

用户要求先调查能提高训练成果的方法，再继续真实研究。检索日期2026-09-27，仅引用原论文和作者实现。基础数据、DLinear、32→8、15列、Adam、3000步及NMSE不变，原总预算继续累计。

## 文献与当前问题的对应

- Katharopoulos & Fleuret, ICML2018, [Not All Samples Are Created Equal](https://proceedings.mlr.press/v80/katharopoulos18a.html)，[论文](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)，[作者实现](https://github.com/idiap/importance-sampling)。按梯度范数分配采样并校正单例损失可降低随机梯度方差。论文还强调评分计算代价和随模型变化更新分布。本项目DLinear可直接计算全参数梯度范数，因此不需要另训练评分网络。论文任务与本数据不同，其收益不视为本项目的收益保证。
- Killamsetty et al., ICML2021, [GRAD-MATCH](https://proceedings.mlr.press/v139/killamsetty21a.html)：选择能保留整体梯度的加权子集。它支持“保留训练信号而非按描述性分数删样本”的动机，但本轮不声称复现该算法；百万高度重叠窗口的子集求解与支持丢失使其暂不优先。
- Ren et al., ICML2018, [Learning to Reweight Examples](https://proceedings.mlr.press/v80/ren18a.html)：利用干净无偏参考集学习样本权重。当前内部后续时段存在漂移，此前一次局部方向评分已失败；不能把参考集无偏的前提直接套用，也不将正式验证标签放入权重训练。

## I-D001：评分诊断运行前

Observation：删除/降权/未校正过采样大多改动经验目标；候选压缩也减少状态支持。尚未研究全支持、损失反概率校正的动态采样。固定3000步可能仍受到梯度噪声影响，但这是待验证假设。

Hypothesis：按当前模型的单窗口梯度范数分配采样，保留所有窗口并补偿抽样概率，可以更有效地用相同768000次反向传播呈现优化原目标。

Action：先A004，只读原train.csv与S000固定3000步latest.pt及训练尺度；计算每个原合法窗口全参数MSE梯度范数。对真实样本用autograd独立核验解析范数。固定q_i=0.5/N+0.5*g_i/sum(g)，w_i=1/(N*q_i)，无样本删除，权重上界2。核对q_i*w_i=1/N，并比较均匀与校正采样的梯度二阶矩（明确不是Adam或验证提升保证）。评分诊断成本计入原预算，源行和分数保存。

推进：解析校验通过且该固定分布在已有训练终点将梯度二阶矩降低至少20%，才做一次2026种子正式试验。否则依据结果重新判断，不随意扫混合强度。训练方案将在诊断后另行登记。此次方向由用户新提出的文献调查与此前目标偏移证据驱动，不是重试已停止的梯度符号降权或去重间隔。
