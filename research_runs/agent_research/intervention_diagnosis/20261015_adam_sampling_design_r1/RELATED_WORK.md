# 直接近邻与候选贡献边界

2026-10-15。以下为原论文页面/正文核查；没有运行或声称复现这些方法。表述针对核查段落，不声称完成穷尽性创新检索。

| 原始来源 | 已有内容与我们的边界 |
|---|---|
| [Lahire, OPT 2023](https://opt-ml.org/papers/2023/paper20.pdf), §3 | 已分析梯度范数最优采样不能直接推广到Adam/RMSProp。不能把“优化器感知的重要性采样”作为首次贡献。当前候选具体优化开发风险的条件二阶近似。 |
| [AdamBS, NeurIPS 2020](https://arxiv.org/abs/2010.12986) | 已将自适应采样用于Adam。后续理论争议意味着不能照搬其收敛保证，也不能借此忽略其基线价值。 |
| [AdamCB, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6f125214c86439d107ccb58e549e828f-Paper-Conference.pdf), Algorithm 1–3, §4 | 以组合bandit选择不重复批次，并有自己的矩状态更新和理论条件。原算法与本项目有放回、标准偏差校正Adam并不相同。将采样思想适配到固定骨干时须明确改动，不能称作者复现或直接继承证明。其目标为强近邻，不能只比较均匀采样。 |
| [GREATS, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ed165f2ff227cf36c7e3ef88957dadd9-Paper-Conference.pdf), §4.1–4.2 | 已使用开发损失Taylor近似及Hessian交互做在线贪心批次选择。开发风险/二阶信息/在线选择均非首次。当前候选是固定平均训练梯度下的概率分配，且显式包含Adam更新映射的二阶项。 |
| [Filter-then-Weight, arXiv:2604.00001](https://arxiv.org/html/2604.00001), §3与附录B | 已有优化器感知的在线选择与非负重权、目标方向匹配和二阶效用联系。仅“加入Adam状态和目标验证集”不足以支持创新。当前候选不以改变平均梯度匹配目标方向，而以校正采样分布优化局部条件风险；这个区别的实用价值尚未实证。 |
| [Towards Efficient LLMs Annealing with Principled Sample Selection, arXiv:2605.31175](https://arxiv.org/html/2605.31175), §4.1–4.2 | 已按验证Hessian的几何结构构造后期选择/重权，并明确改变训练目标。不能把利用验证曲率保护重要方向称为新思想。 |
| [In-Run Data Shapley for Adam, arXiv:2602.00329v1](https://arxiv.org/html/2602.00329v1), §4 | 已研究Adam下的数据贡献和固定状态的一阶效用。当前目标是无偏梯度约束下优化采样概率，差别仍需理论边界和真实收益支撑。 |

此外，项目历史已运行静态校正梯度范数采样、动态hard/progress采样与同均值S/W对照。它们分别限制“没有旧覆盖”“动态策略必然更好”“均值失明就证明选择缺口”的说法。旧池的收益门槛和关闭决定不变。

本阶段可支持的贡献描述仅为：**一个可执行、可证伪的Adam条件风险采样候选，使用完整复合风险曲率和带符号概率分配。**标准delta展开、链式求导与平方根分配本身不构成新定理。若完整风险与几何/范数规则在真实状态下无法产生可靠且值得成本的决策差异，则应停止该方向的完整训练准备。
