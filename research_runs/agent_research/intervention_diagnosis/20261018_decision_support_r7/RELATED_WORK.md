# 定向近邻核对：共享响应与模块化更新

此为只读定向检索，不是穷尽性新颖性判断，也不将下列方法变成新的研究目标。

- Carmona & Nicholls (2020), *Semi-Modular Inference: enhanced learning in multi-modular models by tempering the influence of components*：[作者论文，§2.1、§3–4](https://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf)。Cut改变共享参数接收哪些模块的信息；半模块化推断调节可疑模块影响。分开更新或单向信息流本身已有先例。
- Plummer (2015), *Cuts in Bayesian graphical models*：[出版方](https://link.springer.com/article/10.1007/s11222-014-9503-z)。本次只核到出版方摘要，指出部分cut MCMC未必有定义良好的极限分布。不能将任意冻结消息/权重称为有理论保证的cut后验。
- Bonilla, Chai & Williams, *Multi-task Gaussian Process Prediction*，NIPS 2007：[论文，§2及§2.3](https://papers.nips.cc/paper/3189-multi-task-gaussian-process-prediction.pdf)。跨输出协方差可让辅助输出影响主输出；无噪声共同输入的可分离核也可能发生迁移抵消，因此非零相关不保证新增价值。
- Boyle & Frean, *Dependent Gaussian Processes*，NIPS 2004：[论文，§2–3](https://proceedings.neurips.cc/paper/2004/file/59eb5dd36914c29b299c84b7ddaf08ec-Paper.pdf)。共享潜在驱动加输出私有项构建跨输出依赖。该分解本身不是新方法。
- Álvarez, Luengo & Lawrence (2009), *Latent Force Models*：[作者论文及出版信息](https://proceedings.mlr.press/v5/alvarez09a.html)。机制方程与GP潜驱动力共同构造输出关系，说明“引入机制约束共享参数”也已有直接近邻。

对同一先验和联合分布，先用p(Y)再用p(Z|Y)与一次用p(Y,Z)等价。只用条件因子更新诊断是不同规则，但不自动构成规范cut后验。当前实现的诊断条件评分只是一种预测证据累计，保持这一有限表述。

这些文献提示需要明确算法接口和经验增量；它们不能为当前方法补出不存在的真实收益。此次检索不修改原数据、模型或已完成实验。
