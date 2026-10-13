# SOURCE比较边界的定向核对

2026-10-13，在本轮预测运行期间只读核对；不修改冻结实验、候选或门槛，不安装或执行第三方代码。

SOURCE使用分段梯度/曲率统计近似训练展开；原文附录C给出预条件扩展，附录D.2说明AdamW矩阵指数近似。该工作已覆盖非收敛轨迹及训练数据反事实，因此“考虑优化器或中间轨迹”本身不能作为本项目的新颖性。出处：[NeurIPS 2024原文§3、附录C/D](https://papers.neurips.cc/paper_files/paper/2024/file/7af60ccb99c7a434a0d9d9c1fb00ca94-Paper-Conference.pdf)。

作者参考实现src/source.py的说明及代码显示：当前SourceComputer按SGD形式计算；Adam感知版本未实现；全局1/N因子省略；动量学习率缩放由调用方提供。它接收分段检查点、迭代数和学习率，并返回查询—训练点分数。出处：[固定提交源文件](https://github.com/pomonam/simple-influence/blob/2e4e8f04752f89fabd55db2f6a6ba278e3dee4d3/src/source.py)。提交和文件SHA见source_reference_receipt.json。

本项目的判断：直接以lr=.001调用该实现不足以形成匹配原Adam训练语义的强基线。后续适配须先冻结以下边界，再作数值与成本验收：

- 使用相同父轨迹与A/B/C信息；不能额外读取完整子响应来设分段或阻尼。
- 明确原Adam状态的处理、概率变化的归一化，以及从单点归因到有限群组变化的符号和尺度。
- 沿同一个B检查点选择器输出C净响应；只给固定终点排名不能替代本项目的最终决策目标。
- 计入曲率、各检查点梯度与评分成本；与M1和真实短训使用同一共同父成本。

这些是待验收的适配要求，不是已实现的SOURCE对照，也不证明M1优于SOURCE。本轮仍按已冻结的RAW/FIXED/SHORT矩阵完成，不把这份后补资料写成新增基线的事前协议。
