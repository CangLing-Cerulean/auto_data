# 作者实现的Adam边界

2026-10-14定向只读调查。固定[SourceComputer源码](https://github.com/pomonam/simple-influence/blob/2e4e8f04752f89fabd55db2f6a6ba278e3dee4d3/src/source.py#L31-L37)明确SGD形式及未实现Adam-aware变体；动量学习率缩放说明针对SGDm。当前main仍保留同一边界。其分段平均针对曲率、校正特征值与训练梯度，检查点只加载模型状态，不读取Adam第二矩。

调查还覆盖simple-influence当前主分支50份代码/文档、Kronfluence当前主分支144份代码/文档及作者入口。[Kronfluence v1.0.1说明](https://github.com/pomonam/kronfluence/releases/tag/v1.0.1)将SOURCE集成列作后续计划。本次未找到比论文附录更明确的公开Adam预条件器、bias correction或第一矩处理代码；未穷尽其他分支、历史和非公开实现，不能据此断言作者从未发布。

因此本工程的右端点统计、由已保存v构造带偏差校正的P以及不跟踪m/v扰动，均明确登记为项目适配假设。不能在论文中把它标为作者Adam实现的完全复现，也不能把Adam或轨迹响应本身作为本项目独有创新。

数值接口依据：[PyTorch Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)；独立矩阵指数参考使用[torch.linalg.matrix_exp](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html)。服务器现有Python环境未安装SciPy，工程沿用现有NumPy/PyTorch，没有为此安装依赖。
