# 相近方法的协议与源码审查

2026-10-13。这是定向审查，不是穷尽检索或复现实验。第三方论文与源码仅本地阅读，不安装、不执行、不随本阶段再发布。文件身份见references.json。

## FAF / AdaRho

[正式论文，AISTATS 2026](https://proceedings.mlr.press/v300/taga26a.html)，核对正式PDF第1–5、7–9页。AdaRho按目标模型与参考模型的损失差选择训练样本，并在线更新参考模型；FAF叠加增强。参考模型使用训练标签，目标与参考更新的样本排名区间不同。论文实验以验证集选择保留比例，并使用预训练参考模型；完整FAF报告约两倍单epoch成本，不能只按目标模型步数计成本。其预测阶段不增加过滤开销。论文页和本次公开检索未定位到可确认的官方实现，因此源码核验标记为未完成，不能声称已复现。

本项目的适配判断：AdaRho属于必须比较的模型感知数据选择路线；历史RHO/AdaRho实验不能改名成为新贡献。应分别登记参考模型预训练、在线更新、保留比例选择和最终训练成本。完整FAF的增强与当前固定输入处理协议范围不同，须分别报告忠实复现与受限适配；当前没有启动任何一种。

## TSAA

[论文](https://arxiv.org/abs/2405.00319)与[官方仓库](https://github.com/azencot-group/TSAA)。源码锁定`aa03b0a5a576aebbdbddaa285127ce1bbb107aa9`，以下链接均绑定该提交。

|核对点|源码事实与出处|本项目比较的要求|
|---|---|---|
|搜索与目标|[policy_search.py L405–428、L460–505](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/autoaugment/policy_search.py#L405)：可配置TPE/随机搜索与SuccessiveHalving；目标为末次验证损失|统一验证查询与训练预算，保留简单搜索对照|
|恢复训练|[policy_search.py L95–155](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/autoaugment/policy_search.py#L95)和[loggers.py L188–215](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/autoaugment/loggers.py#L188)：从基线中间检查点继续训练|区别于从头完整训练；共享预训练和试验成本均入账|
|增强对象|[exp_main.py L246–320](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/exp/exp_main.py#L246)：拼接输入与未来训练目标后增强，再拆回两者|当前协议保持目标不变，直接搬运会改变研究问题|
|测试接触|[exp_main.py L338–367](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/exp/exp_main.py#L338)：每epoch计算并记录测试损失；early stopping用验证损失|使用前必须移除开发期测试加载/评估；现有代码事实不能据此推断测试参与自动选择|
|剪枝|[loggers.py L120–128](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/autoaugment/loggers.py#L120)：向trial报告验证损失|剪枝成本与未完成trial必须保留|
|最终策略及推理|[policy_search.py L430–455](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/autoaugment/policy_search.py#L430)保留优于基线的策略；[exp_main.py L389–447](https://github.com/azencot-group/TSAA/blob/aa03b0a5a576aebbdbddaa285127ce1bbb107aa9/exp/exp_main.py#L389)普通test路径直接预测|本项目对开发输入也作处理，需明确训练增强与部署处理的差别|

## 对下一方法的约束

“自动选择处理”已不足以构成创新。待论证的区别应是：能否在完整子训练前，估计固定模型响应及训练适应造成的收益变化，并在独立任务中提升选择质量和成本效率。基本平方误差恒等式、模型投影、TPE、影响函数或已有RHO方法本身均不算新贡献。

初步审查中的[Datamodels](https://arxiv.org/abs/2202.00622)、[组影响函数](https://proceedings.neurips.cc/paper/2019/file/a78482ce76496fcf49085f2190e675b4-Paper.pdf)和[影响函数局限](https://deepmind.google/research/publications/56635/)仍作为后续创新查重边界。本轮未完成它们的全面源码复现。候选设计需明确与低预算直接验证、影响近似和历史梯度冲突方法的差异；若只是重新命名已有量，就不进入新训练。
