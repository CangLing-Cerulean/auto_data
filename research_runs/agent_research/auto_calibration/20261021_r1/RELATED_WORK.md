# 首轮直接参照

核对日期：2026-10-21。以下是首轮实现定位，尚未完成系统性创新性审查。

[COSA官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/2a8ce71baac4c89bf9ff479d8240c7d9-Paper-Conference.pdf)描述冻结骨干、带门控的输出残差修正及依据已观测真值更新适配器。

官方当前代码固定提交为[527c0feb9e997dd85af485ee027616b446e4ae77](https://github.com/bigbases/COSA_ICLR2026/commit/527c0feb9e997dd85af485ee027616b446e4ae77)，新增POGT和MLP。最初提交43a8c8da4de74d5745a8713f6130c523b7df2694与当前提交的标签时序不同。

- [核心网络](https://github.com/bigbases/COSA_ICLR2026/blob/527c0feb9e997dd85af485ee027616b446e4ae77/tta/cosa.py#L19-L85)：逐变量或共享Linear(L+K,L)，tanh标量门，W Xavier gain0.1、b=0、gate=0。
- [上下文与成熟处理](https://github.com/bigbases/COSA_ICLR2026/blob/527c0feb9e997dd85af485ee027616b446e4ae77/tta/cosa.py#L444-L475)：官方上下文来自历史已成熟批次真值的全局均值；当前代码保存预测时上下文并处理延迟完整标签及部分标签。
- [预测替换](https://github.com/bigbases/COSA_ICLR2026/blob/527c0feb9e997dd85af485ee027616b446e4ae77/tta/cosa.py#L553-L573)：当前代码包含部分适应后的预测修订。因此复现实验必须明确原点发出预测与批末修订预测的区别。

本项目从上述公开结构独立实现固定网络，未拷贝官方源文件。上下文使用每通道输入均值/末值/标准差，偏置置于门外以保持零输入下偏移可学习；训练是F拟合、V选择、C后续评分。无预测修订、无部分未来标签、无官方自适应学习率/周期批大小。这是受COSA启发的结构对照，不能声称复现论文全部算法或作者报告收益。

官方仓库根LICENSE为CC BY-NC-SA 4.0，README的MIT描述与其不一致；未将该仓库登记为普通MIT依赖，后续如引入其源码须核清授权条件。

[PETSA](https://arxiv.org/abs/2506.23424)也研究小型校准模块，并包含鲁棒、频域与局部结构损失。自动训练研究不能将采用小适配器、门控或改损失本身视为新增贡献；需要验证Agent对训练配方选择的额外作用。
