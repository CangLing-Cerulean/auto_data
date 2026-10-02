# MetroPT3 固定实验基础数据集

已根据 `research.md` 建立 `metropt3_v1`。全部记录来自用户提供的 `share_20260727.zip` 中 v0.7.0 包的原始 MetroPT3 CSV，没有合成、插值、补点、清洗、重采样或数值改写。没有混用不同版本，没有引入问答评测答案。

## 下一位 agent 从这里开始（2026-10-02）

当前目标仍是探索**如何处理训练数据来提升训练效果**，具体延续训练窗口保留、删除、损失权重和采样概率研究。尚未找到通过既定门槛的新筛选方案；默认保留全部合法窗口并均匀采样。最新状态与建议见 [接续指南](research_runs/CONTINUE.md)，完整历史见 [HANDOFF](research_runs/HANDOFF.md)。

数据和研究资产已配套发布到 [handoff-20261002 Release](https://github.com/CangLing-Cerulean/auto_data/releases/tag/handoff-20261002)。只克隆 Git 仓库不会自动下载这些大文件。在项目根目录运行：

```powershell
python scripts/download_handoff.py --include-test
python scripts/verify_dataset.py
python scripts/audit_assets.py
python scripts/verify_selection_metadata.py
python scripts/verify_transfer_assets.py
python scripts/verify_importance_assets.py
```

下载与完整性校验仅需 Python 3.11+ 标准库。脚本按 [资产清单](research_runs/handoff_assets.json) 校验压缩包及每个原文件 SHA256，拒绝覆盖内容不同的现有文件。`--include-test` 用于完整归档恢复和校验；**不授权测试分布探索、调参或评分**。开发用训练服务器可省略该参数，完整三分区校验应在持有测试归档的机器执行。

Release 包括固定 train/validation/test CSV、123 个检查点文件、79 个 NPZ、16 份压缩窗口清单和73份研究日志。原始输入包内其他数据集和问答 ground_truth 不属于本研究，不随接续包分发。三份固定 CSV 已包含全部 MetroPT3 原始行，校验器可以验证其字节级重构源文件的哈希，无须下载整个混合输入包。

## 固定划分

按原始时间顺序划分 **记录条数**，不是按日历时长或预测窗口数。训练取 floor(N×0.7)，验证取 floor(N×0.2)，其余归测试，因此仅有不可避免的整数舍入。

| 分区 | 记录数 | 起始时间 | 结束时间 |
|---|---:|---|---|
| train | 1,061,863 | 2020-02-01 00:00:00 | 2020-06-29 16:18:07 |
| validation | 303,389 | 2020-06-29 16:18:17 | 2020-08-10 21:17:42 |
| test | 151,696 | 2020-08-10 21:17:52 | 2020-09-01 03:59:50 |

文件位于 `data/metropt3_v1/`，每份保留原 CSV 表头与原始数据行字节。总计 1,516,948 条、17 列：无名原始索引列、timestamp 和 15 个传感器。无名列读取时可在内存中命名为 source_index，不得作为模型传感器输入。源时间没有时区标记，不擅自转换时区。

## 探索结论

- 全量结构校验：时间戳和源索引严格递增，无重复时间戳，15 个传感器均无空值、NaN 或无穷值。不存在记录不等于不存在时间缺口。
- 训练段相邻间隔众数为 10 秒，同时存在 9、12、13 秒等间隔，最大间断为 172,918 秒（约 48.03 小时）。未填补缺失时间。后续窗口协议需要处理间断。
- 按随包字段说明，包含 7 个连续信号和 8 个离散信号，完整字段统计见 `train_profile.json`，传感器分布只探索训练段。
- 训练段油温范围 15.4–83.125 °C，电流范围约 0.02–9.295 A；TP2、H1、DV_pressure 存在负值。本次保留这些真实观测，不凭范围判定或删除异常。
- 训练段 LPS 均值约 0.00246，离散状态明显不均衡；不据此改变基础数据构成。

## 使用与重现

Python 3.11 或更高版本，仅使用标准库。在项目根目录运行（不要使用 `python -O`，校验使用 assert）：

```powershell
python scripts/verify_dataset.py
```

在新的空目录中复制代码及用户原始 ZIP 后，可重现：

```powershell
python scripts/extract_source.py
python scripts/build_dataset.py
python scripts/verify_dataset.py
```

构建器拒绝覆盖已有版本。校验器检查各文件 SHA256、行数、时间边界、分区连续性和 7:2:1 比例，并将三个文件的数据行按顺序串联，验证其与原始 CSV 的 SHA256 完全一致，证明无遗漏、无新增、无字节改写。

数据集构建阶段按用户确认未固定输入窗口或预测长度。2026-09-26 用户进一步授权远程训练并确认首轮研究协议，见 `research_runs/PROTOCOL.md`。后续所有实验必须遵守 `AGENTS.md`：固定基础版本，训练参数仅从训练集拟合，验证集用于开发，测试集只用于最终评价。当前测试隔离是项目流程约定，不是操作系统权限隔离。

研究接续从 `research_runs/CONTINUE.md` 和 `research_runs/HANDOFF.md` 开始；实际状态见 `research_runs/state.json`，各阶段汇总与决策见对应目录。完整配置、源码快照、结果和可恢复检查点保存在 `research_runs/results/`；检查点通过 Release 下载恢复。

用户随后指出中心化不等于数据筛选。训练窗口保留、删除、损失降权、采样概率研究独立保存在 `research_runs/selection/`，最新已完成阶段在 `research_runs/cluster_selection/`。各阶段 NPZ/压缩 CSV 是可追溯的筛选元数据，已纳入 Release。

Git 管理代码、文档、固定清单、来源许可和训练结果；CSV、检查点及二进制研究资产由同仓库 Release 分发。`.gitattributes` 禁止自动换行转换，保留已有 manifest、协议和实验快照的字节身份。不会修改历史哈希来适应另一份数据。

## 来源与署名

随包元数据声明：Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). MetroPT-3 Dataset. UCI Machine Learning Repository. DOI: 10.24432/C5VW3R，CC BY 4.0。

随包清单说明完整 CSV 来自 CC BY 4.0 镜像，曾与获取到的 UCI 前缀逐字节比较；本次验证用户提供包的哈希，不将该说明扩大为独立验证了上游全文件。源文件 SHA256 为 `db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24`。

原始来源说明、字段字典和许可保留于 `source/metropt3/` 并纳入 Git。对外分发时保留上述署名、许可和来源说明；随包来源清单提到的其他数据集不包含在此次发布中。
