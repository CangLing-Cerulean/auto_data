# SKAB v1：多传感器预测基础数据

来源为发送包内 SKAB v0.9 的全部 35 个原始实验 CSV，共 46,860 行、8 个传感器通道。身份与边界以 [manifest.json](manifest.json) 为准。归档层原文件在 `data/raw_sources/20261011/skab/`，逐文件保持原始字节与上游 SHA256。

| 分区 | 行数 | 首时间 | 末时间 |
|---|---:|---|---|
| train | 32,802 | 2020-02-08 13:30:47 | 2020-03-09 12:43:11 |
| validation | 9,372 | 2020-03-09 12:43:12 | 2020-03-09 15:28:11 |
| test | 4,686 | 2020-03-09 15:28:12 | 2020-03-09 17:14:09 |

按 `(datetime, source_file, source_row)` 排序后固定切分为 `floor(N*0.7)`、`floor(N*0.2)`、余量。部分原始实验的时间范围重叠；保留全部记录及各自实验身份，三个分区的时间范围严格不相交。原实验可能在边界被切成两个片段，当前划分用于未来时间预测，不代表独立设备或整实验留出。

预测输入只能取 manifest 的 `sensor_columns`。`source_file`、零基 `source_row`、`datetime` 仅用于溯源和窗口边界。原始 `anomaly`、`changepoint` 保留在原文件中，准备层 CSV 不包含它们。原始数值字符串逐个保持；将分号格式转为逗号格式并加入溯源字段，因此准备层 CSV 不宣称与原 CSV 字节相同。

加载后先按分区、`source_file` 分组，再按 `source_row` 排序构窗。禁止跨原文件、跨分区或跨超过 30 秒的相邻间断。不能把整体 CSV 中相邻两行直接当作同一实验的连续观测。`train_profile.json` 只统计训练段；标准化、筛选和采样参数以后仍只能由训练段拟合。

本版本只冻结基础数据，`window_protocol=null`。输入长度、预测长度和训练方案须另行冻结。1Hz 附近的观测与 MetroPT3 的采样时间尺度不同；不能把相同步数当成相同预测秒数或直接混比结果。测试仅在方案冻结后评分。

校验：在项目根目录用 PowerShell 7 运行 `./scripts/build_additional_datasets.ps1 -VerifyOnly`。来源署名见 [LICENSE_AND_CITATION.md](LICENSE_AND_CITATION.md)，上游 GPL-3.0 全文见 [LICENSE](LICENSE)。详细建库验收见 `research_runs/dataset_build/20261011/REPORT.md`。
