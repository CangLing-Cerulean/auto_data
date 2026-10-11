# UCI Hydraulic v1：8通道周期内预测基础数据

来源为发送包内 8 个 1Hz/虚拟传感器原始文件：CE、CP、SE、TS1、TS2、TS3、TS4、VS1。每个文件是 2,205 个周期 × 每周期 60 个样本，按同一周期与采样位置对齐为 132,300 行。当前包缺少更高采样率的压力、功率和流量原始文件，本版本只覆盖现有 8 通道。身份与边界见 [manifest.json](manifest.json)。原始字节副本在 `data/raw_sources/20261011/hydraulic/`。

| 分区 | 行数 | 首位置：零基周期:秒偏移 | 末位置 |
|---|---:|---|---|
| train | 92,610 | 000000:00 | 001543:29 |
| validation | 26,460 | 001543:30 | 001984:29 |
| test | 13,230 | 001984:30 | 002204:59 |

依原文件周期顺序、周期内采样顺序展开，固定 `floor(N*0.7)`、`floor(N*0.2)`、余量。原矩阵的每个数值字符串逐个保留，逐行可通过 `cycle_id` 与 `sample_offset_seconds` 映射回各传感器文件的原始行和列。没有插值、补值、归一化、重采样、删除或合成传感器读数。准备层格式改变，原始文件另行保持字节相同。

`cycle_id` 和 `sample_offset_seconds` 仅作溯源和窗口约束，不作为传感器。没有创建绝对时间戳或假设周期间连续采样。`profile.txt` 中的工况/状态标签未提取或读取为预测输入。

按分区和 `cycle_id` 构窗，禁止跨周期或跨分区。精确行数切分会将周期 1543、1984 各分成两个 30 秒片段；片段全部保留，后续不能跨分区补足历史。完整周期长度只有 60 条，必须据此另行冻结输入与预测长度。这个版本用于已观测周期序列的后续样本预测，不代表独立整周期留出。

`train_profile.json` 只统计训练段。基础版本没有冻结窗口/训练任务；`window_protocol=null`。标准化、筛选、采样仅使用训练段，测试留待方案冻结后最终评价。不得直接沿用 MetroPT3 的绝对时间窗口解释或混比模型结果。

校验：项目根目录 PowerShell 7 运行 `./scripts/build_additional_datasets.ps1 -VerifyOnly`。署名与许可见 [LICENSE_AND_CITATION.md](LICENSE_AND_CITATION.md)，验收见 `research_runs/dataset_build/20261011/REPORT.md`。
