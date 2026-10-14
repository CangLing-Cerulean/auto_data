# 服务器清理候选：等待用户确认

最新存储约定：用户明确禁止使用`fs`、`fs_public`。已撤销本次在fs创建的失败探针及其系统空检查点，共2个0字节文件、2个空目录；未修改其他文件，见[撤销收据](fs_probe_removal_receipt.json)。后续默认`/root/data/auto_data_research/`，或使用独立新建项目目录，禁止软链接间接指向fs/fs_public。历史配额观测仅作失败记录，不再检查这两处挂载。下文“删除0”专指A/B/C候选；A/B/C仍待确认。

2026-10-14。用户要求先列出已备份目录，确认后才清理。当前删除文件数为0。本清单合计逻辑字节2,690,906,019，约2.5061 GiB（十进制2.691 GB）；实际已分配块约2,690,945,024字节。清理后实际可用空间需重新测量。

## A：阶段1目录中的四个归档副本

所在目录：`/tmp/auto_data_intervention_stage1_20261012_r1/`。本组仅含下表四个文件，总计1,156,214,961字节（1.0768 GiB）；该目录其余代码、数据、运行记录和检查点保留。

|精确文件名|字节|已在线核验的GitHub备份|
|---|---:|---|
|stage1-fixed-references-20261012-r1.tar.gz|194332279|[固定参照](https://github.com/CangLing-Cerulean/auto_data/releases/tag/stage1-fixed-references-20261012-r1)|
|stage1-construction-initial-20261012-r1.tar.gz|174304523|[初始构建](https://github.com/CangLing-Cerulean/auto_data/releases/tag/stage1-construction-initial-20261012-r1)|
|stage1-confirmation-interrupted-20261012-r1.tar.gz|388173172|[中断证据](https://github.com/CangLing-Cerulean/auto_data/releases/tag/stage1-confirmation-interrupted-20261012-r1)|
|stage1-confirmation-recovered-20261012-r1.tar.gz|399404987|[恢复确认](https://github.com/CangLing-Cerulean/auto_data/releases/tag/stage1-confirmation-recovered-20261012-r1)|

本地第二份备份位于`research_runs/agent_research/intervention_diagnosis/20261012_stage1a_r1/archives/`，文件名相同，均重新实测大小与SHA256，和服务器及在线Release完全一致。

## B：阶段0的归档副本

精确文件：`/tmp/auto_data_intervention_phase0_20261012/intervention-phase0-20261012-runtime-r1.tar`，767,447,040字节（0.7147 GiB）。

本地备份：`research_runs/agent_research/intervention_diagnosis/20261012_runtime_r1/archives/intervention-phase0-20261012-runtime-r1.tar`。GitHub备份：[阶段0运行验收Release](https://github.com/CangLing-Cerulean/auto_data/releases/tag/intervention-phase0-20261012-runtime-r1)。三处字节数与SHA256一致。

## C：阶段0已归档的成对评估目录

精确目录：`/tmp/auto_data_intervention_phase0_20261012/runtime_r1/paired_evaluation/`，13个文件，共767,244,018字节（0.7146 GiB）。包含3种子×3模型的9个`errors_*.npz`，以及`receipt.json`、`response_collection.json`、`response_long.csv`、`window_identity.npz`。

该目录的全部服务器文件均匹配B项本地归档对应成员的大小与SHA256；本地tar的34个成员全部重新读取内容验哈希，其中13个完整覆盖本目录。恢复时从B项本地或Release归档提取`runtime_r1/paired_evaluation/`即可。

## 确认范围与执行条件

建议确认A+B+C，共5个归档文件和1个13文件的子目录。基础数据、训练运行目录、模型检查点、当前M1代理父轨迹、本轮选择恢复目录和本地/GitHub备份均保持。本清单不扩大到两处顶层实验目录，也不包含用户持久盘上的预检文件。

执行前须收到用户明确确认；再次核对18个文件的精确绝对路径、普通文件身份、字节数和SHA256，发现新增或变化即停止。按文件清单删除，最后只移除已空的C目录，不递归删除其他内容；保存执行收据、空间变化和剩余资产检查，再验收下一轮存储。删除授权当前为false。

A/B/C释放根文件系统空间，不扩大`/root/data`的独立卷。若确认清理后使用新项目目录，应建在实际已有足够容量的位置，并记录精确路径；不把根盘释放量误记为data可用空间。

证据：[服务器清单](server_inventory.json)、[GitHub在线核验](online_backup_verification.json)、[本地备份复核](local_backup_review.json)、[归档成员内容核验](phase0_member_verification.json)、[待确认执行清单](cleanup_plan.json)。
