# 下一位 agent 的接续指南

更新时间：2026-10-02。用户当前要求：“把数据和研究方法都上传，我要让下一个agent接续任务，现在还是先探索怎么处理数据可以提高训练效果”。本次发布仅补齐数据和研究资产、整理接续入口，没有运行新实验。

## 目标、现状与边界

继续研究训练窗口的保留/删除、损失权重和采样概率如何影响固定训练预算下的效果。沿用最近确认的“训练样本筛选、原始传感器表示”范围；来源替换、PCA或中心化的历史收益不能算成筛选收益。新假设先登记，再运行；不要为了找到改善反复扫描已失败阈值。

已完成55次训练，当前无待运行队列。默认仍为保留全部1,053,433个合法训练窗口并均匀采样；正式筛选uniform三种子平均NMSE=0.2877519517131628，固定seed2026参考检查点为 `results/S000_uniform_s2026/best.pt`。最近四组删除中，组0因保留率限制未运行；删除组1/2/3的内部NMSE分别退化23.7248%/33.9311%/16.8012%，NMAE也退化。后续随机对照、第二原点和正式确认未触发，不要自动补跑。

累计预算上限仍为7200秒，已用4114.524273328483秒，剩3085.475726671517秒（约51.4分钟）。以 `state.json` 和各运行状态为账本；不能把迁移到GitHub视作新额度。聚类阶段的1800秒是历史单轮限制，不是额外预算。基础数据身份及任何历史实验文件不得重写。

## 恢复完整项目

```powershell
git clone https://github.com/CangLing-Cerulean/auto_data.git
cd auto_data
python scripts/download_handoff.py --include-test
python scripts/verify_dataset.py
python scripts/audit_assets.py
python scripts/verify_selection_metadata.py
python scripts/verify_transfer_assets.py
python scripts/verify_importance_assets.py
python scripts/download_handoff.py --include-test --verify-only
```

下载器不需要GitHub账号或额外Python库。数据和资产来自 [handoff-20261002](https://github.com/CangLing-Cerulean/auto_data/releases/tag/handoff-20261002)，哈希在 `handoff_assets.json`。下载器拒绝覆盖不同内容的文件；遇到不匹配先查原因，不删数据、不重建manifest。离线还原可将四个ZIP放在同一目录，增加 `--archive-dir <目录>`。

测试归档独立提供。完整性校验可以读取测试字节/时间边界，但不可输出测试分布、预测得分或选择反馈。开发服务器运行下载器时省略 `--include-test`；在完整备份机器上运行全量 `verify_dataset.py`。数据发布不等于解封最终测试，`state.json` 的历史 `test_uploaded=false` 仍指原训练服务器。

开发数据为 `data/metropt3_v1/train.csv` 和 `validation.csv`。窗口必须在单一分区内，40条相邻记录间断不超过30秒，32条输入预测8条；记录步数不能当作固定秒数。标准化只在完整训练分区（内部诊断则完整历史前缀）拟合，不能因删样改变尺度。固定DLinear 7920参数、Adam 3000步、batch256；NMSE主指标与NMAE副指标、完整验证真值均不变。详细规则以冻结协议为准。

## 必读顺序与方法索引

1. 根目录 `AGENTS.md`、`HANDOFF.md`、`state.json`、`PROTOCOL.md`、`protocol.json`、`decisions.md`。
2. `selection/PROTOCOL.md`、`selection/protocol.json`、`selection/decisions.md` 和 `selection/REPORT.md`：成员关系、损失降权、采样概率的定义及三种子对照。
3. `cluster_selection/PROTOCOL.md`、`REPORT.md`、`decisions.md`、`recommended_policy.json`、`verification.json`：最新失败证据及冻结分组。
4. 需要时按下表读此前机制研究，避免重复已经失败的方案。

| 方向 | 方法与结果目录 | 解释限制 |
|---|---|---|
| 输入变化量、按月匹配、稀有状态覆盖 | `selection/` | 局部指标改善不能替代总体NMSE；均未采用为新默认 |
| 时间块迁移、梯度冲突、重叠窗口稀疏化 | `transfer/` | 内部时间原点与正式验证绝对分数不能混排 |
| 重要性采样 | `importance/` | I001是数值修正；I000和I001不是两个独立种子 |
| PCA、跨传感器输入来源 | `representation/` | 属于表示研究，不能宣称数据筛选有效 |
| 状态组完整删除 | `cluster_selection/` | 全部已运行候选退化；没有证据把组标成普遍好/坏 |

`results/<run_id>/` 保存真实结果、history、identity、environment、源码和协议快照、最佳/最新检查点。恢复完成后，NPZ包含源行、概率、权重、实际抽取次数，压缩CSV包含逐窗口保留/删除清单；这些都是衍生产物，不是新传感器数据。不要只凭目录存在判断运行成功，也不要把失败实验删除名单应用到基础CSV。

## 建议的下一步

先利用已有训练侧分组和映射检查：是否能在保留各月份与输入状态覆盖的前提下，识别组内冗余或可替代窗口。这个方向尚未登记为已执行方案，也未证明有效。新规则必须给出训练侧依据、源行映射、等数量且匹配状态/月份的随机控制，以及保持模型/尺度/更新预算一致的比较。

先在训练内部按时间原点获得开发反馈；只有通过新方案预先写明的门槛，才进入完整验证与配对种子确认。沿用既有1%主指标改善、NMAE不退化等门槛时需明确引用；若改变任务、指标、门槛或训练协议，应单独说明并获得对应授权，不能与旧结果混比。任何新训练都要使用新run ID，并在相应decisions记录观察、假设、动作、推翻条件和预算。

## 环境与执行注意

完整性校验只需Python 3.11+标准库。训练代码依赖NumPy和支持CUDA的PyTorch，无需pandas；现有训练入口明确要求CUDA，不含CPU训练模式。历史环境为Python3.12.9、NumPy2.4.4、PyTorch2.14.0+cu130、CUDA13.0、RTX4090，逐实验准确版本以environment.json为准。新机器安装与驱动兼容的CUDA版PyTorch后记录实际环境；不要声称跨环境逐位一致已经验证。

主要入口：`scripts/run_selection.py`（正式筛选）、`scripts/run_temporal_probe.py`（训练内部时间诊断）、`scripts/cluster_selection.py`（冻结分组/掩码）。旧批处理仅对应历史登记队列，不能当作下一轮启动命令。`scripts/remote_research.ps1` 固定了原服务器与用户SSH路径，只适用于旧环境；仓库不包含私钥、密码或云机器权限。恢复旧任务必须核对其源码/协议/manifest身份，已完成ID不能覆盖。

`.gitattributes` 保留文件原始换行，防止Windows检出改变SHA256。不要运行会批量格式化历史快照或JSON的工具。迁移上传、CPU字节校验不计作新增GPU研究时间；本次未重置或补充训练预算。

## 本次移交核验记录

从暂存Git树导出干净副本，再从四个Release包恢复294个文件。完整基础数据校验、55次训练回执、23项selection元数据、14项transfer资产、30项importance资产均通过。下载器另有6项无网络测试，覆盖原始换行还原、文件/压缩包哈希失败、路径越界和拒绝覆盖。

额外逐条扫描全部历史 `*checksums.json`：1089项匹配，1项历史不匹配。`representation/asset_checksums.json` 对 `representation/decisions.md` 记录的旧哈希为 `6f23de8b09524bc092fdcfa929971b37a02f628f54cd4751e68b46abf4822c0a`，当前本地原件与干净还原副本均为 `e16fe453489ca2c11fd7b04346bae86458bf06a9985f9cfdf38e4c74ff363795`。当前决策文档包含后续R-D003—R-D005段落，但未据此推断其精确改写历史。发布保留原文件和旧回执，不伪造通过结果、不刷新旧哈希。
