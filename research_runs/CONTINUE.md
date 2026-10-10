# 下一位 agent 的接续指南

2026-10-05整理后先读[总体数据结论](OVERVIEW.md)和[下一步计划](NEXT_STEPS.md)。完整研究记录保留在原路径，本次未启动训练。旧Release离线包现位于 `archives/20261005/release_20261002/`，下载器可用 `--archive-dir` 指定；历史下载tar.gz已在758项解包资产核验后清理，详见[归档说明](../archives/20261005/README.md)。下文2026-10-03为最新实验完成时点。

最新入口：[completion_study/REPORT.md](completion_study/REPORT.md)及其PROTOCOL.md、POLICIES.md、FOLLOWUP_PROTOCOL.md、summary.json。原95次训练的综合复盘保留在[RETROSPECTIVE_20261003.md](RETROSPECTIVE_20261003.md)，其中“未尝试/未确认”必须结合本次补充结果更新理解。

更新时间：2026-10-03。用户要求补齐未完成研究并明确“不设预算”。已新增39次固定DLinear主训练和两次评分器诊断；当前累计131次主预测、3次历史辅助，共134次训练。另完成全训练CPU测量质量诊断。没有采用新的数据处理默认，没有新测试，MLP成果与原冻结结论完整保留。下文Release恢复说明对应55次训练快照，不能覆盖新增成果。

服务器逐运行资产覆盖134次训练，本轮758项新资产已有本地哈希核验备份；MLP最终freeze/CPU验证/test及最新总状态、补充报告在本地。此前7份总状态文档回写未重试，远程全局state仍为65次。此次后续长训练代码/协议上传也被自动审批拒绝两次，等待本次服务器确认；不换路径绕过。以本地状态与逐实验完成文件为准，不重复训练。

## 目标、现状与边界

模型探索作为独立成果保留，后续研究以数据处理为主。主对照使用原固定DLinear、原传感器表示与训练协议，研究训练样本的保留、删除、损失降权和采样概率；MLP保持冻结，可作独立参考，不继续搜索网络结构。来源替换、PCA、中心化和模型的历史收益不计为样本筛选收益。新数据假设先登记，依据训练内部与验证反馈开发；基础数据和真值保持不变，不能利用已揭示的测试结果选通道、调阈值或选模型。

当前没有后台训练或自动等待队列，但任务尚未全部完成：long_plan.json的六次长训练、70%原点、bilevel正式三种子及固定MLP数据迁移已准备，等待服务器授权确认；输入表示分支另待范围答复。默认仍全量1053433个合法训练窗口均匀采样，主对照S000/S005/S010，三种子验证NMSE均值0.2877519517。保留的MLP入口M001/best.pt，三种子验证均值0.1741860175，原冻结测试NMSE降低29.2698%、NMAE退化0.7691%。模型收益不计为数据处理收益。

补充结果：最早历史删除三种子正式MSE退化1.0503%；重要性采样平均改善0.3508%，洗牌只改善0.0253%且种子反号。新样本外窗口/目标位置降权两原点均使MSE退化；输入代表及监督代表子集未稳定胜过随机控制。双层权重两原点改善0.5283%/0.5751%，略胜匹配控制，是唯一符合正式推进条件的方法。候选已由固定规则确定，未进行其正式确认。原节拍/杠杆等失败结果保留于各历史报告，不扫描旧参数。

本补充研究无计算上限。原累计6566.439622849226秒未重置，本轮服务器进程增加2796.922931138426秒，合计9363.362553987652秒；CPU测量诊断另记3.9014313998632133秒。旧7200秒协议及其原剩余额度仅作历史记录。以state.json及逐运行状态为账本；基础数据身份、冻结方案和历史实验文件不得改写。

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

测试归档独立提供。完整性校验可以读取测试字节/时间边界；开发服务器运行下载器时省略 `--include-test`，完整备份机器运行全量 `verify_dataset.py`。本测试已在model_exploration冻结后正式评估，结果见test_result.json；它不能继续充当未见的开发比较集。`state.json` 的 `test_uploaded=false` 指原训练服务器，最终评估在本地CPU进行。

开发数据为 `data/metropt3_v1/train.csv` 和 `validation.csv`。窗口必须在单一分区内，40条相邻记录间断不超过30秒，32条输入预测8条；记录步数不能当作固定秒数。标准化只在完整训练分区（内部诊断则完整历史前缀）拟合。旧筛选研究固定DLinear 7920参数；新模型协议单独规定MLP 77048参数，Adam 3000步、batch256、NMSE主指标、NMAE副指标及完整验证真值保持一致。不是等参数量或等FLOPs对照。

## 必读顺序与方法索引

1. 根目录 `AGENTS.md`、`HANDOFF.md`、`state.json` 和 `decisions.md`。
2. `selection/PROTOCOL.md`、`selection/protocol.json`、`selection/decisions.md` 和 `selection/REPORT.md`：数据处理主线的原协议与决策。
3. `completion_study/`的报告、协议、summary与各progress保存最新39次训练；再读cadence_selection/internal_error_deltas.json、leverage_sampling等历史反馈，避免重复失败成员集合。新成员方案先核对历史重合和局部支持。
4. `model_exploration/PROTOCOL.md`、`REPORT.md`、`decisions.md`、`summary.json`、`freeze.json`、`test_result.json`：完整保留的独立模型成果，`exports/`保存推理权重。旧报告中的“测试未访问”与“最新”按当时时点理解。

| 方向 | 方法与结果目录 | 解释限制 |
|---|---|---|
| 时间间隔偏差剔除/软降权 | `cadence_selection/` | 失败；与旧整簇删除集合高度重合，匹配控制扰动幅度小 |
| 输入几何杠杆采样、有/无逆概率校正 | `leverage_sampling/` | 校正首折小幅收益没有跨第二原点保持；无正式验证/测试 |
| 跨传感器线性/残差MLP | `model_exploration/` | 新模型验证与最终测试收益；容量变化，不能归因样本筛选 |
| 输入变化量、按月匹配、稀有状态覆盖 | `selection/` | 局部指标改善不能替代总体NMSE；均未采用为新默认 |
| 时间块迁移、梯度冲突、重叠窗口稀疏化 | `transfer/` | 内部时间原点与正式验证绝对分数不能混排 |
| 重要性采样 | `importance/`、`completion_study/` | I001是数值修正；新增2027/2028补齐三种子平均MSE改善0.3508%，仍待长训练/等墙钟 |
| PCA、跨传感器输入来源 | `representation/` | 属于表示研究，不能宣称数据筛选有效 |
| 状态组完整删除 | `cluster_selection/` | 全部已运行候选退化；没有证据把组标成普遍好/坏 |
| 全支持状态平衡、近期偏向与组合 | `coverage_sampling/` | 入选组合/近期没有跨原点NMSE优势；单平衡后续补测见support_confirmation |
| 当前困难/近期学习进展采样 | `learning_sampling/` | 全支持且保持分层质量仍可出现高单窗口概率；两条规则首折均退化 |
| 参考误差差值、自适应参考、同池困难排名 | `reducible_sampling/` | 论文启发的适配；含额外评分和参考训练，未改善首折NMSE |
| 不放回洗牌、均匀CDF时间分层组批 | `batch_sampling/`、`completion_study/` | 分层未跨第二原点保持；洗牌现已补正式三种子，平均MSE仅改善0.0253%且反号 |
| 近期输入密度比、状态质量固定版本及分层打乱 | `covariate_sampling/` | 只用训练历史的近期输入代理；首折小幅收益没有跨第二原点保持 |
| 状态平衡/洗牌补测与正式确认 | `support_confirmation/` | 历史阶段两者内部原点均小幅改善；状态平衡正式退化；洗牌后来在completion_study补齐 |
| 严格样本外权重、双层权重、覆盖/监督代表子集 | `completion_study/` | 两原点及匹配控制均完成，仅双层权重具备正式确认资格，后续训练受审批阻止 |

`results/<run_id>/` 保存真实结果、history、identity、environment、源码和协议快照、最佳/最新检查点。恢复完成后，NPZ包含源行、概率、权重、实际抽取次数，压缩CSV包含逐窗口保留/删除清单；这些都是衍生产物，不是新传感器数据。不要只凭目录存在判断运行成功，也不要把失败实验删除名单应用到基础CSV。

## 建议的下一步

下一步先解决当前服务器上传/启动确认，再执行已准备的long_plan和run_completion_followup；候选固定为bilevel。来源/中心化补充确认与更宽输入表示方向另待范围答复。不要重新发起已完成矩阵或改变候选/阈值。新假设仍须先核对历史掩码重合、局部覆盖并登记。测试已用于最终评价，不能依据其通道误差开发新规则；新的独立最终评价需要新的合适资料，禁止重划或修改基础数据。

保留原模型和筛选方向全部失败证据，不回改旧门槛。模型探索采用MSE主指标门槛是在训练前登记，测试NMAE退化如实保留。后续训练必须新run ID，记录数据身份、源码、模型范围、决策与累计计时；本补充研究预算无上限。不得重启旧批次或按测试挑随机种子。

## 环境与执行注意

完整性校验只需Python 3.11+标准库。训练代码依赖NumPy和支持CUDA的PyTorch，无需pandas；现有训练入口明确要求CUDA，不含CPU训练模式。历史环境为Python3.12.9、NumPy2.4.4、PyTorch2.14.0+cu130、CUDA13.0、RTX4090，逐实验准确版本以environment.json为准。新机器安装与驱动兼容的CUDA版PyTorch后记录实际环境；不要声称跨环境逐位一致已经验证。

最新已完成流程为run_completion_confirmation、fit_completion_policies、run_completion_policy及run_completion_policies。queue_completion_long、long_plan和run_completion_followup仅在本地准备，当前尚未上传/启动；必须先获得正在等待的服务器确认。原cadence/leverage等study入口是已完成冻结流程，不能直接重跑。forecast_models保留原模型结构；evaluate_frozen_models拒绝覆盖已完成测试，接续读保存结果。remote_research.ps1只适用于原服务器，仓库不包含私钥、密码或云机器权限。已完成ID及freeze不能覆盖。

`.gitattributes`保留文件原始换行，防止Windows检出改变SHA256。不要批量格式化历史快照或JSON。迁移上传、CPU字节校验不计作新增GPU研究时间；本轮CPU测量诊断单独记录。取消计算上限不清零历史计时。

## 历史：55次训练时的Release移交核验记录

从暂存Git树导出干净副本，再从四个Release包恢复294个文件。完整基础数据校验、55次训练回执、23项selection元数据、14项transfer资产、30项importance资产均通过。下载器另有6项无网络测试，覆盖原始换行还原、文件/压缩包哈希失败、路径越界和拒绝覆盖。

额外逐条扫描全部历史 `*checksums.json`：1089项匹配，1项历史不匹配。`representation/asset_checksums.json` 对 `representation/decisions.md` 记录的旧哈希为 `6f23de8b09524bc092fdcfa929971b37a02f628f54cd4751e68b46abf4822c0a`，当前本地原件与干净还原副本均为 `e16fe453489ca2c11fd7b04346bae86458bf06a9985f9cfdf38e4c74ff363795`。当前决策文档包含后续R-D003—R-D005段落，但未据此推断其精确改写历史。发布保留原文件和旧回执，不伪造通过结果、不刷新旧哈希。
