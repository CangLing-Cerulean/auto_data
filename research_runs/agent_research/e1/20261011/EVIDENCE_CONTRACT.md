# E1 逐次响应与冻结决策证据契约

本契约描述待实现的数据交换格式。配套 [record_templates.json](record_templates.json) 仅含空白模板，所有 null 表示尚未登记；禁止当作真实记录、实验结果或成功冻结。协议状态由 [protocol.json](protocol.json) 给出。

## 存储和身份

正式记录采用 UTF-8 JSONL 长表，一行一个对象，不只保留平均分。建议文件为 tasks.jsonl、actions.jsonl、decisions.jsonl、responses.jsonl、experiences.jsonl、visibility.jsonl、costs.jsonl。文件尚未创建，当前没有响应数据。

时间为 UTC ISO 8601，日期展示可转 Asia/Shanghai。ID 唯一且不可复用，冻结后追加新版本及 supersedes_id，不原位覆盖。数据、代码、配置、配方、输入信息集和实际锁定制品均保存 SHA256；空值不冒充哈希。最终冻结哈希写入独立收据，不能对包含自身哈希的 JSON 做自引用校验。

所有正式记录必须将 is_template 设为 false，填齐该阶段必需身份。未知历史字段维持 null 并填写 missing_reason；已知“无”与“未知”区分。空集合不能默认为已验证无数据或无故障。

## 五类主要记录

| 类型 | 主键或粒度 | 核心规则 |
|---|---|---|
| task | task_id | 固定 T/D/审计边界、来源组、目标和协议；历史 D 明示已使用 |
| action_rule / action_instance | 各自 ID | 规则与任务实例分开；匹配控制和配对随机性显式绑定 |
| decision | decision_id | 一个策略在一个时点可见的信息及其预测、推荐；不可事后回填 |
| response | task × action_instance × training_repeat × evaluation_partition | 保留逐重复、逐分区指标、配对基线、状态及成本 |
| experience | experience_id + version | 原条件、动作、响应、反例、适用边界及更新证据 |

response 的 metrics 是结构化数组，包含 scope_id、NMSE、NMAE、window_count 和 metric_artifact 引用。这样总体和局部指标不破坏响应主键；局部组重叠须标明，不将其误差贡献相加为互斥总量。实际运行恢复尝试存在多个回执时，追加 attempt_receipts，不重复把同一个逻辑训练结果计入均值；不同真实训练使用不同 training_repeat/run_id，保留全部费用。

evaluation_partition 只允许 development、confirmation_development、audit。confirmation_development 表示在已用开发数据上用独立训练重复确认，不获得隐藏审计身份。当前 MetroPT3 不产生 audit 行。

## 状态字段

- execution_status：not_started、running、succeeded、code_failed、platform_failed、timeout、interrupted。
- protocol_validity：pending、valid、invalid、control_invalid、information_condition_invalid。
- scientific_status：unassessed、confirmed_beneficial、confirmed_harmful、confirmed_equivalent、insufficient_evidence、not_assessable。
- guardrail_status：not_assessed、pass、fail、unknown。

response 单行只记录观测，默认 scientific_status 为 unassessed。确认类标签必须指向 adjudication_receipt，该收据列出配对响应 ID、重复、比较范围、门槛、区间和规则版本；不能用一次分数自动确认。有害、无效和代码失败不合并。约束未通过单独记录，即使 NMSE 有收益也不自动标成可接受方案。

decision 的 candidate_predictions 每项包含 action_instance_id、预期效果/排序、uncertainty 及 abstain；至少有一种预测形式。predicted_effect 必须声明单位和参照。pair_predictions 登记候选对、方向和不确定性；没有明确区分时记录 undecided。最终推荐只能来自冻结候选池；弃权必须指向已登记不处理实例。

## 可见性与缓存

visible_to_policy_at 是按 policy_run_id 存储的映射，尚未揭示时为空对象，不是所有策略共享的全局时间。visibility 长表保存 information_item_id、policy_run_id、request_id、available_at、visible_to_policy_at、feedback_view、payload_sha256 和 decision_prefix_id。部分响应使用不同 payload 身份：完整响应不能与仅整体分数共用载荷哈希。

结果物理上已计算不等于策略可见。E1 锁定前不允许任何当前任务候选响应；E2 缓存只有在请求对应动作并完成逻辑成本记账后才可揭示。not_revealed 不能通过文件名、目录、排行榜、错误日志、其他策略决策或后续动作间接泄漏。对照组可以知道相同的执行事实与合法性状态，科学分数按视图提供。

每个 decision 绑定 visible_information_ids 和 visible_information_manifest_sha256；其引用必须属于该信息集，且揭示时间不晚于锁定时间。禁止复用另一个策略的访问映射。E2 分叉只共用冻结前缀和指定探针，之后信息集、检索和动作轨迹独立。原生目录约定不构成强隔离证明；正式运行前须验收真实可见性路径，隐藏审计不进入执行分区。

## 锁定与成本节点

decision 中记录 created_at、locked_at、policy_version、prompt_sha256、memory_sha256、candidate_pool_sha256、推荐、cost_snapshot 及 lock_receipt。真实锁定须先写独立快照，再在收据记录路径、字节哈希和锁定时点；模板中的 lock_receipt=null 明确不是已锁定。

E1 锁定早于当前任务候选揭示。成本节点绑定预登记的账本维度、阈值和累计值，选择截至该节点最新已锁定推荐；长作业跨越节点时只计已发生的成本和可见结果，不能偷看完成后的输出。未运行到的节点标记 uncovered，不补造推荐。重复审计不能回流到后续开发决策。

cost 保存物理与逻辑两套量，诊断、基线、候选、控制、决策 API、决策模型训练、复验和共享离线矩阵构建均注明。共享物理费用只记一次，策略重放逻辑费用按预登记规则分配；不可重复累计物理训练，也不可让缓存成为免费额外信息。未知 token/价格为 null 并标明 known/reserved/unresolved，不以预约值替代实际值。

## 经验更新

每条经验保存 pre_intervention_state_ref、state_timing、prior_prediction_ref、response_ids、counterexample_ids 和 applicability。缺少事前材料时设为 missing 或 retrospective，不进入前瞻预测评分分母。历史背景可查询，但须遵守来源组过滤，不能通过反例引用读到隐藏任务。

更新保存 parent_version、conditions_added、conditions_removed、conditions_narrowed、supporting_evidence_ids 和行为比较引用。改变条件是待验证更新，不自动成为正确机制。规则的正确拒绝和正确复用都须在未参与更新的任务上评价；未验证状态显式保留。不得把经验条数当作独立任务数。

## 正式运行前的语义验收

1. 任务/规则/实例/基线/响应/经验引用存在且身份一致；模板不进入数据集。
2. 基础分区、源行和时间边界合法；audit 身份真实、无旧测试替用。
3. 配对训练协议与评价窗口相同，训练重复和来源分组分别记录。
4. 每个响应完整保留执行与科学状态；失败指标为 null，不写成零误差。
5. 决策引用仅来自自身可见信息；E1 无锁定前候选反馈；E2 不共享未来轨迹。
6. 缓存结果、完整/总分/屏蔽视图、成本和锁定顺序可追溯。
7. 科学标签可回溯配对确认和已冻结门槛；证据不足不标为等效。
8. 所有 pending 登记项已解决，再生成独立冻结收据并验收执行器。

当前仅完成格式草案和本地文档静态检查。上述运行期验收尚未实现或执行，不以 JSON 可解析替代这些检查。
