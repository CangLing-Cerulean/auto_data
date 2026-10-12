# 两个最小原型的入口

代码：`online_ts/agent_research/intervention.py`；离线CLI：`scripts/intervention_prototype.py`。两种模式均不直接调用模型或训练器。使用已有R1/R2的固定LLM＋离线JSON传输方式，调用者向相同执行器提交决定；公开信息哈希必须一致。

以下命令是后续已获准运行环境中的使用方式，本阶段没有运行这些Python命令：

```text
python -B scripts/intervention_prototype.py prepare --variant strong_prompt --input prototype_input.json --output baseline_context.json
python -B scripts/intervention_prototype.py prepare --variant full --input prototype_input.json --output full_context.json
python -B scripts/intervention_prototype.py register --variant full --input decision_input.json --output prediction_lock.json
python -B scripts/intervention_prototype.py review --variant full --input feedback_input.json --output dual_update.json
python -B scripts/intervention_prototype.py screen-signal --variant full --input signal_input.json --output signal_screen.json
```

所有输出create-only，不覆盖已有登记。`prototype_input.json`是14条真实历史汇总构成的回放输入，明确response_visible=true；它可以生成上下文，但不能为这些已见结果新登记事前预测。新任务的事前输入必须是独立事实快照，不能把本回放的可见性改为false。

register输入含facts与decision，decision字段为parent_id、action、slice、directions、falsifiers和evidence_ids；directions必须包含global/target/complement，值为improve/unchanged/harm/uncertain。创建时间、事实身份和决定一起计算摘要。review需要原登记、evaluation_registration_sha256、prestate_sha256、slice_sha256、action_sha256；不一致即拒绝。原型不能独自证明用户提供的时间或反馈是真实的，正式执行需接入既有可信执行器的顺序收据与权限控制。

配对反馈含global/target/complement三组，各有elements、parent、child、delta和gain_pct。损失空间、目标和顺序必须相同。matched_control_passed、independent_confirmation、additional_falsifiers_passed只能由可信审查/评价器给出。文字否证条件尚未审查时，假设保持pending_falsifier_review，不能自动判支持。

screen-signal的输入含construction_sources、selection_records、rule、allowed_features、if_true、if_false。选择记录包含source_family、锁定的prestate、baseline_prediction和真实observed_direction；标签为-1/0/+1（有害/不确定/改善）。构建与筛选来源重合立即拒绝。rule为1至2个已声明数值特征条件（gt/le）与all/any组合。先计算预测，再读取响应标签评分；记录方向准确率、决策覆盖和伤害漏报。调用方仍需把选择响应与提出规则的Agent隔离；这个适配器没有新建一套权限平台。

历史缺失状态不进入counterexamples；符合资格的预测错误才进入反例索引。新增规则只进入待检验，不自动成为通用经验。full允许固定LLM根据反例提出新规则；本阶段没有生成或验证新的有效规则。

公共去重比较相同来源下的start_row、实际probability、loss_weight和输入程序身份。exact可用于重复提示；near_review需复核；不同程序不会仅因探针相似就被判等价。

测试：`tests/test_intervention_phase0.py`包含9项边界/代数测试，覆盖元素补集、非有限值/对齐、双重验收、预测矛盾、共同事实、登记不可覆盖、去重、反例资格、来源隔离。测试张量仅为工程单元fixture，不进入传感器基础数据。受本次本地范围和既有Python运行约束，运行状态为NOT_RUN。

待执行的真实配对导出入口是`scripts/export_intervention_pairs.py`，读取冻结R2的9个最佳检查点，输出每个种子/配方的逐窗口×步长×通道误差、源行映射、整体/目标/元素补集响应、校验与额外评估成本。它没有优化器或测试读取路径；尚未上传或运行，因此当前没有新逐窗口文件。
