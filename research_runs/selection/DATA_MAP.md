# 哪些窗口被保留、删除、降权或提高采样概率

本阶段操作单元是 **40 条记录组成的训练窗口**，不是物理删除 CSV 记录；一条原始记录可能属于多个重叠窗口。基础数据从未改写。

`window_table.npz` 的两个数组按相同下标对应：

- `start_row_zero_based[i]`：窗口在原 train.csv 数据行中的起始位置 s，表头不计数。输入行区间 [s,s+32)，标签行区间 [s+32,s+40)。
- `input_variation[i]`：只由 32 条输入记录及固定训练标准化得到的变化分数。未使用验证标签生成筛选分数。

每个实验的 `sampling_weights.npz` 按相同 i 保存：

- `sampling_weight`：未归一化的抽样权重，0 表示从该次候选池剔除。
- `probability`：归一化后的每窗口实际抽样概率。
- `loss_weight`：抽取后在损失中的相对权重，与抽样权重分开。
- `draw_count`：本次 768,000 次呈现中该窗口实际出现的次数。

draw_count 统计完整 3,000 步运行，包括最佳检查点之后的训练；最佳检查点步数另存于 result.json。不要把完整运行的呈现次数误称为最佳检查点之前的次数。

原始训练 CSV 哈希、窗口表哈希、保留/排除数量、期望和实际抽样组份额见 `sampling_audit.json`。因为有限预算下有放回抽样，“保留全部候选”不代表每个候选在单次训练中都一定出现。

可用远程既有 Python 导出某次实验的全部直接受影响窗口为压缩 CSV：

```text
/opt/miniconda3/bin/python scripts/export_window_policy.py --run-id S001_drop_quiet_s2026
```

输出为 `selection/<run_id>_affected_windows.csv.gz`。其中只含源行引用和筛选元数据，没有新造的传感器数据。概率因归一化而间接变化的其他窗口仍可从完整 NPZ 查询。

新增 month_matched_drop：S015–S017 按输入末记录所属月份匹配低变化删除的月内数量，再固定随机删除。S015 的压缩 CSV 列出共同的 263,359 个排除窗口；三次训练掩码一致，实际抽取次数分别查看各自 NPZ。逐月匹配计数保存在 sampling_audit.json 的 month_matching。

新增 S018–S023 的 *_swaps.csv.gz：相对同种子 month_matched_drop，逐行列出 422 个恢复窗口（restored）及 422 个替代排除窗口（replacement_excluded）。含真实源行区间、输入是否含 LPS=1、最终概率及该次实际呈现次数。两个策略使用相同替代删除名单；保护策略恢复 LPS 窗口，普通恢复控制只恢复非 LPS 窗口。全池仍以各 sampling_weights.npz 为准，lps_support 计数保存在 sampling_audit.json。

新增 S024/S025：保留全部候选并保持uniform概率，分别按训练内部未来梯度分数或月内打乱控制，将210646个窗口损失权重设为0.25。两份 affected_windows.csv.gz 给出完整受影响源行引用，NPZ保存全池权重和实际抽样次数。../transfer/alignment_scores.npz 的 source_eligible_indices 索引对应本文件所述固定窗口表，score只对应这些可评分历史窗口；其余候选权重保持1。selection_recipe.json与identity.json记录评分资产哈希。
