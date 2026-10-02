# 动态采样的源行映射

所有候选均来自固定`selection/window_table.npz`，其SHA256记入各运行的sampling_audit.json。`window_draws.npz`的start_row为原train.csv除表头后的零起始行s；输入[s,s+32)，目标[s+32,s+40)。没有新建传感器记录，没有删除原始行。

每个`phase_0000/0500/1000/1500/2000/2500.npz`按同一窗口顺序保存gradient_norm、probability、loss_weight、draw_count，分别对应第1–500、501–1000等六段。首段均匀；后五段在当时模型上以原训练窗口评分。若g大于该阶段平均g则提高抽样概率；小于则降低，但所有概率>0。loss_weight=1/(N*probability)，所以更多抽样不意味着更大期望损失权重。

`scoring_*.pt`是评分时的当前模型，不是按验证选出的教师。`best.pt`只用于正式最终开发评分。`latest.pt`保存优化器、全部draw_counts与sampling_rng以续跑固定总步数。I000仅保留数值历史，当前配方实现核验使用I001；两者不可覆盖或作为独立种子合并。

默认政策并未采用本方案：原1053433个窗口仍全量保留、均匀采样。这些阶段概率只属于相应派生实验。
