# 提案代码交付接口

提交一个不含文件、网络、进程、环境读取的纯计算Python模块，仅使用numpy和torch。模块不是直接执行授权；主代理必须先检查、冻结并验证。可以提交null及理由。

给出最多两个配方，每个包含主操作与匹配控制。代码接口：

```python
def fit(train, starts, strata, seed):
    # train: float32 [rows, channels], already normalized by task T only
    # starts: int64 [N], legal 40-row window starts; inputs are offsets 0..31
    # strata: int [N], original source sequence x chronological quartile
    # seed: fixed control RNG seed 2026101499
    # Return JSON-serializable params and per-candidate train arrays.
    return {"params": {...}, "candidates": {
        "P1": {"target": boolean_array_N, "control": boolean_array_N,
               "weight": float_array_N, "control_weight": float_array_N,
               "probability": float_array_N, "control_probability": float_array_N}
    }}

def transform(x, params, candidate, control=False):
    # x: torch [batch, 32, channels], AFTER fixed parent representation.
    # Return same shape/device/dtype, without mutating x; identity allowed.
    return x

def target_mask(raw_x, params, candidate):
    # raw_x: torch normalized input windows BEFORE parent representation.
    # Return torch bool [batch]; threshold/params from fit(T), never fit(D).
    # For an honestly global operation, return all True and report no complement.
    ...
```

概率非负、至少一项正值，归一化后总和为1；权重非负，按当前采样概率加权均值归一化为1，不能丢失全部训练窗口。主操作/控制应保持相同有效处理数量或概率质量；结构化输入变换使用明确的机制控制，不要求虚假的稀疏掩码匹配。参数必须对不同通道数自然定义。原始索引、文件/数据集名不得进入公式。

请同时返回每个C任务的整体/目标/补集四类别预测与理由、必要的对照验收检查、可能失败条件。历史经验用于明确机制假说，不能把历史不同任务的NMSE直接混排。本轮没有新候选响应，所有预测都将在执行前保存。
