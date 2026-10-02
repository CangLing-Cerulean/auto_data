# 研究启动记录

日期：2026-09-26。

## 已确定

- 根据 research.md 开展基于真实训练反馈的逐轮数据研究。
- 基础数据集固定为 metropt3_v1，沿用原始 7:2:1 时间划分。
- 本次启动前已通过 scripts/verify_dataset.py 完整性校验。
- 用户指定训练服务器：root@instance-lcpdyopw.zju.smartml.cn:20000。
- 未开始训练，未访问测试集分布或计算测试指标；验证脚本仅执行完整性检查。

## 待确认的首轮协议提案（未生效）

- DLinear，15 个传感器多变量预测，输入 32 条、预测 8 条。
- 以训练集均值和标准差进行逐变量标准化，标准化空间的 MAE/MSE 分别定义为本研究的 NMAE/NMSE；分母固定，禁止随方案改变。零方差变量需在训练前明确统一规则。
- 不跨分区、不跨超过 30 秒的相邻时间间断构造窗口；保留基础数据不变。记录跨度不等同于固定秒数。
- 基线后根据实际反馈决定下一方案，最多 3 轮数据方案，总预算 2 GPU 小时。
- 在训练前进一步冻结学习率、批量、训练步数、随机种子、窗口步长、检查点选择和分组评价协议，使各方案可比。
- 不预先列出并执行批量优化方案；每轮记录 Observation → Hypothesis → Action → Result → Belief Update → Next Action。

## 启动时的阻塞（已解决）

SSH 已到达目标服务器，BatchMode 登录返回 Permission denied (publickey,password)。等待可用 SSH Host 别名、私钥路径或用户在本机完成认证配置。密码不写入对话、代码或日志。

首次训练协议和预算等待用户确认，未将提案视为已授权的固定实验边界。

更新：用户已完成 SSH 配置，使用 `C:/Users/31227/.ssh/smartml_codex` 显式指定密钥可登录。用户已确认首轮协议与预算，正式版本见 `protocol.json`；远程 Python 为 `/opt/miniconda3/bin/python`，GPU 为 RTX 4090。此前阻塞仅为历史记录，不再有效。
