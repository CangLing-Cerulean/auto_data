# 时序科研平台

统一入口为 `python -m online_ts.platform`；安装项目后也可使用 `ts-platform`。平台提供两条可组合的执行链：

- **研究工作流**：冻结协议、声明输入、顺序执行依赖步骤、保存每次尝试、核验产物、恢复和人工对账。
- **滚动预测**：持续接入观测、按固定频率预测、后台训练、等待标签成熟、误差监测、候选比较和模型发布。

原有科研批次保持自己的协议、预算和执行身份。平台不会自动恢复 r11、读取封存 D/test 或启动原服务器实验。

## 安装与环境

Python 3.11 及以上；基础功能使用 NumPy、Pandas、SQLite 和标准库。DLinear/PatchTST 等神经模型另需 PyTorch 与对应运行环境。

```powershell
$ErrorActionPreference = 'Stop'
python -m pip install -e .
python -m online_ts.platform doctor
```

`doctor` 只检查本机依赖。源码安装、当前依赖版本和运行路径仍需随研究环境一起保留；每个新 study 会记录解释器、NumPy/Pandas/PyTorch 版本、平台源码和声明输入的 SHA256。协议、外部脚本、额外依赖锁文件应作为输入登记。其他工具、系统库和远程服务由所调用的研究 runner 记录。

## 跑通一个研究工作流

在项目根目录执行：

```powershell
python -m online_ts.platform study validate --config examples/platform/study.json
python -m online_ts.platform study create --config examples/platform/study.json --run outputs/study_demo
python -m online_ts.platform study run --run outputs/study_demo
python -m online_ts.platform study report --run outputs/study_demo
python -m online_ts.platform study resume --run outputs/study_demo
```

示例使用四个合成数值完成测量和评估，不调用外部模型。再次恢复时核验并复用已完成产物。

配置要求：

| 字段 | 含义 |
| --- | --- |
| `version/name/workspace` | 配置版本、实验名、相对配置文件的工作区 |
| `inputs` | 名称到 `{path, role}` 的映射；相对工作区文件，角色为 training、validation、response、protocol、code |
| `steps[].id/depends_on` | 唯一步骤名、必须在列表前方出现的依赖 |
| `command` | 参数数组，直接执行；支持 `{python}`、`{workspace}`、`{attempt}`、`{context}` |
| `timeout_seconds/outputs` | 单次时限、相对当前尝试目录的必需文件 |
| `max_attempts/retry_exit_codes` | 默认 1 次、无重试；只有声明过的确定失败退出码允许新尝试，最多 5 次 |

每次尝试有独立目录、`context.json`、开始记录、标准输出/错误及最终收据。context 提供声明输入和已完成依赖目录；程序将结果写入当前目录或 `PLATFORM_OUTPUT_DIR`。研究步骤采用可信程序接口；生成代码继续由已有隔离执行器检验后执行。输入角色是协议与审计约束，操作系统文件权限仍需单独落实。

`run/resume` 使用同一恢复逻辑。命令断线后子 worker 可继续保存收据。若 worker 已失联且没有最终收据，状态为 `uncertain`，自动执行停止；明确失败、超时、产物不全及未知结果各自保留。超时不自动重试。成本报告包含每次实体命令耗时，LLM tokens 和完整训练次数从研究 runner 的业务产物读取。

确认未知执行结果后，操作者可以附上外部证据：

```powershell
python -m online_ts.platform study reconcile --run outputs/study_demo --step measure --attempt 1 --outcome completed --returncode 0 --evidence outputs/external_receipt.json --actor operator --reason "已核对外部运行收据和产物"
```

这会保存证据副本和对账记录；耗时仍为未知，不覆盖原尝试。只有原冻结重试政策允许时，人工确认失败后才能发起新尝试。源码或输入变化须创建新 study。

## 合成历史回放

```powershell
python examples/platform/prepare_demo.py --output outputs/platform_demo
python -m online_ts.platform stream replay --config outputs/platform_demo/replay.json --runtime outputs/replay_demo --max-steps 40
python -m online_ts.platform stream status --runtime outputs/replay_demo --task demo_replay
python -m online_ts.platform stream predictions --runtime outputs/replay_demo --task demo_replay --limit 3
```

回放与实时模式共用 Engine、模型产物、训练门禁和监视逻辑。示例 CPU 训练属于工程演示，不计入科学实验结果。

## 接入连续数据流

在分别打开的终端中运行日志服务、滚动任务和生产者。环境变量设置在滚动任务所在终端。

```powershell
# 终端 1：持久到达日志
python -m online_ts.platform stream serve --database outputs/live_demo/journal.db --variables signal
```

```powershell
# 终端 2：持续预测与后台训练
$env:PLATFORM_STREAM_URL = 'http://127.0.0.1:18761'
python -m online_ts.platform stream run --config outputs/platform_demo/live.json --runtime outputs/live_demo/runtime
```

```powershell
# 终端 3：用合成生产者验证接入
python examples/platform/produce.py --seconds 240
```

首次训练需要约 112 个按时到达的观测，示例实际约两分钟。服务以收到记录的真实时间登记可用性；批量补入旧数据不会伪造成历史时刻已知。首次模型通过离线检查后会进入 `INITIAL_APPROVAL`。在第四个终端检查并发布：

```powershell
python -m online_ts.platform stream monitor --runtime outputs/live_demo/runtime --task demo_stream
python -m online_ts.platform stream approve --runtime outputs/live_demo/runtime --task demo_stream --actor $env:USERNAME
python -m online_ts.platform stream predictions --runtime outputs/live_demo/runtime --task demo_stream --limit 5
```

示例准备程序将当前操作系统用户登记为发布者；服务仍核对实际用户身份和权限。真实任务应把协作权限写入任务配置。预测读取只返回已发布结果，按完整预测原点分页；将 `next_after_issue` 传给 `--after-issue` 可增量消费。相同原点在重启后不会重复发布。

业务生产者向本机 `POST /records` 发送：

```json
{
  "idempotency_key": "sensor-7:sequence-428",
  "event_ns": 1792454400000000000,
  "values": {"signal": 12.3},
  "maturity_delay_seconds": 2
}
```

`event_ns` 是 UTC Unix 纳秒，不能晚于服务接收时间。变量集合与服务启动配置完全一致，值必须有限。接收时间由服务写入；成熟时间为接收时间加延迟。重复键且相同载荷返回原收据；相同键不同载荷返回 409；修订用新键和相同事件时间。标签先解析最新修订再检查成熟度。未成熟修订以明确的撤回版本暂停旧标签参与当前评价；修订成熟后恢复，历史 as-of 记录保留。停流期间继续结算旧预测的标签。

既有 JSONL 可用 `stream ingest --input <文件> --url http://127.0.0.1:18761` 逐条提交。中断后可重送同一文件。`GET /health` 返回契约/变量；`GET /records?asof=<纳秒>&start=0&limit=256` 返回固定上界的快照分页，响应的 `upper_sequence` 和 `next_after` 用于后续页。日志使用 `server-receipt-journal-v2`，拒绝旧格式数据库；请为新服务创建新数据库。

服务仅绑定回环地址。Kafka、MQTT 或设备 SDK 可由独立 producer 转成上述消息，保留业务幂等键、采样时间和标签成熟规则；当前没有内置这些 broker 连接器，也没有外网认证接口。

新生命周期库使用 schema 9，支持标签撤回和历史版本查询。旧 schema 8 库保留原样，使用其原冻结版本查看和执行；本版本要求新运行目录，不做自动迁移。

## 持续训练、监视与运维

```powershell
python -m online_ts.platform stream retrain --runtime outputs/live_demo/runtime --task demo_stream
python -m online_ts.platform stream rollback --runtime outputs/live_demo/runtime --task demo_stream
python -m online_ts.platform stream stop --runtime outputs/live_demo/runtime --task demo_stream
```

- 训练在独立受预算约束的 worker 中执行。当前 Champion 持续服务；新模型须通过验证和离线评价，后续经过影子、灰度、审批及发布后观察。回滚要求存在先前模型。
- 自动训练由初始模型缺失、成熟误差持续退化或模型年龄触发；冷却时间、新数据量、当前任务和候选状态共同限制启动频率。固定阈值来自配置，不能因结果随意修改研究协议。
- `status/monitor` 是只读查询，包含训练队列、状态、事件、标签覆盖及最近监测窗口；不创建缺失任务。没有标签时等待，不把缺失当作零误差。
- `stream run` 写持久心跳，并以 OS 锁拒绝第二个同任务 runner。心跳过期仅表示需要检查，不能据此擅自重复外部任务。
- Ctrl+C 暂停 runner；显式 `stop` 请求停止任务并取消训练。重新运行相同配置和目录恢复状态。源码变更的上线需另外验证当前任务与模型兼容性。
- 使用操作系统服务管理器托管上述前台命令。保留日志库、运行目录与内容寻址模型产物；备份 SQLite 时使用其 backup API 或停止写入后备份。

当前输入读取按滚动窗口限制；为了完整接收历史修订，标签仍读取保留日志的全部可见历史，运行库也保留完整预测与事件。长周期高吞吐部署需先明确留存期、修订期，再实现带持久游标的标签同步与归档。本轮验证为单机合成场景，未完成生产压力、长时间运行、断电或多机故障验证。

## 扩展边界

| 扩展项 | 接入位置 | 必须保留的语义 |
| --- | --- | --- |
| 新科研流程 | 新 study 配置与独立 runner | 声明输入/产物、确定退出码、失败成本、科学门槛 |
| 新数据源 | producer；或 `read_until/labels_until/discover` source adapter | 事件/到达/成熟/修订时间，as-of 无未来泄漏 |
| 新模型 | `Candidate`、`ForecastModel`、`load_forecast_model` 及配置白名单 | 固定输出形状、训练内拟合、不可变模型包、重载探针 |
| 训练资源 | `execution.py` 的 local/server 协议 | 资源预留、超时、幂等收据和返回资产核验 |
| Agent 策略 | `agent_runtime` 的 Model/Context/Gateway/Repository | prepared 决策恢复、操作幂等、权限和预算 |
| 监测/业务指标 | `monitoring.py` 与配置 | 相同原点/标签配对、覆盖率、关键变量保护、固定阈值 |

共用 Engine/Store 管理预测和训练状态；平台层负责启动、冻结、恢复与查询，避免再建立一套模型生命周期。已有冻结研究 runner 继续按自身身份独立运行。平台工作流中的通用 `reconcile` 不改变 r11 的科学判定或恢复授权。

## 验证

```powershell
python -B -m unittest discover -s tests -p test_platform_studies.py -v
python -B -m unittest discover -s tests -p test_stream_journal.py -v
python -B -m unittest discover -s tests -p test_platform_streams.py -v
```

全部测试使用合成数据和临时目录。回环测试需要本机网络权限；集成测试启动真实小型 CPU worker，并验收模型审批、持续发布、延迟/修订标签、监视、并行训练和重启去重。
