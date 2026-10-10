# 数据筛选 Agent

采用现有Agent代码，新增 `online_ts.selection` 研究入口；不依赖AgentFlow或LangGraph。现成控制器足以编排当前单Agent工具循环，训练与评价边界由确定性工具执行。

## 本地准备

在项目根目录运行 PowerShell 7：

```powershell
python -B -m unittest discover -s tests -p test_selection_harness.py -v
python -B -m online_ts.selection init
python -B -m online_ts.selection prepare
# 检查并调整 config.json 的开发预算和覆盖门槛后冻结
python -B -m online_ts.selection freeze
python -B -m online_ts.selection preflight
```

prepare只生成训练输入诊断、源行映射和历史掩码引用，不训练、不分析测试。任务资产保存于tasks/metropt3_dev。freeze生成含源码、提示词、数据及配置身份的harness_manifest.json；修改代码或配置应使用新manifest和新run_id。

## DeepSeek与执行

API密钥仅从当前进程的 `DEEPSEEK_API_KEY` 读取，禁止写入代码、配置、日志或提交记录。官方基础地址为 `https://api.deepseek.com`；可通过 `DEEPSEEK_BASE_URL` 使用官方 `/v1` 路径。实现依据[官方Chat Completions文档](https://api-docs.deepseek.com/api/create-chat-completion/)，请求JSON对象动作，程序执行本地工具；没有自动启用服务端文件或联网工具。

训练宿主需提供requirements.lock对应依赖及CUDA，并在该宿主生成新manifest，固定实际Python/PyTorch/CUDA环境。本机没有PyTorch，生成的本地manifest只用于工程检查，不能冒充训练环境冻结。检查点使用与现有实验相同的DLinear实现，依赖预检不足会明确报错，不自动安装。以下命令会产生真实API和GPU成本，只在已配置训练宿主执行：

```powershell
python -B -m online_ts.selection run --run-id agent_01 --arm agent
python -B -m online_ts.selection run --run-id random_01 --arm random
# 完成预登记的两组各5次，交错运行
python -B -m online_ts.selection run-batch
python -B -m online_ts.selection report
python -B -m online_ts.selection freeze-batch
python -B -m online_ts.selection confirm
```

重复同一run命令从账本恢复，禁止更换manifest或组别。每次GPU工作用科学配置、清单和seed哈希去重；错误记录仍计入运行结果。最终政策、完整工具轨迹、模型调用、实际token、训练成本、精确窗口清单和曝光记录可追溯。

所有正式预定提交齐全才能进入开发确认。确认结果是独立训练种子下的D成绩，不是独立E成绩。代码没有旧test.csv评价入口。现有数据已被多轮探索，真正独立的泛化结论需要新资料。项目当前尚未启动任何真实API调用或新训练。
