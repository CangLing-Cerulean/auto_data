# 复现与状态解释

生产候选为`scripts/adam_risk_sampling.py`，只需NumPy；它没有数据加载器、训练器、远程访问或旧标签读取入口。参考检查另需SymPy、SciPy。本地本阶段使用既有`20261015_response_extensions_engineering_r1/private/venv/Scripts/python.exe`，没有新增依赖安装。

在仓库根目录运行，输出须是该阶段目录中的新文件名，不能覆写已有记录：

```powershell
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
& 'research_runs/agent_research/intervention_diagnosis/20261015_response_extensions_engineering_r1/private/venv/Scripts/python.exe' scripts/check_adam_risk_sampling.py --out 'research_runs/agent_research/intervention_diagnosis/20261015_adam_sampling_design_r1/reproduction_new.json'
```

原运行`algebra_check_r2.json`为111/112通过，整体保留FAIL，退出码1。混合符号案例的两次SLSQP状态false，而解析解、枚举网格及四个参考目标值一致于登记容差内；重现可能受SciPy版本影响，不能要求SLSQP状态跨版本固定。

`artifact_audit.json`由`scripts/audit_adam_sampling_design.py --stage <本阶段绝对目录>`产生，已有输出时拒绝覆写。它核验原锁、全部记录、首次失败源码，并为唯一失败案例提供全局最优的精确有理数证书。它不重跑整套数值检查，也不把原FAIL改记为PASS。

首次尝试发生结果输入清单的相对/绝对路径错误，未生成数值结果；原checker保留于`failed_check_v1.py.txt`，记录见`attempt1_failure.json`，原源码锁保留。修复仅对输出路径调用resolve，第二次锁见`source_lock_r2.json`。阈值、样例、种子与算法均未改变。

三个有限样例的全部15行精确条件风险与近似值仅用于检查公式和记录余项；不代表真实传感器实验、方法排名或科学选择通过。真实输入、运行协议与完整成本尚未准备，本阶段不能直接运行科学训练。
