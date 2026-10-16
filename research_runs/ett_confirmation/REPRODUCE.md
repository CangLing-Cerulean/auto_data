# ETT确认复现入口

科学设计以PROTOCOL.md为准；data_lock.json和execution_lock.json分别绑定数据身份、原科学设计及服务器训练源码。运行代码为scripts/ett_confirmation.py，原始DLinear类直接来自scripts/run_research.py；launch_ett_confirmation.py负责监督，manage_ett_confirmation.py仅负责传输。

1. 从data/ett_v1/manifest.json记录的官方固定提交获取ETTh1.csv、ETTh2.csv，核对大小和SHA256。原始测量表不放入研究结果Release，标准化表也不发布。
2. 使用记录的PyTorch/CUDA/NumPy环境。复制锁定运行源码、协议、manifest、data_lock、execution_lock及两个时间索引到专用研究目录。运行器固定目录见execution_lock；在另一机器复现需单独记录路径修改与源码身份，不能覆盖原ID或冒充原冻结运行。
3. 首先只提供ETTh1测量表，依次调用launch_ett_confirmation.py的preflight ETTh1、train ETTh1、audit ETTh1。每一阶段必须等监督exit.json为0再进入下一阶段，失败日志保留。
4. 完整ETTh1结果及运行源码发布并核对提交后，生成ETTh2_unlock.json绑定summary/audit哈希，再提供ETTh2测量表；使用完全相同的运行源码依次完成ETTh2三阶段。科学负结果不改变确认矩阵。
5. collect下载的归档与ETTh1/ETTh2_archive_receipt.json核对大小、SHA256；report_ett_confirmation.py ETTh1、ETTh2、all生成逐种子表、全部通道表、汇总、成本和图。best是主口径，last是预定次要结果。

服务器连接信息单独放在本地private/connection.json，字段host、port、identity_file；密钥文件只供SSH使用，不进入Git或Release。全局研究索引仍仅本地维护。原确认的存储禁区、测试隔离及旧批次关闭状态保持。

报告中的运行耗时包括每站预检、训练、B选点、保存、C推理和审计的进程墙钟时间；发布、传输及人工实现时间未单独计时。粗略100步外推只用于执行可行性，不能作为实际耗时或等预算优势证明。
