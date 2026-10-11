# SKAB / UCI Hydraulic 基础数据 v1

本阶段从原发送包挑选两个真实多传感器来源，固定顺序 7:2:1 划分、原始数值与源行映射、仅训练统计和测试隔离。

- SKAB：全部35个原始实验CSV，46860行，8通道；32802/9372/4686行。
- UCI Hydraulic：当前包内8个1Hz/虚拟传感器文件，2205×60=132300行；92610/26460/13230行。
- 逐值溯源、源哈希、分区哈希及静态输入/边界契约通过。旧MetroPT3文件保持原样。
- 没有新训练或测试评分。具体窗口/任务尚未冻结。SKAB窗口不得跨原实验，Hydraulic窗口不得跨周期；两者均禁止跨分区。

资产包含选定原始文件、准备层CSV、manifest、训练profile、许可/署名、建库与校验代码及阶段证据。排除完整发送包、ground_truth、benchmark答案和私有运行记录。包内package_files.json给出每个文件的大小与SHA256，Git中的assets.json给出ZIP身份。发布目标提交登记在同步回执中。

按资产中的项目相对路径恢复到现有auto_data工作区，已有MetroPT3和原发送包仍沿用原位置；不要覆盖不同身份的文件。校验命令为PowerShell 7运行 `./scripts/build_additional_datasets.ps1 -VerifyOnly`。
