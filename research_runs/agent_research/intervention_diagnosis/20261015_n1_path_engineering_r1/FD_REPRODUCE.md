# N1有限差分证据复核入口

本文件说明已完成资产的阅读与复核顺序，不激活新训练。冻结任务、数值定义和停止规则以PROTOCOL.md及fd_plan.json为准；当前结果见FD_REPORT.md，远端交付以fd_sync_receipt.json为准。

1. 核对fd_source_sync_receipt.json中的源码提交、fd_plan与代码SHA256。路径阶段使用其独立source_sync_receipt，有限差分执行器有单独的源码锁。
2. 核对fd.json中的16个固定运行ID及各result SHA；fd_status.json只记录运行完成状态。独立审计fd_audit.json将运行完整性status与numerical_gate_pass分开，status=PASS不等于数值门槛通过。
3. 审计覆盖16条完整轨迹、112个模型/Adam/RNG状态、192组B/C残差，以及全部96导数与48双步长比较。第四质量方向及低信号项不能删除，任何一项超0.02均使原数值门槛失败。
4. fd_derivative_table.csv与fd_two_h_table.csv逐行保留误差分子、分母、两侧范数、父残差范数、低信号标志和原评分。fd_direction_table.csv只按预登记方向汇总；fd_comparisons.png/pdf显示全部144比较点。
5. Release归档包含原输入、冻结源码、全部父和FD状态、运行日志、控制退出记录和独立审计。先核对压缩包大小/SHA256，再用scripts/verify_fourgroup_m1_archive.py在新的本地目录逐成员检查；它拒绝覆盖既有恢复目录、符号链接及特殊文件。

汇总可以在一个新的本地报告目录执行：将原fd_plan.json、fd.json、fd_audit.json逐字节复制进去，再运行scripts/summarize_n1_fd.py --stage该目录；输出采用新建模式。绘图入口scripts/plot_n1_fd.py使用同一审计数据，不读取训练数据或检查点。先核验summary的输入绑定，再绘图。

服务器原执行器具有create-only锁。不得在既有项目上再次启动fd或audit_fd，不得修改锁来重跑已完成ID。独立环境中的完整复现需要单独声明运行目录及环境适配，保留原结果和历史计时；跨设备无法预设逐位相同。

运行完整性、有限差分精度、真实数据干预收益是三个不同证据层次。本阶段仅回答前两项；既有M1标签关闭状态与N2未满足的科学前置不因此改变。
