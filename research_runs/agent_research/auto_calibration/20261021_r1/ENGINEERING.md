# 工程验收记录

核心模块首轮10项测试通过，独立检查另核对9种实际更新步数/训练窗口排除/热启动复制/状态验收/通道隔离，未复现缺陷。父检查点与已发布归档成员哈希一致；NumPy及Torch双精度骨干参考最大差4.44e-16。F/V已准备，C尚未解析。

编排新增7项检查后首次共17项：13通过，4项因Windows系统临时目录创建后无法读写/清理而报PermissionError。没有进入真实研究LLM或校准拟合，也没有读取C。测试临时目录改为已授权工作区内后补验；原失败不计作算法失败或通过记录。最终补验输出另存engineering_tests.txt。

CLI首次将--ignore-user-config置于顶层命令导致参数错误；查阅本机帮助后确认它属于exec子命令，实际入口使用正确位置。仅帮助/特性查询，未调用研究模型。配置依据官方Config Basics及本机feature list；实际信息隔离还需逐次事件验收。

独立编排审查发现两项：原工具事件黑名单漏掉collab_tool_call，合成事件被接纳；F推理与V推理耗时未分别记录。修正为仅允许reasoning/agent_message的白名单，拒绝未知item及turn.failed；训练、F评价、V评价分别计时。修复在真实校准拟合和LLM调用之前完成，新增协作/未知工具/failed-turn回归覆盖。最终17项补验另存engineering_tests_final.txt；独立复核结果见orchestration_review.json。
