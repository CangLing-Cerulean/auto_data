# 复核范围与步骤

本阶段仅依赖Python和NumPy。实测NumPy版本为2.4.2，无GPU、Torch、基础CSV或服务器要求。

1. 按source_lock.json核对DESIGN.md、check_plan.json与两个核心/检查脚本的原始字节SHA256。
2. 从仓库根目录运行`python scripts/check_gradient_transport_quadratic.py`，重建algebra_check.json。它自行核对锁，检查全部固定样例；失败即停止，不搜索替代常数。engineering_seconds按本次运行更新，数值与计数应一致至声明精度。
3. 运行`python scripts/audit_gradient_transport_hand_case.py`，重建hand_check.json。它用Fraction计算一个独立手算例，核对非零梯度缺陷，并汇总全部构造结果。
4. 阅览ANALYSIS.md的条件化限制。固定候选下的单步梯度性质，不等于对事后选择候选、Adam响应或真实float32程序的保证。

未生成或修改任何基础数据记录，未读取基础CSV、A/B/C/D/test数组或模型检查点。checkpoint_review.json只归纳此前已揭示的审计摘要，包括旧推荐收益，记录其哈希。真实训练新增0、累计434；检查脚本中的2810次合成核心更新及2810次逐坐标参考更新单列，不是研究数据训练，也不是其他方法的基准复现。
