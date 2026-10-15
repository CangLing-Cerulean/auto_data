# 复现范围

Python 3.14及NumPy 2.4.2；代码主动将OpenBLAS/MKL/OMP/BLIS线程环境设为1。实际NumPy构建配置记录于kernel_check.json，线程未通过外部运行时探针二次确认。首轮用本项目已有venv运行，无依赖安装。NumPy配置输出提示可安装pyyaml以改善显示，未影响结果、未据此安装依赖。

在项目根目录执行scripts/check_gradient_transport_feasibility.py --stage 新输出目录。新目录先复制PLAN.md、source_lock.json，保持锁内相对源码路径可用。程序验证锁并拒绝覆盖已有kernel_check.json。固定18行全部保留，源码中没有真实数据读取或Adam更新。

source_lock先于首次执行生成并验证；本轮没有依结果调整公式、形状、种子、容差或重复数。kernel_check.json是第一次执行产物。summary.json和artifact_audit.json从该产物和固定公式重算；不重复性能计时、不重跑旧实验。

单步核对仅支持公式重排及原坐标块计算，不支持完整DLinear float32路径逐位一致。CPU计时只用于暴露共享开销；跨机器复现时耗时可以变化，不作为科学通过门槛。本阶段没有模型训练、传感器样本、候选效果预测或新的真实实验授权。
