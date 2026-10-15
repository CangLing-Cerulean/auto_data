# 复核研究初稿

1. 从evidence_inventory.json定位20份已存在的报告或审计文件，核对大小和SHA256。其路径相对仓库根目录；源数据、训练资产和完整运行复现步骤仍以各原阶段协议及Release收据为准。
2. 使用带Matplotlib及NumPy的Python运行`python scripts/build_paper_evidence.py`，重建metrics.json、claim_evidence.csv、evidence_inventory.json、artifact_audit.json与evidence_overview.png/pdf。该程序只读取已揭示报告/审计JSON，不访问基础CSV、D/test，不运行训练。图像字节可能因绘图库版本而异，数值必须一致。
3. 在支持amsmath、amssymb、booktabs、array、xcolor、hyperref、microtype的LaTeX环境编译manuscript.tex。该文件可独立编译，参考文献和表格都在文件内，不依赖外部图或bib文件。Codex内置编辑器可提供实时PDF预览；compile_receipt.json记录本次实际编译结果。
4. 审阅claim_evidence.csv中的scope及boundary列。检查不止是表格数字：跨来源验证未完成、源码适配比较为事后、N1原数值门槛失败、强元重权对照缺失，都必须保留在论文表述中。

本制稿阶段不改变任何旧协议、预测、评分、训练ID或历史失败。新训练0，累计434。原始大资产沿用已有Release，本阶段没有新的检查点归档。
