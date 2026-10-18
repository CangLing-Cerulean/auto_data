# 与当前实现直接相关的既有工作

检索日期2026-10-18。只记录已核对的作者论文或正式出版页面；未声称系统综述，也未复现这些方法。此表用于限定稿件主张，不改冻结实验。

|既有工作及主来源|已核对内容|对当前研究的边界|
|---|---|---|
|Gneiting & Raftery, 2007, [JASA](https://doi.org/10.1198/016214506000001437)|概率预测的适当评分规则，包括对数评分；评分根据预测分布与实现结果评价预测质量。|对数评分及概率预测质量评价不能作为本研究独创。条件评分改善也不能直接代替行动收益。|
|Bissiri, Holmes & Walker, 2016, [JRSSB作者全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5082587/)|用损失而非完整似然连接观测与参数，建立累积损失下的一般信念更新框架。|指数累积评分形式本身不足以建立创新；我们未验证一般贝叶斯解释、后验校准或因果含义。|
|Hollmann, Müller & Hutter, 2023, [CAAFE正式论文](https://papers.neurips.cc/paper_files/paper/2023/file/8c2df4c35cdbee764ebb9e9d0acd5197-Paper-Conference.pdf)|利用数据语义描述让LLM生成附加特征的Python代码与用途说明，并反馈评价。|可执行LLM特征生成已有直接先例；本研究的候选差别是干预响应失配输入、父模型窗口测量和诊断复用，但其独立收益尚未证明。r4普通生成对照不是作者CAAFE复现。|
|Liu, Astorga, Seedat & van der Schaar, 2024, [LLAMBO正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/84b8d9fcb4e262fcd429544697e1e720-Paper-Conference.pdf)|利用上下文、历史评价与LLM进行优化，覆盖初始化、代理建模和候选采样。|基于完整历史反馈让LLM提议下一次查询已有先例。r5是本研究信息条件下的强提示词对照，不能称为LLAMBO复现或对其胜出。|

这些对应关系是本阶段依据论文内容作出的范围判断。当前稿件还需结合既有DoVer/LADDER核对结果讨论诊断验证与错误切片；不能只对比两个较远的Agent工作而忽略评分理论、自动特征工程和序贯黑箱优化。
