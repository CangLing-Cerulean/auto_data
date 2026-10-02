# 许可与引用

MetroPT-3 原始数据由 UCI Machine Learning Repository 发布，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本仓库的代码和评测元数据不改变该原始数据的署名要求。

建议数据集引用：

> Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). MetroPT-3 Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R

来源：<https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B>

相关论文：

- Davari et al., “Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry,” DSAA 2021.
- Veloso et al., “The MetroPT dataset for predictive maintenance,” Scientific Data 9, 764 (2022).

`raw/` 中的官方文件不纳入 Git。再分发时必须同时保留本文件、DOI、来源 URL 和 CC BY 4.0 署名。

## P2 数据源

SCANIA Component X（CC BY 4.0）：

> Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., & Andersson Reyna, O. (2025). SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance (Version 3). Scania CV AB. https://doi.org/10.5878/bnh5-ka77

UCI Hydraulic Systems（CC BY 4.0）：

> Helwig, N., Pignanelli, E., & Schütze, A. (2015). Condition monitoring of hydraulic systems [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CW21

NASA IMS Bearings：NASA PCoE 公开数据，由 University of Cincinnati IMS 提供。规范入口：<https://data.nasa.gov/dataset/ims-bearings>；PHM Society 镜像：<https://data.phmsociety.org/nasa/>。

`raw_p2/` 不纳入 Git。P2 使用的具体文件、镜像、哈希、许可与子集限制以 `p2_dataset_manifest.yaml` 为准。

## SKAB v0.9

Skoltech Anomaly Benchmark（SKAB）上游仓库声明为 GPL-3.0。本项目扩展版使用完整 35 个 CSV 实验文件，并保留原始归档及逐文件哈希。

> Katser, I. D., & Kozitsin, V. O. (2020). Skoltech Anomaly Benchmark (SKAB). Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1693952

规范仓库：<https://github.com/waico/SKAB>。再分发 SKAB 数据或其衍生准备层时必须遵守 GPL-3.0；精确文件范围和哈希见 `expanded_dataset_manifest.yaml`。

## NASA C-MAPSS Turbofan

v0.7.0 使用 NASA Prognostics Center of Excellence 发布的 C-MAPSS 涡扇发动机退化模拟数据。该数据是高保真模拟的运行至失效序列，不得表述为真实机队现场测量。

> Saxena, A., & Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames Prognostics Data Repository.

NASA 规范入口：<https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>。NASA 数据页未为该条目明确列出许可证；发布包保留 NASA 署名、官方入口、镜像来源、完整镜像哈希和不完整官方传输哈希，不额外推定许可。
