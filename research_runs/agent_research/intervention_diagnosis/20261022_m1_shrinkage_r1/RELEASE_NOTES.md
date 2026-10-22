# M1 prefix shrinkage development evaluation

Closed-form scalar and channel-wise prefix calibration on the completed Hydraulic experiment, with the original 500-step prefix and fixed evaluation rules. No additional training or LLM calls; historical completed full runs remain 593.

Channel shrinkage reduces the primary 36-cell response MAE from M1's 9.564270 pp to 6.730335 pp (29.6304%), preserving M1's own task-wise selected true gains. Both predeclared variants pass the development criteria. This is post-hoc known-source development, with no new selection gain or independent-source confirmation.

Independent audit: 72 coefficient fits, 864 forecast matrices, 1,728 scores and 432 RAW scores passed. Local CPU timing is reported separately from reused historical GPU work.

The attached archive contains this stage's source, forecasts, coefficients, curves, logs and reports. Original ABC arrays and real checkpoints are provided by the previously published [prefix experiment archive](https://github.com/CangLing-Cerulean/auto_data/releases/tag/research-20261022-m1-contribution-r1), SHA256 `dc4778da7453b63e818029ae9b92b5400073e37fec76b3a6287f81c277f78119`.
