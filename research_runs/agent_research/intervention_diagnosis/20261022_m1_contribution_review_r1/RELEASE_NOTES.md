# M1 contribution review and trajectory attribution

The review identifies M1's fixed-parent parameter correction as a task-specific use of established zeroth-order additive multifidelity correction. The completed Hydraulic study supports local decision value, not a general accuracy improvement or established algorithmic novelty.

Read-only attribution covered all 36 candidates and 216 fixed checkpoints. RAW/M1/PREFIX response MAE was 8.074144/9.564270/9.244630 percentage points; selected real gains were 17.768573/18.863060/18.863060 percent. M1 avoided RAW's harmful G3 recommendation despite higher aggregate prediction error. Verification covered 918 original metric values, 216 parameter formulas, 766 input hashes, and 108 independently recomputed response decompositions. No new training was run; the historical full-training count remains 593.

This release attaches the unchanged, previously sealed source-study archive used by the analysis. It contains 2,381 files and includes the original failed audit and successful recovery. The archive is 84,728,844 bytes; SHA256 is `dc4778da7453b63e818029ae9b92b5400073e37fec76b3a6287f81c277f78119`. Its publication does not change any original scientific status, prediction lock, or score.

To reproduce attribution, extract the source archive, then run `scripts/analyze_m1_contribution.py` with `--root` pointing to its `20261022_prefix_response_r1` directory and `--output` pointing to a new file. This requires the recorded source environment and compatible CUDA inference. It never trains or updates model parameters. Local `scripts/verify_m1_contribution.py` independently checks the committed results against an extracted copy of the same source. Outputs are create-only.

Reports, analysis code, result JSON, source hashes, and verification receipts are bound to the release commit. RAW is explicitly a post-hoc ablation, not an additional pre-registered prediction. D/test remain untouched; no old closed study is reopened.
