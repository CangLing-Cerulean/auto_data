# Transfer diagnosis: fixed 18-scenario replay

Complete known-source Hydraulic development replay: 54 auxiliary fits, 54,000 updates, 18 ridge fits; no new backbone training or LLM calls. Independent reconstruction passed for all 18 folds, 72 models and 864 predictions.

Unprobed response MAE: OVERALL 7.879503 pp, JOINT 7.097952 pp, SEPARATE 7.252321 pp. Local diagnostic supervision improves over overall feedback by 9.9188% (17 of 18 scenarios). Separate updates do not improve on joint updates. Both diagnostic methods preserve M1's per-task recommendation gains, averaging 18.863060%; no incremental selection gain over M1. The complete H1+H2 criterion is false. This is same-source development replay, not independent confirmation or an originality claim.

The archive contains 141 scientific files: frozen protocol and source, 18 derived input folds, 72 auxiliary model states, full predictions, tables, report, figure, and original preparation failure/recovery logs. Archive SHA256: `0b637d4968554896610d098a7937c35ba2dcc9fe4236ad174d95ca8954854ce8`; size: 3,113,013 bytes. All members were verified locally and on the research server. Original raw ABC arrays and backbone checkpoints are referenced through the prior published archives, not duplicated here.

The first cross-runtime preparation failed coefficient consistency; the identical frozen program passed in the original CPU environment and the recovery completed without changing tolerances or scientific settings. See REPORT.md and PREPARATION_RECOVERY.md. Publication authorization and delivery receipts were recorded after the immutable scientific archive was sealed; use the repository metadata for the final publication state.
