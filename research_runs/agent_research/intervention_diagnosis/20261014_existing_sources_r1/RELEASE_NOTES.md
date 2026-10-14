# Existing-source forecast batch stopped before complete child labels

The registered MetroPT3/SKAB batch completed 6 full parent trainings, 30 proxy rollouts and 24 short runs. The 24 four-method predictions are preserved. MetroPT3 SHORT costs exceeded the frozen 27-second per-action and 108-second selection caps, so no complete child label was started. Cumulative complete trainings remain 373.

M1, RAW and SHORT recommend the same action on both sources. The registered positive recommendation-gain increment is therefore unattainable for this batch, regardless of unknown full child labels. Prediction accuracy and realized gains remain unknown. Prior Hydraulic results are unchanged.

The stop audit verified 1,570 asset hashes, counts, costs and recommendation identity. It is not a full numerical reforward audit of the candidates. All 1,620 archive members were individually verified on the server and after local download. The archive includes frozen code, prepared mappings and arrays, all saved model/optimizer/RNG checkpoints, residual digests, timing, predictions, stop report and audit records; it excludes base validation/test CSVs and the already separately archived MetroPT3 base training file. Included SKAB training data retain their source license/citation.

The prepared r2 final auditor and figure generator were never executed. No model queue remains. This release preserves a failed budget acceptance without changing the protocol or rerunning IDs.
