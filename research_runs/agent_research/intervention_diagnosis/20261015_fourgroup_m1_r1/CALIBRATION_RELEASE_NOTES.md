# Four-group M1 parent calibration

Frozen Hydraulic G1/G2/G3 development tasks, three existing seeds, four fixed groups and eight fixed loss-weight actions. The nine real and nine proxy parents were replayed; independent audit passed 126 states, 216 residual summaries and 18 moment checks. All three tasks select 1000-step SHORT by the frozen timing rule. Candidate forecasts and full labels are not included in this calibration phase.

The archive contains all 760 server-stage ordinary files at the closed calibration boundary: exact input copies, source/configuration locks, runtime and storage receipts, prepared weights, complete initial/checkpoint/Adam/RNG states, residual identities and metrics, timing, control logs and audit records. Archive and every member are independently hash-verified before publication. Scientific source commit remains b61621133a48a560bd32c461ddbd43477365c2a0; this release binds the audited calibration result commit.

No new complete scientific training; cumulative count remains 418. The 54,000 parent replay updates are charged separately. D/test sensor values were not accessed. Calibration readiness does not establish prediction accuracy, selection improvement or cross-source generalization. N1/N2 real-response extensions remain separately gated.
