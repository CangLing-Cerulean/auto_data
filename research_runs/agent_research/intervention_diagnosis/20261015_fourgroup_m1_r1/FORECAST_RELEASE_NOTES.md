# Four-group M1 audited forecasts and frozen stop

The fixed Hydraulic development batch completed 72 proxy rollouts, 72 SHORT runs, and SOURCE predictions. Independent replay passed for 144 trajectories, 288,000 updates, 1,008 states, 2,880 residual checks, 216 moment files, and 27,648 SOURCE propagation blocks.

All nine M1-versus-SHORT cost gates passed. M1 selected the same action as both RAW and SHORT on all three tasks, triggering the preregistered stop before full child labels. No full child training was started; cumulative full training count remains 418. Actual response accuracy, gains, NMAE/channel guards, oracle regret, and the full-enumeration cost gate remain unknown.

The archive contains 4,085 ordinary files (228,118,355 uncompressed bytes), including the locked inputs, code, states, optimizer/RNG state, timing, predictions, audits and six closed control phases. It excludes the duplicate transport bundle and older delivery directories. No D sensor values or test data were accessed by this batch. The stage report, prediction lock and readable tables accompany this release in Git.

Archive: `fourgroup_m1_forecast_20261015.tar.gz`, 172,403,440 bytes, SHA256 `6befe9053fa4771640d7eecff14db5413ff6c571f65f35aa46f37e931c6abe11`.

This is a closed same-source development batch, not evidence of real child gains or independent-source superiority. N1/N2 synthetic engineering remains separate and does not reopen this batch's labels.
