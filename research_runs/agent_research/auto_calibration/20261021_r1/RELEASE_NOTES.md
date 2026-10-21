# Automatic calibrator training: verified partial prefix

Fixed-backbone ETTh1 calibration development pilot. This release preserves five completed calibrator fits (560 Adam updates), six checkpoints including identity, validation predictions, independent audit evidence, frozen protocol, and initialization failure records.

The Agent comparison is incomplete: the first CLI attempt failed during local app-server initialization before a model turn. C/D remain unparsed. Random search selects RANDOM_02 on V, with 0.119761% MSE improvement over identity; this is not a future-block or Agent result. No new backbone training; historical total remains 515.

The public archive excludes F/V source arrays and private configuration. The private reproducibility bundle remains local because the existing research server closed the SSH connection before a verified upload receipt. See REPORT.md and sync receipts for final status.
