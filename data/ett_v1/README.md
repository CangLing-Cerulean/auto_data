# ETT immutable source registration

Original source: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`. The original CSV bytes remain in `raw/` and are not modified by research code. `manifest.json` records SHA256 identities; `upstream_git_objects.json` permits independent Git-blob verification. Upstream README and license are retained locally without edits.

Attribution: Haoyi Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. The upstream repository carries **CC BY-ND 4.0**; consult the retained license. Public research artifacts contain recipes, indexes and aggregate results, not transformed sensor tables.

Only ETTh1 and ETTh2 are registered. Each contains 17,420 hourly rows with a date column and seven measurements. Registration examined raw byte identities, column names and the time column only. It did not parse measurement values, fit normalization, train a model or produce evaluation feedback. ETTh2 remains reserved for confirmation; all source files remain local.

The scientific task, access rules and prospective time partitions are fixed in `research_runs/ett_confirmation/PROTOCOL.md`. These are project-specific partitions and do not claim a reproduction of a leaderboard split. Base MetroPT3/SKAB/Hydraulic data and their protected partitions are unchanged.
