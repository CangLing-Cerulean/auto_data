# UCI Electricity cold-start base data v1

Official source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
Fixed download: https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
Citation: Trindade, A. (2015). ElectricityLoadDiagrams20112014. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Official metadata describes 370 clients, 15-minute power measurements in kW during 2011–2014. Values before client creation are zero. Portuguese daylight-saving spring gaps are zero and autumn repeated periods are combined. Registration performs no interpolation or value edits.

`manifest.json` records original archive and every member identity, including unextracted metadata members. Only the unique root `LD2011_2014.txt` is saved to `raw/`; archive paths are never used as extraction destinations. Existing raw data are never overwritten.

Registration inspects only the header, field counts and timestamps; measurement fields remain unparsed. All 370 channels are retained. The fixed interval begins 2013-01-01 00:15:00 and contains 210 × 96 records: A 120 days, B 30 days, C 30 days and sealed D 30 days. Windows use 32 input and 8 target steps, stride 1, without crossing partitions. Counts are A 11481, B/C/D 2841 each. Coordinates and exact dates are in the manifest and stage `source_index.json`.

No training arrays, numerical measurement summaries, model evaluations or LLM calls are produced. D measurement values remain sealed; old sources are untouched. Training and measurement parsing require the separately frozen experiment protocol.
