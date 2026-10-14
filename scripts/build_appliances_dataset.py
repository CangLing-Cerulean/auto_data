"""Create an immutable UCI 374 source version; inspect held-out rows only for integrity."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://archive.ics.uci.edu/static/public/374/appliances%2Benergy%2Bprediction.zip'
EXPECTED_ROWS = 19735


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out != Path('/root/auto_data_experiments/datasets/uci_appliances_v1') or out.exists():
        raise RuntimeError('Unexpected or existing source directory')
    out.mkdir(parents=True, mode=0o700)
    request = urllib.request.Request(URL, headers={'User-Agent': 'auto-data-research/1.0'})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read(30 * 1024 * 1024)
        if response.read(1):
            raise RuntimeError('Source archive exceeds registered size limit')
    (out / 'source.zip').write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if archive.namelist() != ['energydata_complete.csv']:
            raise RuntimeError('Unexpected archive membership')
        raw = archive.read('energydata_complete.csv')
    (out / 'energydata_complete.csv').write_bytes(raw)
    lines = raw.splitlines(keepends=True)
    header = next(csv.reader([lines[0].decode('utf-8-sig')]))
    expected = ['date', 'Appliances', 'lights']
    for number in range(1, 10):
        expected += [f'T{number}', f'RH_{number}']
    expected += ['T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint', 'rv1', 'rv2']
    if header != expected or len(lines)-1 != EXPECTED_ROWS:
        raise RuntimeError('Source schema or row count changed')
    times = []
    for line in lines[1:]:
        row = next(csv.reader([line.decode('utf-8')]))
        if len(row) != len(header):
            raise RuntimeError('Malformed record')
        timestamp = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
        if times and (timestamp-times[-1]).total_seconds() != 600:
            raise RuntimeError('Unexpected time gap; stop without imputing or repartitioning')
        # Held-out numeric values are never summarized or exposed to the model.
        times.append(timestamp)
    bounds = [('train', 0, 15788), ('validation', 15788, 17761), ('test', 17761, 19735)]
    splits = []
    for name, lo, hi in bounds:
        data = lines[0] + b''.join(lines[lo+1:hi+1])
        (out / f'{name}.csv').write_bytes(data)
        splits.append(dict(name=name, base_row_start_inclusive=lo, base_row_end_exclusive=hi,
                           rows=hi-lo, bytes=len(data), sha256=sha(data),
                           first_timestamp=times[lo].isoformat(), last_timestamp=times[hi-1].isoformat()))
    manifest = dict(dataset_id='uci_appliances_v1', source_url=URL,
                    source_page='https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction',
                    citation='Candanedo, L. (2017). Appliances Energy Prediction. UCI. DOI:10.24432/C5VC8G',
                    license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
                    source_zip_sha256=sha(payload), source_zip_bytes=len(payload),
                    original_csv_sha256=sha(raw), original_csv_bytes=len(raw),
                    columns=header, rows=EXPECTED_ROWS, timestamp_interval_seconds=600,
                    row_mapping='Each split contains original CSV lines unchanged; base row indices are zero based data rows.',
                    excluded_model_columns=['date','rv1','rv2'],
                    split_policy='First 80% train; next floor(10%) validation; remainder test, frozen before download.',
                    splits=splits, sensor_records_generated=0,
                    heldout_access='byte copying, hashing, schema/time integrity only; no numeric summaries or model feedback',
                    model_training_started=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('dataset_id','rows','columns','source_zip_sha256','original_csv_sha256')}))


if __name__ == '__main__':
    main()
