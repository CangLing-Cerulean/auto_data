"""Register upstream byte identities and time-only indexes; never parse sensors."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/ett_v1'
STUDY = ROOT/'research_runs/ett_confirmation'
COMMIT = '1d16c8f4f943005d613b5bc962e9eeb06058cf07'
COLUMNS = ['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
SPLITS = {'A':(0,8640),'B':(8640,10080),'C':(10080,11520),'D':(11520,14400)}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    assert not (DATA/'manifest.json').exists()
    files, indexes = [], {}
    for name in ('ETTh1','ETTh2'):
        path = DATA/'raw'/(name+'.csv')
        assert path.resolve() == path
        dates = []
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream)
            assert next(reader) == COLUMNS
            for row in reader:
                assert len(row) == len(COLUMNS)
                # Sensor strings are not converted, retained, summarized or printed.
                stamp = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                if dates:
                    assert stamp-dates[-1] == timedelta(hours=1)
                dates.append(stamp)
        assert len(dates) >= 14400
        boundaries = {s:dict(row_start=a,row_stop=b,first_date=dates[a].isoformat(),last_date=dates[b-1].isoformat()) for s,(a,b) in SPLITS.items()}
        files.append(dict(path='raw/'+name+'.csv', bytes=path.stat().st_size, sha256=sha(path),
            url=f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{COMMIT}/ETT-small/{name}.csv',
            rows=len(dates), first_date=dates[0].isoformat(), last_date=dates[-1].isoformat(),
            hourly_continuous=True, partitions=boundaries, reserve_rows=len(dates)-14400,
            sensor_values_parsed=False))
        rows = []
        for split,(lo,hi) in SPLITS.items():
            if split == 'D':
                continue
            for start in range(lo,hi-192+1):
                rows.append(dict(split=split,input_start=start,target_start=start+96,
                    target_stop=start+192,group=('EARLY' if start+96 < 4320 else 'LATE') if split == 'A' else ''))
        index = STUDY/(name+'_windows.csv')
        with index.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['split','input_start','target_start','target_stop','group'])
            writer.writeheader(); writer.writerows(rows)
        counts = {s:sum(r['split'] == s for r in rows) for s in ('A','B','C')}
        assert counts == {'A':8449,'B':1249,'C':1249}
        groups = {g:sum(r['split']=='A' and r['group']==g for r in rows) for g in ('EARLY','LATE')}
        assert groups == {'EARLY':4224,'LATE':4225}
        indexes[name] = dict(path=index.relative_to(ROOT).as_posix(),sha256=sha(index),counts=counts,training_groups=groups)
    support = {name:dict(bytes=(DATA/name).stat().st_size,sha256=sha(DATA/name)) for name in ('UPSTREAM_README.md','UPSTREAM_LICENSE')}
    manifest = dict(dataset_id='ett_v1',upstream_commit=COMMIT,source_repository='https://github.com/zhouhaoyi/ETDataset',
        license='CC BY-ND 4.0',files=files,support_files=support,columns=COLUMNS,
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        access='integrity bytes, schema and time only; no measurement statistics or model feedback',
        ETTh2_measurements_sealed=True,original_files_unchanged=True)
    save(DATA/'manifest.json',manifest)
    save(STUDY/'data_lock.json',dict(status='PASS_time_and_identity_only',manifest_sha256=sha(DATA/'manifest.json'),
        protocol_sha256=sha(STUDY/'PROTOCOL.md'),registration_code_sha256=sha(Path(__file__)),indexes=indexes,
        training_enabled=False,new_full_trainings=0,cumulative_full_trainings=434,
        benchmark_reproduction_claim=False,measurement_values_parsed=False))
    print(json.dumps(dict(status='PASS_time_and_identity_only',files=[{k:v[k] for k in ('path','rows','bytes','sha256')} for v in files],
        indexes=indexes,measurement_values_parsed=False),indent=2))


if __name__ == '__main__':
    main()
