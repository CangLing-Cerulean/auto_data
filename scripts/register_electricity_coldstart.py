"""Register UCI Electricity byte identities, header and time fields only."""
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/uci_electricity_v1'
STUDY = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10'
ARCHIVE = DATA / 'raw/electricityloaddiagrams20112014.zip'
TEXT = DATA / 'raw/LD2011_2014.txt'
START = datetime(2013, 1, 1, 0, 15)
STEP = timedelta(minutes=15)
SPLIT_DAYS = {'A': 120, 'B': 30, 'C': 30, 'D': 30}
INPUT_STEPS, TARGET_STEPS = 32, 8
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014'
DOWNLOAD_URL = 'https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def register_members():
    members = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = archive.infolist()
        candidates = [entry for entry in entries if entry.filename == 'LD2011_2014.txt']
        if len(candidates) != 1:
            raise ValueError('Expected exactly one root LD2011_2014.txt member')
        if sum(Path(entry.filename).name == 'LD2011_2014.txt' for entry in entries) != 1:
            raise ValueError('Ambiguous data member basename')
        for entry in entries:
            digest = hashlib.sha256()
            size = 0
            destination = None
            # Only this fixed literal destination may be written; no member path is extracted.
            if entry is candidates[0] and not TEXT.exists():
                destination = TEXT.open('xb')
            try:
                with archive.open(entry) as source:
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        if destination is not None:
                            destination.write(chunk)
            finally:
                if destination is not None:
                    destination.close()
            if size != entry.file_size:
                raise ValueError('Archive member size mismatch')
            identity = dict(name=entry.filename, bytes=size, compressed_bytes=entry.compress_size,
                            crc32=f'{entry.CRC:08x}', sha256=digest.hexdigest(),
                            is_directory=entry.is_dir(), extracted=entry is candidates[0])
            members.append(identity)
            if entry is candidates[0]:
                if TEXT.stat().st_size != size or sha(TEXT) != identity['sha256']:
                    raise ValueError('Saved text identity mismatch')
    return members


def time_index():
    selected = []
    previous = None
    first = None
    count = 0
    with TEXT.open('rb') as stream:
        header = next(csv.reader([stream.readline().decode('utf-8-sig').rstrip('\r\n')], delimiter=';'))
        expected = [''] + [f'MT_{number:03d}' for number in range(1, 371)]
        if header != expected:
            raise ValueError('Expected empty time header followed by exactly MT_001 through MT_370')
        for row_number, line in enumerate(stream):
            if line.count(b';') != 370:
                raise ValueError(f'Incorrect field count at data row {row_number}')
            # Never decode, convert, retain, summarize or print measurement fields.
            time_bytes = line.partition(b';')[0].strip(b'"')
            stamp = datetime.strptime(time_bytes.decode('ascii'), '%Y-%m-%d %H:%M:%S')
            if previous is not None and stamp - previous != STEP:
                raise ValueError(f'Non-quarter-hour time index at data row {row_number}')
            if first is None:
                first = stamp
            previous = stamp
            count += 1
            if START <= stamp < START + STEP * (210 * 96):
                selected.append((row_number, stamp))
    if len(selected) != 210 * 96 or selected[0][1] != START:
        raise ValueError('Fixed 210-day interval is unavailable')
    offset = selected[0][0]
    splits = {}
    cursor = 0
    for split, days in SPLIT_DAYS.items():
        stop = cursor + days * 96
        splits[split] = dict(row_start=offset + cursor, row_stop=offset + stop,
                             first_date=selected[cursor][1].isoformat(),
                             last_date=selected[stop - 1][1].isoformat(),
                             records=stop - cursor,
                             windows=stop - cursor - INPUT_STEPS - TARGET_STEPS + 1,
                             measurements_sealed=True)
        cursor = stop
    if {name: split['windows'] for name, split in splits.items()} != {'A': 11481, 'B': 2841, 'C': 2841, 'D': 2841}:
        raise ValueError('Unexpected split window counts')
    return dict(rows=count, first_date=first.isoformat(), last_date=previous.isoformat(),
                quarter_hour_continuous=True, columns=header, selected_start=START.isoformat(),
                selected_records=len(selected), partitions=splits,
                input_steps=INPUT_STEPS, target_steps=TARGET_STEPS,
                window_stride=1, windows_cross_partitions=False,
                row_coordinates='zero-based data rows, excluding header; stop exclusive')


def main():
    receipt = STUDY / 'download_receipt.json'
    if not receipt.is_file():
        raise FileNotFoundError('Download receipt required before registration')
    downloaded = json.loads(receipt.read_text(encoding='utf-8-sig'))
    if downloaded['status'] != 'downloaded' or downloaded['url'] != DOWNLOAD_URL:
        raise ValueError('Unexpected download receipt status or source')
    if ARCHIVE.stat().st_size != downloaded['bytes'] or sha(ARCHIVE) != downloaded['sha256']:
        raise ValueError('Archive does not match download receipt')
    for output in (DATA / 'manifest.json', STUDY / 'source_index.json', DATA / 'README.md'):
        if output.exists():
            raise FileExistsError(f'Create-only output already exists: {output}')
    members = register_members()
    index = time_index()
    manifest = dict(dataset_id='uci_electricity_v1', source_url=SOURCE_URL,
                    download_url=DOWNLOAD_URL, doi='10.24432/C58C86', license='CC BY 4.0',
                    registered_at_utc=datetime.now(timezone.utc).isoformat(),
                    archive=dict(path=ARCHIVE.relative_to(DATA).as_posix(),
                                 bytes=ARCHIVE.stat().st_size, sha256=sha(ARCHIVE)),
                    archive_members=members,
                    text=dict(path=TEXT.relative_to(DATA).as_posix(), bytes=TEXT.stat().st_size,
                              sha256=sha(TEXT)),
                    download_receipt_sha256=sha(receipt), time_index=index,
                    measurement_values_parsed=False, original_archive_unchanged=True)
    save(DATA / 'manifest.json', manifest)
    save(STUDY / 'source_index.json', dict(
        status='PASS_time_and_identity_only', dataset_id=manifest['dataset_id'],
        manifest_path=(DATA / 'manifest.json').relative_to(ROOT).as_posix(),
        manifest_sha256=sha(DATA / 'manifest.json'),
        registration_code_sha256=sha(Path(__file__)),
        source_protocol_sha256=sha(STUDY / 'SOURCE_PROTOCOL.md'),
        user_scope_sha256=sha(STUDY / 'USER_SCOPE.md'), **index,
        measurement_values_parsed=False, D_measurements_sealed=True,
        training_enabled=False, llm_enabled=False, new_full_trainings=0))
    readme = '''# UCI Electricity cold-start base data v1

Official source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
Fixed download: https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
Citation: Trindade, A. (2015). ElectricityLoadDiagrams20112014. UCI Machine Learning Repository. https://doi.org/10.24432/C58C86
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Official metadata describes 370 clients, 15-minute power measurements in kW during 2011–2014. Values before client creation are zero. Portuguese daylight-saving spring gaps are zero and autumn repeated periods are combined. Registration performs no interpolation or value edits.

`manifest.json` records original archive and every member identity, including unextracted metadata members. Only the unique root `LD2011_2014.txt` is saved to `raw/`; archive paths are never used as extraction destinations. Existing raw data are never overwritten.

Registration inspects only the header, field counts and timestamps; measurement fields remain unparsed. All 370 channels are retained. The fixed interval begins 2013-01-01 00:15:00 and contains 210 × 96 records: A 120 days, B 30 days, C 30 days and sealed D 30 days. Windows use 32 input and 8 target steps, stride 1, without crossing partitions. Counts are A 11481, B/C/D 2841 each. Coordinates and exact dates are in the manifest and stage `source_index.json`.

No training arrays, numerical measurement summaries, model evaluations or LLM calls are produced. D measurement values remain sealed; old sources are untouched. Training and measurement parsing require the separately frozen experiment protocol.
'''
    with (DATA / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(readme)
    print(json.dumps(dict(status='PASS_time_and_identity_only', rows=index['rows'],
                          windows={name: split['windows'] for name, split in index['partitions'].items()},
                          measurement_values_parsed=False), indent=2))


if __name__ == '__main__':
    main()
