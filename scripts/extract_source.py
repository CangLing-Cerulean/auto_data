"""Extract only the pinned Metro source; never extract evaluator ground truth."""
import hashlib
import io
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = 'industrial_timeseries_eval_120_v0.7.0.zip'
EXPECTED = 'b0c2489d1103f224c55e070da956f00e7905d4389850f31f5d2a34f42227a940'


def main():
    with zipfile.ZipFile(ROOT / 'share_20260727.zip') as outer:
        payload = outer.read('share_20260727/' + ARCHIVE)
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        for name in ('raw/MetroPT3(AirCompressor).csv', 'dataset_manifest.yaml',
                     'LICENSE_AND_CITATION.md', 'prepared/sensor_dictionary.csv'):
            target = ROOT / 'source/metropt3' / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                with inner.open('version2_v0.7.0/' + name) as src, target.open('rb') as dst:
                    assert hashlib.file_digest(src, 'sha256').digest() == hashlib.file_digest(dst, 'sha256').digest()
                continue
            with inner.open('version2_v0.7.0/' + name) as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
    print('PASS: pinned archive verified; Metro source extracted or verified.')


if __name__ == '__main__':
    main()
