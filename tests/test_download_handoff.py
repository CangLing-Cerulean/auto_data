"""Exercise restoration and safeguards without network or research data access."""
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'download_handoff', Path(__file__).resolve().parents[1] / 'scripts/download_handoff.py')
handoff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handoff)


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.root_patch = patch.object(handoff, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.payload = b'original\r\nbytes\n'
        self.entry = {'path': 'data/train.csv', 'bytes': len(self.payload),
                      'sha256': hashlib.sha256(self.payload).hexdigest()}
        archive = self.cache / 'sample.zip'
        with zipfile.ZipFile(archive, 'w') as output:
            output.writestr(self.entry['path'], self.payload)
        self.asset = {'name': archive.name, 'bytes': archive.stat().st_size,
                      'sha256': handoff.digest(archive), 'files': [self.entry]}

    def test_restore_and_verify_preserves_bytes(self):
        handoff.restore(self.asset, self.cache, False)
        self.assertEqual((self.root / self.entry['path']).read_bytes(), self.payload)
        handoff.restore(self.asset, self.cache, True)

    def test_existing_different_file_is_not_overwritten(self):
        target = self.root / self.entry['path']
        target.parent.mkdir()
        target.write_bytes(b'keep this')
        with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
            handoff.restore(self.asset, self.cache, False)
        self.assertEqual(target.read_bytes(), b'keep this')

    def test_archive_checksum_failure_writes_no_data(self):
        (self.cache / self.asset['name']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            handoff.restore(self.asset, self.cache, False)
        self.assertFalse((self.root / 'data').exists())

    def test_file_checksum_failure_writes_no_data(self):
        self.entry['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Extracted file checksum mismatch'):
            handoff.restore(self.asset, self.cache, False)
        self.assertFalse((self.root / self.entry['path']).exists())

    def test_verify_only_does_not_download(self):
        with patch.object(handoff.urllib.request, 'urlopen') as download:
            with self.assertRaises(FileNotFoundError):
                handoff.restore(self.asset, self.cache, True)
            download.assert_not_called()

    def test_rejects_unsafe_paths(self):
        for path in ('../outside', '/absolute', 'C:/outside', 'data\\..\\outside'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                handoff.destination(path)


if __name__ == '__main__':
    unittest.main()
