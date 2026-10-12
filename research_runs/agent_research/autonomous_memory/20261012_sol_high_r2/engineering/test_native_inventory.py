"""Output inventory races and resource boundaries, without sensor data."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from online_ts.agent_research import native
from online_ts.config import GateError


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def test_atomic_temporary_file_disappears_after_enumeration(self):
        path = self.folder/'history.json.tmp'
        path.write_text('{}', encoding='utf-8')
        def renamed(_folder, _pattern):
            path.replace(self.folder/'history.json')
            yield path
        with patch.object(Path, 'rglob', renamed):
            self.assertEqual(native.inventory(self.folder, 100, 10), ([], 0))
        self.assertEqual(native.inventory(self.folder, 100, 10)[1], 2)

    def test_regular_nested_files_counted(self):
        (self.folder/'nested').mkdir()
        path = self.folder/'nested'/'result.json'
        path.write_bytes(b'123')
        self.assertEqual(native.inventory(self.folder, 3, 1), ([path], 3))

    def test_byte_limit_retained(self):
        (self.folder/'a').write_bytes(b'123')
        with self.assertRaisesRegex(GateError, 'NATIVE_OUTPUT_LIMIT'):
            native.inventory(self.folder, 2, 10)

    def test_file_limit_retained(self):
        (self.folder/'a').touch()
        (self.folder/'b').touch()
        with self.assertRaisesRegex(GateError, 'NATIVE_OUTPUT_LIMIT'):
            native.inventory(self.folder, 100, 1)

    @unittest.skipUnless(os.name == 'posix', 'Linux native worker')
    def test_dangling_symlink_rejected(self):
        (self.folder/'alias').symlink_to(self.folder/'missing')
        with self.assertRaisesRegex(GateError, 'NATIVE_OUTPUT_LINK'):
            native.inventory(self.folder, 100, 10)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'Linux native worker')
    def test_fifo_rejected(self):
        os.mkfifo(self.folder/'pipe')
        with self.assertRaisesRegex(GateError, 'NATIVE_OUTPUT_TYPE'):
            native.inventory(self.folder, 100, 10)

    def test_unexpected_stat_error_propagates(self):
        path = self.folder/'a'
        with patch.object(Path, 'rglob', return_value=iter([path])), \
                patch.object(Path, 'lstat', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                native.inventory(self.folder, 100, 10)


if __name__ == '__main__':
    unittest.main()
