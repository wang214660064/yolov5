from pathlib import Path
import tempfile
import unittest

from project.nozzle_inspection.data.deduplicate import Deduplicator


class DeduplicatorTest(unittest.TestCase):
    def test_exact_duplicate_keeps_first_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            c = root / "c.jpg"
            a.write_bytes(b"same-image")
            b.write_bytes(b"same-image")
            c.write_bytes(b"different-image")

            report = Deduplicator().find_exact_duplicates([b, c, a])

            self.assertEqual(report.keep_files, [a, c])
            self.assertEqual(report.duplicate_files, [b])
            self.assertEqual(report.duplicate_pairs, [(a, b)])


if __name__ == "__main__":
    unittest.main()
