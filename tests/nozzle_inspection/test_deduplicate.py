from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from project.nozzle_inspection.data import deduplicate
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

    def test_ssim_duplicate_marks_visually_similar_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            c = root / "c.jpg"
            base = np.full((128, 128, 3), 220, dtype=np.uint8)
            cv2.rectangle(base, (20, 20), (60, 60), (30, 30, 30), -1)
            similar = base.copy()
            similar[24:28, 24:28] = 35
            different = np.zeros((128, 128, 3), dtype=np.uint8)
            cv2.imwrite(str(a), base)
            cv2.imwrite(str(b), similar)
            cv2.imwrite(str(c), different)

            report = Deduplicator().find_duplicates([a, b, c], ssim_threshold=0.85)

            self.assertIn(a, report.keep_files)
            self.assertIn(c, report.keep_files)
            self.assertIn(b, report.duplicate_files)
            self.assertEqual(report.similar_duplicate_pairs[0][0], a)
            self.assertEqual(report.similar_duplicate_pairs[0][1], b)
            self.assertGreaterEqual(report.similar_duplicate_pairs[0][2], 0.85)

    def test_similar_duplicate_requires_phash_prefilter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            base = np.full((128, 128, 3), 220, dtype=np.uint8)
            cv2.rectangle(base, (20, 20), (60, 60), (30, 30, 30), -1)
            changed = base.copy()
            cv2.rectangle(changed, (62, 20), (70, 60), (30, 30, 30), -1)
            cv2.imwrite(str(a), base)
            cv2.imwrite(str(b), changed)

            report = Deduplicator().find_duplicates([a, b], ssim_threshold=0.85, phash_threshold=5)

            self.assertEqual(report.keep_files, [a, b])
            self.assertEqual(report.duplicate_files, [])
            self.assertEqual(report.similar_duplicate_pairs, [])

    def test_similar_duplicate_reuses_precomputed_phash_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            base = np.full((128, 128, 3), 220, dtype=np.uint8)
            cv2.rectangle(base, (20, 20), (60, 60), (30, 30, 30), -1)
            similar = base.copy()
            similar[24:28, 24:28] = 35
            cv2.imwrite(str(a), base)
            cv2.imwrite(str(b), similar)
            deduplicator = Deduplicator()
            phash_cache = deduplicator.precompute_phashes([a, b])

            with patch.object(deduplicator, "precompute_phashes", side_effect=AssertionError("不应重复计算 pHash")):
                report = deduplicator.find_similar_duplicates([a, b], phash_cache=phash_cache)

            self.assertEqual(report.duplicate_files, [b])

    def test_exact_duplicate_uses_process_pool_when_workers_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            a = root / "a.jpg"
            b = root / "b.jpg"
            a.write_bytes(b"same-image")
            b.write_bytes(b"same-image")

            class InlineExecutor:
                def __init__(self, max_workers):
                    self.max_workers = max_workers

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return False

                def map(self, func, items):
                    return [func(item) for item in items]

            created_workers = []

            def build_executor(max_workers):
                created_workers.append(max_workers)
                return InlineExecutor(max_workers)

            with patch.object(deduplicate, "ProcessPoolExecutor", side_effect=build_executor):
                report = Deduplicator(workers=2).find_exact_duplicates([b, a])

            self.assertEqual(created_workers, [2])
            self.assertEqual(report.duplicate_files, [b])


if __name__ == "__main__":
    unittest.main()
