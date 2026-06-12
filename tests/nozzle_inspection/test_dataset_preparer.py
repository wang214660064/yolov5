from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from project.nozzle_inspection.data.dataset_preparer import DatasetPreparer


class DatasetPreparerTest(unittest.TestCase):
    def test_prepare_deduplicates_converts_and_splits_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset_2"
            output = root / "prepared"
            self._write_pair(source, "train", "nozzle_tip_wrapped_0001", b"ng-1", "3 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "train", "nozzle_tip_wrapped_0002", b"ng-2", "3 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "train", "nozzle_clean_0001", b"ok-1", "1 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "val", "nozzle_clean_0002", b"ok-2", "1 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "val", "nozzle_clean_0003", b"ok-2", "1 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "test", "nozzle_clean_0099", b"test", "1 0.5 0.5 0.2 0.2\n")

            report = DatasetPreparer(source, output, val_ratio=0.5, seed=1).prepare()

            self.assertEqual(report.total_candidates, 6)
            self.assertEqual(report.duplicate_count, 1)
            self.assertTrue((output / "images" / "train").exists())
            self.assertTrue((output / "images" / "val").exists())
            self.assertTrue((output / "images" / "test").exists())
            all_labels = list((output / "labels").rglob("*.txt"))
            self.assertGreaterEqual(len(all_labels), 4)
            label_text = "\n".join(path.read_text(encoding="utf-8") for path in all_labels)
            self.assertIn("0 0.500000 0.500000 0.200000 0.200000", label_text)
            self.assertIn("1 0.500000 0.500000 0.200000 0.200000", label_text)
            self.assertTrue((output / "data_prepare_report.md").exists())
            self.assertTrue((output / "deduplicate_report.csv").exists())

    def test_prepare_merges_train_val_test_before_deduplication_and_resplit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset_2"
            output = root / "prepared"
            self._write_pair(source, "train", "nozzle_tip_wrapped_0001", b"ng-a", "3 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "train", "nozzle_tip_wrapped_0002", b"ng-b", "3 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "val", "nozzle_clean_0001", b"ok-a", "1 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "val", "nozzle_clean_0002", b"ok-b", "1 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "test", "nozzle_tip_wrapped_0099", b"ng-a", "3 0.5 0.5 0.2 0.2\n")
            self._write_pair(source, "test", "nozzle_clean_0099", b"ok-c", "1 0.5 0.5 0.2 0.2\n")

            report = DatasetPreparer(source, output, val_ratio=0.25, seed=3).prepare()

            output_images = list((output / "images").rglob("*.jpg"))
            output_names = {path.name for path in output_images}
            self.assertEqual(report.total_candidates, 6)
            self.assertEqual(report.exact_duplicate_count, 1)
            self.assertEqual(len(output_images), 5)
            duplicated_names = {"nozzle_tip_wrapped_0001.jpg", "nozzle_tip_wrapped_0099.jpg"}
            self.assertEqual(len(output_names & duplicated_names), 1)
            self.assertEqual(report.train_count + report.val_count + report.test_count, 5)
            self.assertGreater(report.test_count, 0)

    def test_prepare_removes_exact_and_ssim_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "dataset_2"
            output = root / "prepared"
            base = np.full((128, 128, 3), 220, dtype=np.uint8)
            cv2.rectangle(base, (20, 20), (60, 60), (30, 30, 30), -1)
            similar = base.copy()
            similar[24:28, 24:28] = 35
            different = np.zeros((128, 128, 3), dtype=np.uint8)
            self._write_image_pair(source, "train", "nozzle_tip_wrapped_0001", base, "3 0.5 0.5 0.2 0.2\n")
            self._write_image_pair(source, "train", "nozzle_tip_wrapped_0002", base, "3 0.5 0.5 0.2 0.2\n")
            self._write_image_pair(source, "train", "nozzle_tip_wrapped_0003", similar, "3 0.5 0.5 0.2 0.2\n")
            self._write_image_pair(source, "train", "nozzle_clean_0001", different, "1 0.5 0.5 0.2 0.2\n")
            self._write_image_pair(source, "val", "nozzle_clean_0002", np.full((64, 64, 3), 30, dtype=np.uint8), "1 0.5 0.5 0.2 0.2\n")

            report = DatasetPreparer(source, output, val_ratio=0.5, seed=1, ssim_threshold=0.85).prepare()

            self.assertEqual(report.exact_duplicate_count, 1)
            self.assertEqual(report.similar_duplicate_count, 1)
            self.assertEqual(report.duplicate_count, 2)
            report_text = (output / "data_prepare_report.md").read_text(encoding="utf-8")
            self.assertIn("SSIM 相似阈值：0.85", report_text)
            self.assertIn("pHash 汉明距离阈值：4", report_text)
            self.assertIn("SSIM 相似重复样本数：1", report_text)

    @staticmethod
    def _write_pair(root: Path, split: str, stem: str, image_bytes: bytes, label_text: str) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(image_bytes)
        (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    @staticmethod
    def _write_image_pair(root: Path, split: str, stem: str, image: np.ndarray, label_text: str) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_dir / f"{stem}.jpg"), image)
        (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
