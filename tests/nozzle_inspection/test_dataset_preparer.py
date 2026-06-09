from pathlib import Path
import tempfile
import unittest

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

            self.assertEqual(report.total_candidates, 5)
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

    @staticmethod
    def _write_pair(root: Path, split: str, stem: str, image_bytes: bytes, label_text: str) -> None:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / f"{stem}.jpg").write_bytes(image_bytes)
        (label_dir / f"{stem}.txt").write_text(label_text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
