from pathlib import Path
import tempfile
import unittest

from project.nozzle_inspection.data.dataset_analyzer import DatasetAnalyzer


class DatasetAnalyzerTest(unittest.TestCase):
    def test_analyzer_counts_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "train" / "images").mkdir(parents=True)
            (root / "train" / "labels").mkdir(parents=True)
            (root / "train" / "images" / "nozzle_clean_0001.jpg").write_bytes(b"image")
            (root / "train" / "labels" / "nozzle_clean_0001.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            summary = DatasetAnalyzer(root).analyze()

            self.assertEqual(summary["train"]["images"], 1)
            self.assertEqual(summary["train"]["labels"], 1)
            self.assertEqual(summary["train"]["prefixes"]["nozzle_clean"], 1)

    def test_analyzer_renders_markdown(self):
        summary = {"train": {"images": 1, "labels": 1, "prefixes": {"nozzle_clean": 1}}}

        markdown = DatasetAnalyzer.render_markdown(summary)

        self.assertIn("数据集分析报告", markdown)
        self.assertIn("nozzle_clean", markdown)


if __name__ == "__main__":
    unittest.main()
