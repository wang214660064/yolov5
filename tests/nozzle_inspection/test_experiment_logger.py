from pathlib import Path
import tempfile
import unittest

from project.nozzle_inspection.utils.experiment_logger import ExperimentLogger
from project.nozzle_inspection.evaluation.error_analyzer import ErrorAnalyzer


class ExperimentLoggerTest(unittest.TestCase):
    def test_experiment_logger_writes_markdown_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "experiment.md"

            ExperimentLogger(log_path).log_event("训练开始", {"epochs": 3, "optimizer": "AdamW"})

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("训练开始", text)
            self.assertIn("AdamW", text)

    def test_error_analyzer_splits_prediction_status(self):
        analyzer = ErrorAnalyzer(conf_threshold=0.7)
        rows = analyzer.summarize([
            {"image": "a.jpg", "gt": "NG", "pred": "NG", "conf": 0.8},
            {"image": "b.jpg", "gt": "NG", "pred": "OK", "conf": 0.9},
            {"image": "c.jpg", "gt": "OK", "pred": "NG", "conf": 0.6},
        ])

        self.assertEqual(rows["correct"], 1)
        self.assertEqual(rows["missed_ng"], 1)
        self.assertEqual(rows["low_confidence"], 1)


if __name__ == "__main__":
    unittest.main()
