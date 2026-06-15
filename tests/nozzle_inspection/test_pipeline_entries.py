from pathlib import Path
import unittest

from project.nozzle_inspection.factories.data_factory import DataFactory
from project.nozzle_inspection.factories.augmentation_factory import AugmentationFactory
from project.nozzle_inspection.factories.evaluator_factory import EvaluatorFactory
from project.nozzle_inspection.training.experiment_runner import ExperimentRunner


class PipelineEntriesTest(unittest.TestCase):
    def test_factories_create_core_components(self):
        self.assertIsNotNone(DataFactory().create_label_converter())
        self.assertIsNotNone(DataFactory().create_deduplicator())
        self.assertTrue(callable(AugmentationFactory().create_box_scaler()))

    def test_evaluator_command_contains_threshold(self):
        command = EvaluatorFactory().build_val_command(
            data_yaml=Path("data/nozzle_ng_ok.yaml"),
            weights=Path("runs/train/nozzle/weights/best.pt"),
            conf=0.7,
        )

        self.assertIn("val.py", command)
        self.assertIn("--conf-thres", command)
        self.assertIn("0.7", command)

    def test_experiment_runner_dry_run_returns_command(self):
        runner = ExperimentRunner()
        command = ["python", "train.py", "--epochs", "1"]

        result = runner.run(command, dry_run=True)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.command, command)
        self.assertIn("dry-run", result.stdout)


if __name__ == "__main__":
    unittest.main()
