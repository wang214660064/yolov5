import unittest
from pathlib import Path

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig


class ScriptRunnerTest(unittest.TestCase):
    def test_default_config_uses_prepare_data_action(self):
        config = RunConfig()

        self.assertEqual(config.action, "prepare_data")
        self.assertEqual(config.val_ratio, 0.2)
        self.assertEqual(config.split_seed, 42)

    def test_build_train_argv_from_config(self):
        config = RunConfig(action="train", dry_run=True, epochs=5)

        argv = run_project.build_argv(config)

        self.assertEqual(
            argv,
            [
                "train",
                "--data",
                "project/nozzle_inspection/configs/dataset.yaml",
                "--hyp",
                "project/nozzle_inspection/configs/train_ng_ok.yaml",
                "--epochs",
                "5",
                "--dry-run",
            ],
        )

    def test_build_report_argv_from_config(self):
        config = RunConfig(
            action="report",
            report_output=Path("../outputs/reports/demo.md"),
            pptx_output=Path("../outputs/reports/demo.pptx"),
        )

        argv = run_project.build_argv(config)

        self.assertEqual(
            argv,
            [
                "report",
                "--output",
                "../outputs/reports/demo.md",
                "--pptx",
                "../outputs/reports/demo.pptx",
            ],
        )

    def test_build_prepare_data_argv_from_config(self):
        config = RunConfig(
            action="prepare_data",
            dataset_root=Path("../dataset_2"),
            generated_dataset_root=Path("../outputs/datasets/nozzle_ng_ok_v1"),
            val_ratio=0.25,
            split_seed=7,
        )

        argv = run_project.build_argv(config)

        self.assertEqual(
            argv,
            [
                "prepare-data",
                "--dataset",
                "../dataset_2",
                "--output",
                "../outputs/datasets/nozzle_ng_ok_v1",
                "--val-ratio",
                "0.25",
                "--seed",
                "7",
            ],
        )


if __name__ == "__main__":
    unittest.main()
