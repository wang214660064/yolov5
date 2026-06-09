import unittest
from pathlib import Path

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig


class ScriptRunnerTest(unittest.TestCase):
    def test_default_config_uses_train_action_and_dry_run(self):
        config = RunConfig()

        self.assertEqual(config.action, "train")
        self.assertTrue(config.dry_run)
        self.assertEqual(config.epochs, 3)

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


if __name__ == "__main__":
    unittest.main()
