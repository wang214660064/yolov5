import unittest
from pathlib import Path

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig


class ScriptRunnerTest(unittest.TestCase):
    def test_default_config_keeps_current_script_defaults(self):
        config = RunConfig()

        self.assertEqual(config.val_ratio, 0.2)
        self.assertEqual(config.split_seed, 42)
        self.assertEqual(config.ssim_threshold, 0.85)
        self.assertEqual(config.phash_threshold, 5)
        self.assertEqual(config.deduplicate_workers, 0)
        self.assertFalse(config.enable_augmentation)
        self.assertEqual(config.eval_task, "val")

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
                "--workers",
                "2",
                "--dry-run",
            ],
        )

    def test_build_train_argv_can_enable_augmentation(self):
        config = RunConfig(action="train", dry_run=True, epochs=5, enable_augmentation=True)

        argv = run_project.build_argv(config)

        self.assertIn("--enable-augmentation", argv)

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

    def test_build_val_argv_can_target_test_split(self):
        config = RunConfig(
            action="val",
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            conf=0.7,
            eval_task="test",
        )

        argv = run_project.build_argv(config)

        self.assertEqual(
            argv,
            [
                "val",
                "--data",
                "project/nozzle_inspection/configs/dataset.yaml",
                "--weights",
                "runs/train/exp/weights/best.pt",
                "--conf",
                "0.7",
                "--task",
                "test",
            ],
        )

    def test_build_prepare_data_argv_from_config(self):
        config = RunConfig(
            action="prepare_data",
            dataset_root=Path("../dataset_2"),
            generated_dataset_root=Path("../outputs/datasets/nozzle_ng_ok_v1"),
            val_ratio=0.25,
            split_seed=7,
            ssim_threshold=0.85,
            phash_threshold=4,
            deduplicate_workers=4,
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
                "--ssim-threshold",
                "0.85",
                "--phash-threshold",
                "4",
                "--deduplicate-workers",
                "4",
            ],
        )


if __name__ == "__main__":
    unittest.main()
