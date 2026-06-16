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
        self.assertIn(config.eval_task, {"train", "val", "test"})

    def test_build_train_argv_from_config(self):
        config = RunConfig(
            action="train",
            dry_run=True,
            epochs=5,
            batch_size=16,
            enable_augmentation=False,
            train_name=None,
        )

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
                "--batch-size",
                "16",
                "--dry-run",
            ],
        )

    def test_build_train_argv_can_enable_augmentation(self):
        config = RunConfig(action="train", dry_run=True, epochs=5, enable_augmentation=True)

        argv = run_project.build_argv(config)

        self.assertIn("--enable-augmentation", argv)

    def test_build_train_argv_can_use_custom_output_name(self):
        config = RunConfig(action="train", dry_run=True, epochs=5, train_name="EXP003_no_aug")

        argv = run_project.build_argv(config)

        self.assertIn("--name", argv)
        name_index = argv.index("--name")
        self.assertEqual(argv[name_index + 1], "EXP003_no_aug")

    def test_build_val_argv_can_target_test_split(self):
        config = RunConfig(
            action="val",
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            conf=0.7,
            eval_task="test",
            val_name=None,
            save_txt=False,
            save_conf=False,
            save_json=False,
            export_error_samples=False,
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

    def test_build_val_argv_can_use_custom_output_name(self):
        config = RunConfig(
            action="val",
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            conf=0.25,
            eval_task="val",
            val_name="EXP003_val_conf025",
        )

        argv = run_project.build_argv(config)

        self.assertIn("--name", argv)
        name_index = argv.index("--name")
        self.assertEqual(argv[name_index + 1], "EXP003_val_conf025")

    def test_build_val_argv_can_enable_prediction_and_error_exports(self):
        config = RunConfig(
            action="val",
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            conf=0.25,
            eval_task="train",
            val_name="EXP_train_review",
            save_txt=True,
            save_conf=True,
            save_json=True,
            export_error_samples=True,
            error_samples_dir=Path("runs/error_samples"),
            error_iou_threshold=0.5,
        )

        argv = run_project.build_argv(config)

        self.assertIn("--save-txt", argv)
        self.assertIn("--save-conf", argv)
        self.assertIn("--save-json", argv)
        self.assertIn("--export-error-samples", argv)
        self.assertEqual(argv[argv.index("--task") + 1], "train")
        self.assertEqual(argv[argv.index("--error-samples-dir") + 1], "runs/error_samples")
        self.assertEqual(argv[argv.index("--error-iou-thres") + 1], "0.5")

    def test_default_badcase_output_uses_task_specific_badcase_folder(self):
        config = RunConfig()

        self.assertEqual(config.error_samples_dir, Path("runs/train/BadCase"))

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
