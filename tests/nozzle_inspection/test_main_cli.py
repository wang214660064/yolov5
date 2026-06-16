import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from project.nozzle_inspection.main import build_parser, main
from project.nozzle_inspection.training.experiment_runner import ExperimentResult


class MainCliTest(unittest.TestCase):
    def test_parser_contains_chinese_description(self):
        parser = build_parser()

        self.assertIn("3D打印机喷头检测", parser.description)

    def test_parser_accepts_dry_run_train(self):
        parser = build_parser()

        args = parser.parse_args(["train", "--dry-run", "--epochs", "3"])

        self.assertEqual(args.command, "train")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.epochs, 3)

    def test_parser_accepts_analyze_data(self):
        parser = build_parser()

        args = parser.parse_args(["analyze-data", "--dataset", "../dataset_2"])

        self.assertEqual(args.command, "analyze-data")
        self.assertEqual(args.dataset, "../dataset_2")

    def test_parser_rejects_removed_report_command(self):
        parser = build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["report"])

    def test_val_command_executes_yolov5_val_script(self):
        result = ExperimentResult(
            command=["python", "val.py"],
            returncode=0,
            stdout="Results saved to runs/val/exp",
            stderr="",
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.ExperimentRunner") as runner_cls:
            runner_cls.return_value.run.return_value = result

            returncode = main([
                "val",
                "--data",
                "project/nozzle_inspection/configs/dataset.yaml",
                "--weights",
                "runs/train/exp/weights/best.pt",
                "--conf",
                "0.7",
                "--task",
                "val",
                "--save-txt",
                "--save-conf",
            ])

        self.assertEqual(returncode, 0)
        runner_cls.return_value.run.assert_called_once()
        command = runner_cls.return_value.run.call_args.args[0]
        self.assertIn("val.py", command)
        self.assertIn("--conf-thres", command)
        self.assertIn("0.7", command)
        self.assertIn("--save-txt", command)
        self.assertIn("--save-conf", command)

    def test_parser_accepts_train_split_for_error_review(self):
        parser = build_parser()

        args = parser.parse_args([
            "val",
            "--weights",
            "runs/train/exp/weights/best.pt",
            "--task",
            "train",
            "--export-error-samples",
        ])

        self.assertEqual(args.task, "train")
        self.assertTrue(args.export_error_samples)

    def test_val_error_export_defaults_to_task_badcase_folder(self):
        result = ExperimentResult(
            command=["python", "val.py"],
            returncode=0,
            stdout="Results saved to runs/val/exp",
            stderr="",
        )

        with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
             patch("project.nozzle_inspection.main.ExperimentRunner") as runner_cls, \
             patch("project.nozzle_inspection.main.ErrorSampleExporter") as exporter_cls:
            runner_cls.return_value.run.return_value = result
            exporter_cls.return_value.export.return_value = {
                "missed_target": 0,
                "false_alarm": 0,
                "class_error": 0,
                "images_with_errors": 0,
            }

            returncode = main([
                "val",
                "--weights",
                "runs/train/exp/weights/best.pt",
                "--task",
                "train",
                "--save-txt",
                "--export-error-samples",
            ])

        self.assertEqual(returncode, 0)
        self.assertEqual(exporter_cls.call_args.kwargs["output_root"], Path("runs/train/BadCase"))

    def test_train_disables_augmentation_by_default(self):
        result = ExperimentResult(
            command=["python", "train.py"],
            returncode=0,
            stdout="train ok",
            stderr="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            hyp_path = Path(tmpdir) / "hyp.yaml"
            hyp_path.write_text(
                yaml.safe_dump(
                    {
                        "hsv_h": 0.015,
                        "hsv_s": 0.7,
                        "hsv_v": 0.4,
                        "degrees": 12,
                        "translate": 0.1,
                        "scale": 0.5,
                        "shear": 0.1,
                        "perspective": 0.0,
                        "flipud": 0.5,
                        "fliplr": 0.5,
                        "mosaic": 1.0,
                        "mixup": 0.3,
                        "copy_paste": 0.2,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with patch("project.nozzle_inspection.main.detect_device", return_value="cpu"), \
                 patch("project.nozzle_inspection.main.ExperimentRunner") as runner_cls:
                runner_cls.return_value.run.return_value = result

                returncode = main([
                    "train",
                    "--data",
                    "project/nozzle_inspection/configs/dataset.yaml",
                    "--hyp",
                    str(hyp_path),
                    "--epochs",
                    "1",
                    "--workers",
                    "0",
                ])

        self.assertEqual(returncode, 0)
        command = runner_cls.return_value.run.call_args.args[0]
        effective_hyp = Path(command[command.index("--hyp") + 1])
        self.assertNotEqual(effective_hyp, hyp_path)

        hyp = yaml.safe_load(effective_hyp.read_text(encoding="utf-8"))
        for key in (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "mosaic",
            "mixup",
            "copy_paste",
        ):
            self.assertEqual(hyp[key], 0.0)


if __name__ == "__main__":
    unittest.main()
