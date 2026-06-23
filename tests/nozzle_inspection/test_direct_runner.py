import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from project.nozzle_inspection import run_project
from project.nozzle_inspection.run_config import RunConfig


class DirectRunnerTest(unittest.TestCase):
    def test_train_action_calls_yolov5_train_directly(self):
        train_module = SimpleNamespace(run=Mock(return_value=object()))
        config = RunConfig(
            action="train",
            epochs=1,
            workers=0,
            enable_augmentation=True,
            train_name="direct-train",
        )

        with patch.dict(sys.modules, {"train": train_module}), \
             patch.object(run_project, "main", return_value=0, create=True), \
             patch.object(run_project, "_detect_device", return_value="cpu"):
            returncode = run_project.run(config)

        self.assertEqual(returncode, 0)
        train_module.run.assert_called_once()
        self.assertEqual(train_module.run.call_args.kwargs["epochs"], 1)

    def test_val_action_calls_yolov5_val_directly(self):
        val_module = SimpleNamespace(run=Mock(return_value=object()))
        config = RunConfig(
            action="val",
            eval_task="val",
            val_name="direct-val",
            save_txt=False,
            save_conf=False,
            save_json=False,
            export_error_samples=False,
        )

        with patch.dict(sys.modules, {"val": val_module}), \
             patch.object(run_project, "main", return_value=0, create=True), \
             patch.object(run_project, "_detect_device", return_value="cpu"):
            returncode = run_project.run(config)

        self.assertEqual(returncode, 0)
        val_module.run.assert_called_once()
        self.assertEqual(val_module.run.call_args.kwargs["task"], "val")


if __name__ == "__main__":
    unittest.main()
