from pathlib import Path
import unittest

from project.nozzle_inspection.factories.trainer_factory import TrainerFactory
from project.nozzle_inspection.factories.model_factory import ModelFactory
from project.nozzle_inspection.factories.evaluator_factory import EvaluatorFactory


class FactoriesTest(unittest.TestCase):
    def test_trainer_command_uses_adamw_and_two_class_data(self):
        command = TrainerFactory(repo_root=Path("E:/repo/yolov5")).build_train_command(
            data_yaml=Path("data/nozzle_ng_ok.yaml"),
            hyp_yaml=Path("data/hyps/nozzle_ng_ok.yaml"),
            epochs=3,
        )

        self.assertIn("train.py", command)
        self.assertIn("--optimizer", command)
        self.assertIn("AdamW", command)
        self.assertIn("data/nozzle_ng_ok.yaml", command)
        self.assertIn("--epochs", command)
        self.assertIn("3", command)

    def test_model_factory_renders_two_class_dataset_yaml(self):
        yaml_text = ModelFactory().build_dataset_yaml(
            path="../outputs/datasets/nozzle_ng_ok_v1",
            train="images/train",
            val="images/val",
            test="images/test",
        )

        self.assertIn("nc: 2", yaml_text)
        self.assertIn("0: NG", yaml_text)
        self.assertIn("1: OK", yaml_text)

    def test_evaluator_command_can_target_test_split(self):
        command = EvaluatorFactory(device="0").build_val_command(
            data_yaml=Path("project/nozzle_inspection/configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            conf=0.7,
            task="test",
        )

        self.assertIn("--task", command)
        task_index = command.index("--task")
        self.assertEqual(command[task_index + 1], "test")


if __name__ == "__main__":
    unittest.main()
