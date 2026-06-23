import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from project.nozzle_inspection.run_config import RunConfig
from project.nozzle_inspection.run_project import _build_training_hyp
from utils.metrics import bbox_iou
from utils.loss import ComputeLoss, FocalLoss


class LossSwitchesTest(unittest.TestCase):
    def test_siou_is_finite_and_supports_backward(self):
        prediction = torch.tensor([[0.5, 0.5, 0.4, 0.3]], requires_grad=True)
        target = torch.tensor([[0.6, 0.55, 0.35, 0.25]])

        siou = bbox_iou(prediction, target, SIoU=True).mean()
        (1.0 - siou).backward()

        self.assertTrue(torch.isfinite(siou))
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_config_builds_siou_and_focal_hyperparameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hyp_path = Path(tmpdir) / "hyp.yaml"
            hyp_path.write_text(yaml.safe_dump({"fl_gamma": 0.0}), encoding="utf-8")
            config = RunConfig(
                hyp_yaml=hyp_path,
                iou_type="siou",
                use_focal_loss=True,
                focal_gamma=2.0,
                focal_alpha=0.3,
            )

            hyp = _build_training_hyp(config)

        self.assertEqual(hyp["iou_type"], "siou")
        self.assertEqual(hyp["fl_gamma"], 2.0)
        self.assertEqual(hyp["fl_alpha"], 0.3)

    def test_compute_loss_uses_configured_siou_and_focal_alpha(self):
        detect = type(
            "Detect",
            (),
            {
                "nl": 3,
                "stride": torch.tensor([8, 16, 32]),
                "na": 3,
                "nc": 2,
                "anchors": [torch.ones(3, 2) for _ in range(3)],
            },
        )()
        model = nn.Module()
        model.register_parameter("dummy", nn.Parameter(torch.zeros(1)))
        model.model = [detect]
        model.hyp = {
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "label_smoothing": 0.0,
            "fl_gamma": 2.0,
            "fl_alpha": 0.3,
            "iou_type": "siou",
        }

        loss = ComputeLoss(model)

        self.assertEqual(loss.iou_type, "siou")
        self.assertIsInstance(loss.BCEcls, FocalLoss)
        self.assertEqual(loss.BCEcls.alpha, 0.3)


if __name__ == "__main__":
    unittest.main()
