from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainerFactory:
    """构建 YOLOv5 训练命令，默认使用 AdamW。"""

    repo_root: Path
    weights: str = "yolov5s.pt"
    cfg: str = "models/yolov5s.yaml"
    imgsz: int = 640
    batch_size: int = 8
    optimizer: str = "AdamW"

    def build_train_command(
        self,
        data_yaml: Path,
        hyp_yaml: Path,
        epochs: int,
        project: str = "runs/train",
        name: str = "nozzle_ng_ok",
    ) -> list[str]:
        return [
            "python",
            "train.py",
            "--weights",
            self.weights,
            "--cfg",
            self.cfg,
            "--data",
            str(data_yaml).replace("\\", "/"),
            "--hyp",
            str(hyp_yaml).replace("\\", "/"),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(self.batch_size),
            "--imgsz",
            str(self.imgsz),
            "--optimizer",
            self.optimizer,
            "--project",
            project,
            "--name",
            name,
        ]
