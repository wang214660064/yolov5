from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluatorFactory:
    """构建 YOLOv5 验证命令。"""

    imgsz: int = 640
    iou: float = 0.45

    def build_val_command(self, data_yaml: Path, weights: Path, conf: float = 0.25) -> list[str]:
        return [
            "python",
            "val.py",
            "--data",
            str(data_yaml).replace("\\", "/"),
            "--weights",
            str(weights).replace("\\", "/"),
            "--imgsz",
            str(self.imgsz),
            "--conf-thres",
            str(conf),
            "--iou-thres",
            str(self.iou),
            "--verbose",
        ]
