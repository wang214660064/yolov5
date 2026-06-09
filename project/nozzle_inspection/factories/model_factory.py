from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelFactory:
    """生成 YOLOv5 项目配置。"""

    names: tuple[str, str] = ("NG", "OK")

    def build_dataset_yaml(self, path: str, train: str, val: str, test: str | None = None) -> str:
        lines = [
            f"path: {path}",
            f"train: {train}",
            f"val: {val}",
        ]
        if test:
            lines.append(f"test: {test}")
        lines.extend([
            "nc: 2",
            "names:",
            f"  0: {self.names[0]}",
            f"  1: {self.names[1]}",
            "",
        ])
        return "\n".join(lines)

    def write_dataset_yaml(self, output_path: Path, path: str, train: str, val: str, test: str | None = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.build_dataset_yaml(path, train, val, test), encoding="utf-8")
        return output_path
