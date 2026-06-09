from collections import Counter
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetAnalyzer:
    """统计 YOLO 数据集图片、标签和样本前缀分布。"""

    dataset_root: Path

    def analyze(self, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict]:
        root = Path(self.dataset_root)
        summary: dict[str, dict] = {}
        for split in splits:
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
            labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
            prefixes = Counter(_sample_prefix(path.stem) for path in images)
            summary[split] = {
                "images": len(images),
                "labels": len(labels),
                "prefixes": dict(sorted(prefixes.items())),
            }
        return summary

    @staticmethod
    def render_markdown(summary: dict[str, dict]) -> str:
        lines = ["# 数据集分析报告", ""]
        for split, values in summary.items():
            lines.extend([
                f"## {split}",
                "",
                f"- 图片数量：{values.get('images', 0)}",
                f"- 标签数量：{values.get('labels', 0)}",
                "",
                "| 前缀 | 数量 |",
                "| --- | ---: |",
            ])
            for prefix, count in values.get("prefixes", {}).items():
                lines.append(f"| {prefix} | {count} |")
            lines.append("")
        return "\n".join(lines)

    def write_markdown(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render_markdown(self.analyze()), encoding="utf-8")
        return output_path


def _sample_prefix(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem
