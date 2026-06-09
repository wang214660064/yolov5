from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


NG_PREFIXES = (
    "nozzle_tip_wrapped",
    "nozzle_no_extrusion",
    "nozzle_heavy_contamination",
    "nozzle_slight_contamination",
    "camera_occlusion",
)

OK_PREFIXES = (
    "nozzle_clean",
    "nozzle_extrusion_normal",
)


@dataclass
class LabelConverter:
    """将原始多类别标签归并为 NG/OK 两类。"""

    ng_prefixes: tuple[str, ...] = field(default_factory=lambda: NG_PREFIXES)
    ok_prefixes: tuple[str, ...] = field(default_factory=lambda: OK_PREFIXES)

    def class_id_for_stem(self, stem: str) -> int:
        for prefix in self.ng_prefixes:
            if stem.startswith(prefix):
                return 0
        for prefix in self.ok_prefixes:
            if stem.startswith(prefix):
                return 1
        raise ValueError(f"未知样本前缀，无法映射 NG/OK：{stem}")

    def convert_lines(self, stem: str, lines: Iterable[str]) -> list[str]:
        class_id = self.class_id_for_stem(stem)
        converted: list[str] = []
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 5:
                raise ValueError(f"标签格式非法：{stem} 第 {line_number} 行")
            try:
                _, x, y, w, h = parts
                values = [float(x), float(y), float(w), float(h)]
            except ValueError as exc:
                raise ValueError(f"标签数值非法：{stem} 第 {line_number} 行") from exc
            if not self._is_valid_box(values):
                raise ValueError(f"标签坐标非法：{stem} 第 {line_number} 行")
            converted.append(
                f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}\n"
            )
        return converted

    def convert_file(self, label_path: Path, output_path: Path) -> int:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        converted = self.convert_lines(label_path.stem, lines)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("".join(converted), encoding="utf-8")
        return len(converted)

    @staticmethod
    def _is_valid_box(values: list[float]) -> bool:
        x, y, w, h = values
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
