from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ExperimentLogger:
    """将实验事件写入 Markdown，便于报告复盘。"""

    log_path: Path

    def log_event(self, title: str, fields: Mapping[str, object] | None = None) -> Path:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fields = fields or {}
        lines = [
            f"## {title}",
            "",
            f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        ]
        for key, value in fields.items():
            lines.append(f"- {key}：{value}")
        lines.append("")
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")
        return self.log_path
