from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ErrorAnalyzer:
    """统计 NG/OK 预测中的正确、漏检、误检和低置信度样本。"""

    conf_threshold: float = 0.7

    def summarize(self, rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
        summary = {
            "correct": 0,
            "missed_ng": 0,
            "false_alarm": 0,
            "low_confidence": 0,
        }
        for row in rows:
            gt = str(row.get("gt"))
            pred = str(row.get("pred"))
            conf = float(row.get("conf", 0.0))
            if conf < self.conf_threshold:
                summary["low_confidence"] += 1
            if gt == pred:
                summary["correct"] += 1
            elif gt == "NG" and pred != "NG":
                summary["missed_ng"] += 1
            elif gt == "OK" and pred == "NG":
                summary["false_alarm"] += 1
        return summary
