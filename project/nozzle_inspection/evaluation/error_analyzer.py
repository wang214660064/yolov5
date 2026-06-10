"""
错误分析器模块 - 预测结果统计分析

该模块用于分析模型预测结果，统计正确预测、漏检、误检和低置信度样本的数量。

核心功能：
- 统计正确预测的样本数
- 统计漏检的 NG 样本数（实际是 NG 但预测为 OK）
- 统计误检的 OK 样本数（实际是 OK 但预测为 NG）
- 统计低置信度样本数（置信度低于阈值）

使用示例：
    analyzer = ErrorAnalyzer(conf_threshold=0.7)
    
    predictions = [
        {"gt": "NG", "pred": "NG", "conf": 0.9},   # correct
        {"gt": "NG", "pred": "OK", "conf": 0.6},   # missed_ng + low_confidence
        {"gt": "OK", "pred": "NG", "conf": 0.8},   # false_alarm
        {"gt": "OK", "pred": "OK", "conf": 0.5},   # correct + low_confidence
    ]
    
    result = analyzer.summarize(predictions)
    # result: {"correct": 2, "missed_ng": 1, "false_alarm": 1, "low_confidence": 2}
"""

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ErrorAnalyzer:
    """
    错误分析器，用于统计预测结果的各类错误
    
    属性：
        conf_threshold: 置信度阈值，低于此值的预测视为低置信度
    
    方法：
        summarize(): 统计预测结果
    """

    conf_threshold: float = 0.7  # 置信度阈值

    def summarize(self, rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
        """
        统计预测结果中的各类情况
        
        参数：
            rows: 预测结果列表，每个元素是包含 "gt", "pred", "conf" 的字典
        
        返回值：
            dict[str, int]: 统计结果字典，包含以下字段：
                - correct: 正确预测的样本数
                - missed_ng: 漏检的 NG 样本数（实际 NG，预测非 NG）
                - false_alarm: 误检的 OK 样本数（实际 OK，预测 NG）
                - low_confidence: 低置信度样本数（置信度 < conf_threshold）
        
        统计逻辑：
            1. 遍历每个预测结果
            2. 判断置信度是否低于阈值（低置信度）
            3. 判断预测是否正确
            4. 如果不正确，判断是漏检还是误检
        
        示例：
            analyzer = ErrorAnalyzer(conf_threshold=0.7)
            predictions = [
                {"gt": "NG", "pred": "NG", "conf": 0.9},   # correct
                {"gt": "NG", "pred": "OK", "conf": 0.6},   # missed_ng + low_confidence
                {"gt": "OK", "pred": "NG", "conf": 0.8},   # false_alarm
                {"gt": "OK", "pred": "OK", "conf": 0.5},   # correct + low_confidence
            ]
            result = analyzer.summarize(predictions)
            # result: {"correct": 2, "missed_ng": 1, "false_alarm": 1, "low_confidence": 2}
        """
        # 初始化统计结果
        summary = {
            "correct": 0,           # 正确预测
            "missed_ng": 0,         # 漏检（NG 被预测为 OK）
            "false_alarm": 0,       # 误检（OK 被预测为 NG）
            "low_confidence": 0,    # 低置信度样本
        }
        
        # 遍历每个预测结果
        for row in rows:
            # 获取真实标签、预测标签和置信度
            gt = str(row.get("gt"))       # 真实标签（ground truth）
            pred = str(row.get("pred"))   # 预测标签（prediction）
            conf = float(row.get("conf", 0.0))  # 预测置信度
            
            # 判断是否为低置信度
            if conf < self.conf_threshold:
                summary["low_confidence"] += 1
            
            # 判断预测是否正确
            if gt == pred:
                summary["correct"] += 1
            elif gt == "NG" and pred != "NG":
                # 实际是 NG，预测为其他类别（漏检）
                summary["missed_ng"] += 1
            elif gt == "OK" and pred == "NG":
                # 实际是 OK，预测为 NG（误检）
                summary["false_alarm"] += 1
        
        return summary
