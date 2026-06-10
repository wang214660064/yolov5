"""
数据处理模块包 - 所有数据处理组件的统一导出

该模块将所有数据处理组件统一导出，方便其他模块导入使用。

数据处理组件说明：
    - DatasetAnalyzer: 数据集分析器，分析数据集结构
    - DatasetPreparer: 数据集准备器，整合去重、标签转换、划分
    - Deduplicator: 去重器，基于 SHA256 精确去重
    - DuplicateReport: 去重结果报告
    - LabelConverter: 标签转换器，多类别转 NG/OK
    - scale_yolo_box_on_canvas: 边界框缩放函数
    - stratified_split: 分层划分函数

使用示例：
    from nozzle_inspection.data import (
        DatasetPreparer,
        DatasetAnalyzer,
        stratified_split
    )
    
    preparer = DatasetPreparer()
    analyzer = DatasetAnalyzer()
"""

from .dataset_analyzer import DatasetAnalyzer
from .dataset_preparer import DatasetPreparer
from .deduplicate import Deduplicator, DuplicateReport
from .label_converter import LabelConverter
from .scale_augment import scale_yolo_box_on_canvas
from .split_dataset import stratified_split

# 导出所有数据处理组件
__all__ = [
    "DatasetAnalyzer",
    "DatasetPreparer",
    "Deduplicator",
    "DuplicateReport",
    "LabelConverter",
    "scale_yolo_box_on_canvas",
    "stratified_split",
]
