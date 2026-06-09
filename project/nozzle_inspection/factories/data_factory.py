from dataclasses import dataclass

from project.nozzle_inspection.data.deduplicate import Deduplicator
from project.nozzle_inspection.data.label_converter import LabelConverter


@dataclass
class DataFactory:
    """创建数据处理组件。"""

    def create_label_converter(self) -> LabelConverter:
        return LabelConverter()

    def create_deduplicator(self) -> Deduplicator:
        return Deduplicator()
