"""
数据工厂模块 - 数据处理组件创建

该模块负责创建数据处理相关的组件，遵循工厂模式设计。

当前功能：
- 创建标签转换器（用于将多类别标签转换为 NG/OK 两类）
- 创建去重器（用于数据集精确去重）

使用示例：
    factory = DataFactory()
    converter = factory.create_label_converter()
    deduplicator = factory.create_deduplicator()
    
    # 使用转换器
    converter.convert_file(Path("input.txt"), Path("output.txt"))
    
    # 使用去重器
    report = deduplicator.find_exact_duplicates([Path("image1.jpg"), Path("image2.jpg")])
"""

from dataclasses import dataclass

from ..data.deduplicate import Deduplicator
from ..data.label_converter import LabelConverter


@dataclass
class DataFactory:
    """
    数据处理组件工厂
    
    负责创建数据处理相关的组件，遵循工厂模式设计。
    
    方法：
        create_label_converter(): 创建标签转换器
        create_deduplicator(): 创建去重器
    """

    def create_label_converter(self) -> LabelConverter:
        """
        创建标签转换器
        
        返回值：
            LabelConverter: 标签转换器实例，用于将多类别标签归并为 NG/OK 两类
        
        LabelConverter 主要功能：
            - class_id_for_stem(): 根据文件名前缀获取类别ID
            - convert_file(): 转换标签文件
        
        使用场景：
            将原始多类别数据集转换为二分类（NG/OK）数据集
        
        示例：
            factory = DataFactory()
            converter = factory.create_label_converter()
            converter.convert_file(
                Path("labels/train/nozzle_clean_001.txt"),
                Path("output/labels/train/nozzle_clean_001.txt")
            )
        """
        return LabelConverter()

    def create_deduplicator(self) -> Deduplicator:
        """
        创建去重器
        
        返回值：
            Deduplicator: 去重器实例，用于数据集精确去重
        
        Deduplicator 主要功能：
            - find_exact_duplicates(): 查找精确重复的文件
        
        使用场景：
            检测数据集中完全相同的图片文件，避免训练过拟合
        
        示例：
            factory = DataFactory()
            deduplicator = factory.create_deduplicator()
            image_paths = [Path("images/img1.jpg"), Path("images/img2.jpg")]
            report = deduplicator.find_exact_duplicates(image_paths)
            print(f"重复文件数：{len(report.duplicate_files)}")
        """
        return Deduplicator()
