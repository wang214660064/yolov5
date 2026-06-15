"""
工厂模块包 - 所有工厂类的统一导出

该模块将所有工厂类统一导出，方便其他模块导入使用。

工厂类说明：
    - AugmentationFactory: 数据增强组件工厂
    - DataFactory: 数据处理组件工厂（标签转换、去重）
    - EvaluatorFactory: YOLOv5 验证命令工厂
    - ModelFactory: YOLOv5 配置生成工厂
    - TrainerFactory: YOLOv5 训练命令工厂

使用示例：
    from nozzle_inspection.factories import (
        ModelFactory,
        TrainerFactory,
        EvaluatorFactory
    )
    
    model_factory = ModelFactory()
    trainer_factory = TrainerFactory(repo_root=Path.cwd())
    evaluator_factory = EvaluatorFactory()
"""

from .augmentation_factory import AugmentationFactory
from .data_factory import DataFactory
from .evaluator_factory import EvaluatorFactory
from .model_factory import ModelFactory
from .trainer_factory import TrainerFactory

# 导出所有工厂类
__all__ = [
    "AugmentationFactory",
    "DataFactory",
    "EvaluatorFactory",
    "ModelFactory",
    "TrainerFactory",
]
