"""
增强工厂模块 - 数据增强组件创建

该模块负责创建数据增强相关的组件，遵循工厂模式设计。

当前功能：
- 创建边界框缩放器（用于数据增强时同步调整边界框）

使用示例：
    factory = AugmentationFactory()
    box_scaler = factory.create_box_scaler()
    # 使用缩放器
    scaled_box = box_scaler((0, 0.5, 0.5, 0.2, 0.2), 0.8, 0.1, 0)
"""

from dataclasses import dataclass

from ..data.scale_augment import scale_yolo_box_on_canvas


@dataclass
class AugmentationFactory:
    """
    数据增强组件工厂
    
    负责创建数据增强相关的组件，遵循工厂模式设计。
    
    方法：
        create_box_scaler(): 创建边界框缩放器
    """

    def create_box_scaler(self):
        """
        创建边界框缩放器
        
        返回值：
            function: scale_yolo_box_on_canvas 函数，用于在归一化画布上缩放和偏移 YOLO 边界框
        
        函数签名：
            scale_yolo_box_on_canvas(box, scale, offset_x, offset_y, min_size=1e-4) -> YoloBox | None
        
        使用场景：
            当对图片进行裁剪、缩放等数据增强操作时，需要同步调整边界框坐标
        
        示例：
            factory = AugmentationFactory()
            scaler = factory.create_box_scaler()
            original_box = (0, 0.5, 0.5, 0.2, 0.2)  # (类别, x, y, w, h)
            scaled_box = scaler(original_box, scale=0.8, offset_x=0.1, offset_y=0)
        """
        return scale_yolo_box_on_canvas
