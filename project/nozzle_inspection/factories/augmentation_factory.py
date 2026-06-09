from dataclasses import dataclass

from project.nozzle_inspection.data.scale_augment import scale_yolo_box_on_canvas


@dataclass
class AugmentationFactory:
    """创建数据增强组件。"""

    def create_box_scaler(self):
        return scale_yolo_box_on_canvas
