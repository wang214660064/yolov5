"""
评估器工厂模块 - YOLOv5 验证命令构建

该模块负责构建 YOLOv5 验证命令，用于评估训练好的模型性能。

核心功能：
- 构建完整的验证命令列表
- 支持自定义置信度阈值和 IoU 阈值
- 默认输出详细信息
- 支持指定验证设备

使用示例：
    factory = EvaluatorFactory(device="0")
    command = factory.build_val_command(
        data_yaml=Path("configs/dataset.yaml"),
        weights=Path("runs/train/exp/weights/best.pt"),
        conf=0.7
    )
    # 生成的命令类似：
    # python val.py --data configs/dataset.yaml --weights runs/train/exp/weights/best.pt --device 0 ...
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluatorFactory:
    """
    YOLOv5 验证命令工厂
    
    属性：
        imgsz: 输入图像尺寸，默认 640
        iou: IoU（交并比）阈值，默认 0.45
        device: 验证设备（"0" 表示GPU，"cpu" 表示CPU）
    
    方法：
        build_val_command(): 构建验证命令
    """

    imgsz: int = 640      # 输入图像尺寸
    iou: float = 0.45     # IoU 阈值，用于 NMS（非极大值抑制）
    device: str = "cpu"   # 验证设备

    def build_val_command(
        self,
        data_yaml: Path,
        weights: Path,
        conf: float = 0.25,
        task: str = "val",
        project: str = "runs/val",
        name: str = "exp",
        save_txt: bool = False,
        save_conf: bool = False,
        save_json: bool = False,
    ) -> list[str]:
        """
        构建 YOLOv5 验证命令
        
        参数：
            data_yaml: 数据集配置文件路径
            weights: 模型权重文件路径
            conf: 置信度阈值，用于过滤检测结果，默认 0.25
        
        返回值：
            list[str]: 验证命令列表，每个元素是命令的一部分
        
        生成的命令格式：
            python val.py --data configs/dataset.yaml --weights runs/train/exp/weights/best.pt 
            --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --device 0 --verbose
        
        关键参数说明：
            --conf-thres: 置信度阈值，低于此值的检测结果会被过滤
            --iou-thres: IoU 阈值，用于非极大值抑制
            --device: 训练设备（GPU/CPU）
            --verbose: 输出详细信息
        
        示例：
            factory = EvaluatorFactory()
            cmd = factory.build_val_command(
                data_yaml=Path("configs/dataset.yaml"),
                weights=Path("runs/train/exp/weights/best.pt"),
                conf=0.5
            )
        """
        command = [
            "python",                      # Python 解释器
            "val.py",                      # YOLOv5 验证脚本
            "--data",                      # 指定数据集配置
            str(data_yaml).replace("\\", "/"),  # 统一路径分隔符
            "--weights",                   # 指定模型权重
            str(weights).replace("\\", "/"),
            "--imgsz",                     # 指定输入图像尺寸
            str(self.imgsz),
            "--conf-thres",                # 指定置信度阈值
            str(conf),
            "--task",                      # 指定评估 val 集或 test 集
            task,
            "--iou-thres",                 # 指定 IoU 阈值
            str(self.iou),
            "--device",                    # 指定验证设备
            self.device,
            "--project",                   # 指定验证输出根目录
            str(project).replace("\\", "/"),
            "--name",                      # 指定验证输出文件夹名称
            name,
            "--verbose",                   # 输出详细信息
        ]
        if save_txt:
            command.append("--save-txt")
        if save_conf:
            command.append("--save-conf")
        if save_json:
            command.append("--save-json")
        return command
