"""
训练器工厂模块 - YOLOv5 训练命令构建

该模块负责构建 YOLOv5 训练命令，封装了训练所需的所有参数。

核心功能：
- 构建完整的训练命令列表
- 支持自定义权重、配置、批次大小等参数
- 默认使用 AdamW 优化器

使用示例：
    factory = TrainerFactory(repo_root=Path.cwd())
    command = factory.build_train_command(
        data_yaml=Path("configs/dataset.yaml"),
        hyp_yaml=Path("configs/hyp.yaml"),
        epochs=50
    )
    # 生成的命令类似：
    # python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml ...
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainerFactory:
    """
    YOLOv5 训练命令工厂
    
    属性：
        repo_root: 代码仓库根目录
        weights: 预训练权重文件，默认 yolov5s.pt
        cfg: 模型配置文件路径，默认 models/yolov5s.yaml
        imgsz: 输入图像尺寸，默认 640
        batch_size: 批次大小，默认 8
        optimizer: 优化器类型，默认 AdamW
    
    方法：
        build_train_command(): 构建训练命令
    """

    repo_root: Path       # 代码仓库根目录
    weights: str = "yolov5s.pt"      # 预训练权重
    cfg: str = "models/yolov5s.yaml"  # 模型配置文件
    imgsz: int = 640                  # 输入图像尺寸
    batch_size: int = 8               # 批次大小
    optimizer: str = "AdamW"          # 优化器

    def build_train_command(
        self,
        data_yaml: Path,
        hyp_yaml: Path,
        epochs: int,
        project: str = "runs/train",
        name: str = "nozzle_ng_ok",
    ) -> list[str]:
        """
        构建 YOLOv5 训练命令
        
        参数：
            data_yaml: 数据集配置文件路径
            hyp_yaml: 超参数配置文件路径
            epochs: 训练轮数
            project: 训练结果输出目录，默认 runs/train
            name: 实验名称，默认 nozzle_ng_ok
        
        返回值：
            list[str]: 训练命令列表，每个元素是命令的一部分
        
        生成的命令格式：
            python train.py --weights yolov5s.pt --cfg models/yolov5s.yaml 
            --data configs/dataset.yaml --hyp configs/hyp.yaml 
            --epochs 50 --batch-size 8 --imgsz 640 --optimizer AdamW 
            --project runs/train --name nozzle_ng_ok
        
        示例：
            factory = TrainerFactory(repo_root=Path("/workspace/yolov5"))
            cmd = factory.build_train_command(
                data_yaml=Path("configs/dataset.yaml"),
                hyp_yaml=Path("configs/hyp.yaml"),
                epochs=30
            )
        """
        return [
            "python",                      # Python 解释器
            "train.py",                    # YOLOv5 训练脚本
            "--weights",                   # 指定预训练权重
            self.weights,
            "--cfg",                       # 指定模型配置
            self.cfg,
            "--data",                      # 指定数据集配置
            str(data_yaml).replace("\\", "/"),  # 统一路径分隔符
            "--hyp",                       # 指定超参数配置
            str(hyp_yaml).replace("\\", "/"),
            "--epochs",                    # 指定训练轮数
            str(epochs),
            "--batch-size",                # 指定批次大小
            str(self.batch_size),
            "--imgsz",                     # 指定输入图像尺寸
            str(self.imgsz),
            "--optimizer",                 # 指定优化器
            self.optimizer,
            "--project",                   # 指定输出目录
            project,
            "--name",                      # 指定实验名称
            name,
        ]
