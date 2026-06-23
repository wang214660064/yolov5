"""
运行配置模块 - 脚本式配置管理。

用户只需要修改本文件中的 CONFIG/RunConfig 字段，再运行固定入口
`python -u run_nozzle_project.py` 即可。

运行前请先在 PowerShell 中进入 yolov5 目录并激活环境：
`conda activate yolov5`
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunConfig:
    """
    项目运行配置。

    action 可选：
    - prepare_data: 数据准备
    - analyze_data: 数据分析
    - write_config: 生成 dataset.yaml
    - train: 训练模型
    - val: 训练后评估模型，可通过 eval_task 选择 train、val 或 test
    """

    # 数据地址。当前推荐 EXP002 baseline 使用 out_cross_hash_clean。
    data_path_full = r"../out_cross_hash_clean"
    data_path_small = r"project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_downsampled"
    data_path = data_path_full

    # 动作选择。
    action: str = "train"

    # 数据准备配置。
    dataset_root: Path = Path("../dataset_2")
    generated_dataset_root: Path = Path(data_path)
    val_ratio: float = 0.2
    split_seed: int = 42
    ssim_threshold: float = 0.85
    phash_threshold: int = 5
    deduplicate_workers: int = 0

    # 数据分析配置。
    dataset_report: Path = Path(data_path + "/dataset_report.md")

    # dataset.yaml 生成配置。
    generated_dataset_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")

    # 训练配置。
    dry_run: bool = False
    epochs: int = 20
    data_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")
    hyp_yaml: Path = Path("project/nozzle_inspection/configs/train_ng_ok.yaml")
    batch_size: int = 32  # 批次大小，根据显存调整（16/32/64）
    workers: int = 2
    enable_augmentation: bool = True
    # 边界框回归损失，可选 "ciou" 或 "siou"。
    iou_type: str = "siou"
    # Focal Loss 仅作用于分类和目标置信度 BCE；关闭时使用普通 BCE。
    use_focal_loss: bool = True
    focal_gamma: float = 1.5
    focal_alpha: float = 0.25
    # 模型选择。model_weights 为预训练权重，model_cfg 为模型结构配置。
    model_weights: str = "runs/train/nozzle_ng_ok-aug-focal/weights/best.pt"  # 可选 yolov5s.pt 或自定义路径
    model_cfg: str = "models/yolov5s.yaml"  # 对应 models/ 下的 yaml 配置文件
    # 训练输出文件夹名称。None 或 "" 表示使用 YOLOv5 默认名称 nozzle_ng_ok，并自动递增。
    train_name: Optional[str] = "nozzle_ng_ok-aug-focal-continue"

    # 训练后评估配置。
    # eval_task="train" 表示用验证逻辑检查训练集；"val" 表示验证集；"test" 表示测试集。
    weights: Path = Path("runs/train/"+train_name+"/weights/best.pt")
    conf: float = 0.7 # 分别跑 conf=0.25 / 0.5 / 0.7验证集的AP，官方默认 conf=0.001
    eval_task: str = "test"  # 评估任务，可选 train、val、test的数据集
    # 验证输出文件夹名称。None 或 "" 表示使用 YOLOv5 默认 exp，并自动递增。
    val_name: Optional[str]= f"{train_name}/exp-conf-{conf}" if train_name else None
    # 保存 YOLOv5 原生预测结果，错误样本复查需要 save_txt=True。
    save_txt: bool = True
    save_conf: bool = True
    save_json: bool = True
    # 导出错误样本到 runs/<eval_task>/BadCase，便于人工复查和后续报告整理。
    export_error_samples: bool = True
    error_iou_threshold: float = 0.5
    error_samples_dir: Path = Path("runs/"+eval_task+"/BadCase")

CONFIG = RunConfig()
