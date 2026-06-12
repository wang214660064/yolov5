"""
运行配置模块 - 脚本式配置管理。

用户只需要修改本文件中的 CONFIG/RunConfig 字段，再运行固定入口
`conda run -n yolov5 python run_nozzle_project.py` 即可。
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """
    项目运行配置。

    action 可选：
    - prepare_data: 数据准备
    - analyze_data: 数据分析
    - write_config: 生成 dataset.yaml
    - train: 训练模型
    - val: 训练后评估模型，可通过 eval_task 选择 val 或 test
    - report: 生成报告
    """

    # 数据地址。当前推荐 EXP002 baseline 使用 out_cross_hash_clean。
    data_path_full = r"../out_cross_hash_clean"
    data_path_small = r"project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_downsampled"
    data_path = data_path_full

    # 动作选择。
    action: str = "val"

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
    workers: int = 2
    enable_augmentation: bool = False

    # 训练后评估配置。
    # eval_task="val" 表示评估验证集；eval_task="test" 表示评估测试集。
    weights: Path = Path("runs/train/nozzle_ng_ok4/weights/best.pt")
    conf: float = 0.25 # 分别跑 conf=0.25 / 0.5 / 0.7验证集的AP
    eval_task: str = "val"

    # 报告输出配置。
    report_output: Path = Path(data_path + "/reports/nozzle_report.md")
    pptx_output: Path = Path(data_path + "/reports/nozzle_report.pptx")


CONFIG = RunConfig()
