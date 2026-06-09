from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """脚本式运行配置；日常使用时优先修改这里的默认值。"""

    # 可选：prepare_data、train、val、report、write_config、analyze_data
    action: str = "prepare_data"

    # 训练配置
    dry_run: bool = False
    epochs: int = 3
    data_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")
    hyp_yaml: Path = Path("project/nozzle_inspection/configs/train_ng_ok.yaml")

    # 验证配置
    weights: Path = Path("runs/train/nozzle_ng_ok/weights/best.pt")
    conf: float = 0.7

    # 数据分析与配置生成
    dataset_root: Path = Path("../dataset_2")
    dataset_report: Path = Path("project/nozzle_inspection/data/dataset_report.md")
    generated_dataset_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")
    generated_dataset_root: Path = Path("../outputs/datasets/nozzle_ng_ok_v1")
    val_ratio: float = 0.2
    split_seed: int = 42

    # 报告输出
    report_output: Path = Path("../outputs/reports/nozzle_report.md")
    pptx_output: Path = Path("../outputs/reports/nozzle_report.pptx")


# 运行脚本时默认读取这个配置。你只需要改这里，不需要拼命令行参数。
CONFIG = RunConfig()
