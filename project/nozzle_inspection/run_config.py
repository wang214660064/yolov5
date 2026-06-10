"""
运行配置模块 - 脚本式配置管理

该模块定义了项目运行所需的所有配置参数，使用 dataclass 装饰器使其更加简洁。
用户只需修改 CONFIG 实例的属性值，即可配置整个项目的运行行为，无需手动拼接命令行参数。

配置项按实验流程分为以下几类：
1. 动作选择：指定要执行的操作
2. 数据准备配置：数据集去重、转换标签、划分相关参数（实验流程第1步）
3. 数据分析配置：数据集分析和报告生成相关参数（实验流程第2步）
4. 生成数据配置：生成 YOLO 数据配置文件相关参数（实验流程第3步）
5. 训练配置：模型训练相关参数（实验流程第4步）
6. 验证配置：模型验证相关参数（实验流程第5步）
7. 报告配置：项目报告输出相关参数（实验流程第6步）
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """
    运行配置类，用于存储项目运行的所有参数
    
    使用 @dataclass(frozen=True) 装饰器使其成为不可变对象，保证配置的稳定性。
    字段按实验流程顺序排列。
    
    主要字段说明：
        action: 要执行的操作，按实验流程可选值包括：
                - prepare_data: 准备数据集（实验流程第1步）
                - analyze_data: 分析数据集（实验流程第2步）
                - write_config: 生成数据配置文件（实验流程第3步）
                - train: 训练模型（实验流程第4步）
                - val: 验证模型（实验流程第5步）
                - report: 生成报告（实验流程第6步）
        
        # 实验流程第1步：数据准备配置
        dataset_root: 原始数据集根目录路径
        generated_dataset_root: 处理后的数据集输出目录
        val_ratio: 验证集占比
        split_seed: 数据集划分的随机种子
        
        # 实验流程第2步：数据分析配置
        dataset_report: 数据集分析报告输出路径
        
        # 实验流程第3步：生成数据配置
        generated_dataset_yaml: 生成的数据配置文件输出路径
        
        # 实验流程第4步：训练配置
        dry_run: 是否为试运行模式（只打印命令，不实际执行）
        epochs: 训练轮数
        data_yaml: YOLO 数据配置文件路径
        hyp_yaml: 训练超参数配置文件路径
        
        # 实验流程第5步：验证配置
        weights: 待验证的模型权重文件路径
        conf: 置信度阈值
        
        # 实验流程第6步：报告配置
        report_output: Markdown 报告输出路径
        pptx_output: PPTX 汇报文件输出路径
    """

    # 动作选择：指定要执行的操作
    action: str = "val"

    # 实验流程第1步：数据准备配置
    dataset_root: Path = Path("../dataset_2")  # 原始数据集目录
    generated_dataset_root: Path = Path("project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1")  # 处理后数据集输出目录
    val_ratio: float = 0.2          # 验证集占比（0.2 表示 20%）
    split_seed: int = 42            # 随机种子，确保划分结果可重复

    # 实验流程第2步：数据分析配置
    dataset_report: Path = Path("project/nozzle_inspection/outputs/dataset_report.md")  # 分析报告路径

    # 实验流程第3步：生成数据配置
    generated_dataset_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")  # 生成的配置文件

    # 实验流程第4步：训练配置
    dry_run: bool = False           # 是否只打印命令不执行
    epochs: int = 2                 # 训练轮数，默认2轮（测试用），实际训练建议50轮
    data_yaml: Path = Path("project/nozzle_inspection/configs/dataset.yaml")  # 数据配置文件
    hyp_yaml: Path = Path("project/nozzle_inspection/configs/train_ng_ok.yaml")  # 超参数配置文件

    # 实验流程第5步：验证配置
    weights: Path = Path("runs/train/nozzle_ng_ok7/weights/best.pt")  # 模型权重文件
    conf: float = 0.7               # 置信度阈值，过滤低置信度检测结果

    # 实验流程第6步：报告输出配置
    report_output: Path = Path("project/nozzle_inspection/outputs/reports/nozzle_report.md")  # Markdown 报告路径
    pptx_output: Path = Path("project/nozzle_inspection/outputs/reports/nozzle_report.pptx")  # PPTX 报告路径


# 默认配置实例，运行脚本时会读取此配置
# 用户只需修改这个配置实例的属性值即可，无需手动拼接命令行参数
CONFIG = RunConfig()
