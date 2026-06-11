"""
运行项目模块 - 脚本式运行入口

该模块提供了一个简化的项目运行方式，通过配置类 RunConfig 来驱动项目执行，
无需手动拼接复杂的命令行参数。

核心功能：
1. 将 RunConfig 配置转换为命令行参数列表
2. 调用 main 函数执行相应操作
3. 提供统一的运行入口

按实验流程支持的操作：
1. prepare_data: 准备数据集
2. analyze_data: 分析数据集
3. write_config: 生成数据配置
4. train: 训练模型
5. val: 验证模型
6. report: 生成报告

支持两种运行方式：
1. 作为模块运行：python -m project.nozzle_inspection.run_project
2. 直接运行脚本：python project/nozzle_inspection/run_project.py
"""

import os
import sys
from pathlib import Path

# 设置 UTF-8 编码环境变量，解决 Windows 上 subprocess 的编码问题
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 修正 OMP_NUM_THREADS，避免 libgomp 报错（必须为正整数）
omp_threads = os.environ.get("OMP_NUM_THREADS", "")
if omp_threads == "0" or not omp_threads.isdigit():
    os.environ.pop("OMP_NUM_THREADS", None)

# 支持相对导入和直接运行两种方式
try:
    from .main import main
    from .run_config import CONFIG, RunConfig
except ImportError:
    # 如果直接运行脚本，添加上级目录到路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from project.nozzle_inspection.main import main
    from project.nozzle_inspection.run_config import CONFIG, RunConfig


def build_argv(config: RunConfig) -> list[str]:
    """
    将 RunConfig 配置对象转换为命令行参数列表
    
    根据配置中的 action 字段，按实验流程构建对应的命令行参数列表，
    然后传递给 main 函数执行。
    
    参数：
        config: RunConfig 配置对象，包含所有运行参数
    
    返回值：
        list[str]: 命令行参数列表，格式为 ["command", "--arg1", "value1", ...]
    
    按实验流程支持的 action 值：
        prepare_data: 准备数据集（实验流程第1步）
        analyze_data: 分析数据集（实验流程第2步）
        write_config: 生成数据配置（实验流程第3步）
        train: 训练模型（实验流程第4步）
        val: 验证模型（实验流程第5步）
        report: 生成报告（实验流程第6步）
    """

    # 实验流程第1步：构建数据准备命令参数
    if config.action == "prepare_data":
        return [
            "prepare-data",
            "--dataset",
            _path(config.dataset_root),
            "--output",
            _path(config.generated_dataset_root),
            "--val-ratio",
            str(config.val_ratio),
            "--seed",
            str(config.split_seed),
        ]

    # 实验流程第2步：构建数据分析命令参数
    if config.action == "analyze_data":
        return [
            "analyze-data",
            "--dataset",
            _path(config.dataset_root),
            "--output",
            _path(config.dataset_report),
        ]

    # 实验流程第3步：构建生成配置文件命令参数
    if config.action == "write_config":
        return [
            "write-config",
            "--output",
            _path(config.generated_dataset_yaml),
            "--path",
            _path(config.generated_dataset_root),
        ]

    # 实验流程第4步：构建训练命令参数
    if config.action == "train":
        argv = [
            "train",
            "--data",
            _path(config.data_yaml),
            "--hyp",
            _path(config.hyp_yaml),
            "--epochs",
            str(config.epochs),
        ]
        # 如果是试运行模式，添加 --dry-run 参数
        if config.dry_run:
            argv.append("--dry-run")
        return argv

    # 实验流程第5步：构建验证命令参数
    if config.action == "val":
        return [
            "val",
            "--data",
            _path(config.data_yaml),
            "--weights",
            _path(config.weights),
            "--conf",
            str(config.conf),
        ]

    # 实验流程第6步：构建生成报告命令参数
    if config.action == "report":
        return [
            "report",
            "--output",
            _path(config.report_output),
            "--pptx",
            _path(config.pptx_output),
        ]

    # 如果 action 值不在支持列表中，抛出异常
    raise ValueError(f"未知运行动作：{config.action}")


def run(config: RunConfig = CONFIG) -> int:
    """
    项目运行入口函数
    
    参数：
        config: RunConfig 配置对象，默认为 CONFIG
    
    返回值：
        int: 程序退出码
    """
    # 将配置转换为命令行参数，然后调用 main 函数
    return main(build_argv(config))


def _path(path: Path) -> str:
    """
    将 Path 对象转换为字符串路径，并统一路径分隔符
    
    参数：
        path: Path 对象
    
    返回值：
        str: 标准化的路径字符串（使用 '/' 作为分隔符）
    """
    return str(path).replace("\\", "/")


if __name__ == "__main__":
    # 脚本直接运行时，调用 run 函数并将返回值作为系统退出码
    raise SystemExit(run())
