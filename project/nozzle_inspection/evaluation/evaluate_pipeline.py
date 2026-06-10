"""
评估管道模块 - 多阈值验证命令构建

该模块提供多阈值验证命令的构建功能，用于评估不同置信度阈值下的模型性能。

核心功能：
- 为多个置信度阈值构建验证命令
- 便于对比不同阈值下的模型表现

使用示例：
    commands = build_threshold_commands(
        data_yaml=Path("configs/dataset.yaml"),
        weights=Path("runs/train/exp/weights/best.pt"),
        thresholds=(0.3, 0.5, 0.7)
    )
    # 返回包含3个命令的列表，分别使用 0.3, 0.5, 0.7 的置信度阈值
"""

from pathlib import Path

from ..factories.evaluator_factory import EvaluatorFactory


def build_threshold_commands(
    data_yaml: Path, 
    weights: Path, 
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.7)
) -> list[list[str]]:
    """
    为多个置信度阈值构建验证命令
    
    参数：
        data_yaml: 数据集配置文件路径
        weights: 模型权重文件路径
        thresholds: 置信度阈值元组，默认 (0.25, 0.5, 0.7)
    
    返回值：
        list[list[str]]: 验证命令列表，每个元素是一个完整的验证命令
    
    使用场景：
        当需要评估模型在不同置信度阈值下的性能时使用，
        可以对比精确率、召回率等指标随阈值的变化。
    
    示例：
        commands = build_threshold_commands(
            data_yaml=Path("configs/dataset.yaml"),
            weights=Path("runs/train/exp/weights/best.pt"),
            thresholds=(0.3, 0.5, 0.7)
        )
        
        # 执行所有命令
        runner = ExperimentRunner()
        for cmd in commands:
            result = runner.run(cmd, cwd="/workspace/yolov5")
            print(f"阈值 {cmd[cmd.index('--conf-thres')+1]}:")
            print(result.stdout)
    """
    # 创建评估器工厂
    factory = EvaluatorFactory()
    
    # 为每个阈值构建验证命令
    return [
        factory.build_val_command(data_yaml=data_yaml, weights=weights, conf=conf)
        for conf in thresholds
    ]
