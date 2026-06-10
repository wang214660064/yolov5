"""
训练模块包 - 所有训练相关组件的统一导出

该模块将所有训练相关组件统一导出，方便其他模块导入使用。

训练组件说明：
    - ExperimentRunner: 实验运行器，执行训练/验证命令

使用示例：
    from nozzle_inspection.training import ExperimentRunner
    
    runner = ExperimentRunner()
    result = runner.run(["python", "train.py", "--epochs", "50"])
"""

from .experiment_runner import ExperimentRunner

__all__ = ["ExperimentRunner"]
