"""
评估模块包 - 所有评估相关组件的统一导出

该模块将所有评估相关组件统一导出，方便其他模块导入使用。

评估组件说明：
    - ErrorAnalyzer: 错误分析器，统计预测结果
    - evaluate_pipeline: 评估管道，构建多阈值验证命令

使用示例：
    from nozzle_inspection.evaluation import ErrorAnalyzer
    
    analyzer = ErrorAnalyzer()
    result = analyzer.summarize(predictions)
"""

from .error_analyzer import ErrorAnalyzer

__all__ = ["ErrorAnalyzer"]
