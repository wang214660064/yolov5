"""
喷嘴检测项目层 - 3D打印机喷头检测项目主模块

该模块是项目的核心入口，提供以下功能：
- 数据集准备（去重、标签转换、分层划分）
- YOLOv5 模型训练和验证
- 实验报告生成（Markdown 和 PPT）
- 错误分析和结果统计

项目结构：
    - data/: 数据处理模块（去重、标签转换、数据集分析）
    - factories/: 工厂模式模块（创建训练器、评估器、报告器等）
    - training/: 训练相关模块
    - evaluation/: 评估和错误分析模块
    - reporting/: 报告生成模块
    - utils/: 工具函数模块

核心工作流程：
    1. 数据准备：使用 DatasetPreparer 处理原始数据
    2. 模型训练：使用 TrainerFactory 构建训练命令
    3. 模型验证：使用 EvaluatorFactory 构建验证命令
    4. 报告生成：使用 ReportBuilder 生成项目报告
"""

