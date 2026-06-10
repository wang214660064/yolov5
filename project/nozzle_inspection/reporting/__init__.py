"""
报告模块包 - 所有报告生成组件的统一导出

该模块将所有报告生成组件统一导出，方便其他模块导入使用。

报告生成组件说明：
    - ReportBuilder: 报告构建器，生成 Markdown 和 PPT 报告

使用示例：
    from nozzle_inspection.reporting import ReportBuilder
    
    builder = ReportBuilder()
    builder.write_markdown(Path("outputs/report.md"))
"""

from .report_builder import ReportBuilder

__all__ = ["ReportBuilder"]
