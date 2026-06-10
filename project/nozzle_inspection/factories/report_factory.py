"""
报告工厂模块 - 报告生成组件创建

该模块负责创建报告生成相关的组件，遵循工厂模式设计。

当前功能：
- 创建报告构建器（用于生成 Markdown 和 PPT 报告）

使用示例：
    factory = ReportFactory()
    builder = factory.create_report_builder()
    markdown_path = builder.write_markdown(Path("report.md"))
    pptx_path = builder.write_pptx(Path("report.pptx"))
"""

from dataclasses import dataclass

from ..reporting.report_builder import ReportBuilder


@dataclass
class ReportFactory:
    """
    报告生成组件工厂
    
    负责创建报告生成相关的组件，遵循工厂模式设计。
    
    方法：
        create_report_builder(): 创建报告构建器
    """

    def create_report_builder(self) -> ReportBuilder:
        """
        创建报告构建器
        
        返回值：
            ReportBuilder: 报告构建器实例，用于生成项目报告
        
        ReportBuilder 主要功能：
            - build_markdown(): 构建 Markdown 报告内容
            - write_markdown(): 将 Markdown 报告写入文件
            - write_pptx(): 将 PPT 报告写入文件
        
        使用示例：
            factory = ReportFactory()
            builder = factory.create_report_builder()
            
            # 生成 Markdown 报告
            builder.write_markdown(Path("outputs/report.md"))
            
            # 生成 PPT 报告
            builder.write_pptx(Path("outputs/report.pptx"))
        """
        return ReportBuilder()
