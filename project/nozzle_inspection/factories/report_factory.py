from dataclasses import dataclass

from project.nozzle_inspection.reporting.report_builder import ReportBuilder


@dataclass
class ReportFactory:
    """创建报告生成组件。"""

    def create_report_builder(self) -> ReportBuilder:
        return ReportBuilder()
