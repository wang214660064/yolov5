import tempfile
from pathlib import Path
import unittest

from project.nozzle_inspection.reporting.report_builder import ReportBuilder


class ReportBuilderTest(unittest.TestCase):
    def test_report_builder_renders_required_sections(self):
        report = ReportBuilder().build_markdown(
            title="3D打印机喷头检测项目报告",
            metrics={"NG_P": 0.98, "NG_R": 0.85},
        )

        self.assertIn("项目背景", report)
        self.assertIn("数据集分析", report)
        self.assertIn("算法原理", report)
        self.assertIn("结果分析", report)
        self.assertIn("NG_P", report)

    def test_report_builder_writes_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.md"

            ReportBuilder().write_markdown(output_path, metrics={"NG_P": 0.98})

            self.assertTrue(output_path.exists())
            self.assertIn("3D打印机喷头检测项目报告", output_path.read_text(encoding="utf-8"))

    def test_report_builder_writes_pptx_even_without_optional_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.pptx"

            result = ReportBuilder().write_pptx(output_path, metrics={"NG_P": 0.98})

            self.assertEqual(result, output_path)
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
