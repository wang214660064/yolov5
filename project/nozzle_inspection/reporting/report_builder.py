from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from xml.sax.saxutils import escape
import zipfile


@dataclass
class ReportBuilder:
    """生成项目 Markdown 报告和可选 PPT 汇报文档。"""

    default_title: str = "3D打印机喷头检测项目报告"

    def build_markdown(self, title: str | None = None, metrics: Mapping[str, object] | None = None) -> str:
        report_title = title or self.default_title
        metrics = metrics or {}
        metric_lines = "\n".join(f"- `{key}`: {value}" for key, value in metrics.items()) or "- 暂无训练指标，等待实验完成后补充。"
        return f"""# {report_title}

## 项目背景

本项目面向消费级 3D 打印机场景，识别喷头 OK/NG 状态，为打印前报警和停机策略提供算法依据。

## 数据集分析

数据集由外层 `dataset_2` 提供，训练前合并原始训练集和验证集，完成去重、NG/OK 标签归并和分层重划分。

## 算法原理

项目基于 YOLOv5 目标检测框架，通过两类检测头输出 `NG` 和 `OK`。业务上重点关注 NG 类精确率、召回率和低置信度样本。

## 模型设计

项目层使用工厂模式封装数据处理、模型配置、训练、评估和报告生成，YOLOv5 核心源码保持相对稳定。

## 实验过程

实验过程记录数据版本、清洗报告、增强策略、训练命令、权重路径和验证指标。

## 结果分析

{metric_lines}

## 结论与展望

后续应结合误检、漏检和低置信度样本继续修订标注规范、增强困难样本，并评估端侧推理耗时。
"""

    def write_markdown(self, output_path: Path, metrics: Mapping[str, object] | None = None, title: str | None = None) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.build_markdown(title=title, metrics=metrics), encoding="utf-8")
        return output_path

    def write_pptx(self, output_path: Path, metrics: Mapping[str, object] | None = None, title: str | None = None) -> Path | None:
        try:
            from pptx import Presentation
        except ImportError:
            return self._write_minimal_pptx(output_path, metrics=metrics, title=title)

        report_title = title or self.default_title
        prs = Presentation()
        self._add_title_slide(prs, report_title)
        self._add_bullet_slide(prs, "项目目标", ["两类检测：NG / OK", "优先保障高精确率", "保留 YOLOv5 核心能力，新增工厂模式项目层"])
        self._add_bullet_slide(prs, "实验指标", [f"{key}: {value}" for key, value in (metrics or {}).items()] or ["暂无训练指标"])
        self._add_bullet_slide(prs, "后续优化", ["复查误检漏检样本", "补充困难样本增强", "验证端侧推理耗时"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(output_path)
        return output_path

    def _write_minimal_pptx(self, output_path: Path, metrics: Mapping[str, object] | None = None, title: str | None = None) -> Path:
        report_title = title or self.default_title
        slides = [
            (report_title, ["YOLOv5 工厂模式项目汇报"]),
            ("项目目标", ["两类检测：NG / OK", "优先保障高精确率", "外层数据不纳入 Git 管理"]),
            ("实验指标", [f"{key}: {value}" for key, value in (metrics or {}).items()] or ["暂无训练指标"]),
            ("后续优化", ["复查误检漏检样本", "补充困难样本增强", "验证端侧推理耗时"]),
        ]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as pptx:
            pptx.writestr("[Content_Types].xml", _content_types(len(slides)))
            pptx.writestr("_rels/.rels", _root_rels())
            pptx.writestr("ppt/presentation.xml", _presentation_xml(len(slides)))
            pptx.writestr("ppt/_rels/presentation.xml.rels", _presentation_rels(len(slides)))
            for index, (slide_title, bullets) in enumerate(slides, start=1):
                pptx.writestr(f"ppt/slides/slide{index}.xml", _slide_xml(slide_title, bullets))
        return output_path

    @staticmethod
    def _add_title_slide(prs, title: str) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        slide.shapes.title.text = title
        slide.placeholders[1].text = "YOLOv5 工厂模式项目汇报"

    @staticmethod
    def _add_bullet_slide(prs, title: str, bullets: list[str]) -> None:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        body = slide.placeholders[1].text_frame
        body.clear()
        for index, bullet in enumerate(bullets):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            paragraph.text = bullet
            paragraph.level = 0


def _content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  {slide_overrides}
</Types>'''


def _root_rels() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>'''


def _presentation_xml(slide_count: int) -> str:
    slide_ids = "\n".join(f'<p:sldId id="{255 + i}" r:id="rId{i}"/>' for i in range(1, slide_count + 1))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldSz cx="12192000" cy="6858000" type="wide"/>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
</p:presentation>'''


def _presentation_rels(slide_count: int) -> str:
    rels = "\n".join(
        f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>'''


def _slide_xml(title: str, bullets: list[str]) -> str:
    bullet_text = "".join(
        f'<a:p><a:r><a:rPr lang="zh-CN" sz="2400"/><a:t>{escape(bullet)}</a:t></a:r></a:p>'
        for bullet in bullets
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="685800" y="457200"/><a:ext cx="10668000" cy="900000"/></a:xfrm></p:spPr>
        <p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="zh-CN" sz="3600" b="1"/><a:t>{escape(title)}</a:t></a:r></a:p></p:txBody>
      </p:sp>
      <p:sp>
        <p:nvSpPr><p:cNvPr id="3" name="Content"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr><a:xfrm><a:off x="914400" y="1600200"/><a:ext cx="10058400" cy="4267200"/></a:xfrm></p:spPr>
        <p:txBody><a:bodyPr/><a:lstStyle/>{bullet_text}</p:txBody>
      </p:sp>
    </p:spTree>
  </p:cSld>
</p:sld>'''
