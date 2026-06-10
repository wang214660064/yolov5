"""
数据集分析模块 - 统计数据集信息

该模块用于分析 YOLO 格式数据集的结构，包括：
1. 统计各数据集划分（train/val/test）的图片和标签数量
2. 分析样本前缀分布（如 nozzle_clean_001.jpg 的前缀为 nozzle_clean）
3. 生成 Markdown 格式的分析报告

使用示例：
    analyzer = DatasetAnalyzer(Path("../dataset_2"))
    summary = analyzer.analyze()
    analyzer.write_markdown(Path("dataset_report.md"))
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# 支持的图片文件后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetAnalyzer:
    """
    数据集分析器，用于统计 YOLO 数据集的图片、标签和样本前缀分布
    
    属性：
        dataset_root: 数据集根目录路径
    
    方法：
        analyze(): 分析数据集，返回统计结果
        render_markdown(): 将统计结果转换为 Markdown 格式
        write_markdown(): 将分析报告写入文件
    """

    dataset_root: Path  # 数据集根目录

    def analyze(self, splits: tuple[str, ...] = ("train", "val", "test")) -> dict[str, dict]:
        """
        分析数据集，统计各划分的图片、标签和前缀分布
        
        参数：
            splits: 要分析的数据集划分，默认为 ("train", "val", "test")
        
        返回值：
            dict[str, dict]: 统计结果字典，格式如下：
                {
                    "train": {
                        "images": 图片数量,
                        "labels": 标签数量,
                        "prefixes": {"前缀名": 数量, ...}
                    },
                    ...
                }
        
        示例：
            输入目录结构：
                dataset_2/
                    train/
                        images/
                            nozzle_clean_001.jpg
                            nozzle_clean_002.jpg
                            nozzle_ng_001.jpg
                        labels/
                            nozzle_clean_001.txt
                            nozzle_clean_002.txt
            
            输出结果：
                {
                    "train": {
                        "images": 3,
                        "labels": 2,
                        "prefixes": {"nozzle_clean": 2, "nozzle_ng": 1}
                    },
                    ...
                }
        """
        root = Path(self.dataset_root)
        summary: dict[str, dict] = {}
        
        # 遍历每个数据集划分
        for split in splits:
            image_dir = root / split / "images"
            label_dir = root / split / "labels"
            
            # 获取图片文件列表（只包含指定后缀的文件）
            if image_dir.exists():
                images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            else:
                images = []
            
            # 获取标签文件列表
            if label_dir.exists():
                labels = sorted(label_dir.glob("*.txt"))
            else:
                labels = []
            
            # 统计样本前缀分布
            # 前缀是文件名去掉数字后缀后的部分，如 nozzle_clean_001 -> nozzle_clean
            prefixes = Counter(_sample_prefix(path.stem) for path in images)
            
            # 保存统计结果
            summary[split] = {
                "images": len(images),
                "labels": len(labels),
                "prefixes": dict(sorted(prefixes.items())),
            }
        
        return summary

    @staticmethod
    def render_markdown(summary: dict[str, dict]) -> str:
        """
        将统计结果转换为 Markdown 格式
        
        参数：
            summary: analyze() 方法返回的统计结果字典
        
        返回值：
            str: Markdown 格式的报告文本
        
        生成的报告包含：
            - 各划分的图片数量
            - 各划分的标签数量
            - 各前缀的样本数量表格
        """
        lines = ["# 数据集分析报告", ""]
        
        # 遍历每个划分的统计结果
        for split, values in summary.items():
            lines.extend([
                f"## {split} 数据集",
                "",
                f"- 图片数量：{values.get('images', 0)}",
                f"- 标签数量：{values.get('labels', 0)}",
                "",
                "| 样本前缀 | 数量 |",
                "| --- | ---: |",
            ])
            
            # 添加前缀分布表格
            for prefix, count in values.get("prefixes", {}).items():
                lines.append(f"| {prefix} | {count} |")
            lines.append("")
        
        return "\n".join(lines)

    def write_markdown(self, output_path: Path) -> Path:
        """
        将分析报告写入 Markdown 文件
        
        参数：
            output_path: 输出文件路径
        
        返回值：
            Path: 实际输出的文件路径
        """
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告内容并写入文件
        output_path.write_text(self.render_markdown(self.analyze()), encoding="utf-8")
        
        return output_path


def _sample_prefix(stem: str) -> str:
    """
    从文件名（不含扩展名）中提取样本前缀
    
    文件名格式：前缀_数字，如 nozzle_clean_001
    
    参数：
        stem: 文件名（不含扩展名）
    
    返回值：
        str: 样本前缀
    
    示例：
        _sample_prefix("nozzle_clean_001") -> "nozzle_clean"
        _sample_prefix("image_123") -> "image"
        _sample_prefix("test") -> "test"  # 如果没有下划线分隔，返回原字符串
    """
    # 从右边开始分割，只分割一次
    parts = stem.rsplit("_", 1)
    
    # 如果分割后有两部分，且第二部分是数字，则返回第一部分作为前缀
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    else:
        return stem
