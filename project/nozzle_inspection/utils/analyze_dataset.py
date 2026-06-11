"""
数据集分析工具 - 分析 nozzle_ng_ok_v1 数据集

分析数据集的图片数量、标签数量和样本前缀分布，
生成 markdown 格式的分析报告。

使用示例：
    python -m project.nozzle_inspection.utils.analyze_dataset \
        --dataset project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1 \
        --output project/nozzle_inspection/outputs/dataset_report.md
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


class DatasetAnalyzer:
    """
    数据集分析器，用于分析 YOLO 格式数据集的统计信息

    属性：
        dataset_dir: 数据集根目录
        splits: 数据集分割列表

    方法：
        analyze(): 执行分析并返回结果
        _get_label_files(): 获取指定分割的标签文件列表
        _extract_prefix(): 从文件名提取样本前缀
    """

    def __init__(self, dataset_dir: Path):
        """
        初始化分析器

        参数：
            dataset_dir: 数据集根目录路径
        """
        self.dataset_dir = dataset_dir
        self.splits = ["train", "val", "test"]

    def analyze(self) -> Dict[str, Dict]:
        """
        执行数据集分析

        返回值：
            Dict: 每个分割的分析结果
        """
        results = {}

        for split in self.splits:
            results[split] = self._analyze_split(split)

        return results

    def _analyze_split(self, split: str) -> Dict:
        """
        分析单个数据分割

        参数：
            split: 分割名称（train/val/test）

        返回值：
            Dict: 包含图片数量、标签数量、前缀分布的分析结果
        """
        labels_dir = self.dataset_dir / "labels" / split
        images_dir = self.dataset_dir / "images" / split

        label_files = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        image_count = 0

        if images_dir.exists():
            image_count = len([f for f in images_dir.glob("*") 
                              if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])

        label_count = len(label_files)

        # 统计前缀分布
        prefix_counter: Counter = Counter()
        for label_file in label_files:
            prefix = self._extract_prefix(label_file.stem)
            prefix_counter[prefix] += 1

        return {
            "image_count": image_count,
            "label_count": label_count,
            "prefix_distribution": dict(prefix_counter)
        }

    def _extract_prefix(self, stem: str) -> str:
        """
        从文件名提取样本前缀

        参数：
            stem: 文件名（不含扩展名）

        返回值：
            str: 样本前缀
        """
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return parts[0]

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """
        生成 markdown 格式的报告

        参数：
            results: analyze() 返回的分析结果

        返回值：
            str: markdown 格式的报告字符串
        """
        lines = ["# 数据集分析报告", ""]

        for split in self.splits:
            data = results[split]
            lines.append(f"## {split} 数据集")
            lines.append("")
            lines.append(f"- 图片数量：{data['image_count']}")
            lines.append(f"- 标签数量：{data['label_count']}")
            lines.append("")
            lines.append("| 样本前缀 | 数量 |")
            lines.append("| --- | ---: |")

            for prefix, count in sorted(data["prefix_distribution"].items()):
                lines.append(f"| {prefix} | {count} |")

            lines.append("")

        return "\n".join(lines)


def main():
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(
        description="数据集分析工具 - 分析 nozzle_ng_ok_v1 数据集"
    )

    parser.add_argument(
        "--dataset", "-d",
        default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1",
        type=str,
        help="数据集目录路径"
    )

    parser.add_argument(
        "--output", "-o",
        default="project/nozzle_inspection/outputs/nozzle_ng_ok_v1/dataset_report.md",
        type=str,
        help="输出报告文件路径"
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_dir.exists():
        print(f"错误：数据集目录不存在: {dataset_dir}")
        return

    print("=" * 50)
    print("数据集分析工具")
    print("=" * 50)
    print(f"数据集目录: {dataset_dir}")
    print(f"输出报告: {output_path}")
    print("-" * 50)

    # 执行分析
    analyzer = DatasetAnalyzer(dataset_dir)
    results = analyzer.analyze()

    # 打印统计信息
    for split in analyzer.splits:
        data = results[split]
        print(f"\n{split} 数据集:")
        print(f"  图片数量：{data['image_count']}")
        print(f"  标签数量：{data['label_count']}")
        if data['prefix_distribution']:
            print(f"  样本前缀分布：")
            for prefix, count in sorted(data['prefix_distribution'].items()):
                print(f"    {prefix}: {count}")

    # 生成并保存报告
    report = analyzer.generate_report(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")

    print("\n" + "-" * 50)
    print(f"报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
