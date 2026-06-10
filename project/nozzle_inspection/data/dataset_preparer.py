"""
数据集准备模块 - 数据预处理管道

该模块提供完整的数据预处理功能，包括：
1. 收集数据集样本（图片+标签对）
2. 精确去重（基于 SHA256 哈希）
3. 标签转换（将多类别标签归并为 NG/OK 两类）
4. 分层划分（保持类别比例）
5. 复制文件到输出目录
6. 生成处理报告

使用示例：
    preparer = DatasetPreparer(
        source_root=Path("../dataset_2"),
        output_root=Path("../outputs/datasets/nozzle_ng_ok_v1"),
        val_ratio=0.2,
        seed=42
    )
    report = preparer.prepare()
"""

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, rmtree
import csv

from .deduplicate import Deduplicator
from .label_converter import LabelConverter
from .split_dataset import stratified_split

# 支持的图片文件后缀
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class PrepareReport:
    """
    数据准备报告，记录处理过程的统计信息
    
    属性：
        total_candidates: 候选样本总数（去重前）
        duplicate_count: 重复样本数
        train_count: 训练集样本数
        val_count: 验证集样本数
        test_count: 测试集样本数
        output_root: 输出目录路径
    """
    total_candidates: int  # 候选样本总数
    duplicate_count: int   # 重复样本数
    train_count: int       # 训练集样本数
    val_count: int         # 验证集样本数
    test_count: int        # 测试集样本数
    output_root: Path      # 输出目录


@dataclass(frozen=True)
class DatasetSample:
    """
    数据集样本，包含图片路径、标签路径和类别ID
    
    属性：
        image_path: 图片文件路径
        label_path: 标签文件路径
        class_id: 类别ID（0=NG, 1=OK）
    """
    image_path: Path   # 图片路径
    label_path: Path   # 标签路径
    class_id: int      # 类别ID


@dataclass
class DatasetPreparer:
    """
    数据集准备器，一键完成数据预处理流程：
    1. 收集样本
    2. 精确去重
    3. 分层划分训练/验证集
    4. 转换标签（多类别→NG/OK）
    5. 复制文件到输出目录
    6. 生成报告
    
    属性：
        source_root: 原始数据集根目录
        output_root: 处理后数据集输出目录
        val_ratio: 验证集占比（默认0.2）
        seed: 随机种子（确保划分结果可重复）
    """

    source_root: Path  # 原始数据集目录
    output_root: Path  # 输出目录
    val_ratio: float = 0.2  # 验证集比例
    seed: int = 42          # 随机种子

    def prepare(self) -> PrepareReport:
        """
        执行完整的数据准备流程
        
        返回值：
            PrepareReport: 处理报告，包含各阶段的统计信息
        
        处理流程：
            1. 检查输入目录是否存在
            2. 清理输出目录（如果存在）
            3. 创建输出目录结构
            4. 收集训练集和验证集候选样本
            5. 执行精确去重
            6. 分层划分训练/验证集
            7. 收集测试集样本（不参与划分）
            8. 复制文件并转换标签
            9. 生成去重报告和数据准备报告
        """
        source_root = Path(self.source_root)
        output_root = Path(self.output_root)
        
        # 检查输入目录是否存在
        if not source_root.exists():
            raise FileNotFoundError(f"数据集目录不存在：{source_root}")

        # 如果输出目录已存在，删除它（确保输出目录干净）
        if output_root.exists():
            rmtree(output_root)
        
        # 创建输出目录结构
        self._make_output_dirs(output_root)

        # 步骤1：收集候选样本（合并原始训练集和验证集）
        candidates = self._collect_samples(source_root, ("train", "val"))
        
        # 步骤2：执行精确去重（基于 SHA256 哈希）
        dedup_report = Deduplicator().find_exact_duplicates(
            [sample.image_path for sample in candidates]
        )
        duplicate_set = set(dedup_report.duplicate_files)
        kept_samples = [sample for sample in candidates if sample.image_path not in duplicate_set]

        # 步骤3：分层划分训练/验证集（保持类别比例）
        train_items, val_items = stratified_split(
            [(sample, sample.class_id) for sample in kept_samples],
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        train_samples = [sample for sample, _ in train_items]
        val_samples = [sample for sample, _ in val_items]
        
        # 步骤4：收集测试集样本（测试集不参与划分）
        test_samples = self._collect_samples(source_root, ("test",))

        # 步骤5：复制文件并转换标签
        converter = LabelConverter()
        self._copy_split(train_samples, output_root, "train", converter)
        self._copy_split(val_samples, output_root, "val", converter)
        self._copy_split(test_samples, output_root, "test", converter)
        
        # 步骤6：生成去重报告
        self._write_deduplicate_report(output_root / "deduplicate_report.csv", dedup_report.duplicate_pairs)

        # 步骤7：生成数据准备报告
        report = PrepareReport(
            total_candidates=len(candidates),
            duplicate_count=len(duplicate_set),
            train_count=len(train_samples),
            val_count=len(val_samples),
            test_count=len(test_samples),
            output_root=output_root,
        )
        self._write_prepare_report(output_root / "data_prepare_report.md", report)
        
        return report

    def _collect_samples(self, source_root: Path, splits: tuple[str, ...]) -> list[DatasetSample]:
        """
        收集指定划分中的样本（图片+标签对）
        
        参数：
            source_root: 数据集根目录
            splits: 要收集的划分列表，如 ("train", "val")
        
        返回值：
            list[DatasetSample]: 样本列表
        
        收集规则：
            1. 遍历每个划分目录下的 images 文件夹
            2. 对于每个图片文件，查找对应的标签文件
            3. 使用 LabelConverter 确定样本的类别ID
            4. 只保留有对应标签的图片
        """
        converter = LabelConverter()
        samples: list[DatasetSample] = []
        
        for split in splits:
            image_dir = source_root / split / "images"
            label_dir = source_root / split / "labels"
            
            # 如果图片目录不存在，跳过
            if not image_dir.exists():
                continue
            
            # 遍历所有图片文件
            for image_path in sorted(path for path in image_dir.iterdir() 
                                    if path.suffix.lower() in IMAGE_SUFFIXES):
                # 构建对应的标签文件路径
                label_path = label_dir / f"{image_path.stem}.txt"
                
                # 如果标签文件不存在，跳过此图片
                if not label_path.exists():
                    continue
                
                # 获取样本的类别ID（0=NG, 1=OK）
                class_id = converter.class_id_for_stem(image_path.stem)
                
                # 添加到样本列表
                samples.append(DatasetSample(
                    image_path=image_path, 
                    label_path=label_path, 
                    class_id=class_id
                ))
        
        return samples

    @staticmethod
    def _make_output_dirs(output_root: Path) -> None:
        """
        创建输出目录结构
        
        目录结构：
            output_root/
                images/
                    train/
                    val/
                    test/
                labels/
                    train/
                    val/
                    test/
        
        参数：
            output_root: 输出目录根路径
        """
        for kind in ("images", "labels"):
            for split in ("train", "val", "test"):
                (output_root / kind / split).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy_split(samples: list[DatasetSample], output_root: Path, split: str, converter: LabelConverter) -> None:
        """
        复制一个划分的所有样本到输出目录
        
        参数：
            samples: 样本列表
            output_root: 输出目录根路径
            split: 划分名称（train/val/test）
            converter: 标签转换器
        """
        for sample in samples:
            # 复制图片文件
            copy2(sample.image_path, output_root / "images" / split / sample.image_path.name)
            
            # 转换并复制标签文件（将多类别标签转换为 NG/OK）
            converter.convert_file(
                sample.label_path, 
                output_root / "labels" / split / sample.label_path.name
            )

    @staticmethod
    def _write_deduplicate_report(output_path: Path, pairs: list[tuple[Path, Path]]) -> None:
        """
        写入去重报告（CSV格式）
        
        参数：
            output_path: 输出文件路径
            pairs: 重复文件对列表，每个元组为(保留文件, 重复文件)
        """
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["保留文件", "重复文件"])
            for keep_path, duplicate_path in pairs:
                writer.writerow([keep_path, duplicate_path])

    @staticmethod
    def _write_prepare_report(output_path: Path, report: PrepareReport) -> None:
        """
        写入数据准备报告（Markdown格式）
        
        参数：
            output_path: 输出文件路径
            report: 数据准备报告对象
        """
        output_path.write_text(
            "\n".join([
                "# 数据准备报告",
                "",
                f"- 候选样本数：{report.total_candidates}",
                f"- 精确重复样本数：{report.duplicate_count}",
                f"- 训练集样本数：{report.train_count}",
                f"- 验证集样本数：{report.val_count}",
                f"- 测试集样本数：{report.test_count}",
                f"- 输出目录：{report.output_root}",
                "",
            ]),
            encoding="utf-8",
        )
