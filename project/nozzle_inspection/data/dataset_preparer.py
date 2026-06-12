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
from collections import defaultdict

from .deduplicate import Deduplicator, DuplicateReport
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
        exact_duplicate_count: 精确重复样本数
        similar_duplicate_count: SSIM 相似重复样本数
        train_count: 训练集样本数
        val_count: 验证集样本数
        test_count: 测试集样本数
        output_root: 输出目录路径
        ssim_threshold: SSIM 相似去重阈值
        phash_threshold: pHash 汉明距离阈值
    """
    total_candidates: int  # 候选样本总数
    exact_duplicate_count: int   # 精确重复样本数
    similar_duplicate_count: int # SSIM 相似重复样本数
    train_count: int       # 训练集样本数
    val_count: int         # 验证集样本数
    test_count: int        # 测试集样本数
    output_root: Path      # 输出目录
    ssim_threshold: float  # SSIM 相似去重阈值
    phash_threshold: int   # pHash 汉明距离阈值

    @property
    def duplicate_count(self) -> int:
        return self.exact_duplicate_count + self.similar_duplicate_count


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
        ssim_threshold: SSIM 相似去重阈值
        deduplicate_workers: 去重阶段工作进程数，0 表示自动按 CPU 选择
    """

    source_root: Path  # 原始数据集目录
    output_root: Path  # 输出目录
    val_ratio: float = 0.2  # 验证集比例
    seed: int = 42          # 随机种子
    ssim_threshold: float = 0.85  # SSIM 相似去重阈值
    phash_threshold: int = 4  # pHash 汉明距离阈值，越小越严格
    deduplicate_workers: int = 0  # 去重阶段工作进程数，0 表示自动

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
            5. 执行精确去重和 SSIM 相似去重
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

        # 步骤1：收集候选样本（先合并原始训练集、验证集和测试集，避免跨集合重复导致数据泄漏）
        candidates = self._collect_samples(source_root, ("train", "val", "test"))
        
        # 步骤2：执行两级去重（先 SHA256 精确重复，再按类别内做 SSIM 相似去重）
        dedup_report = self._deduplicate_candidates(candidates)
        duplicate_set = set(dedup_report.duplicate_files)
        kept_samples = [sample for sample in candidates if sample.image_path not in duplicate_set]

        # 步骤3：从去重后的全集重新分层划分训练/验证/测试集（保持类别比例）
        train_items, val_items, test_items = self._split_train_val_test(kept_samples)
        train_samples = [sample for sample, _ in train_items]
        val_samples = [sample for sample, _ in val_items]
        test_samples = [sample for sample, _ in test_items]

        # 步骤5：复制文件并转换标签
        converter = LabelConverter()
        self._copy_split(train_samples, output_root, "train", converter)
        self._copy_split(val_samples, output_root, "val", converter)
        self._copy_split(test_samples, output_root, "test", converter)
        
        # 步骤6：生成去重报告
        self._write_deduplicate_report(output_root / "deduplicate_report.csv", dedup_report)

        # 步骤7：生成数据准备报告
        report = PrepareReport(
            total_candidates=len(candidates),
            exact_duplicate_count=len(dedup_report.duplicate_pairs),
            similar_duplicate_count=len(dedup_report.similar_duplicate_pairs or []),
            train_count=len(train_samples),
            val_count=len(val_samples),
            test_count=len(test_samples),
            output_root=output_root,
            ssim_threshold=self.ssim_threshold,
            phash_threshold=self.phash_threshold,
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

    def _deduplicate_candidates(self, candidates: list[DatasetSample]) -> DuplicateReport:
        """
        对候选样本做两级去重。

        注意：SSIM 相似去重按类别内比较，避免 OK 和 NG 图像相似时被误删。
        """
        exact_report = Deduplicator(workers=self.deduplicate_workers).find_exact_duplicates(
            [sample.image_path for sample in candidates]
        )
        exact_duplicate_set = set(exact_report.duplicate_files)
        remaining_by_class: dict[int, list[Path]] = defaultdict(list)
        for sample in candidates:
            if sample.image_path not in exact_duplicate_set:
                remaining_by_class[sample.class_id].append(sample.image_path)

        visual_keep_files: list[Path] = []
        similar_duplicate_files: list[Path] = []
        similar_pairs: list[tuple[Path, Path, float]] = []
        deduplicator = Deduplicator(workers=self.deduplicate_workers)
        remaining_paths = [path for paths in remaining_by_class.values() for path in paths]
        phash_cache = deduplicator.precompute_phashes(remaining_paths)
        for paths in remaining_by_class.values():
            class_report = deduplicator.find_similar_duplicates(
                paths,
                ssim_threshold=self.ssim_threshold,
                phash_threshold=self.phash_threshold,
                phash_cache=phash_cache,
            )
            visual_keep_files.extend(class_report.keep_files)
            similar_duplicate_files.extend(class_report.duplicate_files)
            similar_pairs.extend(class_report.similar_duplicate_pairs or [])

        exact_duplicate_files = list(exact_report.duplicate_files)
        return DuplicateReport(
            keep_files=sorted(visual_keep_files),
            duplicate_files=sorted(exact_duplicate_files + similar_duplicate_files),
            duplicate_pairs=exact_report.duplicate_pairs,
            similar_duplicate_pairs=similar_pairs,
        )

    def _split_train_val_test(
        self,
        samples: list[DatasetSample],
    ) -> tuple[list[tuple[DatasetSample, int]], list[tuple[DatasetSample, int]], list[tuple[DatasetSample, int]]]:
        """
        从统一去重后的样本池重新划分 train/val/test。

        val_ratio 同时作为最终验证集和测试集的目标占比。例如默认 0.2 时，
        先划出约 20% 测试集，再从剩余样本中按换算比例划出约 20% 验证集。
        当 val_ratio 过大时，退化为对剩余样本继续按相同比例划分，避免比例非法。
        """
        labeled_samples = [(sample, sample.class_id) for sample in samples]
        train_val_items, test_items = stratified_split(
            labeled_samples,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        if not train_val_items:
            return [], [], test_items

        remaining_val_ratio = self.val_ratio / (1.0 - self.val_ratio)
        if not 0.0 < remaining_val_ratio < 1.0:
            remaining_val_ratio = self.val_ratio

        train_items, val_items = stratified_split(
            train_val_items,
            val_ratio=remaining_val_ratio,
            seed=self.seed + 1,
        )
        return train_items, val_items, test_items

    @staticmethod
    def _write_deduplicate_report(output_path: Path, report: DuplicateReport) -> None:
        """
        写入去重报告（CSV格式）
        
        参数：
            output_path: 输出文件路径
            report: 去重报告
        """
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["去重类型", "保留文件", "重复文件", "相似度"])
            for keep_path, duplicate_path in report.duplicate_pairs:
                writer.writerow(["exact_hash", keep_path, duplicate_path, "1.000000"])
            for keep_path, duplicate_path, score in report.similar_duplicate_pairs or []:
                writer.writerow(["ssim", keep_path, duplicate_path, f"{score:.6f}"])

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
                f"- SSIM 相似阈值：{report.ssim_threshold}",
                f"- pHash 汉明距离阈值：{report.phash_threshold}",
                f"- 精确重复样本数：{report.exact_duplicate_count}",
                f"- SSIM 相似重复样本数：{report.similar_duplicate_count}",
                f"- 总重复样本数：{report.duplicate_count}",
                f"- 训练集样本数：{report.train_count}",
                f"- 验证集样本数：{report.val_count}",
                f"- 测试集样本数：{report.test_count}",
                f"- 输出目录：{report.output_root}",
                "",
            ]),
            encoding="utf-8",
        )
