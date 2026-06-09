from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, rmtree
import csv

from project.nozzle_inspection.data.deduplicate import Deduplicator
from project.nozzle_inspection.data.label_converter import LabelConverter
from project.nozzle_inspection.data.split_dataset import stratified_split


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class PrepareReport:
    total_candidates: int
    duplicate_count: int
    train_count: int
    val_count: int
    test_count: int
    output_root: Path


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path
    class_id: int


@dataclass
class DatasetPreparer:
    """一键完成去重、NG/OK 标签转换和数据集重划分。"""

    source_root: Path
    output_root: Path
    val_ratio: float = 0.2
    seed: int = 42

    def prepare(self) -> PrepareReport:
        source_root = Path(self.source_root)
        output_root = Path(self.output_root)
        if not source_root.exists():
            raise FileNotFoundError(f"数据集目录不存在：{source_root}")

        if output_root.exists():
            rmtree(output_root)
        self._make_output_dirs(output_root)

        candidates = self._collect_samples(source_root, ("train", "val"))
        dedup_report = Deduplicator().find_exact_duplicates([sample.image_path for sample in candidates])
        duplicate_set = set(dedup_report.duplicate_files)
        kept_samples = [sample for sample in candidates if sample.image_path not in duplicate_set]

        train_items, val_items = stratified_split(
            [(sample, sample.class_id) for sample in kept_samples],
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        train_samples = [sample for sample, _ in train_items]
        val_samples = [sample for sample, _ in val_items]
        test_samples = self._collect_samples(source_root, ("test",))

        converter = LabelConverter()
        self._copy_split(train_samples, output_root, "train", converter)
        self._copy_split(val_samples, output_root, "val", converter)
        self._copy_split(test_samples, output_root, "test", converter)
        self._write_deduplicate_report(output_root / "deduplicate_report.csv", dedup_report.duplicate_pairs)

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
        converter = LabelConverter()
        samples: list[DatasetSample] = []
        for split in splits:
            image_dir = source_root / split / "images"
            label_dir = source_root / split / "labels"
            if not image_dir.exists():
                continue
            for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    continue
                class_id = converter.class_id_for_stem(image_path.stem)
                samples.append(DatasetSample(image_path=image_path, label_path=label_path, class_id=class_id))
        return samples

    @staticmethod
    def _make_output_dirs(output_root: Path) -> None:
        for kind in ("images", "labels"):
            for split in ("train", "val", "test"):
                (output_root / kind / split).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy_split(samples: list[DatasetSample], output_root: Path, split: str, converter: LabelConverter) -> None:
        for sample in samples:
            copy2(sample.image_path, output_root / "images" / split / sample.image_path.name)
            converter.convert_file(sample.label_path, output_root / "labels" / split / sample.label_path.name)

    @staticmethod
    def _write_deduplicate_report(output_path: Path, pairs: list[tuple[Path, Path]]) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["保留文件", "重复文件"])
            for keep_path, duplicate_path in pairs:
                writer.writerow([keep_path, duplicate_path])

    @staticmethod
    def _write_prepare_report(output_path: Path, report: PrepareReport) -> None:
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
