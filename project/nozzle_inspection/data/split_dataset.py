import random
from collections import defaultdict
from pathlib import Path
from shutil import copy2
from typing import Iterable, TypeVar


T = TypeVar("T")


def stratified_split(samples: list[tuple[T, int]], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[tuple[T, int]], list[tuple[T, int]]]:
    """按类别分层拆分样本，返回训练集和验证集。"""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("验证集比例必须在 0 到 1 之间")

    grouped: dict[int, list[tuple[T, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    train: list[tuple[T, int]] = []
    val: list[tuple[T, int]] = []
    for label, items in sorted(grouped.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])

    return sorted(train, key=lambda item: str(item[0])), sorted(val, key=lambda item: str(item[0]))


def copy_yolo_pair(image_path: Path, label_path: Path, output_image_dir: Path, output_label_dir: Path) -> tuple[Path, Path]:
    """复制图片和标签对，保证输出目录存在。"""

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    image_target = output_image_dir / image_path.name
    label_target = output_label_dir / label_path.name
    copy2(image_path, image_target)
    copy2(label_path, label_target)
    return image_target, label_target
