"""
数据集划分模块 - 分层划分功能

该模块提供数据集划分功能，支持按类别分层划分，确保训练集和验证集的类别分布一致。

核心功能：
1. stratified_split(): 分层划分数据集，保持类别比例
2. copy_yolo_pair(): 复制图片和标签文件对

使用示例：
    samples = [(sample1, 0), (sample2, 1), (sample3, 0)]
    train, val = stratified_split(samples, val_ratio=0.2, seed=42)
"""

import random
from collections import defaultdict
from pathlib import Path
from shutil import copy2
from typing import Iterable, TypeVar

# 泛型类型变量，用于支持任意类型的样本
T = TypeVar("T")


def stratified_split(samples: list[tuple[T, int]], val_ratio: float = 0.2, seed: int = 42) -> tuple[list[tuple[T, int]], list[tuple[T, int]]]:
    """
    按类别分层拆分样本，返回训练集和验证集
    
    参数：
        samples: 样本列表，每个元素为 (样本, 类别ID) 元组
        val_ratio: 验证集占比，默认 0.2（20%）
        seed: 随机种子，确保划分结果可重复
    
    返回值：
        tuple[list, list]: (训练集, 验证集)
    
    算法流程：
        1. 按类别ID分组
        2. 对每个类别的样本单独打乱
        3. 按比例分配到训练集和验证集
        4. 确保每个类别在训练集和验证集中都有样本（至少1个）
    
    关键特性：
        - 保持类别比例（分层抽样）
        - 固定随机种子确保可重复性
        - 每个类别至少有1个样本在验证集（如果该类有多个样本）
    
    示例：
        输入: [(A, 0), (B, 0), (C, 1), (D, 1), (E, 1)]
        val_ratio=0.2, seed=42
        输出: 
            train: [(A, 0), (C, 1), (D, 1)]
            val: [(B, 0), (E, 1)]
    """
    # 验证比例范围
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("验证集比例必须在 0 到 1 之间")

    # 按类别ID分组
    grouped: dict[int, list[tuple[T, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    # 使用指定种子创建随机数生成器
    rng = random.Random(seed)
    
    # 存储训练集和验证集
    train: list[tuple[T, int]] = []
    val: list[tuple[T, int]] = []
    
    # 遍历每个类别
    for label, items in sorted(grouped.items()):
        # 复制并打乱顺序
        shuffled = list(items)
        rng.shuffle(shuffled)
        
        # 计算验证集数量
        # 如果该类别只有1个样本，不放入验证集；否则至少放入1个
        if len(shuffled) > 1:
            val_count = max(1, round(len(shuffled) * val_ratio))
        else:
            val_count = 0
        
        # 分配样本
        val.extend(shuffled[:val_count])
        train.extend(shuffled[val_count:])

    # 按样本的字符串表示排序（确保输出稳定）
    return sorted(train, key=lambda item: str(item[0])), sorted(val, key=lambda item: str(item[0]))


def copy_yolo_pair(image_path: Path, label_path: Path, output_image_dir: Path, output_label_dir: Path) -> tuple[Path, Path]:
    """
    复制图片和标签文件对
    
    参数：
        image_path: 源图片文件路径
        label_path: 源标签文件路径
        output_image_dir: 目标图片目录
        output_label_dir: 目标标签目录
    
    返回值：
        tuple[Path, Path]: (目标图片路径, 目标标签路径)
    
    功能说明：
        1. 确保目标目录存在
        2. 复制图片文件
        3. 复制标签文件
        4. 返回目标路径
    
    示例：
        copy_yolo_pair(
            Path("images/train/img.jpg"),
            Path("labels/train/img.txt"),
            Path("output/images/train"),
            Path("output/labels/train")
        )
    """
    # 确保输出目录存在
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建目标路径
    image_target = output_image_dir / image_path.name
    label_target = output_label_dir / label_path.name
    
    # 复制文件（保留元数据）
    copy2(image_path, image_target)
    copy2(label_path, label_target)
    
    return image_target, label_target
