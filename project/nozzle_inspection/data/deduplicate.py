"""
数据去重模块 - 精确去重功能

该模块提供基于 SHA256 哈希算法的精确去重功能，用于检测数据集中完全相同的图片文件。

核心思想：
1. 计算每个文件的 SHA256 哈希值
2. 将相同哈希值的文件分组
3. 每组只保留一个文件，其余标记为重复

使用示例：
    deduplicator = Deduplicator()
    report = deduplicator.find_exact_duplicates([Path("image1.jpg"), Path("image2.jpg")])
    print(f"重复文件数：{len(report.duplicate_files)}")
"""

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os

import cv2
import imagehash
import numpy as np
from PIL import Image

from ..utils.image_hash import sha256_file


@dataclass(frozen=True)
class DuplicateReport:
    """
    去重报告，记录去重结果
    
    属性：
        keep_files: 保留的文件列表（每组第一个文件）
        duplicate_files: 重复的文件列表
        duplicate_pairs: 重复文件对列表，每个元组为(保留文件, 重复文件)
    """
    keep_files: list[Path]              # 保留的文件
    duplicate_files: list[Path]         # 重复的文件
    duplicate_pairs: list[tuple[Path, Path]]  # 重复文件对
    similar_duplicate_pairs: list[tuple[Path, Path, float]] | None = None  # SSIM 相似重复对


class Deduplicator:
    """
    数据去重器，负责数据集精确去重
    
    基于 SHA256 哈希算法实现精确去重，只有完全相同的文件才会被标记为重复。
    后续可扩展支持视觉相似去重（感知哈希）。
    
    方法：
        find_exact_duplicates(): 查找精确重复的文件
    """

    def __init__(self, workers: int | None = 1) -> None:
        """
        初始化去重器。

        workers=1 表示单进程；workers=0 或 None 表示按 CPU 数和任务量自动选择；
        workers>1 表示显式启用指定数量的多进程。
        """
        self.workers = workers

    def find_exact_duplicates(self, image_paths: list[Path]) -> DuplicateReport:
        """
        查找精确重复的图片文件
        
        参数：
            image_paths: 图片文件路径列表
        
        返回值：
            DuplicateReport: 去重报告，包含保留文件、重复文件和重复对
        
        算法流程：
            1. 遍历所有文件，计算每个文件的 SHA256 哈希值
            2. 将相同哈希值的文件分组
            3. 每组保留第一个文件，其余标记为重复
            4. 记录重复文件对（保留文件, 重复文件）
        
        示例：
            输入：[image_a.jpg, image_b.jpg, image_c.jpg]
            假设 image_a.jpg 和 image_c.jpg 内容完全相同
            输出：
                keep_files: [image_a.jpg, image_b.jpg]
                duplicate_files: [image_c.jpg]
                duplicate_pairs: [(image_a.jpg, image_c.jpg)]
        """
        sorted_paths = sorted(Path(p) for p in image_paths)
        worker_count = _resolve_worker_count(self.workers, len(sorted_paths))

        # 字典：key 是哈希值，value 是具有相同哈希值的文件列表
        groups: dict[str, list[Path]] = {}

        # 遍历所有图片路径，按哈希值分组。文件较多时并行计算 SHA256，降低读盘和哈希耗时。
        if worker_count > 1:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                hash_results = executor.map(_hash_image_path, sorted_paths)
                for path, hash_value in hash_results:
                    groups.setdefault(hash_value, []).append(path)
        else:
            for path in sorted_paths:
                hash_value = sha256_file(path)
            # 将路径添加到对应哈希值的组中
                groups.setdefault(hash_value, []).append(path)

        # 整理去重结果
        keep_files: list[Path] = []           # 保留的文件
        duplicate_files: list[Path] = []      # 重复的文件
        duplicate_pairs: list[tuple[Path, Path]] = []  # 重复对

        # 遍历每个哈希组
        for paths in groups.values():
            # 保留每组的第一个文件
            keeper = paths[0]
            keep_files.append(keeper)
            
            # 将其余文件标记为重复
            for duplicate in paths[1:]:
                duplicate_files.append(duplicate)
                duplicate_pairs.append((keeper, duplicate))

        # 返回去重报告
        return DuplicateReport(
            keep_files=sorted(keep_files),
            duplicate_files=sorted(duplicate_files),
            duplicate_pairs=duplicate_pairs,
            similar_duplicate_pairs=[],
        )

    def find_duplicates(
        self,
        image_paths: list[Path],
        ssim_threshold: float = 0.85,
        phash_threshold: int = 4,
    ) -> DuplicateReport:
        """
        查找精确重复和 SSIM 视觉相似重复的图片文件

        参数：
            image_paths: 图片文件路径列表
            ssim_threshold: SSIM 相似度阈值，默认 0.85

        返回值：
            DuplicateReport: 同时包含精确重复和视觉相似重复

        处理顺序：
            1. 先用 SHA256 剔除完全重复样本
            2. 再在剩余样本中用 SSIM 检测视觉相似样本
        """
        exact_report = self.find_exact_duplicates(image_paths)
        exact_duplicate_set = set(exact_report.duplicate_files)
        exact_keep_files = [Path(p) for p in exact_report.keep_files if Path(p) not in exact_duplicate_set]

        visual_report = self.find_similar_duplicates(
            exact_keep_files,
            ssim_threshold=ssim_threshold,
            phash_threshold=phash_threshold,
        )

        duplicate_files = sorted(list(exact_report.duplicate_files) + visual_report.duplicate_files)
        return DuplicateReport(
            keep_files=visual_report.keep_files,
            duplicate_files=duplicate_files,
            duplicate_pairs=exact_report.duplicate_pairs,
            similar_duplicate_pairs=visual_report.similar_duplicate_pairs,
        )

    def find_similar_duplicates(
        self,
        image_paths: list[Path],
        ssim_threshold: float = 0.85,
        phash_threshold: int = 4,
        phash_cache: dict[Path, str] | None = None,
    ) -> DuplicateReport:
        """
        仅执行 SSIM 相似去重。

        该方法用于数据准备中“已完成精确哈希去重”的样本池，避免重复计算 SHA256。
        """
        visual_keep_files: list[Path] = []
        similar_duplicate_files: list[Path] = []
        similar_pairs: list[tuple[Path, Path, float]] = []

        sorted_paths = sorted(Path(p) for p in image_paths)
        worker_count = _resolve_worker_count(self.workers, len(sorted_paths))
        phashes = phash_cache or self.precompute_phashes(sorted_paths)
        executor = ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
        try:
            for candidate in sorted_paths:
                matched = False
                candidate_hash = phashes.get(candidate)
                comparable_keepers = [
                    keeper for keeper in visual_keep_files
                    if _phash_distance(phashes.get(keeper), candidate_hash) <= phash_threshold
                ]
                if executor is not None and len(visual_keep_files) > 1:
                    pair_args = [(keeper, candidate, 128) for keeper in comparable_keepers]
                    scores = executor.map(_compute_ssim_pair, pair_args)
                    for keeper, _, score in scores:
                        if score >= ssim_threshold:
                            similar_duplicate_files.append(candidate)
                            similar_pairs.append((keeper, candidate, score))
                            matched = True
                            break
                else:
                    for keeper in comparable_keepers:
                        score = compute_ssim(keeper, candidate)
                        if score >= ssim_threshold:
                            similar_duplicate_files.append(candidate)
                            similar_pairs.append((keeper, candidate, score))
                            matched = True
                            break
                if not matched:
                    visual_keep_files.append(candidate)
        finally:
            if executor is not None:
                executor.shutdown()

        return DuplicateReport(
            keep_files=sorted(visual_keep_files),
            duplicate_files=sorted(similar_duplicate_files),
            duplicate_pairs=[],
            similar_duplicate_pairs=similar_pairs,
        )

    def precompute_phashes(self, image_paths: list[Path]) -> dict[Path, str]:
        """
        批量预计算 pHash。

        参考老师代码的并行思路：先把所有图片的感知哈希一次性算好，
        后续相似比较只查缓存，避免在不同类别或不同阶段重复读图。
        """
        sorted_paths = sorted(Path(p) for p in image_paths)
        worker_count = _resolve_worker_count(self.workers, len(sorted_paths))
        return _build_phash_map(sorted_paths, worker_count)


def compute_ssim(left_path: Path, right_path: Path, image_size: int = 128) -> float:
    """
    计算两张图片的 SSIM 相似度。

    为了让流程在没有额外 skimage 依赖时也可运行，这里使用标准 SSIM 公式的灰度实现。
    """
    left = _read_gray_resized(left_path, image_size)
    right = _read_gray_resized(right_path, image_size)
    if left is None or right is None:
        return 0.0

    left = left.astype(np.float64)
    right = right.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu_left = left.mean()
    mu_right = right.mean()
    sigma_left = ((left - mu_left) ** 2).mean()
    sigma_right = ((right - mu_right) ** 2).mean()
    sigma_lr = ((left - mu_left) * (right - mu_right)).mean()

    numerator = (2 * mu_left * mu_right + c1) * (2 * sigma_lr + c2)
    denominator = (mu_left ** 2 + mu_right ** 2 + c1) * (sigma_left + sigma_right + c2)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _resolve_worker_count(workers: int | None, task_count: int) -> int:
    if task_count < 2:
        return 1
    if workers is None or workers == 0:
        return max(1, min(os.cpu_count() or 1, task_count))
    return max(1, min(workers, task_count))


def _hash_image_path(path: Path) -> tuple[Path, str]:
    return path, sha256_file(path)


def _build_phash_map(paths: list[Path], worker_count: int) -> dict[Path, str]:
    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(_compute_phash_path, paths)
            return {path: hash_value for path, hash_value in results if hash_value is not None}
    return {
        path: hash_value
        for path, hash_value in (_compute_phash_path(path) for path in paths)
        if hash_value is not None
    }


def _compute_phash_path(path: Path) -> tuple[Path, str | None]:
    try:
        with Image.open(path) as image:
            return path, str(imagehash.phash(image))
    except Exception:
        return path, None


def _phash_distance(left_hash: str | None, right_hash: str | None) -> int:
    if left_hash is None or right_hash is None:
        return 10**9
    return imagehash.hex_to_hash(left_hash) - imagehash.hex_to_hash(right_hash)


def _compute_ssim_pair(args: tuple[Path, Path, int]) -> tuple[Path, Path, float]:
    left_path, right_path, image_size = args
    return left_path, right_path, compute_ssim(left_path, right_path, image_size=image_size)


def _read_gray_resized(path: Path, image_size: int) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
