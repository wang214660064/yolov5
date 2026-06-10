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


class Deduplicator:
    """
    数据去重器，负责数据集精确去重
    
    基于 SHA256 哈希算法实现精确去重，只有完全相同的文件才会被标记为重复。
    后续可扩展支持视觉相似去重（感知哈希）。
    
    方法：
        find_exact_duplicates(): 查找精确重复的文件
    """

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
        # 字典：key 是哈希值，value 是具有相同哈希值的文件列表
        groups: dict[str, list[Path]] = {}
        
        # 遍历所有图片路径，按哈希值分组
        for path in sorted(Path(p) for p in image_paths):
            # 计算文件的 SHA256 哈希值
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
        )
