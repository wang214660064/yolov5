"""
数据集采样工具 - 按比例生成小子集

该模块提供了一个简单的工具，用于从大数据集中按比例采样生成小子集，
方便在本地电脑上进行快速测试和调试。

核心功能：
1. 支持按比例采样（如采样10%数据）
2. 保持类别分布平衡（分层采样）
3. 同时复制图像和对应的标签文件
4. 支持 train/val/test 三个数据集的同时采样

使用示例：
    # 创建采样器
    sampler = DatasetSampler(
        input_dir=Path("project/outputs/datasets/nozzle_ng_ok_v1"),
        output_dir=Path("project/outputs/datasets/nozzle_ng_ok_sample"),
        sample_ratio=0.1  # 采样10%
    )
    
    # 执行采样
    sampler.sample()
    
    # 查看采样结果
    print(f"采样完成，共采样 {sampler.sampled_count} 个样本")

命令行使用：
    python -m project.nozzle_inspection.utils.sample_dataset \
        --input project/outputs/datasets/nozzle_ng_ok_v1 \
        --output project/outputs/datasets/nozzle_ng_ok_sample \
        --ratio 0.1
"""

import argparse
import random
from pathlib import Path
from shutil import copy2
from typing import List, Tuple, Dict


class DatasetSampler:
    """
    数据集采样器，用于按比例从大数据集中采样生成小子集
    
    属性：
        input_dir: 输入数据集目录，包含 images/ 和 labels/ 子目录
        output_dir: 输出小子集目录
        sample_ratio: 采样比例（0.0-1.0）
        seed: 随机种子，确保采样结果可重复
        stratified: 是否保持类别分布（分层采样）
        sampled_count: 已采样的样本数量
    
    方法：
        sample(): 执行采样
        _get_samples(): 获取所有样本列表
        _stratified_sample(): 分层采样
        _copy_samples(): 复制样本文件
    """
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        sample_ratio: float = 0.1,
        seed: int = 42,
        stratified: bool = True
    ):
        """
        初始化采样器
        
        参数：
            input_dir: 输入数据集目录路径
            output_dir: 输出小子集目录路径
            sample_ratio: 采样比例，默认为0.1（10%）
            seed: 随机种子，默认42
            stratified: 是否分层采样保持类别平衡，默认True
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.stratified = stratified
        self.sampled_count = 0
        
        # 设置随机种子
        random.seed(seed)
    
    def _get_samples(self, split: str) -> List[Tuple[str, Path, Path]]:
        """
        获取指定数据集分割的所有样本
        
        参数：
            split: 数据集分割名称（train/val/test）
        
        返回值：
            List[Tuple[str, Path, Path]]: 样本列表，每个元素包含（文件名前缀, 图像路径, 标签路径）
        """
        images_dir = self.input_dir / "images" / split
        labels_dir = self.input_dir / "labels" / split
        
        samples = []
        
        if not images_dir.exists() or not labels_dir.exists():
            return samples
        
        # 遍历所有图像文件
        for image_path in images_dir.glob("*"):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            
            # 获取对应的标签文件路径
            label_path = labels_dir / (image_path.stem + ".txt")
            
            if label_path.exists():
                samples.append((image_path.stem, image_path, label_path))
        
        return samples
    
    def _stratified_sample(self, samples: List[Tuple[str, Path, Path]]) -> List[Tuple[str, Path, Path]]:
        """
        分层采样，保持类别分布
        
        参数：
            samples: 原始样本列表
        
        返回值：
            List[Tuple[str, Path, Path]]: 采样后的样本列表
        """
        # 按类别分组
        # 假设文件名格式为：nozzle_clean_0001, nozzle_tip_wrapped_0001 等
        # 类别由前缀决定
        groups: Dict[str, List[Tuple[str, Path, Path]]] = {}
        
        for stem, img_path, lbl_path in samples:
            # 提取类别前缀（第一个下划线之前的部分）
            # 或者使用标签文件中的类别信息
            class_id = self._get_class_from_label(lbl_path)
            if class_id not in groups:
                groups[class_id] = []
            groups[class_id].append((stem, img_path, lbl_path))
        
        # 对每个类别进行采样
        sampled = []
        for class_id, class_samples in groups.items():
            sample_size = max(1, int(len(class_samples) * self.sample_ratio))
            sampled.extend(random.sample(class_samples, sample_size))
        
        # 打乱顺序
        random.shuffle(sampled)
        return sampled
    
    def _get_class_from_label(self, label_path: Path) -> int:
        """
        从标签文件中获取类别ID
        
        参数：
            label_path: 标签文件路径
        
        返回值：
            int: 类别ID（0=NG, 1=OK）
        """
        try:
            with open(label_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    if parts:
                        return int(parts[0])
        except Exception:
            pass
        
        # 默认返回0（NG）
        return 0
    
    def _simple_sample(self, samples: List[Tuple[str, Path, Path]]) -> List[Tuple[str, Path, Path]]:
        """
        简单随机采样，不考虑类别分布
        
        参数：
            samples: 原始样本列表
        
        返回值：
            List[Tuple[str, Path, Path]]: 采样后的样本列表
        """
        sample_size = max(1, int(len(samples) * self.sample_ratio))
        return random.sample(samples, sample_size)
    
    def _copy_samples(self, samples: List[Tuple[str, Path, Path]], split: str):
        """
        复制采样后的样本文件到输出目录
        
        参数：
            samples: 要复制的样本列表
            split: 数据集分割名称（train/val/test）
        """
        output_images_dir = self.output_dir / "images" / split
        output_labels_dir = self.output_dir / "labels" / split
        
        # 创建输出目录
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for stem, img_path, lbl_path in samples:
            copy2(img_path, output_images_dir / img_path.name)
            copy2(lbl_path, output_labels_dir / lbl_path.name)
            self.sampled_count += 1
    
    def sample(self) -> Dict[str, int]:
        """
        执行采样，处理所有数据集分割（train/val/test）
        
        返回值：
            Dict[str, int]: 每个分割的采样数量
        """
        splits = ["train", "val", "test"]
        result = {}
        
        for split in splits:
            samples = self._get_samples(split)
            
            if not samples:
                print(f"警告：{split} 数据集为空或不存在")
                result[split] = 0
                continue
            
            # 执行采样
            if self.stratified:
                sampled = self._stratified_sample(samples)
            else:
                sampled = self._simple_sample(samples)
            
            # 复制文件
            self._copy_samples(sampled, split)
            result[split] = len(sampled)
            
            print(f"  {split}: {len(sampled)}/{len(samples)} ({len(sampled)/len(samples)*100:.1f}%)")
        
        return result


def main():
    """
    命令行入口函数
    """
    parser = argparse.ArgumentParser(
        description="数据集采样工具 - 按比例生成小子集用于测试"
    )
    
    parser.add_argument(
        "--input", "-i",
        default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1",
        type=str,
        help="输入数据集目录路径"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_downsampled",
        type=str,
        help="输出小子集目录路径"
    )
    
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.05,
        help="采样比例（0.0-1.0），默认0.05"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子，确保采样结果可重复，默认42"
    )
    
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="不使用分层采样，直接随机采样"
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("数据集采样工具")
    print("=" * 50)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"采样比例: {args.ratio * 100:.1f}%")
    print(f"随机种子: {args.seed}")
    print(f"分层采样: {'是' if not args.no_stratified else '否'}")
    print("-" * 50)
    
    # 创建采样器并执行采样
    sampler = DatasetSampler(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sample_ratio=args.ratio,
        seed=args.seed,
        stratified=not args.no_stratified
    )
    
    result = sampler.sample()
    
    print("-" * 50)
    print("采样完成！")
    total_sampled = sum(result.values())
    print(f"共采样 {total_sampled} 个样本")
    print(f"输出目录: {args.output}")


if __name__ == "__main__":
    main()