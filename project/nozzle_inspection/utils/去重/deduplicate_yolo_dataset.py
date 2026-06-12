"""
YOLO 数据集去重工具（多进程加速版）

对 YOLO 格式数据集进行去重，支持：
1. MD5 完全重复检测：识别完全相同的图片（多进程）
2. pHash + SSIM 相似检测：识别视觉相似的图片（多进程）
3. 同步移动对应的图片和标签文件
4. 保持 train/val/test 目录结构

使用方法：
    python deduplicate_yolo_dataset.py

配置参数在 main() 函数中修改。
"""

import os
import sys
import shutil
import time
import uuid
import multiprocessing as mp
from collections import defaultdict
from functools import partial

# 强制无缓冲输出
os.environ["PYTHONUNBUFFERED"] = "1"

# 导入重复检测模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import filter_duplicate_images as dup


# ==================== 配置参数 ====================

# 数据集根目录
DATASET_DIR = r"E:\Desktop\MAC-WIN\04 OpenCV\98_Practice\3DPrinterNozzleInspection\yolov5\project\nozzle_inspection\outputs\datasets\nozzle_ng_ok_v1"

# 重复文件输出根目录
OUTPUT_DIR = r"E:\Desktop\MAC-WIN\04 OpenCV\98_Practice\3DPrinterNozzleInspection\yolov5\project\nozzle_inspection\outputs\datasets\nozzle_ng_ok_v1_dedup"

# 去重模式：'exact_only' 或 'both'
DEDUP_MODE = "both"

# pHash 汉明距离阈值（越小越严格）
SIMILARITY_THRESHOLD = 4

# SSIM 结构相似性阈值（越大越严格）
SSIM_THRESHOLD = 0.85

# 工作进程数（默认 CPU 核心数）
NUM_PROCESSES = max(1, mp.cpu_count() - 1)

# 数据分割列表
SPLITS = ["train", "val", "test"]

# 支持的图片扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')


def get_split_from_path(file_path: str, dataset_dir: str) -> str:
    """从文件路径中提取数据分割名称（train/val/test）"""
    rel_path = os.path.relpath(file_path, dataset_dir)
    parts = rel_path.replace("\\", "/").split("/")
    for split in SPLITS:
        if split in parts:
            return split
    return "unknown"


def collect_all_images(dataset_dir: str) -> list:
    """收集数据集中所有图片文件（遍历所有分割目录）"""
    all_images = []
    images_dir = os.path.join(dataset_dir, "images")
    if not os.path.exists(images_dir):
        print(f"错误：找不到图片目录 {images_dir}")
        return all_images
    for split in SPLITS:
        split_dir = os.path.join(images_dir, split)
        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    all_images.append(os.path.join(split_dir, f))
    return all_images


def get_label_path(image_path: str, dataset_dir: str) -> str | None:
    """根据图片路径获取对应的标签文件路径"""
    rel_path = os.path.relpath(image_path, dataset_dir)
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "images":
        stem = os.path.splitext(parts[-1])[0]
        label_rel = os.path.join("labels", *parts[1:-1], f"{stem}.txt")
        label_path = os.path.join(dataset_dir, label_rel)
        return label_path if os.path.exists(label_path) else None
    return None


def generate_unique_filename(dest_dir: str, filename: str) -> str:
    """在目标目录中生成唯一的文件名"""
    if not os.path.exists(dest_dir):
        return os.path.join(dest_dir, filename)
    base, ext = os.path.splitext(filename)
    dest_path = os.path.join(dest_dir, filename)
    if not os.path.exists(dest_path):
        return dest_path
    return os.path.join(dest_dir, f"{base}_{str(uuid.uuid4())[:8]}{ext}")


def move_file_with_label(file_path: str, dest_images_dir: str, dest_labels_dir: str, dataset_dir: str) -> bool:
    """移动图片文件及其对应的标签文件"""
    if not os.path.exists(file_path):
        return False
    os.makedirs(dest_images_dir, exist_ok=True)
    if dest_labels_dir:
        os.makedirs(dest_labels_dir, exist_ok=True)
    try:
        filename = os.path.basename(file_path)
        dest_img = generate_unique_filename(dest_images_dir, filename)
        shutil.move(file_path, dest_img)
        if dest_labels_dir:
            label_path = get_label_path(file_path, dataset_dir)
            if label_path and os.path.exists(label_path):
                dest_lbl = generate_unique_filename(dest_labels_dir, os.path.basename(label_path))
                shutil.move(label_path, dest_lbl)
        return True
    except Exception as e:
        return False


# ==================== 多进程辅助函数 ====================

def _compute_md5(file_path: str) -> tuple:
    """计算单个文件的 MD5（多进程 worker）"""
    try:
        return file_path, dup.get_file_hash(file_path)
    except Exception:
        return file_path, None


def _compute_phash(file_path: str) -> tuple:
    """计算单个文件的 pHash（多进程 worker）"""
    try:
        return file_path, dup.get_phash(file_path)
    except Exception:
        return file_path, None


def _compute_phash_batch(file_paths: list, threshold: int) -> list:
    """
    批量计算 pHash 并两两比对相似度（多进程 worker）
    
    返回：[(path1, path2, 汉明距离, SSIM值), ...]
    """
    # 计算这批文件的 pHash
    phash_dict = {}
    for path in file_paths:
        phash = dup.get_phash(path)
        if phash is not None:
            phash_dict[path] = phash

    # 两两比对
    pairs = []
    checked = set()
    for p1, h1 in phash_dict.items():
        for p2, h2 in phash_dict.items():
            if p1 != p2 and p2 not in checked:
                hd = h1 - h2
                if hd <= threshold:
                    ssim_val = dup.ssim_check(p1, p2)
                    if ssim_val >= SSIM_THRESHOLD:
                        pairs.append((p1, p2, hd, ssim_val))
        checked.add(p1)
    return pairs


def deduplicate_yolo_dataset():
    """YOLO 数据集去重主函数（多进程加速）"""
    dataset_dir = DATASET_DIR
    output_dir = OUTPUT_DIR
    mode = DEDUP_MODE
    num_procs = NUM_PROCESSES

    print("=" * 60)
    print("YOLO 数据集去重工具（多进程加速）")
    print("=" * 60)
    print(f"数据集目录: {dataset_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"去重模式:   {'完全重复+视觉相似' if mode == 'both' else '仅完全重复'}")
    print(f"工作进程数: {num_procs}")
    print()

    if not os.path.exists(dataset_dir):
        print(f"错误：数据集目录不存在: {dataset_dir}")
        return

    # 收集所有图片
    print("正在收集图片文件...")
    all_images = collect_all_images(dataset_dir)
    print(f"共找到 {len(all_images)} 张图片")
    print()

    # ===== 阶段1：完全重复检测（多进程）=====
    print("=" * 60)
    print("阶段1：MD5 完全重复检测（多进程）")
    print("-" * 60)
    t0 = time.time()

    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(_compute_md5, all_images)

    md5_groups = defaultdict(list)
    for path, h in results:
        if h is not None:
            md5_groups[h].append(path)

    exact_dup_groups = [files for files in md5_groups.values() if len(files) > 1]
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  找到 {len(exact_dup_groups)} 组完全重复")
    print()

    # 移动完全重复文件
    exact_files_to_move = []
    for group in exact_dup_groups:
        for file_path in group[1:]:
            exact_files_to_move.append((file_path, get_split_from_path(file_path, dataset_dir)))

    if exact_files_to_move:
        print("移动完全重复文件（含对应标签）...")
        exact_images_dir = os.path.join(output_dir, "exact_duplicates", "images")
        exact_labels_dir = os.path.join(output_dir, "exact_duplicates", "labels")
        for file_path, split in exact_files_to_move:
            move_file_with_label(file_path,
                                 os.path.join(exact_images_dir, split),
                                 os.path.join(exact_labels_dir, split),
                                 dataset_dir)
        print(f"  完成: {len(exact_files_to_move)} 个文件")
    else:
        print("未发现完全重复文件。")
    print()

    # ===== 阶段2：视觉相似检测（多进程）=====
    similar_files_to_move = []

    if mode == "both":
        print("=" * 60)
        print("阶段2：pHash + SSIM 视觉相似检测（多进程）")
        print("-" * 60)
        t0 = time.time()

        # 已移动的完全重复文件（不入相似检测）
        moved_exact_set = set(f for f, _ in exact_files_to_move)
        remaining = [f for f in all_images if f not in moved_exact_set]
        print(f"  待检测图片: {len(remaining)} 张")
        print(f"  计算 pHash ({num_procs} 进程)...")

        # 多进程计算 pHash
        with mp.Pool(processes=num_procs) as pool:
            phash_results = pool.map(_compute_phash, remaining)

        phash_dict = {p: h for p, h in phash_results if h is not None}
        phash_elapsed = time.time() - t0
        print(f"  pHash 计算完成: {len(phash_dict)} 个, 耗时: {phash_elapsed:.2f} 秒")
        print(f"  两两比对相似度 ({num_procs} 进程)...")

        # 将文件分块，多进程并行比对
        chunk_size = max(1, len(remaining) // num_procs)
        chunks = [remaining[i:i + chunk_size] for i in range(0, len(remaining), chunk_size)]

        with mp.Pool(processes=num_procs) as pool:
            worker = partial(_compute_phash_batch, threshold=SIMILARITY_THRESHOLD)
            batch_results = pool.map(worker, chunks)

        similar_pairs = []
        for r in batch_results:
            similar_pairs.extend(r)

        compare_elapsed = time.time() - t0 - phash_elapsed
        total_elapsed = time.time() - t0
        print(f"  比对完成: {len(similar_pairs)} 对, 耗时: {total_elapsed:.2f} 秒")
        print()

        if similar_pairs:
            print("移动视觉相似文件（含对应标签）...")
            seen = set()
            for pair in similar_pairs:
                file_to_move = pair[1]
                if file_to_move not in seen and file_to_move not in moved_exact_set:
                    seen.add(file_to_move)
                    split = get_split_from_path(file_to_move, dataset_dir)
                    similar_files_to_move.append((file_to_move, split))

            similar_images_dir = os.path.join(output_dir, "similar_duplicates", "images")
            similar_labels_dir = os.path.join(output_dir, "similar_duplicates", "labels")
            for file_path, split in similar_files_to_move:
                move_file_with_label(file_path,
                                     os.path.join(similar_images_dir, split),
                                     os.path.join(similar_labels_dir, split),
                                     dataset_dir)
            print(f"  完成: {len(similar_files_to_move)} 个文件")
        else:
            print("未发现视觉相似文件。")

    # ===== 输出统计 =====
    print()
    print("=" * 60)
    print("去重完成！")
    print("-" * 60)
    print(f"完全重复文件: {len(exact_files_to_move)} 个")
    if mode == "both":
        print(f"视觉相似文件: {len(similar_files_to_move)} 个")
    total = len(exact_files_to_move) + len(similar_files_to_move)
    print(f"合计移动: {total} 个文件（含对应标签）")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()  # Windows 多进程支持
    deduplicate_yolo_dataset()
