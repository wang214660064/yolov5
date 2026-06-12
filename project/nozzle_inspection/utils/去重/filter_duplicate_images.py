"""
图像重复检测与过滤工具
=============================

功能说明：
    本脚本用于自动检测和整理数据集中的重复图像文件。
    采用双重检测机制：
    1. MD5哈希检测：识别完全相同的文件
    2. pHash + SSIM检测：识别视觉上相似的图片

使用方法：
    修改 main() 函数中的 target_dir 为你的图片目录路径
    设置 mode 参数：
        - 'hash_only': 仅检测完全重复
        - 'both': 同时检测完全重复和视觉相似

依赖库：
    pip install Pillow imagehash opencv-python scikit-image numpy
"""

# ==================== 导入必要的库 ====================
import os              # 文件路径和目录操作
import hashlib         # MD5哈希计算
import shutil          # 文件移动操作
from PIL import Image  # 图像读取（Pillow库）
import imagehash      # 感知哈希算法
import cv2             # OpenCV图像处理
import numpy as np     # 数值计算
from skimage.metrics import structural_similarity as ssim  # 结构相似性算法
from collections import defaultdict  # 字典工具，用于分组


# ==================== 辅助函数 ====================
def imread_chinese(file_path):
    """
    使用OpenCV读取包含中文路径的图像
    
    参数：
        file_path (str): 图像文件路径（支持中文）
    
    返回：
        numpy.ndarray 或 None: 读取的图像，失败返回None
    """
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return None


import multiprocessing as mp        # 多进程并行处理
from functools import partial        # 函数工具
import time                         # 计时工具
import uuid                         # UUID生成，用于唯一文件名


# ==================== 配置参数 ====================
# 这些参数可以根据实际需求进行调整

# pHash感知哈希的汉明距离阈值
# 范围：0-64，值越小表示要求越严格
# 推荐：
#   - 3-4: 非常严格，几乎只匹配完全相同的图片
#   - 5-6: 中等严格，推荐默认值
#   - 7-10: 较宽松，可以匹配经过轻微处理的图片
SIMILARITY_THRESHOLD = 4

# SSIM结构相似性阈值
# 范围：0-1，值越大表示要求越严格
# 推荐：
#   - 0.7-0.9: 高相似度要求
#   - 0.5-0.7: 中等相似度要求
#   - 0.3-0.5: 宽松匹配，可能包含不太相似的图片
SSIM_THRESHOLD = 0.85

# 输出目录名称
EXACT_DUP_DIR = "exact_duplicates"    # 完全重复文件的存放目录
SIMILAR_DUP_DIR = "similar_duplicates"  # 视觉相似文件的存放目录

# 多进程配置
USE_MULTIPROCESSING = True   # 是否启用多进程加速（大量文件时建议开启）
NUM_PROCESSES = 4             # 使用的进程数量（建议设置为CPU核心数）


# ==================== 核心函数定义 ====================

def get_file_hash(file_path):
    """
    计算文件的MD5哈希值

    原理：
        MD5是一种常用的哈希算法，它会对文件内容进行数学运算，
        生成一个128位的哈希值。只要文件内容完全相同，MD5值就相同。

    参数：
        file_path (str): 文件的完整路径

    返回：
        str: 32位的十六进制哈希字符串

    示例：
        >>> hash1 = get_file_hash("image1.jpg")
        >>> hash2 = get_file_hash("image2.jpg")
        >>> if hash1 == hash2:
        ...     print("两个文件完全相同")
    """
    hasher = hashlib.md5()  # 创建MD5哈希对象
    with open(file_path, "rb") as f:  # 以二进制读取模式打开文件
        # 分块读取文件内容（每块4KB），避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)  # 更新哈希值
    return hasher.hexdigest()  # 返回十六进制哈希字符串


def get_phash(file_path):
    """
    生成图像的感知哈希（Perceptual Hash）

    原理：
        pHash通过以下步骤生成图像的"指纹"：
        1. 将图像缩放到极小尺寸（如8x8）
        2. 转换为灰度图
        3. 计算DCT（离散余弦变换）
        4. 取低频部分计算均值
        5. 根据像素是否大于均值生成64位哈希

    特点：
        - 对图像缩放、轻微颜色变化不敏感
        - 相似图像会产生相似的哈希值
        - 计算速度快，适合大规模筛选

    参数：
        file_path (str): 图像文件的完整路径

    返回：
        imagehash.ImageHash 或 None: 感知哈希对象，失败时返回None
    """
    try:
        img = Image.open(file_path)  # 使用Pillow打开图像
        # imagehash.phash() 自动完成缩放、灰度化、DCT等步骤
        return imagehash.phash(img)
    except Exception as e:
        print(f"处理图像时出错 {file_path}: {e}")
        return None


def ssim_check(img1_path, img2_path):
    """
    计算两张图像的结构相似性指数（SSIM）

    原理：
        SSIM是一种衡量两张图像相似度的算法，考虑了：
        - 亮度相似性（luminance）
        - 对比度相似性（contrast）
        - 结构相似性（structure）

    参数：
        img1_path (str): 第一张图像的路径
        img2_path (str): 第二张图像的路径

    返回：
        float: SSIM值，范围0-1，值越大表示越相似

    处理流程：
        1. 读取两张图像
        2. 缩放到128x128加快计算速度
        3. 转换为灰度图
        4. 计算SSIM值
    """
    try:
        # 使用OpenCV读取图像
        img1 = imread_chinese(img1_path)
        img2 = imread_chinese(img2_path)

        # 检查图像是否成功读取
        if img1 is None or img2 is None:
            return 0

        # 缩放图像到128x128，减少计算量
        img1 = cv2.resize(img1, (128, 128))
        img2 = cv2.resize(img2, (128, 128))

        # 转换为灰度图（SSIM通常在灰度图上计算）
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算结构相似性
        # 返回值范围0-1，1表示完全相同
        ssim_value = ssim(gray1, gray2)
        return ssim_value
    except Exception as e:
        print(f"SSIM计算错误: {e}")
        return 0


def generate_unique_path(dest_dir, filename):
    """
    为目标文件生成唯一的文件路径，避免覆盖已有文件

    场景：
        当多个同名文件需要移动到同一目录时，直接移动会覆盖原文件。
        这个函数通过添加UUID或时间戳后缀来生成唯一文件名。

    参数：
        dest_dir (str): 目标目录路径
        filename (str): 原始文件名

    返回：
        str: 唯一的完整文件路径

    示例：
        >>> path = generate_unique_path("/output", "image.jpg")
        >>> # 如果 image.jpg 已存在，可能返回 "/output/image_a1b2c3d4.jpg"
    """
    # 如果目标目录不存在，直接返回原路径
    if not os.path.exists(dest_dir):
        return os.path.join(dest_dir, filename)

    # 分离文件名和扩展名
    base, ext = os.path.splitext(filename)

    # 检查原始文件名是否可用
    original_path = os.path.join(dest_dir, filename)
    if not os.path.exists(original_path):
        return original_path

    # 使用UUID生成唯一标识符（取前8位，足够区分且不会太长）
    unique_id = str(uuid.uuid4())[:8]
    new_filename = f"{base}_{unique_id}{ext}"
    new_path = os.path.join(dest_dir, new_filename)

    # 极小概率下UUID也可能冲突，再加一层保险
    if os.path.exists(new_path):
        import time
        import random
        # 使用高精度时间戳+随机数
        timestamp = int(time.time() * 1000)
        random_num = random.randint(0, 9999)
        new_filename = f"{base}_{timestamp}_{random_num}{ext}"
        new_path = os.path.join(dest_dir, new_filename)

    return new_path


def move_files(files, dest_dir, exclude_files=None):
    """
    将文件列表移动到目标目录

    参数：
        files (list): 需要移动的文件路径列表
        dest_dir (str): 目标目录路径
        exclude_files (set): 需要排除的文件路径集合

    返回：
        list: 成功移动的文件路径列表

    功能：
        1. 跳过已排除或已不存在的文件
        2. 为同名文件生成唯一名称
        3. 打印移动日志
    """
    exclude_files = exclude_files or set()  # 默认空集合
    moved = []  # 记录成功移动的文件

    for file in files:
        # 跳过排除列表中的文件
        if file in exclude_files or not os.path.exists(file):
            continue

        filename = os.path.basename(file)  # 获取文件名
        dest_path = generate_unique_path(dest_dir, filename)  # 生成唯一目标路径

        try:
            shutil.move(file, dest_path)  # 执行移动操作
            moved.append(file)
            print(f"已移动: {file} -> {dest_path}")
        except Exception as e:
            print(f"移动失败 {file}: {str(e)}")

    return moved


def find_exact_duplicates(directory):
    """
    查找完全相同的文件（基于MD5哈希）

    原理：
        1. 遍历目录中所有图像文件
        2. 计算每个文件的MD5哈希值
        3. 将相同哈希值的文件分组
        4. 返回包含2个及以上文件的组（即有重复的文件组）

    参数：
        directory (str): 要搜索的根目录路径

    返回：
        list: 重复文件组列表，每组是一个文件路径列表
              例如：[['a.jpg', 'a_copy.jpg'], ['b.png', 'b_duplicate.png']]
    """
    md5_dict = defaultdict(list)  # 哈希值 -> 文件列表 的映射

    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')

    # 递归遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否为图像格式
            if file.lower().endswith(image_extensions):
                path = os.path.join(root, file)  # 构建完整路径
                file_hash = get_file_hash(path)  # 计算MD5哈希
                md5_dict[file_hash].append(path)  # 加入对应哈希的分组

    # 筛选出有重复的组（文件数量 > 1）
    # 只返回重复的文件组，单独的Unique文件会被过滤掉
    return [files for files in md5_dict.values() if len(files) > 1]


def find_similar_duplicates_worker(file_paths, threshold=SIMILARITY_THRESHOLD):
    """
    工作进程函数：在一组文件中查找视觉相似的图片

    这个函数会在多进程中被调用，处理一批图像文件。

    参数：
        file_paths (list): 要检查的图像文件路径列表
        threshold (int): pHash汉明距离阈值

    返回：
        list: 相似文件对列表，每项是 (文件1, 文件2, 汉明距离, SSIM值)

    检测流程：
        1. 为所有文件计算pHash值
        2. 两两比较，计算汉明距离
        3. 对汉明距离小于阈值的候选对进行SSIM验证
        4. 返回同时满足两个条件的相似对
    """
    similar_pairs = []  # 存储找到的相似对

    # 第一步：计算所有图像的pHash
    phash_dict = {}
    for path in file_paths:
        phash = get_phash(path)
        if phash is not None:
            phash_dict[path] = phash

    # 第二步：两两比较，计算汉明距离
    checked = set()  # 记录已检查过的文件，避免重复检查
    for path1, hash1 in phash_dict.items():
        for path2, hash2 in phash_dict.items():
            # 跳过自己和已检查过的配对
            if path1 != path2 and path2 not in checked:
                # 计算汉明距离（两哈希值不同的位数）
                hamming_distance = hash1 - hash2

                # 汉明距离小于阈值，进入SSIM二次验证
                if hamming_distance <= threshold:
                    ssim_value = ssim_check(path1, path2)
                    # SSIM也超过阈值，确认是相似图片
                    if ssim_value >= SSIM_THRESHOLD:
                        similar_pairs.append((path1, path2, hamming_distance, ssim_value))

        checked.add(path1)  # 标记为已检查

    return similar_pairs


def find_similar_duplicates(directory):
    """
    查找视觉上相似的图片（pHash + SSIM双重验证）

    参数：
        directory (str): 要搜索的根目录路径

    返回：
        list: 相似文件对列表

    处理策略：
        - 文件数量 <= 100: 单进程处理
        - 文件数量 > 100: 启用多进程加速（分成4批并行处理）
    """
    # 收集目录中所有图像文件
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                path = os.path.join(root, file)
                image_files.append(path)

    # 根据文件数量决定处理方式
    if USE_MULTIPROCESSING and len(image_files) > 100:
        # 多进程模式：分割任务并行处理
        print(f"使用多进程处理 {len(image_files)} 个文件...")
        chunk_size = max(1, len(image_files) // NUM_PROCESSES)
        # 将文件列表分成多个块
        chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

        # 创建进程池并执行
        with mp.Pool(processes=NUM_PROCESSES) as pool:
            results = pool.map(find_similar_duplicates_worker, chunks) # todo 多进程处理会导致块之间的文件无法被检测到相似度

        # 合并各进程的结果
        similar_pairs = []
        for result in results:
            similar_pairs.extend(result)

        return similar_pairs
    else:
        # 单进程模式
        return find_similar_duplicates_worker(image_files)


def find_and_move_duplicates(target_dir, mode='hash_only'):
    """
    查找并移动重复文件的主函数

    参数：
        target_dir (str): 目标目录路径
        mode (str): 去重模式
            - 'hash_only': 仅检测和移动完全重复文件
            - 'both': 同时检测和移动完全重复和视觉相似文件

    返回：
        tuple: (移动的完全重复文件数, 移动的视觉相似文件数)

    工作流程：
        1. 创建输出目录
        2. 查找并移动完全重复文件（每组保留第一个）
        3. （可选）查找并移动视觉相似文件
        4. 返回统计信息
    """
    # 构建输出目录路径
    exact_dest = os.path.join(target_dir, EXACT_DUP_DIR)
    similar_dest = os.path.join(target_dir, SIMILAR_DUP_DIR)

    # 创建完全重复文件的输出目录（总是创建）
    os.makedirs(exact_dest, exist_ok=True)

    # 仅在需要时创建视觉相似文件输出目录
    if mode == 'both':
        os.makedirs(similar_dest, exist_ok=True)

    # ===== 阶段1：完全重复检测 =====
    print("查找完全重复文件...")
    start_time = time.time()
    exact_dups = find_exact_duplicates(target_dir)
    exact_time = time.time() - start_time
    print(f"找到 {len(exact_dups)} 组完全重复文件，耗时: {exact_time:.2f}秒")

    # 移动完全重复文件
    # 策略：每组保留第一个文件，移动其余的
    exact_files_to_move = []
    for group in exact_dups:
        exact_files_to_move.extend(group[1:])  # 保留group[0]，移动group[1:]

    moved_exact = move_files(exact_files_to_move, exact_dest)

    # ===== 阶段2：视觉相似检测（仅在mode='both'时执行）=====
    moved_similar = []

    if mode == 'both':
        print("查找视觉相似文件...")
        start_time = time.time()
        similar_dups = find_similar_duplicates(target_dir)
        similar_time = time.time() - start_time
        print(f"找到 {len(similar_dups)} 对视觉相似文件，耗时: {similar_time:.2f}秒")

        # 收集所有相似文件
        similar_files_to_move = set()
        for pair in similar_dups:
            similar_files_to_move.add(pair[0])

        # 排除已经移动的完全重复文件（避免二次处理）
        similar_files_to_move = similar_files_to_move - set(moved_exact)

        # 移动视觉相似文件
        moved_similar = move_files(list(similar_files_to_move), similar_dest)

    return moved_exact, moved_similar


# ==================== 程序入口 ====================
if __name__ == "__main__":
    """
    主程序入口

    使用说明：
        1. 修改 target_dir 为你要处理的数据目录
        2. 设置 mode 参数：
            - 'hash_only': 只去除完全相同的重复文件
            - 'both': 同时去除完全相同和视觉相似的文件
        3. 运行脚本，重复文件会被移动到相应子目录
    """
    # ===== 在这里修改目标目录 =====
    target_dir = r"Data\data"

    # ===== 在这里修改去重模式 =====
    # 'hash_only': 仅去除完全重复的文件（MD5哈希相同）
    # 'both': 同时去除完全重复和视觉相似的文件（推荐）
    moved_exact, moved_similar = find_and_move_duplicates(target_dir, mode='both')

    # ===== 输出统计结果 =====
    print(f"\n处理完成！")
    print(f"- 完全重复文件：{len(moved_exact)} 个")
    print(f"- 视觉相似文件：{len(moved_similar)} 个")
    print(f"\n文件已移动到以下目录：")
    print(f"- {os.path.join(target_dir, 'processed', EXACT_DUP_DIR)}")
    print(f"- {os.path.join(target_dir, 'processed', SIMILAR_DUP_DIR)}")
