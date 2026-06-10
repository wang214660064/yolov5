"""
图像哈希模块 - 文件哈希计算

该模块提供两种哈希计算方法：
1. SHA256 哈希：用于精确去重（文件内容完全相同时哈希值相同）
2. 平均哈希（Average Hash）：用于视觉相似性检测

核心功能：
- sha256_file(): 计算文件的 SHA256 哈希值
- average_hash(): 计算图像的感知哈希值
- hamming_distance(): 计算两个哈希值的汉明距离

使用示例：
    # 精确去重
    hash1 = sha256_file(Path("image1.jpg"))
    hash2 = sha256_file(Path("image2.jpg"))
    if hash1 == hash2:
        print("文件完全相同")
    
    # 视觉相似检测
    hash1 = average_hash(Path("image1.jpg"))
    hash2 = average_hash(Path("image2.jpg"))
    distance = hamming_distance(hash1, hash2)
    if distance < 5:
        print("图像相似")
"""

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    计算文件的 SHA256 哈希值
    
    参数：
        path: 文件路径
        chunk_size: 读取块大小，默认 1MB（1024*1024 字节）
    
    返回值：
        str: 64 位十六进制 SHA256 哈希字符串
    
    算法说明：
        1. 创建 SHA256 哈希对象
        2. 分块读取文件内容
        3. 更新哈希对象
        4. 返回十六进制摘要
    
    使用场景：
        - 精确去重：只有文件内容完全相同时，哈希值才相同
        - 文件完整性验证
    
    示例：
        sha256_file(Path("data/image.jpg")) -> "abc123..." (64个字符)
    """
    # 创建 SHA256 哈希对象
    digest = hashlib.sha256()
    
    # 以二进制模式打开文件
    with Path(path).open("rb") as file:
        # 分块读取文件，避免一次性加载大文件到内存
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    
    # 返回十六进制格式的哈希值
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> int:
    """
    计算图像的平均哈希（Average Hash）
    
    参数：
        path: 图像文件路径
        hash_size: 哈希尺寸，默认 8x8
    
    返回值：
        int: 64 位整数哈希值（8x8=64位）
    
    算法说明：
        1. 将图像转换为灰度图
        2. 缩放到 hash_size x hash_size 大小
        3. 计算所有像素的平均值
        4. 将每个像素与平均值比较，大于等于为1，小于为0
        5. 将二进制序列转换为整数
    
    使用场景：
        - 视觉相似性检测
        - 模糊去重（内容相似但不完全相同的图像）
    
    注意：
        需要安装 Pillow 库才能使用此功能
    
    示例：
        average_hash(Path("image.jpg")) -> 1234567890123456 (64位整数)
    """
    # 尝试导入 PIL 库
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 Pillow，无法执行视觉相似去重") from exc

    # 打开图像并处理
    with Image.open(path) as image:
        # 转换为灰度图
        image = image.convert("L")
        # 缩放到指定大小
        image = image.resize((hash_size, hash_size))
        # 获取所有像素值
        pixels = list(image.getdata())
    
    # 计算平均像素值
    avg = sum(pixels) / len(pixels)
    
    # 构建哈希值
    bits = 0
    for pixel in pixels:
        # 左移一位，然后根据像素值是否大于等于平均值决定最低位
        bits = (bits << 1) | int(pixel >= avg)
    
    return bits


def hamming_distance(left: int, right: int) -> int:
    """
    计算两个整数的汉明距离
    
    参数：
        left: 第一个整数
        right: 第二个整数
    
    返回值：
        int: 汉明距离（两个数二进制表示中不同位的数量）
    
    算法说明：
        1. 对两个数进行异或运算（相同位为0，不同位为1）
        2. 统计结果中1的个数
    
    使用场景：
        - 比较两个平均哈希值的相似程度
        - 汉明距离越小，图像越相似
    
    示例：
        hamming_distance(0b1010, 0b1001) -> 2（第2位和第4位不同）
    """
    # 异或运算得到不同位，然后统计1的个数
    return (left ^ right).bit_count()
