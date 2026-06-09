import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256，用于精确去重。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> int:
    """计算简单感知 HASH；没有图像依赖时由调用方决定是否使用。"""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 Pillow，无法执行视觉相似去重") from exc

    with Image.open(path) as image:
        image = image.convert("L").resize((hash_size, hash_size))
        pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()
