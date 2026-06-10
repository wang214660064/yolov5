"""
标签缩放模块 - YOLO 边界框变换

该模块提供 YOLO 边界框的缩放和偏移功能，用于数据增强场景。

当对图片进行裁剪、缩放等变换时，需要同步调整边界框坐标。
该模块实现了在归一化画布上的边界框变换。

核心功能：
1. 解析 YOLO 格式标签
2. 缩放和偏移边界框
3. 裁剪边界框到有效范围
4. 格式化输出 YOLO 标签

使用示例：
    # 将边界框缩小到原来的 0.8 倍，并向右偏移 0.1
    scaled_box = scale_yolo_box_on_canvas((0, 0.5, 0.5, 0.2, 0.2), 0.8, 0.1, 0)
"""

from pathlib import Path

# YOLO 边界框类型定义
# 格式：(类别ID, x中心, y中心, 宽度, 高度)，所有坐标归一化到 0-1 范围
YoloBox = tuple[int, float, float, float, float]


def scale_yolo_box_on_canvas(
    box: YoloBox,
    scale: float,
    offset_x: float,
    offset_y: float,
    min_size: float = 1e-4,
) -> YoloBox | None:
    """
    在归一化画布上同步缩放和偏移 YOLO 边界框
    
    参数：
        box: 原始边界框，格式 (cls, x, y, w, h)
        scale: 缩放比例（>0）
        offset_x: x 方向偏移量（归一化坐标）
        offset_y: y 方向偏移量（归一化坐标）
        min_size: 最小尺寸阈值，小于此值的边界框将被丢弃
    
    返回值：
        YoloBox | None: 变换后的边界框，如果太小则返回 None
    
    算法说明：
        1. 将中心点坐标转换为左上角和右下角坐标
        2. 应用缩放和偏移变换
        3. 将坐标裁剪到 0-1 范围
        4. 计算新的边界框参数
        5. 如果新边界框太小，返回 None
    
    示例：
        原始框: (0, 0.5, 0.5, 0.2, 0.2)  # 类别0，中心在(0.5,0.5)，宽高0.2
        缩放0.8，x偏移0.1:
        结果: (0, 0.52, 0.5, 0.16, 0.16)
    """
    # 验证缩放比例必须大于 0
    if scale <= 0:
        raise ValueError("缩放比例必须大于 0")

    # 解析边界框参数
    cls, x, y, w, h = box
    
    # 将中心点坐标转换为左上角和右下角坐标
    # x1 = x - w/2, x2 = x + w/2
    x1 = (x - w / 2.0) * scale + offset_x
    y1 = (y - h / 2.0) * scale + offset_y
    x2 = (x + w / 2.0) * scale + offset_x
    y2 = (y + h / 2.0) * scale + offset_y

    # 将坐标裁剪到 0-1 范围（防止越界）
    x1 = _clip01(x1)
    y1 = _clip01(y1)
    x2 = _clip01(x2)
    y2 = _clip01(y2)

    # 计算新的宽度和高度
    new_w = x2 - x1
    new_h = y2 - y1
    
    # 如果新边界框太小，返回 None（表示丢弃此边界框）
    if new_w <= min_size or new_h <= min_size:
        return None

    # 返回变换后的边界框
    return (
        int(cls),                    # 类别ID保持不变
        (x1 + x2) / 2.0,            # 新的 x 中心
        (y1 + y2) / 2.0,            # 新的 y 中心
        new_w,                      # 新的宽度
        new_h,                      # 新的高度
    )


def parse_yolo_box(line: str) -> YoloBox:
    """
    解析 YOLO 标签行
    
    参数：
        line: 标签行字符串，格式 "class_id x_center y_center width height"
    
    返回值：
        YoloBox: 解析后的边界框
    
    异常情况：
        - 标签行必须有且仅有 5 个元素
        - 数值必须能够转换为浮点数
    """
    parts = line.strip().split()
    
    # 检查格式是否正确
    if len(parts) != 5:
        raise ValueError(f"YOLO 标签行格式非法：{line}")
    
    # 解析并转换为对应类型
    cls, x, y, w, h = parts
    return int(float(cls)), float(x), float(y), float(w), float(h)


def format_yolo_box(box: YoloBox) -> str:
    """
    将边界框格式化为 YOLO 标签行字符串
    
    参数：
        box: YOLO 边界框
    
    返回值：
        str: 格式化后的标签行（带换行符）
    
    格式："class_id x_center y_center width height\n"
    坐标保留 6 位小数
    """
    cls, x, y, w, h = box
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"


def scale_label_file(input_path: Path, output_path: Path, scale: float, offset_x: float, offset_y: float) -> int:
    """
    对整个标签文件进行缩放变换
    
    参数：
        input_path: 输入标签文件路径
        output_path: 输出标签文件路径
        scale: 缩放比例
        offset_x: x 方向偏移量
        offset_y: y 方向偏移量
    
    返回值：
        int: 转换后的边界框数量（丢弃的不计入）
    
    处理流程：
        1. 读取输入文件的每一行
        2. 跳过空行
        3. 解析边界框并进行变换
        4. 如果变换后的边界框有效，添加到结果
        5. 写入输出文件
    """
    converted: list[str] = []
    
    # 读取并处理每一行
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        
        # 解析并变换边界框
        scaled = scale_yolo_box_on_canvas(parse_yolo_box(line), scale, offset_x, offset_y)
        
        # 如果变换后的边界框有效，添加到结果
        if scaled is not None:
            converted.append(format_yolo_box(scaled))
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入输出文件
    output_path.write_text("".join(converted), encoding="utf-8")
    
    # 返回转换的边界框数量
    return len(converted)


def _clip01(value: float) -> float:
    """
    将值裁剪到 0-1 范围
    
    参数：
        value: 输入值
    
    返回值：
        float: 裁剪后的值（0 <= result <= 1）
    
    示例：
        _clip01(-0.1) -> 0.0
        _clip01(0.5) -> 0.5
        _clip01(1.2) -> 1.0
    """
    return min(1.0, max(0.0, value))
