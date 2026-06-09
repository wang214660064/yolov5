from pathlib import Path


YoloBox = tuple[int, float, float, float, float]


def scale_yolo_box_on_canvas(
    box: YoloBox,
    scale: float,
    offset_x: float,
    offset_y: float,
    min_size: float = 1e-4,
) -> YoloBox | None:
    """在归一化画布上同步缩放和偏移 YOLO 框。"""

    if scale <= 0:
        raise ValueError("缩放比例必须大于 0")

    cls, x, y, w, h = box
    x1 = (x - w / 2.0) * scale + offset_x
    y1 = (y - h / 2.0) * scale + offset_y
    x2 = (x + w / 2.0) * scale + offset_x
    y2 = (y + h / 2.0) * scale + offset_y

    x1 = _clip01(x1)
    y1 = _clip01(y1)
    x2 = _clip01(x2)
    y2 = _clip01(y2)

    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= min_size or new_h <= min_size:
        return None

    return (
        int(cls),
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        new_w,
        new_h,
    )


def parse_yolo_box(line: str) -> YoloBox:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"YOLO 标签行格式非法：{line}")
    cls, x, y, w, h = parts
    return int(float(cls)), float(x), float(y), float(w), float(h)


def format_yolo_box(box: YoloBox) -> str:
    cls, x, y, w, h = box
    return f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"


def scale_label_file(input_path: Path, output_path: Path, scale: float, offset_x: float, offset_y: float) -> int:
    converted: list[str] = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scaled = scale_yolo_box_on_canvas(parse_yolo_box(line), scale, offset_x, offset_y)
        if scaled is not None:
            converted.append(format_yolo_box(scaled))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(converted), encoding="utf-8")
    return len(converted)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
