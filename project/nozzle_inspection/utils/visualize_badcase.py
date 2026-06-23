"""
BadCase 可视化工具 - 将 GT 框和预测框画在同一张图上。

用法：
    conda run -n yolov5 python project/nozzle_inspection/utils/visualize_badcase.py

默认可视化 runs/train/BadCase/ 下所有的 false_alarm / missed_target / class_error，
也可通过 --error-dir 指定单个错误目录。
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# YOLO 格式：class_id x_center y_center width height (均归一化)
# 预测格式多一列 confidence

CATEGORIES = {0: "NG", 1: "OK"}
COLOR_GT = (0, 255, 0)       # GT 框：绿色
COLOR_PRED = (0, 0, 255)     # 预测框：红色


def _find_config(start: Path) -> Path | None:
    """向上查找 badcase_config.yaml"""
    for parent in [start] + list(start.parents):
        candidate = parent / "badcase_config.yaml"
        if candidate.exists():
            return candidate
    return None


def _load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_yolo_box(line: str, has_conf: bool) -> dict | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if has_conf and len(parts) >= 6 else None
    return {"cls": cls, "x": x, "y": y, "w": w, "h": h, "conf": conf}


def _yolo_to_pixel(x, y, w, h, img_w, img_h):
    """YOLO 归一化坐标 → 像素坐标 (x1, y1, x2, y2)"""
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2


def visualize(error_dir: Path, output_dir: Path, config: dict):
    """对 error_dir 中的每一对 label+prediction 画图。"""
    image_dir = Path(config["image_dir"])
    label_dir = error_dir / "labels"
    pred_dir = error_dir / "predictions"

    output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(label_dir.glob("*.txt"))
    if not txt_files:
        print(f"  在 {label_dir} 中未找到标签文件")
        return

    for gt_path in txt_files:
        stem = gt_path.stem
        pred_path = pred_dir / f"{stem}.txt"
        if not pred_path.exists():
            continue

        # 找图片
        img_candidates = list(image_dir.glob(f"{stem}.*"))
        img_candidates = [p for p in img_candidates if p.suffix.lower() in
                          (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")]
        if not img_candidates:
            print(f"  ⚠ 未找到图片: {stem}")
            continue
        img_path = img_candidates[0]

        # 读取图片
        img = cv2.imdecode(np.frombuffer(img_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ⚠ 无法读取图片: {img_path}")
            continue
        img_h, img_w = img.shape[:2]
        canvas = img.copy()

        # 读取 GT 框
        gt_boxes = []
        for line in gt_path.read_text(encoding="utf-8").strip().splitlines():
            box = _parse_yolo_box(line, has_conf=False)
            if box:
                gt_boxes.append(box)
                x1, y1, x2, y2 = _yolo_to_pixel(box["x"], box["y"], box["w"], box["h"], img_w, img_h)
                label = CATEGORIES.get(box["cls"], str(box["cls"]))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_GT, 2)
                cv2.putText(canvas, f"GT:{label}", (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GT, 2)

        # 读取预测框
        pred_boxes = []
        for line in pred_path.read_text(encoding="utf-8").strip().splitlines():
            box = _parse_yolo_box(line, has_conf=True)
            if box:
                pred_boxes.append(box)
                x1, y1, x2, y2 = _yolo_to_pixel(box["x"], box["y"], box["w"], box["h"], img_w, img_h)
                label = CATEGORIES.get(box["cls"], str(box["cls"]))
                conf_str = f"{box['conf']:.2f}" if box["conf"] is not None else ""
                cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_PRED, 2)
                text = f"Pred:{label} {conf_str}"
                cv2.putText(canvas, text, (x1, max(y2 + 15, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRED, 2)

        # 添加图例（使用英文避免 OpenCV 中文显示问题）
        cv2.putText(canvas, "GT (Green)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)
        cv2.putText(canvas, "Pred (Red)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)
        cv2.putText(canvas, f"GT={len(gt_boxes)}, Pred={len(pred_boxes)}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 保存
        out_path = output_dir / f"{stem}.jpg"
        cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(out_path)


def visualize_badcase(badcase_root: Path) -> int:
    """对指定目录下的错误样本进行可视化。
    
    Args:
        badcase_root: BadCase 根目录，包含 false_alarm/missed_target/class_error 子目录
    
    Returns:
        0 表示成功，1 表示未找到错误目录
    """
    # 扫描该目录下的三个错误类型文件夹
    error_dirs = []
    for error_type in ["false_alarm", "missed_target", "class_error"]:
        error_dir = badcase_root / error_type
        if error_dir.exists():
            error_dirs.append(error_dir)
    
    if not error_dirs:
        print(f"在 {badcase_root} 下未找到 false_alarm/missed_target/class_error 目录")
        return 1

    for error_dir in error_dirs:
        print(f"\n{'='*60}")
        print(f"处理: {error_dir}")
        config_path = _find_config(error_dir)
        if not config_path:
            print(f"  ⚠ 未找到 badcase_config.yaml，跳过")
            continue
        config = _load_config(config_path)

        output_dir = error_dir / "visualizations"
        visualize(error_dir.resolve(), output_dir.resolve(), config)

    print(f"\nBadCase 可视化完成！")
    return 0


def main():
    # 直接指定要处理的根目录
    badcase_root = Path(r"E:\Desktop\MAC-WIN\04 OpenCV\98_Practice\3DPrinterNozzleInspection\yolov8\runs\val\BadCase\nozzle_ng_ok_v8-aug-continue\exp-conf-0.25")
    raise SystemExit(visualize_badcase(badcase_root))


if __name__ == "__main__":
    raise SystemExit(main())
