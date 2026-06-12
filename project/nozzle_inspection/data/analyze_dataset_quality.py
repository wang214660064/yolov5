"""
YOLO 数据集质量分析脚本。

功能：
1. 统计 OK/NG 类别分布。
2. 分析目标框尺寸、宽高比和 anchor 聚类建议。
3. 分析图像亮度、对比度、清晰度、过曝/偏暗比例。
4. 导出代表性困难样本拼图和 CSV 明细。

示例：
    python project/nozzle_inspection/data/analyze_dataset_quality.py \
        --dataset_root ../out_cross_hash_clean \
        --output_dir ../out_cross_hash_clean/quality_analysis
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "NG", 1: "OK"}


@dataclass(frozen=True)
class ImageMetrics:
    split: str
    image: str
    file: str
    width: int
    height: int
    brightness: float
    contrast: float
    sharpness_laplacian_var: float
    overexposed_ratio: float
    dark_ratio: float
    saturation: float
    label_exists: bool
    object_count: int
    class_ids: str
    class_name: str
    max_bbox_area: float
    mean_bbox_area: float
    max_aspect_ratio: float
    mean_center_distance: float


def read_label_metrics(label_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    class_ids: list[int] = []
    bbox_areas: list[float] = []
    aspect_ratios: list[float] = []
    center_distances: list[float] = []

    if not label_path.exists():
        return class_ids, bbox_areas, aspect_ratios, center_distances

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:5])
        except ValueError:
            continue
        class_ids.append(cls)
        bbox_areas.append(width * height)
        aspect_ratios.append(width / height if height else 0.0)
        center_distances.append(math.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2))

    return class_ids, bbox_areas, aspect_ratios, center_distances


def collect_metrics(dataset_root: Path) -> list[ImageMetrics]:
    rows: list[ImageMetrics] = []
    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"

    for split in SPLITS:
        split_image_dir = image_root / split
        if not split_image_dir.exists():
            continue
        for image_path in sorted(split_image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            height, width = gray.shape[:2]

            label_path = label_root / split / f"{image_path.stem}.txt"
            class_ids, bbox_areas, aspect_ratios, center_distances = read_label_metrics(label_path)
            class_name = "unknown"
            if 0 in class_ids:
                class_name = "NG"
            elif 1 in class_ids:
                class_name = "OK"

            rows.append(
                ImageMetrics(
                    split=split,
                    image=str(image_path),
                    file=image_path.name,
                    width=width,
                    height=height,
                    brightness=float(gray.mean()),
                    contrast=float(gray.std()),
                    sharpness_laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    overexposed_ratio=float((gray >= 245).mean()),
                    dark_ratio=float((gray <= 25).mean()),
                    saturation=float(hsv[:, :, 1].mean()),
                    label_exists=label_path.exists(),
                    object_count=len(class_ids),
                    class_ids=";".join(map(str, class_ids)),
                    class_name=class_name,
                    max_bbox_area=max(bbox_areas) if bbox_areas else 0.0,
                    mean_bbox_area=float(np.mean(bbox_areas)) if bbox_areas else 0.0,
                    max_aspect_ratio=max(aspect_ratios) if aspect_ratios else 0.0,
                    mean_center_distance=float(np.mean(center_distances)) if center_distances else 0.0,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(rows: list[ImageMetrics], attr: str, q: float, positive_only: bool = False) -> float:
    values = [float(getattr(row, attr)) for row in rows]
    if positive_only:
        values = [value for value in values if value > 0]
    return float(np.quantile(np.array(values, dtype=float), q))


def summarize(rows: list[ImageMetrics], output_dir: Path) -> dict[str, float]:
    metric_dicts = [row.__dict__ for row in rows]
    write_csv(output_dir / "image_quality_metrics.csv", metric_dicts)

    thresholds = {
        "dark_p10_brightness": quantile(rows, "brightness", 0.10),
        "bright_p90_brightness": quantile(rows, "brightness", 0.90),
        "low_contrast_p10": quantile(rows, "contrast", 0.10),
        "low_sharpness_p10": quantile(rows, "sharpness_laplacian_var", 0.10),
        "overexposed_p90_ratio": quantile(rows, "overexposed_ratio", 0.90),
        "small_target_p10_area": quantile(rows, "max_bbox_area", 0.10, positive_only=True),
        "large_target_p90_area": quantile(rows, "max_bbox_area", 0.90, positive_only=True),
    }

    feature_counts: list[dict] = []
    for split in ("all", *SPLITS):
        group = rows if split == "all" else [row for row in rows if row.split == split]
        feature_counts.append(
            {
                "split": split,
                "count": len(group),
                "dark_p10_count": sum(row.brightness <= thresholds["dark_p10_brightness"] for row in group),
                "bright_p90_count": sum(row.brightness >= thresholds["bright_p90_brightness"] for row in group),
                "low_contrast_p10_count": sum(row.contrast <= thresholds["low_contrast_p10"] for row in group),
                "low_sharpness_p10_count": sum(
                    row.sharpness_laplacian_var <= thresholds["low_sharpness_p10"] for row in group
                ),
                "overexposed_p90_count": sum(
                    row.overexposed_ratio >= thresholds["overexposed_p90_ratio"] for row in group
                ),
                "small_target_p10_count": sum(
                    0 < row.max_bbox_area <= thresholds["small_target_p10_area"] for row in group
                ),
                "large_target_p90_count": sum(
                    row.max_bbox_area >= thresholds["large_target_p90_area"] for row in group
                ),
            }
        )
    write_csv(output_dir / "quality_feature_counts.csv", feature_counts)

    split_class_summary: list[dict] = []
    for split in SPLITS:
        for class_name in ("NG", "OK"):
            group = [row for row in rows if row.split == split and row.class_name == class_name]
            if not group:
                continue
            split_class_summary.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "count": len(group),
                    "brightness_mean": float(np.mean([row.brightness for row in group])),
                    "contrast_mean": float(np.mean([row.contrast for row in group])),
                    "sharpness_mean": float(np.mean([row.sharpness_laplacian_var for row in group])),
                    "overexposed_ratio_mean": float(np.mean([row.overexposed_ratio for row in group])),
                    "bbox_area_mean": float(np.mean([row.max_bbox_area for row in group])),
                }
            )
    write_csv(output_dir / "quality_split_class_summary.csv", split_class_summary)

    return thresholds


def class_distribution(rows: list[ImageMetrics], output_dir: Path) -> None:
    dist_rows: list[dict] = []
    for split in SPLITS:
        group = [row for row in rows if row.split == split]
        counter = Counter(row.class_name for row in group)
        total = len(group)
        dist_rows.append(
            {
                "split": split,
                "NG": counter.get("NG", 0),
                "OK": counter.get("OK", 0),
                "total": total,
                "NG_ratio": counter.get("NG", 0) / total if total else 0,
                "OK_ratio": counter.get("OK", 0) / total if total else 0,
            }
        )
    write_csv(output_dir / "class_distribution.csv", dist_rows)

    labels = [row["split"] for row in dist_rows]
    ng_counts = [row["NG"] for row in dist_rows]
    ok_counts = [row["OK"] for row in dist_rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(8, 5))
    plt.bar(x, ng_counts, label="NG", color="#d95f02")
    plt.bar(x, ok_counts, bottom=ng_counts, label="OK", color="#1b9e77")
    plt.xticks(x, labels)
    plt.ylabel("Images")
    plt.title("Class distribution by split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=180)
    plt.close()


def box_distribution(rows: list[ImageMetrics], output_dir: Path, num_clusters: int = 9) -> list[list[float]]:
    areas = [row.max_bbox_area for row in rows if row.max_bbox_area > 0]
    ratios = [row.max_aspect_ratio for row in rows if row.max_aspect_ratio > 0]

    box_summary = [
        {
            "area_min": float(np.min(areas)),
            "area_max": float(np.max(areas)),
            "area_mean": float(np.mean(areas)),
            "area_p10": float(np.quantile(areas, 0.10)),
            "area_p90": float(np.quantile(areas, 0.90)),
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "ratio_mean": float(np.mean(ratios)),
        }
    ]
    write_csv(output_dir / "box_size_aspect_summary.csv", box_summary)

    plt.figure(figsize=(11, 8))
    plt.subplot(2, 2, 1)
    plt.hist(areas, bins=50, color="#4c78a8", edgecolor="white")
    plt.title("BBox area distribution")
    plt.xlabel("Normalized area")
    plt.ylabel("Count")

    plt.subplot(2, 2, 2)
    plt.hist(ratios, bins=50, color="#f58518", edgecolor="white")
    plt.title("BBox aspect ratio distribution")
    plt.xlabel("Width / height")

    plt.subplot(2, 2, 3)
    plt.scatter([row.max_bbox_area for row in rows], [row.max_aspect_ratio for row in rows], s=5, alpha=0.35)
    plt.title("Area vs aspect ratio")
    plt.xlabel("Normalized area")
    plt.ylabel("Width / height")

    split_names = []
    split_area_means = []
    for split in SPLITS:
        split_areas = [row.max_bbox_area for row in rows if row.split == split and row.max_bbox_area > 0]
        split_names.append(split)
        split_area_means.append(float(np.mean(split_areas)))
    plt.subplot(2, 2, 4)
    plt.bar(split_names, split_area_means, color="#54a24b")
    plt.title("Mean bbox area by split")
    plt.ylabel("Normalized area")
    plt.tight_layout()
    plt.savefig(output_dir / "box_size_aspect_distribution.png", dpi=180)
    plt.close()

    wh: list[list[float]] = []
    for row in rows:
        label_path = Path(row.image.replace("/images/", "/labels/")).with_suffix(".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                wh.append([float(parts[3]), float(parts[4])])

    anchors: list[list[float]] = []
    try:
        from sklearn.cluster import KMeans

        if wh:
            kmeans = KMeans(n_clusters=min(num_clusters, len(wh)), random_state=42, n_init="auto")
            kmeans.fit(np.array(wh, dtype=float))
            centers = kmeans.cluster_centers_
            centers = centers[np.argsort(centers[:, 0] * centers[:, 1])]
            anchors = centers.tolist()
    except Exception as exc:
        if wh:
            anchors = numpy_kmeans(np.array(wh, dtype=float), min(num_clusters, len(wh))).tolist()
            (output_dir / "anchor_suggestions_note.txt").write_text(
                f"sklearn 不可用，已使用 NumPy KMeans 兜底。原始错误：{exc}",
                encoding="utf-8",
            )

    if anchors:
        anchors_array = np.array(anchors, dtype=float)
        anchors_array = anchors_array[np.argsort(anchors_array[:, 0] * anchors_array[:, 1])]
        anchors = anchors_array.tolist()
        write_csv(
            output_dir / "anchor_suggestions.csv",
            [
                {
                    "anchor_id": index + 1,
                    "width": float(anchor[0]),
                    "height": float(anchor[1]),
                    "aspect_ratio": float(anchor[0] / anchor[1]) if anchor[1] else 0.0,
                }
                for index, anchor in enumerate(anchors)
            ],
        )

    return anchors


def numpy_kmeans(points: np.ndarray, num_clusters: int, iterations: int = 80) -> np.ndarray:
    """轻量级 KMeans 兜底实现，避免为了 anchor 建议强依赖 sklearn。"""
    rng = np.random.default_rng(42)
    if len(points) <= num_clusters:
        return points.copy()
    indices = rng.choice(len(points), size=num_clusters, replace=False)
    centers = points[indices].copy()

    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for index in range(num_clusters):
            cluster_points = points[labels == index]
            if len(cluster_points):
                new_centers[index] = cluster_points.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers


def make_sample_sheets(rows: list[ImageMetrics], output_dir: Path) -> None:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    categories = [
        ("brightness_high", "High brightness samples", sorted(rows, key=lambda row: row.brightness, reverse=True)),
        ("brightness_low", "Low brightness samples", sorted(rows, key=lambda row: row.brightness)),
        ("low_contrast", "Low contrast samples", sorted(rows, key=lambda row: row.contrast)),
        ("low_sharpness", "Low sharpness samples", sorted(rows, key=lambda row: row.sharpness_laplacian_var)),
        ("overexposed", "High overexposure samples", sorted(rows, key=lambda row: row.overexposed_ratio, reverse=True)),
        ("small_target", "Small target samples", sorted(rows, key=lambda row: row.max_bbox_area)),
        ("large_target", "Large target samples", sorted(rows, key=lambda row: row.max_bbox_area, reverse=True)),
    ]

    sample_rows: list[dict] = []
    for name, title, samples in categories:
        samples = samples[:8]
        tile_w, tile_h = 260, 210
        cols = 4
        sheet = Image.new("RGB", (cols * tile_w, math.ceil(len(samples) / cols) * tile_h + 34), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), title, fill=(20, 20, 20), font=font)
        for index, row in enumerate(samples):
            x0 = (index % cols) * tile_w
            y0 = 34 + (index // cols) * tile_h
            image = Image.open(row.image).convert("RGB")
            image.thumbnail((tile_w - 12, tile_h - 54))
            sheet.paste(image, (x0 + (tile_w - image.width) // 2, y0 + 4))
            caption1 = f"{row.split} | {row.class_name} | {row.file[:30]}"
            caption2 = (
                f"B {row.brightness:.1f} C {row.contrast:.1f} "
                f"S {row.sharpness_laplacian_var:.0f} OE {row.overexposed_ratio:.2%}"
            )
            caption3 = f"area {row.max_bbox_area:.3f} ratio {row.max_aspect_ratio:.2f}"
            draw.text((x0 + 6, y0 + tile_h - 45), caption1, fill=(20, 20, 20), font=small_font)
            draw.text((x0 + 6, y0 + tile_h - 31), caption2, fill=(20, 20, 20), font=small_font)
            draw.text((x0 + 6, y0 + tile_h - 17), caption3, fill=(20, 20, 20), font=small_font)
            sample_rows.append(
                {
                    "category": name,
                    "rank": index + 1,
                    "split": row.split,
                    "class_name": row.class_name,
                    "file": row.file,
                    "image": row.image,
                    "brightness": row.brightness,
                    "contrast": row.contrast,
                    "sharpness_laplacian_var": row.sharpness_laplacian_var,
                    "overexposed_ratio": row.overexposed_ratio,
                    "max_bbox_area": row.max_bbox_area,
                    "sheet": str(output_dir / f"{name}_samples.jpg"),
                }
            )
        sheet.save(output_dir / f"{name}_samples.jpg", quality=92)
    write_csv(output_dir / "quality_feature_samples.csv", sample_rows)


def write_markdown(dataset_root: Path, output_dir: Path, rows: list[ImageMetrics], thresholds: dict[str, float]) -> None:
    md = [
        "# 数据集质量分析报告",
        "",
        f"- 数据集：`{dataset_root}`",
        f"- 图片总数：{len(rows)}",
        "- 分析内容：类别分布、框尺寸、宽高比、亮度、对比度、清晰度、过曝比例和代表样本。",
        "",
        "## 关键阈值",
        "",
    ]
    for key, value in thresholds.items():
        md.append(f"- `{key}`: {value:.6f}")
    md.extend(
        [
            "",
            "## 输出文件",
            "",
            "| 文件 | 说明 |",
            "| --- | --- |",
            "| `class_distribution.csv` / `class_distribution.png` | OK/NG 类别分布 |",
            "| `box_size_aspect_summary.csv` / `box_size_aspect_distribution.png` | 目标框尺寸与宽高比分布 |",
            "| `anchor_suggestions.csv` | 基于当前标签的 anchor 聚类建议 |",
            "| `image_quality_metrics.csv` | 单图质量指标明细 |",
            "| `quality_feature_counts.csv` | 困难质量特征数量统计 |",
            "| `quality_split_class_summary.csv` | 按 split 和 OK/NG 聚合的质量指标 |",
            "| `quality_feature_samples.csv` | 各特征代表样本清单 |",
            "",
            "## 代表样本拼图",
            "",
        ]
    )
    for name, title in [
        ("brightness_high", "高亮/偏亮样本"),
        ("brightness_low", "偏暗样本"),
        ("low_contrast", "低对比度样本"),
        ("low_sharpness", "低清晰度样本"),
        ("overexposed", "过曝比例高样本"),
        ("small_target", "小目标样本"),
        ("large_target", "大目标样本"),
    ]:
        md.extend([f"### {title}", "", f"![{title}]({name}_samples.jpg)", ""])
    (output_dir / "quality_analysis_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 YOLO 数据集质量")
    parser.add_argument("--dataset_root", type=Path, required=True, help="包含 images/labels 的数据集目录")
    parser.add_argument("--output_dir", type=Path, required=True, help="分析结果输出目录")
    parser.add_argument("--num_clusters", type=int, default=9, help="anchor 聚类数量")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_metrics(args.dataset_root)
    if not rows:
        raise RuntimeError(f"没有在 {args.dataset_root} 中找到可分析图片")

    thresholds = summarize(rows, args.output_dir)
    class_distribution(rows, args.output_dir)
    box_distribution(rows, args.output_dir, num_clusters=args.num_clusters)
    make_sample_sheets(rows, args.output_dir)
    write_markdown(args.dataset_root, args.output_dir, rows, thresholds)

    print(f"分析完成：{len(rows)} 张图片")
    print(f"输出目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
