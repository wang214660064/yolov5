"""YOLOv5 喷头检测项目的最短脚本入口。"""

from __future__ import annotations

from pathlib import Path

import yaml

from .data.dataset_analyzer import DatasetAnalyzer
from .data.dataset_preparer import DatasetPreparer
from .evaluation.error_sample_exporter import ErrorSampleExporter
from .run_config import CONFIG, RunConfig
from .utils.visualize_badcase import visualize_badcase


AUGMENTATION_HYP_KEYS = (
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
)


def run(config: RunConfig = CONFIG) -> int:
    """直接根据配置执行对应动作，不再经过 CLI、argparse 或 Factory。"""
    actions = {
        "prepare_data": _prepare_data,
        "analyze_data": _analyze_data,
        "write_config": _write_config,
        "train": _train,
        "val": _val,
    }
    try:
        action = actions[config.action]
    except KeyError as exc:
        raise ValueError(f"未知运行动作：{config.action}") from exc
    return action(config)


def _prepare_data(config: RunConfig) -> int:
    report = DatasetPreparer(
        source_root=config.dataset_root,
        output_root=config.generated_dataset_root,
        val_ratio=config.val_ratio,
        seed=config.split_seed,
        ssim_threshold=config.ssim_threshold,
        phash_threshold=config.phash_threshold,
        deduplicate_workers=config.deduplicate_workers,
    ).prepare()
    print(f"数据准备完成：{report.output_root}")
    print(
        f"候选样本：{report.total_candidates}，精确重复：{report.exact_duplicate_count}，"
        f"SSIM相似重复：{report.similar_duplicate_count}，总重复：{report.duplicate_count}"
    )
    print(f"训练集：{report.train_count}，验证集：{report.val_count}，测试集：{report.test_count}")
    return 0


def _analyze_data(config: RunConfig) -> int:
    output_path = DatasetAnalyzer(config.dataset_root).write_markdown(config.dataset_report)
    print(f"数据分析报告已生成：{output_path}")
    return 0


def _write_config(config: RunConfig) -> int:
    content = {
        "path": str(config.generated_dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {0: "NG", 1: "OK"},
    }
    config.generated_dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    config.generated_dataset_yaml.write_text(
        yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"数据配置已生成：{config.generated_dataset_yaml}")
    return 0


def _train(config: RunConfig) -> int:
    import train

    hyp_yaml = _resolve_training_hyp(config)
    kwargs = {
        "weights": config.model_weights,
        "cfg": config.model_cfg,
        "data": _path(config.data_yaml),
        "hyp": _path(hyp_yaml),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "imgsz": 640,
        "optimizer": "AdamW",
        "workers": config.workers,
        "device": _detect_device(),
        "project": "runs/train",
        "name": config.train_name or "nozzle_ng_ok",
    }
    if config.dry_run:
        print(f"dry-run train: {kwargs}")
        return 0
    train.run(**kwargs)
    return 0


def _val(config: RunConfig) -> int:
    import val

    requested_dir = Path("runs/val") / (config.val_name or "exp")
    save_dir = _increment_path(requested_dir)
    kwargs = {
        "data": _path(config.data_yaml),
        "weights": _path(config.weights),
        "imgsz": 640,
        "conf_thres": config.conf,
        "iou_thres": 0.45,
        "task": config.eval_task,
        "device": _detect_device(),
        "workers": config.workers,
        "project": _path(save_dir.parent),
        "name": save_dir.name,
        "exist_ok": True,
        "verbose": True,
        "save_txt": config.save_txt,
        "save_conf": config.save_conf,
        "save_json": config.save_json,
    }
    if config.dry_run:
        print(f"dry-run val: {kwargs}")
        return 0

    val.run(**kwargs)
    if config.export_error_samples:
        if not config.save_txt:
            print("未开启 save_txt，跳过 BadCase 导出。")
            return 0
        summary = ErrorSampleExporter(
            data_yaml=config.data_yaml,
            task=config.eval_task,
            val_save_dir=save_dir,
            output_root=config.error_samples_dir,
            iou_threshold=config.error_iou_threshold,
        ).export()
        badcase_dir = config.error_samples_dir / save_dir.relative_to("runs/val")
        print(f"BadCase 已导出：{badcase_dir}")
        print(
            f"漏检：{summary['missed_target']}，误检：{summary['false_alarm']}，"
            f"类别错误：{summary['class_error']}，涉及图片：{summary['images_with_errors']}"
        )
        
        # 可视化错误样本
        visualize_badcase(badcase_dir)
    return 0


def _build_training_hyp(config: RunConfig) -> dict:
    """合并基础 hyp、增强总开关和损失函数配置。"""
    hyp = yaml.safe_load(config.hyp_yaml.read_text(encoding="utf-8")) or {}
    if not config.enable_augmentation:
        for key in AUGMENTATION_HYP_KEYS:
            hyp[key] = 0.0
    iou_type = config.iou_type.lower()
    if iou_type not in {"ciou", "siou"}:
        raise ValueError("iou_type 只支持 'ciou' 或 'siou'")
    hyp["iou_type"] = iou_type
    hyp["fl_gamma"] = config.focal_gamma if config.use_focal_loss else 0.0
    hyp["fl_alpha"] = config.focal_alpha
    return hyp


def _resolve_training_hyp(config: RunConfig) -> Path:
    hyp = _build_training_hyp(config)
    output_path = Path("project/nozzle_inspection/outputs/configs/train_effective.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(hyp, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(
        f"训练损失配置：IoU={hyp['iou_type'].upper()}，"
        f"Focal={'开启' if hyp['fl_gamma'] > 0 else '关闭'}，"
        f"gamma={hyp['fl_gamma']}，alpha={hyp['fl_alpha']}"
    )
    return output_path


def _detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        print(f"检测到 GPU：{torch.cuda.get_device_name(0)}")
        return "0"
    print("未检测到 GPU，使用 CPU")
    return "cpu"


def _path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _increment_path(path: Path) -> Path:
    """目录已存在时追加数字，保持每次实验输出独立。"""
    # 首先检查原始路径是否存在
    if not path.exists():
        return path
    # 如果原始路径存在，尝试递增
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    raise SystemExit(run())
