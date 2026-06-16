"""
3D打印机喷头检测项目 - 命令行主入口

该模块是整个项目的命令行接口，按实验流程提供以下功能：
1. 准备数据集（prepare-data）：去重、转换标签、划分数据集
2. 分析数据集（analyze-data）：分析数据集并生成报告
3. 生成数据配置文件（write-config）：生成 YOLO 数据配置
4. 训练 YOLOv5 模型（train）：训练目标检测模型
5. 验证模型性能（val）：验证训练好的模型

使用示例：
    python main.py prepare-data --dataset ../dataset_2 --output ../outputs/datasets/nozzle_ng_ok_v1
    python main.py analyze-data --dataset ../dataset_2
    python main.py train --epochs 30
"""

import argparse
import os
from pathlib import Path

import yaml

# 设置 UTF-8 编码环境变量，解决 Windows 上 subprocess 的编码问题
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 导入项目各模块（相对导入）
from .data.dataset_analyzer import DatasetAnalyzer
from .data.dataset_preparer import DatasetPreparer
from .factories.evaluator_factory import EvaluatorFactory
from .factories.model_factory import ModelFactory
from .factories.trainer_factory import TrainerFactory
from .evaluation.error_sample_exporter import ErrorSampleExporter
from .training.experiment_runner import ExperimentRunner


AUGMENTATION_HYP_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
)


def resolve_training_hyp(hyp_yaml: Path, enable_augmentation: bool) -> Path:
    """
    根据总开关生成训练实际使用的超参数文件。

    enable_augmentation=False 时，不覆盖用户手写 hyp，而是生成一份增强项全为 0 的临时配置；
    enable_augmentation=True 时，直接使用用户指定的 hyp 文件，让各增强强度由 hyp 控制。
    """
    if enable_augmentation:
        return hyp_yaml

    hyp = yaml.safe_load(hyp_yaml.read_text(encoding="utf-8")) or {}
    for key in AUGMENTATION_HYP_KEYS:
        hyp[key] = 0.0

    output_path = Path("project/nozzle_inspection/outputs/configs/train_no_augment.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(hyp, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def detect_device() -> str:
    """
    集中检测可用设备
    
    返回值：
        str: 设备标识，"0" 表示第一个GPU，"cpu" 表示CPU
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"检测到 GPU: {device_name}")
            return "0"  # 使用第一个GPU
        print("未检测到 GPU，使用 CPU")
        return "cpu"
    except ImportError:
        print("PyTorch 未安装，使用 CPU")
        return "cpu"


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器
    
    返回一个配置好的 argparse.ArgumentParser 对象，按实验流程包含以下子命令：
    - prepare-data: 准备数据集（去重、转换标签、划分）
    - analyze-data: 分析数据集
    - write-config: 生成数据配置
    - train: 训练模型
    - val: 验证模型
    
    返回值：
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description="3D打印机喷头检测 NG/OK 工厂模式项目入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 数据准备子命令配置（实验流程第1步）
    prepare = subparsers.add_parser("prepare-data", help="去重、转换 NG/OK 标签并重新划分数据集")
    prepare.add_argument("--dataset", default="../dataset_2", 
                         help="原始数据集根目录")
    prepare.add_argument("--output", default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1", 
                         help="处理后的数据集输出目录")
    prepare.add_argument("--val-ratio", type=float, default=0.2, 
                         help="验证集占比，默认0.2（20%）")
    prepare.add_argument("--seed", type=int, default=42, 
                         help="随机种子，用于确保数据集划分结果可重复")
    prepare.add_argument("--ssim-threshold", type=float, default=0.85,
                         help="SSIM 相似去重阈值，默认0.85")
    prepare.add_argument("--phash-threshold", type=int, default=4,
                         help="pHash 汉明距离阈值，默认4，越小越严格")
    prepare.add_argument("--deduplicate-workers", type=int, default=0,
                         help="去重阶段工作进程数，0表示自动按CPU和任务量选择")

    # 数据分析子命令配置（实验流程第2步）
    analyze = subparsers.add_parser("analyze-data", help="分析数据集并生成 Markdown 报告")
    analyze.add_argument("--dataset", default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1", 
                         help="待分析的数据集根目录")
    analyze.add_argument("--output", default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1/dataset_report.md", 
                         help="数据分析报告输出路径")

    # 配置生成子命令配置（实验流程第3步）
    config = subparsers.add_parser("write-config", help="写入两类数据配置")
    config.add_argument("--output", default="project/nozzle_inspection/configs/dataset.yaml", 
                       help="生成的配置文件输出路径")
    config.add_argument("--path", default="project/nozzle_inspection/outputs/datasets/nozzle_ng_ok_v1", 
                       help="衍生数据集的根目录路径")

    # 训练子命令配置（实验流程第4步）
    train = subparsers.add_parser("train", help="构建或执行训练命令")
    train.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", 
                       help="YOLO 数据配置文件路径")
    train.add_argument("--hyp", default="project/nozzle_inspection/configs/train_ng_ok.yaml", 
                       help="训练超参数配置文件路径")
    train.add_argument("--epochs", type=int, default=50, help="训练轮数，默认50轮")
    train.add_argument("--workers", type=int, default=8, help="DataLoader 工作进程数，默认8，Windows 建议使用0")
    train.add_argument("--batch-size", type=int, default=16, help="批次大小，根据显存调整（16/32/64），默认16")
    train.add_argument("--enable-augmentation", action="store_true",
                       help="开启训练数据增强；不传该参数时会把 hyp 中的增强项全部置 0")
    train.add_argument("--name", default=None,
                       help="训练输出文件夹名称；不填时使用 YOLOv5 默认 nozzle_ng_ok 并自动递增")
    train.add_argument("--dry-run", action="store_true", 
                       help="仅打印训练命令，不实际执行训练（用于调试）")

    # 验证子命令配置（实验流程第5步）
    val = subparsers.add_parser("val", help="构建验证命令")
    val.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", 
                     help="YOLO 数据配置文件路径")
    val.add_argument("--weights", required=True, help="待验证的模型权重文件路径")
    val.add_argument("--conf", type=float, default=0.25, 
                     help="置信度阈值，用于过滤检测结果，默认0.25")
    val.add_argument("--task", choices=("train", "val", "test"), default="val",
                     help="评估数据划分，train表示训练集复查，val表示验证集，test表示测试集")
    val.add_argument("--name", default=None,
                     help="验证输出文件夹名称；不填时使用 YOLOv5 默认 exp 并自动递增")
    val.add_argument("--save-txt", action="store_true",
                     help="保存预测框 txt，错误样本复查建议开启")
    val.add_argument("--save-conf", action="store_true",
                     help="在预测 txt 中保存置信度，需要配合 --save-txt 使用")
    val.add_argument("--save-json", action="store_true",
                     help="保存 COCO JSON 格式预测结果")
    val.add_argument("--export-error-samples", action="store_true",
                     help="验证结束后按漏检、误检、类别错误复制错误样本到 runs 目录")
    val.add_argument("--error-samples-dir", default=None,
                     help="错误样本输出根目录；不填时默认 runs/<task>/BadCase")
    val.add_argument("--error-iou-thres", type=float, default=0.5,
                     help="错误样本匹配 IoU 阈值，默认0.5")

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    主函数，根据命令行参数执行相应操作
    
    参数：
        argv: 命令行参数列表，默认为None（使用sys.argv）
    
    返回值：
        int: 程序退出码，0表示成功，非0表示失败
    
    按实验流程支持的命令：
        prepare-data: 准备数据集（实验流程第1步）
        analyze-data: 分析数据集（实验流程第2步）
        write-config: 生成数据配置（实验流程第3步）
        train: 训练模型（实验流程第4步）
        val: 验证模型（实验流程第5步）
    """
    # ========== 参数解析 ==========
    # 第1步：创建命令行参数解析器
    # build_parser() 函数返回一个配置好的 argparse.ArgumentParser 对象
    # 这个解析器知道如何解析项目支持的所有子命令和参数
    parser = build_parser()
    
    # 第2步：解析命令行参数
    # parser.parse_args(argv) 会：
    #   1. 读取命令行参数（通常是 sys.argv，即用户在终端输入的命令）
    #   2. 根据 build_parser() 定义的规则验证参数
    #   3. 将解析结果存储在 Namespace 对象中
    #   4. 返回包含所有参数值的 args 对象
    # 
    # 解析后，args 对象会包含类似这样的属性：
    #   - args.command: 用户选择的子命令（如 "prepare-data", "train" 等）
    #   - args.dataset: 数据集路径
    #   - args.epochs: 训练轮数
    #   - 等等...根据不同的子命令有不同的属性
    args = parser.parse_args(argv)

    # ========== 集中设备检测 ==========
    # 在需要 GPU 的操作之前统一检测设备
    # 训练和验证都需要使用 GPU，因此提前检测一次即可
    device = detect_device()

    # 实验流程第1步：准备数据集（去重、转换标签、划分）
    if args.command == "prepare-data":
        report = DatasetPreparer(
            source_root=Path(args.dataset),
            output_root=Path(args.output),
            val_ratio=args.val_ratio,
            seed=args.seed,
            ssim_threshold=args.ssim_threshold,
            phash_threshold=args.phash_threshold,
            deduplicate_workers=args.deduplicate_workers,
        ).prepare()
        print(f"数据准备完成：{report.output_root}")
        print(
            f"候选样本：{report.total_candidates}，"
            f"精确重复：{report.exact_duplicate_count}，"
            f"SSIM相似重复：{report.similar_duplicate_count}，"
            f"总重复：{report.duplicate_count}"
        )
        print(f"训练集：{report.train_count}，验证集：{report.val_count}，测试集：{report.test_count}")
        return 0

    # 实验流程第2步：分析数据集
    if args.command == "analyze-data":
        output_path = DatasetAnalyzer(Path(args.dataset)).write_markdown(Path(args.output))
        print(f"数据分析报告已生成：{output_path}")
        return 0

    # 实验流程第3步：生成数据配置文件
    if args.command == "write-config":
        output_path = ModelFactory().write_dataset_yaml(
            output_path=Path(args.output),
            path=args.path,
            train="images/train",
            val="images/val",
            test="images/test",
        )
        print(f"数据配置已生成：{output_path}")
        return 0

    # 实验流程第4步：执行训练命令
    if args.command == "train":
        # 创建训练器工厂，构建训练命令（传递设备参数和 workers 参数）
        hyp_yaml = resolve_training_hyp(Path(args.hyp), args.enable_augmentation)
        command = TrainerFactory(repo_root=Path.cwd(), device=device, batch_size=args.batch_size).build_train_command(
            data_yaml=Path(args.data),
            hyp_yaml=hyp_yaml,
            epochs=args.epochs,
            workers=args.workers,
            name=args.name or "nozzle_ng_ok",
        )
        # 运行训练命令
        result = ExperimentRunner().run(command, dry_run=args.dry_run, cwd=str(Path.cwd()))
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode

    # 实验流程第5步：执行验证命令
    if args.command == "val":
        command = EvaluatorFactory(device=device).build_val_command(
            Path(args.data), 
            Path(args.weights), 
            conf=args.conf,
            task=args.task,
            name=args.name or "exp",
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_json=args.save_json,
        )
        result = ExperimentRunner().run(command, cwd=str(Path.cwd()))
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0 and args.export_error_samples:
            if not args.save_txt:
                print("未开启 --save-txt，无法导出错误样本；请开启 save_txt 后重新验证。")
                return result.returncode
            val_save_dir = _resolve_val_save_dir(result.stdout, project="runs/val", name=args.name or "exp")
            error_samples_dir = Path(args.error_samples_dir or f"runs/{args.task}/BadCase")
            summary = ErrorSampleExporter(
                data_yaml=Path(args.data),
                task=args.task,
                val_save_dir=val_save_dir,
                output_root=error_samples_dir,
                iou_threshold=args.error_iou_thres,
            ).export()
            print(
                "错误样本已导出："
                f"{error_samples_dir / val_save_dir.relative_to('runs/val') if _is_relative_to(val_save_dir, Path('runs/val')) else error_samples_dir / val_save_dir.name}"
            )
            print(
                f"漏检：{summary['missed_target']}，"
                f"误检：{summary['false_alarm']}，"
                f"类别错误：{summary['class_error']}，"
                f"涉及图片：{summary['images_with_errors']}"
            )
        return result.returncode

    # 未知命令处理
    parser.error("未知命令")
    return 2


def _resolve_val_save_dir(stdout: str, project: str, name: str) -> Path:
    """
    从 YOLOv5 val.py 输出中解析真实保存目录。

    当 YOLO 自动递增 exp2/exp3 时，stdout 中的 Results saved to 更可靠；
    解析失败时回退到 project/name。
    """
    for line in stdout.splitlines():
        if "Results saved to" not in line:
            continue
        raw_path = line.split("Results saved to", 1)[1].strip()
        raw_path = raw_path.split(" labels saved to ", 1)[0].strip()
        raw_path = _strip_ansi(raw_path)
        if raw_path:
            return Path(raw_path)
    return Path(project) / name


def _strip_ansi(text: str) -> str:
    result = ""
    index = 0
    while index < len(text):
        if text[index] == "\x1b":
            end = text.find("m", index)
            if end == -1:
                break
            index = end + 1
            continue
        result += text[index]
        index += 1
    return result


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # 程序入口，调用main函数并将返回值作为系统退出码
    raise SystemExit(main())
