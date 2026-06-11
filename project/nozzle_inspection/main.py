"""
3D打印机喷头检测项目 - 命令行主入口

该模块是整个项目的命令行接口，按实验流程提供以下功能：
1. 准备数据集（prepare-data）：去重、转换标签、划分数据集
2. 分析数据集（analyze-data）：分析数据集并生成报告
3. 生成数据配置文件（write-config）：生成 YOLO 数据配置
4. 训练 YOLOv5 模型（train）：训练目标检测模型
5. 验证模型性能（val）：验证训练好的模型
6. 生成项目报告（report）：生成 Markdown 和 PPT 汇报

使用示例：
    python main.py prepare-data --dataset ../dataset_2 --output ../outputs/datasets/nozzle_ng_ok_v1
    python main.py analyze-data --dataset ../dataset_2
    python main.py train --epochs 30
"""

import argparse
import os
from pathlib import Path

# 设置 UTF-8 编码环境变量，解决 Windows 上 subprocess 的编码问题
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

# 导入项目各模块（相对导入）
from .data.dataset_analyzer import DatasetAnalyzer
from .data.dataset_preparer import DatasetPreparer
from .factories.evaluator_factory import EvaluatorFactory
from .factories.model_factory import ModelFactory
from .factories.trainer_factory import TrainerFactory
from .reporting.report_builder import ReportBuilder
from .training.experiment_runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器
    
    返回一个配置好的 argparse.ArgumentParser 对象，按实验流程包含以下子命令：
    - prepare-data: 准备数据集（去重、转换标签、划分）
    - analyze-data: 分析数据集
    - write-config: 生成数据配置
    - train: 训练模型
    - val: 验证模型
    - report: 生成报告
    
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
    train.add_argument("--dry-run", action="store_true", 
                       help="仅打印训练命令，不实际执行训练（用于调试）")

    # 验证子命令配置（实验流程第5步）
    val = subparsers.add_parser("val", help="构建验证命令")
    val.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", 
                     help="YOLO 数据配置文件路径")
    val.add_argument("--weights", required=True, help="待验证的模型权重文件路径")
    val.add_argument("--conf", type=float, default=0.25, 
                     help="置信度阈值，用于过滤检测结果，默认0.25")

    # 报告生成子命令配置（实验流程第6步）
    report = subparsers.add_parser("report", help="生成 Markdown 和可选 PPT 汇报")
    report.add_argument("--output", default="project/nozzle_inspection/outputs/reports/nozzle_report.md", 
                        help="Markdown 报告输出路径")
    report.add_argument("--pptx", default="project/nozzle_inspection/outputs/reports/nozzle_report.pptx", 
                        help="PPTX 汇报文件输出路径")
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
        report: 生成报告（实验流程第6步）
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

    # 实验流程第1步：准备数据集（去重、转换标签、划分）
    if args.command == "prepare-data":
        report = DatasetPreparer(
            source_root=Path(args.dataset),
            output_root=Path(args.output),
            val_ratio=args.val_ratio,
            seed=args.seed,
        ).prepare()
        print(f"数据准备完成：{report.output_root}")
        print(f"候选样本：{report.total_candidates}，重复样本：{report.duplicate_count}")
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
        # 创建训练器工厂，构建训练命令
        command = TrainerFactory(repo_root=Path.cwd()).build_train_command(
            data_yaml=Path(args.data),
            hyp_yaml=Path(args.hyp),
            epochs=args.epochs,
        )
        # 运行训练命令
        result = ExperimentRunner().run(command, dry_run=args.dry_run, cwd=str(Path.cwd()))
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode

    # 实验流程第5步：执行验证命令
    if args.command == "val":
        command = EvaluatorFactory().build_val_command(
            Path(args.data), 
            Path(args.weights), 
            conf=args.conf
        )
        print(" ".join(command))
        return 0

    # 实验流程第6步：生成报告
    if args.command == "report":
        builder = ReportBuilder()
        markdown_path = builder.write_markdown(Path(args.output))
        pptx_path = builder.write_pptx(Path(args.pptx))
        print(f"Markdown 报告已生成：{markdown_path}")
        if pptx_path:
            print(f"PPT 汇报已生成：{pptx_path}")
        else:
            print("当前环境缺少 python-pptx，已跳过 PPT 汇报生成。")
        return 0

    # 未知命令处理
    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    # 程序入口，调用main函数并将返回值作为系统退出码
    raise SystemExit(main())
