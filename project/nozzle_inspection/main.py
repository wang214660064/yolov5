import argparse
from pathlib import Path

from project.nozzle_inspection.data.dataset_analyzer import DatasetAnalyzer
from project.nozzle_inspection.data.dataset_preparer import DatasetPreparer
from project.nozzle_inspection.factories.evaluator_factory import EvaluatorFactory
from project.nozzle_inspection.factories.model_factory import ModelFactory
from project.nozzle_inspection.factories.trainer_factory import TrainerFactory
from project.nozzle_inspection.reporting.report_builder import ReportBuilder
from project.nozzle_inspection.training.experiment_runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D打印机喷头检测 NG/OK 工厂模式项目入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="构建或执行训练命令")
    train.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", help="YOLO 数据配置路径")
    train.add_argument("--hyp", default="project/nozzle_inspection/configs/train_ng_ok.yaml", help="训练超参数配置路径")
    train.add_argument("--epochs", type=int, default=50, help="训练轮数")
    train.add_argument("--dry-run", action="store_true", help="只打印命令，不启动训练")

    val = subparsers.add_parser("val", help="构建验证命令")
    val.add_argument("--data", default="project/nozzle_inspection/configs/dataset.yaml", help="YOLO 数据配置路径")
    val.add_argument("--weights", required=True, help="待验证权重路径")
    val.add_argument("--conf", type=float, default=0.25, help="置信度阈值")

    report = subparsers.add_parser("report", help="生成 Markdown 和可选 PPT 汇报")
    report.add_argument("--output", default="../outputs/reports/nozzle_report.md", help="Markdown 报告输出路径")
    report.add_argument("--pptx", default="../outputs/reports/nozzle_report.pptx", help="PPTX 汇报输出路径")

    config = subparsers.add_parser("write-config", help="写入两类数据配置")
    config.add_argument("--output", default="project/nozzle_inspection/configs/dataset.yaml", help="配置输出路径")
    config.add_argument("--path", default="../outputs/datasets/nozzle_ng_ok_v1", help="衍生数据集根目录")

    analyze = subparsers.add_parser("analyze-data", help="分析数据集并生成 Markdown 报告")
    analyze.add_argument("--dataset", default="../dataset_2", help="待分析数据集根目录")
    analyze.add_argument("--output", default="project/nozzle_inspection/data/dataset_report.md", help="数据分析报告输出路径")

    prepare = subparsers.add_parser("prepare-data", help="去重、转换 NG/OK 标签并重新划分数据集")
    prepare.add_argument("--dataset", default="../dataset_2", help="原始数据集根目录")
    prepare.add_argument("--output", default="../outputs/datasets/nozzle_ng_ok_v1", help="衍生数据集输出目录")
    prepare.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例")
    prepare.add_argument("--seed", type=int, default=42, help="重划分随机种子")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        command = TrainerFactory(repo_root=Path.cwd()).build_train_command(
            data_yaml=Path(args.data),
            hyp_yaml=Path(args.hyp),
            epochs=args.epochs,
        )
        result = ExperimentRunner().run(command, dry_run=args.dry_run, cwd=str(Path.cwd()))
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return result.returncode

    if args.command == "val":
        command = EvaluatorFactory().build_val_command(Path(args.data), Path(args.weights), conf=args.conf)
        print(" ".join(command))
        return 0

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

    if args.command == "analyze-data":
        output_path = DatasetAnalyzer(Path(args.dataset)).write_markdown(Path(args.output))
        print(f"数据分析报告已生成：{output_path}")
        return 0

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

    parser.error("未知命令")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
