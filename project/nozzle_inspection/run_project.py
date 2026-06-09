from pathlib import Path

from project.nozzle_inspection.main import main
from project.nozzle_inspection.run_config import CONFIG, RunConfig


def build_argv(config: RunConfig) -> list[str]:
    """把脚本配置转换为内部入口参数。"""

    if config.action == "train":
        argv = [
            "train",
            "--data",
            _path(config.data_yaml),
            "--hyp",
            _path(config.hyp_yaml),
            "--epochs",
            str(config.epochs),
        ]
        if config.dry_run:
            argv.append("--dry-run")
        return argv

    if config.action == "val":
        return [
            "val",
            "--data",
            _path(config.data_yaml),
            "--weights",
            _path(config.weights),
            "--conf",
            str(config.conf),
        ]

    if config.action == "report":
        return [
            "report",
            "--output",
            _path(config.report_output),
            "--pptx",
            _path(config.pptx_output),
        ]

    if config.action == "write_config":
        return [
            "write-config",
            "--output",
            _path(config.generated_dataset_yaml),
            "--path",
            _path(config.generated_dataset_root),
        ]

    if config.action == "analyze_data":
        return [
            "analyze-data",
            "--dataset",
            _path(config.dataset_root),
            "--output",
            _path(config.dataset_report),
        ]

    raise ValueError(f"未知运行动作：{config.action}")


def run(config: RunConfig = CONFIG) -> int:
    return main(build_argv(config))


def _path(path: Path) -> str:
    return str(path).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(run())
