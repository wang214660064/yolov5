from pathlib import Path

from project.nozzle_inspection.factories.evaluator_factory import EvaluatorFactory


def build_threshold_commands(data_yaml: Path, weights: Path, thresholds: tuple[float, ...] = (0.25, 0.5, 0.7)) -> list[list[str]]:
    factory = EvaluatorFactory()
    return [factory.build_val_command(data_yaml=data_yaml, weights=weights, conf=conf) for conf in thresholds]
