from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """统一管理代码仓库、外层数据和输出目录路径。"""

    repo_root: Path

    def __post_init__(self):
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())

    @property
    def outer_root(self) -> Path:
        return self.repo_root.parent

    @property
    def dataset_root(self) -> Path:
        return self.outer_root / "dataset_2"

    @property
    def outputs_root(self) -> Path:
        return self.outer_root / "outputs"

    @property
    def project_root(self) -> Path:
        return self.repo_root / "project" / "nozzle_inspection"

    @property
    def project_data_dir(self) -> Path:
        return self.project_root / "data"

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return target
