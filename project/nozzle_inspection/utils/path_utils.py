"""
路径工具模块 - 项目路径管理

该模块提供项目路径的统一管理，使用 dataclass 封装所有关键路径。

目录结构约定：
    repo_root/                    # 代码仓库根目录
        project/
            nozzle_inspection/    # 项目代码目录
    ../dataset_2/                 # 数据集目录（仓库外部）
    ../outputs/                   # 输出目录（仓库外部）

核心功能：
- 统一管理项目所有关键路径
- 自动解析路径关系
- 提供目录创建工具

使用示例：
    paths = ProjectPaths(Path("/path/to/repo"))
    print(paths.dataset_root)   # -> /path/to/dataset_2
    print(paths.outputs_root)   # -> /path/to/outputs
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """
    项目路径管理器，统一管理代码仓库、数据集和输出目录路径
    
    属性：
        repo_root: 代码仓库根目录
    
    派生属性（通过 @property 定义）：
        outer_root: 仓库外部目录（仓库的父目录）
        dataset_root: 数据集目录
        outputs_root: 输出目录
        project_root: 项目代码根目录
        project_data_dir: 项目数据目录
    
    方法：
        ensure_dir(): 确保目录存在（不存在则创建）
    """

    repo_root: Path  # 代码仓库根目录

    def __post_init__(self):
        """
        初始化后处理
        
        将 repo_root 转换为绝对路径并解析符号链接
        由于类是 frozen（不可变），需要使用 object.__setattr__
        """
        object.__setattr__(self, "repo_root", Path(self.repo_root).resolve())

    @property
    def outer_root(self) -> Path:
        """
        仓库外部目录
        
        返回代码仓库的父目录，数据集和输出目录都放在这里
        """
        return self.repo_root.parent

    @property
    def dataset_root(self) -> Path:
        """
        数据集目录
        
        默认路径：repo_root/../dataset_2
        """
        return self.outer_root / "dataset_2"

    @property
    def outputs_root(self) -> Path:
        """
        输出目录
        
        默认路径：repo_root/../outputs
        用于存放训练结果、报告等输出文件
        """
        return self.outer_root / "outputs"

    @property
    def project_root(self) -> Path:
        """
        项目代码根目录
        
        默认路径：repo_root/project/nozzle_inspection
        """
        return self.repo_root / "project" / "nozzle_inspection"

    @property
    def project_data_dir(self) -> Path:
        """
        项目数据目录
        
        默认路径：repo_root/project/nozzle_inspection/data
        用于存放数据集分析报告等
        """
        return self.project_root / "data"

    @staticmethod
    def ensure_dir(path: Path) -> Path:
        """
        确保目录存在
        
        如果目录不存在，创建该目录及其所有父目录
        
        参数：
            path: 目标目录路径
        
        返回值：
            Path: 创建后的目录路径
        
        示例：
            ProjectPaths.ensure_dir(Path("outputs/reports"))
            # 如果 outputs/reports 不存在，会创建它
        """
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        return target
