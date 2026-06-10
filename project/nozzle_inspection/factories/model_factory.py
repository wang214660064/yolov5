"""
模型工厂模块 - YOLOv5 配置生成

该模块负责生成 YOLOv5 所需的数据集配置文件（dataset.yaml）。

YOLOv5 数据集配置文件格式：
    path: 数据集根目录路径
    train: 训练集相对路径
    val: 验证集相对路径
    test: 测试集相对路径（可选）
    nc: 类别数量
    names: 类别名称列表

使用示例：
    factory = ModelFactory()
    yaml_content = factory.build_dataset_yaml(
        path="../outputs/datasets/nozzle_ng_ok_v1",
        train="images/train",
        val="images/val",
        test="images/test"
    )
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelFactory:
    """
    YOLOv5 配置生成工厂
    
    属性：
        names: 类别名称元组，默认 ("NG", "OK")
    
    方法：
        build_dataset_yaml(): 构建数据集配置内容
        write_dataset_yaml(): 将配置写入文件
    """

    names: tuple[str, str] = ("NG", "OK")  # 类别名称，索引0为NG，索引1为OK

    def build_dataset_yaml(self, path: str, train: str, val: str, test: str | None = None) -> str:
        """
        构建 YOLOv5 数据集配置内容
        
        参数：
            path: 数据集根目录路径
            train: 训练集图片目录相对路径
            val: 验证集图片目录相对路径
            test: 测试集图片目录相对路径（可选）
        
        返回值：
            str: YAML 格式的配置内容
        
        生成的配置格式：
            path: ../outputs/datasets/nozzle_ng_ok_v1
            train: images/train
            val: images/val
            test: images/test
            nc: 2
            names:
              0: NG
              1: OK
        
        示例：
            factory = ModelFactory()
            yaml = factory.build_dataset_yaml(
                path="../datasets",
                train="images/train",
                val="images/val"
            )
        """
        # 构建配置行列表
        lines = [
            f"path: {path}",   # 数据集根目录
            f"train: {train}", # 训练集相对路径
            f"val: {val}",     # 验证集相对路径
        ]
        
        # 如果提供了测试集路径，添加测试集配置
        if test:
            lines.append(f"test: {test}")
        
        # 添加类别配置
        lines.extend([
            "nc: 2",                        # 类别数量（NG 和 OK 两类）
            "names:",                        # 类别名称列表
            f"  0: {self.names[0]}",         # 类别0：NG
            f"  1: {self.names[1]}",         # 类别1：OK
            "",                              # 结尾空行
        ])
        
        # 合并为字符串并返回
        return "\n".join(lines)

    def write_dataset_yaml(self, output_path: Path, path: str, train: str, val: str, test: str | None = None) -> Path:
        """
        将数据集配置写入文件
        
        参数：
            output_path: 输出文件路径
            path: 数据集根目录路径
            train: 训练集相对路径
            val: 验证集相对路径
            test: 测试集相对路径（可选）
        
        返回值：
            Path: 实际输出的文件路径
        
        执行步骤：
            1. 确保输出目录存在
            2. 构建配置内容
            3. 写入文件
        """
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 构建配置内容并写入文件
        output_path.write_text(self.build_dataset_yaml(path, train, val, test), encoding="utf-8")
        
        return output_path
