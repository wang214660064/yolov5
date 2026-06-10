"""
实验日志模块 - 实验事件记录

该模块提供实验事件的记录功能，将实验过程中的关键事件写入 Markdown 文件，
便于后续报告复盘和实验追踪。

核心功能：
- 记录实验事件（训练开始、训练结束、参数调整等）
- 自动添加时间戳
- 支持自定义字段

使用示例：
    logger = ExperimentLogger(Path("experiments/log.md"))
    logger.log_event("训练开始", {"epochs": 50, "batch_size": 8})
    logger.log_event("训练结束", {"accuracy": 0.95, "loss": 0.05})
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ExperimentLogger:
    """
    实验日志记录器，将实验事件写入 Markdown 文件
    
    属性：
        log_path: 日志文件路径
    
    方法：
        log_event(): 记录一个实验事件
    """

    log_path: Path  # 日志文件路径

    def log_event(self, title: str, fields: Mapping[str, object] | None = None) -> Path:
        """
        记录一个实验事件
        
        参数：
            title: 事件标题
            fields: 事件相关的字段字典（可选）
        
        返回值：
            Path: 日志文件路径
        
        输出格式：
            ## 事件标题
            
            - 时间：2024-01-01T12:00:00
            - 字段1：值1
            - 字段2：值2
            
        
        示例：
            logger.log_event("训练开始", {
                "epochs": 50,
                "batch_size": 8,
                "learning_rate": 0.001
            })
            
            输出：
                ## 训练开始
                
                - 时间：2024-01-01T12:00:00
                - epochs：50
                - batch_size：8
                - learning_rate：0.001
                
        """
        # 确保日志目录存在
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果没有提供字段，使用空字典
        fields = fields or {}
        
        # 构建日志内容
        lines = [
            f"## {title}",           # 事件标题
            "",                      # 空行
            f"- 时间：{datetime.now().isoformat(timespec='seconds')}",  # 当前时间
        ]
        
        # 添加自定义字段
        for key, value in fields.items():
            lines.append(f"- {key}：{value}")
        
        # 添加结尾空行
        lines.append("")
        
        # 以追加模式写入日志文件
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")
        
        return self.log_path
