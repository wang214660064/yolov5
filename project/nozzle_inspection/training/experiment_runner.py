"""
实验运行器模块 - 命令执行管理

该模块负责执行训练或验证命令，支持 dry-run 模式便于调试和命令复现。

核心功能：
- 执行外部命令（如 YOLOv5 训练/验证脚本）
- 支持 dry-run 模式（只打印命令不执行）
- 捕获命令输出和错误信息
- 使用当前 Python 解释器执行命令，确保环境一致性

使用示例：
    runner = ExperimentRunner()
    
    # 执行命令
    result = runner.run(
        ["python", "train.py", "--epochs", "50"],
        cwd="/workspace/yolov5"
    )
    print(result.stdout)
    
    # dry-run 模式（只打印命令）
    result = runner.run(
        ["python", "train.py", "--epochs", "50"],
        dry_run=True
    )
"""

import sys
from dataclasses import dataclass
import subprocess

# 获取当前 Python 解释器路径，确保使用正确的 conda 环境
_CURRENT_PYTHON = sys.executable


@dataclass(frozen=True)
class ExperimentResult:
    """
    实验执行结果
    
    属性：
        command: 执行的命令列表
        returncode: 命令退出码（0表示成功，非0表示失败）
        stdout: 命令标准输出
        stderr: 命令错误输出
    """
    command: list[str]  # 执行的命令
    returncode: int     # 退出码
    stdout: str         # 标准输出
    stderr: str         # 错误输出


class ExperimentRunner:
    """
    实验运行器，负责执行训练或验证命令
    
    支持 dry-run 模式，便于调试和命令复现。
    
    方法：
        run(): 执行命令（或 dry-run）
    """

    def run(self, command: list[str], dry_run: bool = False, cwd: str | None = None) -> ExperimentResult:
        """
        执行命令（或 dry-run）
        
        参数：
            command: 命令列表，每个元素是命令的一部分
            dry_run: 是否为试运行模式，默认为 False
            cwd: 工作目录，命令将在此目录下执行
        
        返回值：
            ExperimentResult: 执行结果
        
        dry-run 模式说明：
            - 当 dry_run=True 时，不实际执行命令，只返回命令字符串
            - 用于调试和记录实验命令
        
        执行模式说明：
            - 使用 subprocess.run() 执行命令
            - 捕获 stdout 和 stderr
            - 返回退出码
        
        示例：
            runner = ExperimentRunner()
            
            # dry-run 模式
            result = runner.run(
                ["python", "train.py", "--epochs", "50"],
                dry_run=True
            )
            print(result.stdout)  # 输出: dry-run: python train.py --epochs 50
            
            # 实际执行
            result = runner.run(
                ["python", "train.py", "--epochs", "50"],
                cwd="/workspace/yolov5"
            )
            print(f"退出码: {result.returncode}")
            print(f"输出: {result.stdout}")
        """
        # 将命令中的 "python" 替换为当前 Python 解释器路径，确保使用正确的 conda 环境
        # 这样可以避免使用系统默认 Python 导致的环境不一致问题
        resolved_command = [
            _CURRENT_PYTHON if cmd == "python" else cmd
            for cmd in command
        ]

        # dry-run 模式：只打印命令，不执行
        if dry_run:
            return ExperimentResult(
                command=command,
                returncode=0,
                stdout=f"dry-run: {' '.join(resolved_command)}",
                stderr=""
            )

        # 实际执行命令
        completed = subprocess.run(
            resolved_command,  # 使用解析后的命令（替换了 python 路径）
            cwd=cwd,           # 工作目录
            text=True,         # 输出为文本模式
            capture_output=True,  # 捕获 stdout 和 stderr
            check=False,       # 不抛出异常，返回退出码
            encoding="utf-8"   # 显式指定编码，避免 Windows 上的编码问题
        )
        
        return ExperimentResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
