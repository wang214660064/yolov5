from dataclasses import dataclass
import subprocess


@dataclass(frozen=True)
class ExperimentResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class ExperimentRunner:
    """执行训练或验证命令；默认支持 dry-run 便于复现实验命令。"""

    def run(self, command: list[str], dry_run: bool = False, cwd: str | None = None) -> ExperimentResult:
        if dry_run:
            return ExperimentResult(command=command, returncode=0, stdout=f"dry-run: {' '.join(command)}", stderr="")

        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
        return ExperimentResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
