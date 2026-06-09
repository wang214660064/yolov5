from dataclasses import dataclass
from pathlib import Path

from project.nozzle_inspection.utils.image_hash import sha256_file


@dataclass(frozen=True)
class DuplicateReport:
    keep_files: list[Path]
    duplicate_files: list[Path]
    duplicate_pairs: list[tuple[Path, Path]]


class Deduplicator:
    """负责数据集精确去重和后续视觉去重扩展。"""

    def find_exact_duplicates(self, image_paths: list[Path]) -> DuplicateReport:
        groups: dict[str, list[Path]] = {}
        for path in sorted(Path(p) for p in image_paths):
            groups.setdefault(sha256_file(path), []).append(path)

        keep_files: list[Path] = []
        duplicate_files: list[Path] = []
        duplicate_pairs: list[tuple[Path, Path]] = []
        for paths in groups.values():
            keeper = paths[0]
            keep_files.append(keeper)
            for duplicate in paths[1:]:
                duplicate_files.append(duplicate)
                duplicate_pairs.append((keeper, duplicate))

        return DuplicateReport(
            keep_files=sorted(keep_files),
            duplicate_files=sorted(duplicate_files),
            duplicate_pairs=duplicate_pairs,
        )
