"""
标签转换模块 - 多类别标签归并

该模块负责将原始多类别标签转换为 NG/OK 两类标签，用于简化分类任务。

类别映射规则：
- NG（不合格）：喷头有问题的情况
  - nozzle_tip_wrapped: 喷头缠绕
  - nozzle_no_extrusion: 无法挤出
  - nozzle_heavy_contamination: 严重污染
  - nozzle_slight_contamination: 轻微污染
  - camera_occlusion: 相机遮挡

- OK（合格）：喷头正常的情况
  - nozzle_clean: 喷头干净
  - nozzle_extrusion_normal: 挤出正常

YOLO 标签格式：
  每行格式: class_id x_center y_center width height
  所有坐标都是归一化的（0-1范围）

使用示例：
    converter = LabelConverter()
    converter.convert_file(
        Path("input.txt"), 
        Path("output.txt")
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# NG 类别的样本前缀列表（不合格情况）
NG_PREFIXES = (
    "nozzle_tip_wrapped",           # 喷头缠绕
    "nozzle_no_extrusion",          # 无法挤出
    "nozzle_heavy_contamination",   # 严重污染
    "nozzle_slight_contamination",  # 轻微污染
    "camera_occlusion",             # 相机遮挡
)

# OK 类别的样本前缀列表（合格情况）
OK_PREFIXES = (
    "nozzle_clean",                 # 喷头干净
    "nozzle_extrusion_normal",      # 挤出正常
)


@dataclass
class LabelConverter:
    """
    标签转换器，将原始多类别标签归并为 NG/OK 两类
    
    属性：
        ng_prefixes: NG 类别的样本前缀元组
        ok_prefixes: OK 类别的样本前缀元组
    
    方法：
        class_id_for_stem(): 根据文件名前缀获取类别ID
        convert_lines(): 转换标签行
        convert_file(): 转换标签文件
    """

    ng_prefixes: tuple[str, ...] = field(default_factory=lambda: NG_PREFIXES)
    ok_prefixes: tuple[str, ...] = field(default_factory=lambda: OK_PREFIXES)

    def class_id_for_stem(self, stem: str) -> int:
        """
        根据文件名前缀获取类别ID
        
        参数：
            stem: 文件名（不含扩展名）
        
        返回值：
            int: 类别ID（0=NG, 1=OK）
        
        映射规则：
            - 如果文件名以 NG_PREFIXES 中的前缀开头，返回 0
            - 如果文件名以 OK_PREFIXES 中的前缀开头，返回 1
            - 否则抛出 ValueError
        
        示例：
            class_id_for_stem("nozzle_clean_001") -> 1 (OK)
            class_id_for_stem("nozzle_tip_wrapped_002") -> 0 (NG)
        """
        # 检查是否匹配 NG 类别
        for prefix in self.ng_prefixes:
            if stem.startswith(prefix):
                return 0  # NG 类别
        
        # 检查是否匹配 OK 类别
        for prefix in self.ok_prefixes:
            if stem.startswith(prefix):
                return 1  # OK 类别
        
        # 如果都不匹配，抛出异常
        raise ValueError(f"未知样本前缀，无法映射 NG/OK：{stem}")

    def convert_lines(self, stem: str, lines: Iterable[str]) -> list[str]:
        """
        转换标签行，将多类别标签转换为 NG/OK 两类
        
        参数：
            stem: 文件名（不含扩展名），用于确定类别
            lines: 原始标签行迭代器
        
        返回值：
            list[str]: 转换后的标签行列表
        
        标签格式说明：
            原始格式: 原始类别ID x_center y_center width height
            转换格式: NG/OK类别ID x_center y_center width height
            所有坐标必须在 0-1 范围内
        
        异常情况：
            - 每行必须有且仅有 5 个元素
            - 坐标值必须是有效的浮点数
            - 坐标值必须在有效范围内（x,y: 0-1, w,h: 0-1且大于0）
        """
        # 根据文件名确定新的类别ID
        class_id = self.class_id_for_stem(stem)
        converted: list[str] = []
        
        # 遍历每一行标签
        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 按空格分割
            parts = stripped.split()
            
            # 检查格式是否正确（必须有5个元素）
            if len(parts) != 5:
                raise ValueError(f"标签格式非法：{stem} 第 {line_number} 行")
            
            # 解析坐标值
            try:
                # 忽略原始类别ID，使用新的 class_id
                _, x, y, w, h = parts
                values = [float(x), float(y), float(w), float(h)]
            except ValueError as exc:
                raise ValueError(f"标签数值非法：{stem} 第 {line_number} 行") from exc
            
            # 验证坐标是否在有效范围内
            if not self._is_valid_box(values):
                raise ValueError(f"标签坐标非法：{stem} 第 {line_number} 行")
            
            # 构建转换后的标签行（保留6位小数精度）
            converted.append(
                f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}\n"
            )
        
        return converted

    def convert_file(self, label_path: Path, output_path: Path) -> int:
        """
        转换标签文件
        
        参数：
            label_path: 原始标签文件路径
            output_path: 转换后标签文件输出路径
        
        返回值：
            int: 转换的标签数量
        
        执行步骤：
            1. 读取原始标签文件
            2. 转换每行标签
            3. 确保输出目录存在
            4. 写入转换后的标签文件
        """
        # 读取原始标签文件
        lines = label_path.read_text(encoding="utf-8").splitlines()
        
        # 转换标签
        converted = self.convert_lines(label_path.stem, lines)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入转换后的标签文件
        output_path.write_text("".join(converted), encoding="utf-8")
        
        # 返回转换的标签数量
        return len(converted)

    @staticmethod
    def _is_valid_box(values: list[float]) -> bool:
        """
        验证 YOLO 边界框坐标是否有效
        
        参数：
            values: [x_center, y_center, width, height]
        
        返回值：
            bool: True 表示有效，False 表示无效
        
        验证规则：
            - x_center: 0.0 <= x <= 1.0
            - y_center: 0.0 <= y <= 1.0
            - width: 0.0 < w <= 1.0 （宽度必须大于0）
            - height: 0.0 < h <= 1.0 （高度必须大于0）
        """
        x, y, w, h = values
        return (0.0 <= x <= 1.0 and 
                0.0 <= y <= 1.0 and 
                0.0 < w <= 1.0 and 
                0.0 < h <= 1.0)
