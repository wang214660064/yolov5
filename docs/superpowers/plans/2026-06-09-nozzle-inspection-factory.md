# 3D打印机喷头检测工厂模式 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `project/nozzle_inspection/` 下实现 NG/OK 两类 3D 打印机喷头检测项目层，覆盖数据清洗、标签归并、去重、划分、增强、训练评估封装和报告生成。

**Architecture:** 保留 YOLOv5 核心源码，新增项目业务层与工厂层。数据和实验产物默认写入外层 `../outputs/`，代码、配置、测试和报告生成逻辑只放在 `yolov5/` 仓库中。

**Tech Stack:** Python 3、YOLOv5、OpenCV、PyYAML、标准库 `unittest`、可选 `python-pptx`。

---

## File Structure

- Create: `project/nozzle_inspection/__init__.py`，项目包入口。
- Create: `project/nozzle_inspection/utils/path_utils.py`，统一路径解析和目录创建。
- Create: `project/nozzle_inspection/utils/experiment_logger.py`，CSV/Markdown 报告写入。
- Create: `project/nozzle_inspection/utils/image_hash.py`，SHA256 与感知 HASH。
- Create: `project/nozzle_inspection/data/label_converter.py`，原始多类到 NG/OK 转换。
- Create: `project/nozzle_inspection/data/deduplicate.py`，精确去重和视觉近重复识别。
- Create: `project/nozzle_inspection/data/split_dataset.py`，合并 train/val 后分层划分。
- Create: `project/nozzle_inspection/data/dataset_analyzer.py`，数据质量与分布统计。
- Create: `project/nozzle_inspection/data/scale_augment.py`，缩放增强与框同步修复。
- Create: `project/nozzle_inspection/factories/*.py`，数据、增强、模型、训练、评估、报告工厂。
- Create: `project/nozzle_inspection/training/experiment_runner.py`，训练命令构建与执行。
- Create: `project/nozzle_inspection/evaluation/evaluate_pipeline.py`，验证命令构建。
- Create: `project/nozzle_inspection/reporting/report_builder.py`，Markdown 与 PPT 汇报文档生成。
- Create: `project/nozzle_inspection/main.py`，命令行入口。
- Create: `project/nozzle_inspection/configs/*.yaml`，默认配置。
- Create: `tests/nozzle_inspection/*.py`，核心行为测试。

## Task 1: 包结构与路径工具

**Files:**
- Create: `project/nozzle_inspection/__init__.py`
- Create: `project/nozzle_inspection/utils/path_utils.py`
- Create: `tests/nozzle_inspection/test_path_utils.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from project.nozzle_inspection.utils.path_utils import ProjectPaths


def test_project_paths_resolve_outer_dataset_and_outputs(tmp_path):
    repo_root = tmp_path / "yolov5"
    repo_root.mkdir()
    paths = ProjectPaths(repo_root=repo_root)

    assert paths.dataset_root == tmp_path / "dataset_2"
    assert paths.outputs_root == tmp_path / "outputs"
    assert paths.project_data_dir == repo_root / "project" / "nozzle_inspection" / "data"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_path_utils -v`

Expected: FAIL because `ProjectPaths` is not defined.

- [ ] **Step 3: Implement minimal path utility**

Create `ProjectPaths` with `repo_root` normalization and directory helpers. Use Chinese comments only where useful.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_path_utils -v`

Expected: PASS.

## Task 2: NG/OK 标签转换

**Files:**
- Create: `project/nozzle_inspection/data/label_converter.py`
- Create: `tests/nozzle_inspection/test_label_converter.py`

- [ ] **Step 1: Write failing tests**

```python
from project.nozzle_inspection.data.label_converter import LabelConverter


def test_convert_known_prefix_to_ng_and_ok():
    converter = LabelConverter()

    assert converter.class_id_for_stem("nozzle_tip_wrapped_0001") == 0
    assert converter.class_id_for_stem("nozzle_no_extrusion_0001") == 0
    assert converter.class_id_for_stem("nozzle_clean_0001") == 1
    assert converter.class_id_for_stem("nozzle_extrusion_normal_0001") == 1


def test_convert_label_lines_rewrites_class_and_keeps_box():
    converter = LabelConverter()
    lines = ["3 0.5 0.5 0.2 0.1\n"]

    converted = converter.convert_lines("nozzle_tip_wrapped_0001", lines)

    assert converted == ["0 0.500000 0.500000 0.200000 0.100000\n"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_label_converter -v`

Expected: FAIL because converter module is missing.

- [ ] **Step 3: Implement label converter**

Implement prefix mapping, YOLO row validation, unknown-prefix reporting through returned conversion result or raised `ValueError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_label_converter -v`

Expected: PASS.

## Task 3: 图片 HASH 与去重

**Files:**
- Create: `project/nozzle_inspection/utils/image_hash.py`
- Create: `project/nozzle_inspection/data/deduplicate.py`
- Create: `tests/nozzle_inspection/test_deduplicate.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from project.nozzle_inspection.data.deduplicate import Deduplicator


def test_exact_duplicate_keeps_first_file(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    a.write_bytes(b"same-image")
    b.write_bytes(b"same-image")

    report = Deduplicator().find_exact_duplicates([a, b])

    assert report.keep_files == [a]
    assert report.duplicate_files == [b]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_deduplicate -v`

Expected: FAIL because `Deduplicator` is not defined.

- [ ] **Step 3: Implement exact deduplication**

Implement deterministic SHA256 grouping. Keep the lexicographically first file in each group.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_deduplicate -v`

Expected: PASS.

## Task 4: 数据划分与文件复制

**Files:**
- Create: `project/nozzle_inspection/data/split_dataset.py`
- Create: `tests/nozzle_inspection/test_split_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
from project.nozzle_inspection.data.split_dataset import stratified_split


def test_stratified_split_preserves_both_classes():
    samples = [(f"ng_{i}", 0) for i in range(10)] + [(f"ok_{i}", 1) for i in range(10)]

    train, val = stratified_split(samples, val_ratio=0.2, seed=7)

    assert len(train) == 16
    assert len(val) == 4
    assert {label for _, label in val} == {0, 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_split_dataset -v`

Expected: FAIL because split function is missing.

- [ ] **Step 3: Implement split logic**

Implement deterministic class-wise shuffle and split. Keep labels paired with image stems.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_split_dataset -v`

Expected: PASS.

## Task 5: 缩放增强

**Files:**
- Create: `project/nozzle_inspection/data/scale_augment.py`
- Create: `tests/nozzle_inspection/test_scale_augment.py`

- [ ] **Step 1: Write failing tests**

```python
from project.nozzle_inspection.data.scale_augment import scale_yolo_box_on_canvas


def test_scale_yolo_box_keeps_coordinates_valid():
    result = scale_yolo_box_on_canvas(
        box=(0, 0.5, 0.5, 0.2, 0.2),
        scale=1.5,
        offset_x=0.0,
        offset_y=0.0,
    )

    cls, x, y, w, h = result
    assert cls == 0
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_scale_augment -v`

Expected: FAIL because scale augment module is missing.

- [ ] **Step 3: Implement bbox transform**

Implement normalized coordinate scaling, clipping and invalid-box filtering.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_scale_augment -v`

Expected: PASS.

## Task 6: 工厂与命令封装

**Files:**
- Create: `project/nozzle_inspection/factories/data_factory.py`
- Create: `project/nozzle_inspection/factories/augmentation_factory.py`
- Create: `project/nozzle_inspection/factories/model_factory.py`
- Create: `project/nozzle_inspection/factories/trainer_factory.py`
- Create: `project/nozzle_inspection/factories/evaluator_factory.py`
- Create: `project/nozzle_inspection/factories/report_factory.py`
- Create: `project/nozzle_inspection/training/experiment_runner.py`
- Create: `project/nozzle_inspection/evaluation/evaluate_pipeline.py`
- Create: `tests/nozzle_inspection/test_factories.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from project.nozzle_inspection.factories.trainer_factory import TrainerFactory


def test_trainer_command_uses_adamw_and_two_class_data(tmp_path):
    command = TrainerFactory(repo_root=tmp_path).build_train_command(
        data_yaml=Path("data/nozzle_ng_ok.yaml"),
        hyp_yaml=Path("data/hyps/nozzle_ng_ok.yaml"),
        epochs=3,
    )

    assert "train.py" in command
    assert "--optimizer" in command
    assert "AdamW" in command
    assert "data/nozzle_ng_ok.yaml" in command
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_factories -v`

Expected: FAIL because factory module is missing.

- [ ] **Step 3: Implement command factories**

Implement command-list builders without executing training by default.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_factories -v`

Expected: PASS.

## Task 7: 配置与报告生成

**Files:**
- Create: `project/nozzle_inspection/configs/dataset.yaml`
- Create: `project/nozzle_inspection/configs/train_ng_ok.yaml`
- Create: `project/nozzle_inspection/configs/augmentation.yaml`
- Create: `project/nozzle_inspection/configs/report.yaml`
- Create: `project/nozzle_inspection/reporting/report_builder.py`
- Create: `tests/nozzle_inspection/test_report_builder.py`

- [ ] **Step 1: Write failing tests**

```python
from project.nozzle_inspection.reporting.report_builder import ReportBuilder


def test_report_builder_renders_required_sections():
    report = ReportBuilder().build_markdown(
        title="3D打印机喷头检测项目报告",
        metrics={"NG_P": 0.98, "NG_R": 0.85},
    )

    assert "项目背景" in report
    assert "数据集分析" in report
    assert "结果分析" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_report_builder -v`

Expected: FAIL because report builder is missing.

- [ ] **Step 3: Implement report builder**

Generate Markdown. If `python-pptx` is installed, also generate PPTX; otherwise write a clear warning and keep Markdown report.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n yolo python -m unittest tests.nozzle_inspection.test_report_builder -v`

Expected: PASS.

## Task 8: CLI 与全量验证

**Files:**
- Create: `project/nozzle_inspection/main.py`
- Modify: `project/nozzle_inspection/__init__.py`

- [ ] **Step 1: Add CLI smoke test**

Run: `conda run -n yolo python -m project.nozzle_inspection.main --help`

Expected: command prints Chinese help text.

- [ ] **Step 2: Run all unit tests**

Run: `conda run -n yolo python -m unittest discover tests/nozzle_inspection -v`

Expected: PASS.

- [ ] **Step 3: Run syntax check**

Run: `conda run -n yolo python -m py_compile project/nozzle_inspection/main.py project/nozzle_inspection/data/label_converter.py project/nozzle_inspection/data/deduplicate.py project/nozzle_inspection/data/split_dataset.py project/nozzle_inspection/data/scale_augment.py project/nozzle_inspection/reporting/report_builder.py`

Expected: PASS with no output.

- [ ] **Step 4: Check Git status**

Run: `git status --short`

Expected: only intended new project/test/config/plan files plus pre-existing user changes.
