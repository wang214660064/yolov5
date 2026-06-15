# 3D 打印机喷头检测项目

基于 YOLOv5 的 3D 打印机喷头缺陷检测（NG/OK 分类）项目，提供完整的数据处理、模型训练和验证流程。

## 🚀 功能特点

- **数据集分析**：自动分析数据集结构、样本分布、类别统计
- **数据预处理**：去重、标签转换、数据集划分
- **模型训练**：基于 YOLOv5 的端到端训练流程
- **模型验证**：自动评估模型性能

## 📁 项目结构

```
project/nozzle_inspection/
├── configs/              # 配置文件
│   ├── augmentation.yaml # 数据增强配置
│   ├── dataset.yaml      # 数据集配置
│   └── train_ng_ok.yaml  # 训练超参数
├── data/                 # 数据处理模块
│   ├── dataset_analyzer.py    # 数据集分析器
│   ├── dataset_preparer.py    # 数据集准备器
│   ├── deduplicate.py         # 去重工具
│   ├── label_converter.py     # 标签转换
│   ├── scale_augment.py       # 尺度增强
│   └── split_dataset.py       # 数据集划分
├── evaluation/           # 评估模块
│   ├── error_analyzer.py      # 错误分析
│   └── evaluate_pipeline.py   # 评估流程
├── factories/            # 工厂模式模块
│   ├── augmentation_factory.py
│   ├── data_factory.py
│   ├── evaluator_factory.py
│   ├── model_factory.py
│   └── trainer_factory.py
├── training/             # 训练模块
│   └── experiment_runner.py
├── utils/                # 工具模块
├── main.py               # 命令行入口
├── run_config.py         # 脚本运行配置
└── run_project.py        # 项目运行脚本
```

## 🛠️ 环境要求

- Python 3.12+
- Conda 环境：`yolov5`

安装依赖：

```bash
conda activate yolov5
pip install -r yolov5/requirements.txt
```

## 📖 使用方法

### 方式一：命令行模式

从 `yolov5` 目录运行：

```bash
conda activate yolov5
python -m project.nozzle_inspection.main --help
```

### 方式二：脚本模式（推荐）

修改 `run_config.py` 中的配置后运行：

```bash
conda activate yolov5
python -u run_nozzle_project.py
```

## 📋 命令参考

### 1. 数据分析

```bash
python -m project.nozzle_inspection.main analyze-data \
  --dataset ../dataset_2 \
  --output project/nozzle_inspection/data/dataset_report.md
```

### 2. 数据准备

```bash
python -m project.nozzle_inspection.main prepare-data \
  --dataset ../dataset_2 \
  --output ../outputs/datasets/nozzle_ng_ok_v1 \
  --val-ratio 0.2 \
  --seed 42
```

### 3. 生成配置

```bash
python -m project.nozzle_inspection.main write-config \
  --output project/nozzle_inspection/configs/dataset.yaml \
  --path ../outputs/datasets/nozzle_ng_ok_v1
```

### 4. 训练模型

```bash
python -m project.nozzle_inspection.main train \
  --data project/nozzle_inspection/configs/dataset.yaml \
  --hyp project/nozzle_inspection/configs/train_ng_ok.yaml \
  --epochs 50
```

### 5. 验证模型

```bash
python -m project.nozzle_inspection.main val \
  --data project/nozzle_inspection/configs/dataset.yaml \
  --weights runs/train/nozzle_ng_ok/weights/best.pt \
  --conf 0.25
```

## 🔄 工作流程

```
1. 数据分析 → 2. 数据准备 → 3. 生成配置 → 4. 训练 → 5. 验证
    ↓              ↓              ↓           ↓         ↓
analyze-data   prepare-data   write-config   train     val
```

## ⚙️ 配置说明

### 默认配置（run_config.py）

| 参数             | 默认值                                      | 说明           |
| ---------------- | ------------------------------------------- | -------------- |
| `action`       | `prepare_data`                            | 运行动作       |
| `epochs`       | `3`                                       | 训练轮数       |
| `data_yaml`    | `configs/dataset.yaml`                    | 数据集配置路径 |
| `hyp_yaml`     | `configs/train_ng_ok.yaml`                | 超参数配置路径 |
| `weights`      | `runs/train/nozzle_ng_ok/weights/best.pt` | 模型权重       |
| `conf`         | `0.7`                                     | 置信度阈值     |
| `dataset_root` | `../dataset_2`                            | 原始数据集目录 |
| `val_ratio`    | `0.2`                                     | 验证集比例     |

## 📊 输出结构

```
outputs/
├── datasets/
│   └── nozzle_ng_ok_v1/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
```

## 🚀 快速开始

```bash
# 进入项目目录
cd "E:\Desktop\MAC-WIN\04 OpenCV\98_Practice\3DPrinterNozzleInspection\yolov5"
conda activate yolov5

# 分析数据集
python -m project.nozzle_inspection.main analyze-data

# 准备数据
python -m project.nozzle_inspection.main prepare-data

# 生成配置
python -m project.nozzle_inspection.main write-config

# 训练模型
python -u -m project.nozzle_inspection.main train --epochs 50
```

## 📝 许可证

MIT License

---

*项目维护：3D 打印机喷头检测团队*
