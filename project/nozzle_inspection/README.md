# YOLOv5 喷头检测项目

项目只保留一个脚本配置和一个固定入口：

```text
run_config.py -> run_project.py -> train.py / val.py
```

## 使用方式

1. 修改 `project/nozzle_inspection/run_config.py` 中的 `CONFIG`。
2. 激活环境并运行固定入口：

```powershell
conda activate yolov5
python -u run_nozzle_project.py
```

在 IDE 中调试 `run_nozzle_project.py` 时，可以直接进入 YOLOv5 的 `train.py`、`val.py` 及下游源码。

`action` 支持：

- `prepare_data`：数据准备、去重和重新划分。
- `analyze_data`：生成数据分析结果。
- `write_config`：生成 `dataset.yaml`。
- `train`：直接调用 `train.run()`。
- `val`：直接调用 `val.run()`，并按配置导出 BadCase。
