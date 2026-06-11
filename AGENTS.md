# 项目说明

## 运行环境

使用项目 conda 环境进行所有 Python 检查、脚本和测试：

```bash
conda run -n yolov5 python ...
```

除非用户明确要求使用其他环境，否则不要使用系统 Python、基础 conda 环境或 Codex 捆绑的 Python 运行时进行项目级测试。

## 基本检查

进行轻量级代码验证，运行：

```bash
conda run -n yolov5 -m py_compile
```

进行依赖检查，优先使用相同环境：

```bash
conda run -n yolov5 python -c "import torch, cv2, openpyxl; print(torch.__version__, cv2.__version__)"
```

## 注意事项

- 除非用户要求更新生成文件，否则将生成的报告和训练输出排除在源代码编辑之外。
- 全部交流和注释用中文。
