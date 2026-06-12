# 数据集分析报告

## 初始数据总体情况

初始数据集按 `train`、`val`、`test` 三个子集存放，共有图片 4699 张，标签文件 4699 个。

| 子集 | 图片数量 | 标签数量 | OK 数量 | NG 数量 |
| --- | ---: | ---: | ---: | ---: |
| train | 3289 | 3289 | 1692 | 1597 |
| val | 704 | 704 | 364 | 340 |
| test | 706 | 706 | 383 | 323 |
| 合计 | 4699 | 4699 | 2439 | 2260 |

说明：

- OK 由 `nozzle_clean`、`nozzle_extrusion_normal` 合并得到。
- NG 由 `camera_occlusion`、`nozzle_heavy_contamination`、`nozzle_no_extrusion`、`nozzle_slight_contamination`、`nozzle_tip_wrapped` 合并得到。
- 从 OK/NG 二分类角度看，总体数量接近均衡；从原始子类角度看，部分类别样本极少，例如 `camera_occlusion`、`nozzle_slight_contamination`、`nozzle_heavy_contamination`。
- 初始 `train + val` 候选样本为 3993 张，后续数据准备流程会对该部分做去重，并重新划分训练集和验证集；`test` 保持独立用于最终评估。

## train

- 图片数量：3289
- 标签数量：3289

| 前缀 | 归并标签 | 数量 |
| --- | --- | ---: |
| camera_occlusion | NG | 1 |
| nozzle_clean | OK | 645 |
| nozzle_extrusion_normal | OK | 1047 |
| nozzle_heavy_contamination | NG | 12 |
| nozzle_no_extrusion | NG | 633 |
| nozzle_slight_contamination | NG | 4 |
| nozzle_tip_wrapped | NG | 947 |

## val

- 图片数量：704
- 标签数量：704

| 前缀 | 归并标签 | 数量 |
| --- | --- | ---: |
| nozzle_clean | OK | 136 |
| nozzle_extrusion_normal | OK | 228 |
| nozzle_heavy_contamination | NG | 3 |
| nozzle_no_extrusion | NG | 148 |
| nozzle_slight_contamination | NG | 2 |
| nozzle_tip_wrapped | NG | 187 |

## test

- 图片数量：706
- 标签数量：706

| 前缀 | 归并标签 | 数量 |
| --- | --- | ---: |
| nozzle_clean | OK | 140 |
| nozzle_extrusion_normal | OK | 243 |
| nozzle_heavy_contamination | NG | 6 |
| nozzle_no_extrusion | NG | 133 |
| nozzle_slight_contamination | NG | 2 |
| nozzle_tip_wrapped | NG | 182 |
