# 外部输入说明

此模块不包含 Silver 数据、原始 CSV、Parquet、模型权重或 DCS 配置。运行数据工程前需要在同一工程中准备：

- `assets/data/xingrong_3/silver/by_point/point_code=<点位编码>/data.parquet`：按点位保存的 Silver Parquet；
- `assets/profiles/xingrong_3/points/point_classification.xlsx`：唯一的点位和模型变量语义来源。

若单独交接 `offline/forecast`，将上述两项通过受控存储提供，并通过环境变量 `WISBURN_FORECAST_POINT_ROOT`、`WISBURN_FORECAST_CLASSIFICATION_XLSX` 配置输入。数据到位后依次执行数据构造、时间切分和训练；大文件产物写入 `offline/artifacts/forecast/` 或 `WISBURN_FORECAST_ARTIFACT_ROOT` 指定的位置。

当前发布模型仍只包含原有两组各五个预测目标。兴蓉 Profile 中新增的压力、温度、负压、三路水位和一次风温度需先检查历史覆盖和点位质量，再重建数据集、训练和验收；仅同步 Silver 文件不会让现有 bundle 自动具备这些输出。
