# 外部输入说明

此模块不包含 Silver 数据、原始 CSV、Parquet、模型权重或 DCS 配置。运行数据工程前需要在同一工程中准备：

- `assets/data/xingrong_3/silver/by_point/point_code=<点位编码>/data.parquet`：按点位保存的 Silver Parquet；
- `assets/profiles/xingrong_3/points/point_classification.xlsx`：唯一的点位和模型变量语义来源。

若单独交接 `offline/forecast`，将上述两项通过受控存储提供，并通过环境变量 `WISBURN_FORECAST_POINT_ROOT`、`WISBURN_FORECAST_CLASSIFICATION_XLSX` 配置输入。数据到位后依次执行数据构造、时间切分和训练；大文件产物写入 `offline/artifacts/forecast/` 或 `WISBURN_FORECAST_ARTIFACT_ROOT` 指定的位置。

当前发布模型仍各含5个旧预测目标。新训练数据集已按统一10秒口径构建：模型1含14个物理输出点，模型2仍输出5个排放指标；下一步是训练、逐目标验收并生成新 bundle。仅同步 Silver 文件不会让现有发布 bundle 自动具备新增输出，训练输入必须与本轮 metadata、点位表版本和时间切分一致。
