# 外部输入说明

此模块不包含 Silver 数据、原始 CSV、Parquet、模型权重或 DCS 配置。运行数据工程前需要在同一工程中准备：

- `assets/data/xingrong_3/silver/by_point/`：按点位保存的 Silver Parquet；
- `assets/profiles/xingrong_3/points/point_classification.xlsx`：唯一的点位和模型变量语义来源。

若单独交接 `offline/forecast`，将上述两项通过受控存储提供，并在 `data_engineering/config/default_config.json` 中修改本地输入路径。数据到位后依次执行数据构造、时间切分和训练；大文件产物写入 `offline/artifacts/forecast/` 或 `WISBURN_FORECAST_ARTIFACT_ROOT` 指定的位置。
