# 训练入口

本目录是 `offline/forecast` 的训练实现：包含兴蓉 3# 炉预测模型的数据桥接、训练配置和原始 Crossformer 对照实现。它可随代码仓库维护；原始点位数据、分块训练集、checkpoint、预测数组和部署模型权重均不提交。

当前训练入口提供 Model 1、Model 2 两套十秒训练配置，均为过去20分钟预测未来10分钟（`120→60`）。Model 1 使用304个输入特征、输出14个物理点；Model 2 使用113个输入特征、输出5个烟气排放点，并按 `--no_future_covariates` 训练。两套候选已用于后端只读影子路径；Profile 的正式 `bundle_dir` 仍保持旧兼容包（Model 1 5个旧目标、Model 2逐秒5个排放目标），`retrain_required: true` 未解除。实际特征、目标顺序和索引必须以对应训练产物 `metadata.json` 为准。

## 目录职责

- `run_chunked_tslib_training.py`：训练入口；默认只生成训练命令，必须显式将 `INTERACTIVE_RUN_TRAINING` 改为 `True` 才会启动训练。
- `chunked_tslib_dataset.py`：把分块 Parquet 和滑动窗口清单适配为 Time-Series-Library 的 `custom_chunked` 数据集。
- `requirements_vm_training.txt`：训练环境的 Python 依赖。
- `original_crossformer/`：Crossformer 的原始参考实现与配置，保留作模型结构对照。
- 训练就绪产物：写入仓库外置式资产目录 `offline/artifacts/forecast/training_ready/`；其中的 `metadata.json`、`scaler.json` 和 YAML 是验收依据，权重和预测数组不进入 Git。

## 给协作者的最短流程

1. 按数据交付文档取得点位数据，并校验 `SHA256SUMS.txt`。
2. 用 `offline/forecast/data_engineering/build_model_datasets.py` 生成分块数据集；不要生成全年单体 CSV。
3. 运行本目录训练入口，先保持 `INTERACTIVE_RUN_TRAINING = False` 检查命令和 metadata，再显式启动训练。
4. 将训练结果按 `TRAINING_SPEC.md` 的交付清单提供给项目方；由项目方完成 bundle 校验和 SHADOW 环境接入。

默认路径适用于完整工程和单独检出的 `offline/forecast`。训练数据、checkpoint 与预测数组可通过 `WISBURN_FORECAST_ARTIFACT_ROOT` 指向 VM 的受控本地存储；若代码/数据路径不同，还需设置 `WISBURN_FORECAST_POINT_ROOT`、`WISBURN_FORECAST_CLASSIFICATION_XLSX` 和 `WISBURN_TSLIB_ROOT`。环境变量只指定本地文件路径，不涉及 DCS。

VM 迁移时，除 `offline/forecast/` 源码外，还要准备：点位分类 Excel、Silver 输入（或已生成且与新筛选规则一致的 10 秒分块数据集及 splits）。当前配置的停炉筛选已调整，因此旧数据集不能自动获得本次质量修正；建议在 VM 以新的 artifact 输出目录重建数据集和 splits，再运行训练入口。先保持 `INTERACTIVE_RUN_TRAINING = False` 生成并检查命令；默认单 GPU 0，若 VM 有多卡，再显式设置 `INTERACTIVE_USE_MULTI_GPU=True` 和实际设备列表。

## 不可替换的事实

- 模型 1 是 `120` 个 10 秒输入点预测未来 `60` 个 10 秒点，即过去 20 分钟预测未来 10 分钟。
- 它不是 1 秒模型；不得把 60 个输出插值或重复后称为真实的 600 个逐秒预测。
- 特征顺序、目标列、目标索引和 scaler 是一个整体。仅列名相同但顺序不同的模型不能接入。
