# 预测模型契约

预测模型可交接的最小契约由 `metadata.json`、`scaler.json`、训练配置和本目录的 [current_contracts.json](current_contracts.json) 共同组成。

- `feature_columns` 的内容和顺序必须与 scaler 完全一致；列名相同但顺序不同不能接入。
- `target_columns`、`target_indices`、采样步长、输入长度、预测长度必须逐项匹配。
- 本目录 `current_contracts.json` 描述 Profile 当前引用的正式 bundle：Model 1 为原生十秒、5个旧目标；Model 2 为原生逐秒、5个排放目标。现场只读影子候选另由 `contracts/configuration/model_contracts.json` 描述：Model 1 为304输入/14物理输出，Model 2 为113输入/5排放输出；两者均为10秒、120→60，Model 2 训练策略为 `no_future_covariates`。候选已接入 SHADOW 预测，不会自动替换正式 bundle 或解除 `retrain_required`。控制轨迹在需要时统一为十秒节点，平均目标由控制层合成。
- 交付文件、checksum、ONNX 对齐和离线验收要求以 [../training/TRAINING_SPEC.md](../training/TRAINING_SPEC.md) 为准。

该契约仅说明预测数据边界，不授权自动控制。部署运行时仍只支持离线回放、SHADOW 和 RECOMMEND。
