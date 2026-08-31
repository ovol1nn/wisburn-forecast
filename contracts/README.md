# 预测模型契约

预测模型可交接的最小契约由 `metadata.json`、`scaler.json`、训练配置和本目录的 [current_contracts.json](current_contracts.json) 共同组成。

- `feature_columns` 的内容和顺序必须与 scaler 完全一致；列名相同但顺序不同不能接入。
- `target_columns`、`target_indices`、采样步长、输入长度、预测长度必须逐项匹配。
- 模型 1 原生输出为 60 个十秒节点；模型 2 原生输出为 600 个逐秒节点。两者进入控制链前统一成真实的 60 个十秒节点，原始轨迹保留审计。
- 交付文件、checksum、ONNX 对齐和离线验收要求以 [../training/TRAINING_SPEC.md](../training/TRAINING_SPEC.md) 为准。

该契约仅说明预测数据边界，不授权自动控制。部署运行时仍只支持离线回放、SHADOW 和 RECOMMEND。
