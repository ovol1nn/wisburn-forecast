# 预测模型契约

预测模型可交接的最小契约由 `metadata.json`、`scaler.json`、训练配置和本目录的 [current_contracts.json](current_contracts.json) 共同组成。

- `feature_columns` 的内容和顺序必须与 scaler 完全一致；列名相同但顺序不同不能接入。
- `target_columns`、`target_indices`、采样步长、输入长度、预测长度必须逐项匹配。
- 本目录 `current_contracts.json` 描述 Profile 当前引用的已发布 bundle：模型1为原生十秒、5个旧目标；模型2为原生逐秒、5个排放目标。新训练数据集的契约另见本轮 `training_ready/*/metadata.json`：两模型均为十秒，模型1输出14个物理点、模型2输出5个排放点，尚未训练验收并发布。控制链统一使用十秒节点，平均目标由控制层合成。
- 交付文件、checksum、ONNX 对齐和离线验收要求以 [../training/TRAINING_SPEC.md](../training/TRAINING_SPEC.md) 为准。

该契约仅说明预测数据边界，不授权自动控制。部署运行时仍只支持离线回放、SHADOW 和 RECOMMEND。
