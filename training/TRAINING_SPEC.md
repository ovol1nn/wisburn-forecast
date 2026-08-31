# 兴蓉 3# 炉模型训练与交付规范

本文是外部协作者训练后能够被本项目直接接入的统一契约。除非项目方书面确认新版本适配方案，否则所有“必须”项都不得变更。

## 1. 参考实现与数据边界

- 训练源码：本目录的 `run_chunked_tslib_training.py`、`chunked_tslib_dataset.py`，以及上级 `../tslib/` 中已注册的 `custom_chunked` 数据入口。
- 数据工程：`offline/forecast/data_engineering/build_model_datasets.py`。输入为点位级 Silver Parquet；输出为带分块清单和时间顺序切分的训练帧。
- 外部交付的数据包不应复制到仓库：仅使用其 `package_manifest.csv`、`SHA256SUMS.txt`、`README_数据说明.md`、点位分类说明进行校验和追溯。
- 训练必须使用分块 Parquet 和窗口索引；禁止把全年数据拼接成单个 CSV 宽表。

## 2. 当前基准模型：模型 1，10 秒原生预测

| 项目 | 固定值 |
| --- | --- |
| 模型标识 | `model1` / `model1_10s_seq120_pred60` |
| 模型架构 | Time-Series-Library `Crossformer` |
| 原生时间粒度 | 10 秒 |
| 输入窗口 | `seq_len=120`（20 分钟） |
| 解码上下文 | `label_len=60`（10 分钟） |
| 预测窗口 | `pred_len=60`（10 分钟） |
| 输入/输出维度 | `enc_in=dec_in=c_out=300` |
| 目标列 | `B3_F80.PV`、`B3QT00`、`B3_O2_S1`、`B3_TE1119`、`B3_GIC01` |
| 目标索引 | `[28, 29, 30, 31, 32]` |
| 时间特征维度 | `6` |
| 训练结构 | `d_model=192`、`n_heads=6`、`e_layers=2`、`d_layers=1`、`d_ff=384`、`factor=3`、`patch_len=8` |
| 训练超参数 | `batch_size=128`、`learning_rate=2e-5`、`epochs=60`、`patience=8`、AMP、多 GPU `0..7` |
| 目标损失增强 | `loss_trend_weight=0.25`、`loss_endpoint_weight=0.5` |

完整的 300 个特征及其顺序以训练生成的 `metadata.json` 为准。`scaler.json` 的列顺序必须与 `metadata.json.feature_columns` 完全一致。

## 3. 数据与时间处理要求

1. 源数据保留原始 1 秒采样的审计能力；本模型的数据工程必须按右端点聚合为 10 秒点后训练。
2. 训练和验证/测试集按时间顺序切分，并在边界保留 purge gap，不能随机打乱跨时间窗口。
3. 所有输入窗口与预测窗口必须连续、有效；停炉、检修、集体异常、目标缺失和低信息输入应沿用数据工程的过滤规则。
4. 运行时若读取到 1 秒数据，由后端适配器聚合为 120 个 10 秒点；离线数据已经是 10 秒粒度时不得再次聚合。
5. 模型输出为 60 个 10 秒节点。为兼容旧展示接口生成的 600 点零阶保持序列必须标记 `synthetic_upsample=true`，且不能作为策略或 MPC 输入。

## 4. 允许与不允许的变化

允许在不改变本契约的前提下调整训练轮数、batch size、学习率、随机种子或早停点，并在交付说明中记录。

以下变化会导致不能直接部署，必须先由项目方实现并验收适配：

- 改变 `freq_seconds`、`seq_len`、`label_len` 或 `pred_len`；
- 改变模型类别、输入维度、300 个特征的任意列或列顺序；
- 改变五个目标列、目标索引或 scaler 列顺序；
- 输出不是完整的 60 个 10 秒节点，或将其伪装为真实逐秒预测；
- 缺少 checkpoint、metadata、scaler、配置或版本/训练命令记录。

## 5. 必须交付的训练产物

协作者通过共享存储、对象存储或受控文件传输交付一个版本目录；不要把大文件推入常规 Git。目录至少包含：

```text
model1_xingrong3_<version>/
  checkpoint.pth
  metadata.json
  scaler.json
  tslib_chunked_config.yaml
  last_train_command.txt
  metrics.json                 # 训练/验证/测试指标和每目标指标
  model_card.md                # 数据范围、代码 commit、随机种子、已知限制
  pred.npy                     # 建议提供；测试集预测
  true.npy                     # 建议提供；对应真实值
  SHA256SUMS.txt               # 上述文件的 SHA-256
```

`metadata.json` 必须至少含有：`model_slug`、`freq_seconds`、`seq_len`、`label_len`、`pred_len`、`feature_columns`、`target_columns`、`target_indices`、`n_vars`、训练/验证/测试切分摘要。若提供 ONNX，还必须提供 `model.onnx` 和 `onnx_info.json`，其中记录输入/输出 shape、opset 和导出命令。

## 6. 交付前自检

协作者须提供以下检查结果：

1. `feature_columns` 恰为 300 项，与基准 metadata 的顺序逐项一致。
2. 五个目标列及 `[28, 29, 30, 31, 32]` 与本规范一致。
3. 一个 `(batch, 120, 300)` 输入可推理，得到对应的未来 `(batch, 60, 300)` 输出；五个目标从索引 28–32 正确抽取。
4. `freq_seconds=10`，预测时间轴从预测基准点后 10 秒开始，连续 60 点并覆盖未来 10 分钟。
5. `scaler.json` 的列、顺序、统计量和 checkpoint 的训练特征完全匹配。
6. checksum 全部通过；提供用于复现的代码 commit、Python/PyTorch/CUDA 版本、GPU 型号及训练命令。

## 7. 接入边界

模型交付不等于自动控制授权。项目后端会先加载 bundle、验证本规范，并在 shadow/建议模式下回放。模型 1 与旧模型 2 会各自预测，随后统一到真实 10 秒控制网格；MPC 的 5 分钟和 10 分钟节点分别对应约第 30、60 个 10 秒节点。不得新增或启用真实 DCS 写入。
