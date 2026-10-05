# 兴蓉 3# 炉模型训练与交付规范

本文是外部协作者训练后能够被本项目直接接入的统一契约。除非项目方书面确认新版本适配方案，否则所有“必须”项都不得变更。

## 1. 参考实现与数据边界

- 训练源码：本目录的 `run_chunked_tslib_training.py`、`chunked_tslib_dataset.py`，以及上级 `../tslib/` 中已注册的 `custom_chunked` 数据入口。
- 数据工程：`offline/forecast/data_engineering/build_model_datasets.py`。输入为点位级 Silver Parquet；输出为带分块清单和时间顺序切分的训练帧。
- 外部交付的数据包不应复制到仓库：仅使用其 `package_manifest.csv`、`SHA256SUMS.txt`、`README_数据说明.md`、点位分类说明进行校验和追溯。
- 训练必须使用分块 Parquet 和窗口索引；禁止把全年数据拼接成单个 CSV 宽表。

## 2. 当前研究训练配置：模型 1 与模型 2，10 秒原生预测

| 项目 | 固定值 |
| --- | --- |
| 项目 | 模型 1 | 模型 2 |
| --- | --- | --- |
| 数据集标识 | `model1_10s_seq120_pred60` | `model2_10s_seq120_pred60` |
| 模型架构 | Time-Series-Library `Crossformer` | Time-Series-Library `Crossformer` |
| 原生时间粒度 | 10 秒 | 10 秒 |
| 输入窗口 | `seq_len=120`（20 分钟） | `seq_len=120`（20 分钟） |
| 解码上下文 | `label_len=60`（10 分钟） | `label_len=60`（10 分钟） |
| 预测窗口 | `pred_len=60`（10 分钟） | `pred_len=60`（10 分钟） |
| 目标数 | 14 | 5 |
| 输入/输出维度 | 以各自 `metadata.json` 中的特征数为准 | 以各自 `metadata.json` 中的特征数为准 |
| 时间特征维度 | 以当前 TSLib 数据适配器为准 | 以当前 TSLib 数据适配器为准 |
| 训练结构与超参数 | 以 `run_chunked_tslib_training.py` 的 `model1_10s` profile 为准 | 以该脚本的 `model2_10s` profile 为准 |
| GPU | 默认单卡 0；多卡需按 VM 实际设备显式配置 | 默认单卡 0；多卡需按 VM 实际设备显式配置 |

模型 1 当前目标为主蒸汽流量、锅炉氧量/温度相关点位、三路汽包水位、炉膛负压 A/B、主蒸汽压力 A/B、一次风预热器出口温度和主蒸汽温度；模型 2 当前为五个烟气相关目标。准确列名、顺序和索引以每次训练生成的 `metadata.json` 为唯一依据，不可复用旧版五目标索引。

每个模型的输入特征及其顺序以训练生成的 `metadata.json` 为准。`scaler.json` 的列顺序必须与 `metadata.json.feature_columns` 完全一致。

## 3. 数据与时间处理要求

1. 源数据保留原始 1 秒采样的审计能力；本模型的数据工程必须按右端点聚合为 10 秒点后训练。
2. 训练和验证/测试集按时间顺序切分，并在边界保留 purge gap，不能随机打乱跨时间窗口。
3. 所有输入窗口与预测窗口必须连续、有效；停炉、检修、集体异常、目标缺失和低信息输入应沿用数据工程的过滤规则。
4. 本轮两套离线训练数据已经是 10 秒粒度；不得再次聚合或将模型 2 当作逐秒预测。
5. 模型输出为 60 个 10 秒节点。为兼容旧展示接口生成的 600 点零阶保持序列必须标记 `synthetic_upsample=true`，且不能作为策略或 MPC 输入。

## 4. 允许与不允许的变化

允许在不改变本契约的前提下调整训练轮数、batch size、学习率、随机种子或早停点，并在交付说明中记录。

以下变化会影响旧部署契约，必须先由项目方实现并验收适配。本节所列新目标和模型 2 十秒配置属于研究训练口径，不代表已具备部署兼容性：

- 改变 `freq_seconds`、`seq_len`、`label_len` 或 `pred_len`；
- 改变模型类别、输入维度、特征列或列顺序；
- 改变目标列、目标索引或 scaler 列顺序；
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

1. `feature_columns`、`target_columns` 和索引与当前点位配置、训练帧及 scaler 一致；不得套用旧版固定维数和索引。
2. 模型 1 恰有 14 个目标，模型 2 恰有 5 个目标，并核对 metadata 中的实际名称和顺序。
3. 每个模型都应使用 metadata 记录的 `n_vars` 做推理输入/输出维度检查，并按对应 `target_indices` 抽取 14 个或 5 个目标。
4. `freq_seconds=10`，预测时间轴从预测基准点后 10 秒开始，连续 60 点并覆盖未来 10 分钟。
5. `scaler.json` 的列、顺序、统计量和 checkpoint 的训练特征完全匹配。
6. checksum 全部通过；提供用于复现的代码 commit、Python/PyTorch/CUDA 版本、GPU 型号及训练命令。

## 7. 接入边界

模型交付不等于自动控制授权。项目后端会先加载 bundle、验证其对应契约，并在 shadow/建议模式下回放。历史发布模型与当前研究训练版本必须分开管理；不得新增或启用真实 DCS 写入。
