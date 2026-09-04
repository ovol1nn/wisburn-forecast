# 模型数据工程

本目录把平台数据存储工具生成的 Silver 点位 Parquet 处理成预测模型数据集。模型 1 使用 10 秒粒度、`120→60`；模型 2 使用 1 秒粒度、`1200→600`。实际参数以脚本顶部交互参数和生成的 metadata 为准。

## VSCode 直接运行

推荐直接打开 [build_model_datasets.py](build_model_datasets.py)，修改文件顶部的 `INTERACTIVE_*` 参数，然后在 VSCode 里点击运行 Python 文件。

常用参数：

```python
INTERACTIVE_ACTION = "build"      # check-env | list-variables | inspect-points | build
INTERACTIVE_MODEL = "子模型1"
INTERACTIVE_ALL_MODELS = False
INTERACTIVE_START = "2025-11-12"
INTERACTIVE_END = "2025-11-13"
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_CHUNKED = True
INTERACTIVE_CHUNK_DAYS = 1
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_PRED_LEN = 60
INTERACTIVE_WINDOW_STRIDE = 1
```

如果要跑全量时间段，把 `INTERACTIVE_START` 和 `INTERACTIVE_END` 改成空字符串 `""`。如果只想检查变量或覆盖范围，把 `INTERACTIVE_ACTION` 改成 `list-variables` 或 `inspect-points`。

注意：全年 1 秒粒度的宽表会非常大，尤其是子模型1这种 300+ 点位的配置。脚本会先用 `max_wide_table_raw_gb` 估算宽表规模，超过阈值时提前停止，避免 pandas 合并过程中耗尽内存。要构建全年 1 秒训练样本，请启用 `INTERACTIVE_CHUNKED = True`，脚本会按 `INTERACTIVE_CHUNK_DAYS` 分块写训练帧，并为每块保留 `seq_len + pred_len` 所需的未来重叠区。这样不会生成一个全年单体宽表。

如果 `INTERACTIVE_START` / `INTERACTIVE_END` 留空，分块输出写入全量目录，例如 `datasets/chunked/model2_1s_seq1200_pred300`。如果设置了短时间范围做测试，脚本会自动在目录名后增加时间后缀，避免覆盖全量 manifest。

当前模型 1（全年 10 秒）推荐参数：

```python
INTERACTIVE_MODEL = "子模型1"
INTERACTIVE_START = ""
INTERACTIVE_END = ""
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_CHUNKED = True
INTERACTIVE_CHUNK_DAYS = 1
INTERACTIVE_WRITE_CHUNK_WIDE = False
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_PRED_LEN = 60
```

## 当前处理逻辑

脚本保留原始重采样宽表，同时额外生成训练用过滤结果，避免检修段、停炉段、长期 0 值点位污染滑动窗口样本。

主要步骤：

1. 从配置的点位分类 Excel 中读取 `模型变量明细`；完整工程优先使用 Profile/数据资产提供的点位分类文件。
2. 逐点读取 Silver Parquet，按右端点重采样，写入 `offline/artifacts/forecast/data_engineering/outputs/cache/resampled_<freq>s`，支持断点续跑。
3. 拼接模型宽表，保留派生特征：一次风分段/占比、一次/二次风比例、推料器位置近似、炉排速度组均值等。
4. 生成点位质量报告，统计缺失率、近零率、常值率。
5. 对长期近零或常值的输入点位，写入低信息点报告，并从训练帧中剔除。
6. 生成 `valid_for_training` mask：排除停炉/检修候选、集体异常、目标缺失、有效变量比例过低的时间点。
7. 用 `valid_for_training` 生成滑动窗口索引，窗口长度由模型参数决定，并要求输入段、预测段有效且时间连续。

## 检修和异常段

停炉/检修候选由两类规则识别。

第一类是已知运行指标规则：

- 集体缺失率超过 `collective_missing_rate_warn`
- 主蒸汽流量 `B3FT80` 低于阈值
- 一次风流量 `B3FT00` 低于阈值
- 二次风流量 `B3FT07` 低于阈值

阈值在 [config/default_config.json](config/default_config.json) 的 `shutdown_indicators` 中调整。当前默认按新数据量级设置为 `5`。指标缺失本身不会直接判定为停炉，只有“有值且低于阈值”才触发。

第二类是更泛用的多变量共识低位规则 `long_common_low_anomaly`：

- 每个变量先按自身分布自适应计算低位阈值。
- 长期缺失、长期为 0、长期常值的点位不参与投票。
- 同一时间如果足够多变量同时处于低位，则形成共识低位得分。
- 只保留持续时间超过 `min_duration_seconds` 的长异常段，默认 `21600` 秒，即 6 小时。
- 允许中间短暂恢复并合并，默认 `bridge_gap_seconds = 3600` 秒。

这条规则用于识别你图中那类“一年内多个点位共同出现的长时间异常低位段”，不会把普通短时尖峰、短时下探直接视为异常工况。

触发停炉/检修候选后，会按 `training_exclusion_buffer_seconds` 做前后 buffer，默认 `3600` 秒，避免启停过渡段进入滑动窗口。

可以单独审计若干点位的全年共性异常：

```powershell
python "offline\forecast\data_engineering\build_model_datasets.py" audit-anomalies `
  --codes B3TE07D,B3TE19A,B3ZI86,B3TE05,B3_ST19 `
  --freq-seconds 300
```

审计输出：

- `reports/audit_common_low_windows_300s.csv`
- `reports/audit_common_low_candidates_300s.csv`
- `reports/audit_common_low_score_300s.parquet`

## 主要输出

默认输出目录为 `offline/artifacts/forecast/data_engineering/outputs`，不进入 Git。

- `cache/resampled_<freq>s/*.parquet`：逐点重采样缓存，可断点续跑。
- `datasets/model*_1s.parquet`：模型宽表，保留原始清洗后的行。
- `datasets/model*_training_frame_1s.parquet`：训练帧，包含 `valid_for_training`，并剔除长期近零/常值输入列。
- `datasets/chunked/model*_1s_seq1200_pred300/frames/*.parquet`：分块训练帧；全年 1 秒训练推荐读取这里，而不是生成单个全年宽表。
- `datasets/chunked/model*_1s_seq1200_pred300/windows/*_windows.csv`：每个分块对应的滑动窗口索引，窗口起点只落在该块核心时间段内。
- `datasets/chunked/model*_1s_seq1200_pred300/manifest.csv`：分块清单，记录每块的核心时间、加载时间、训练帧路径、窗口索引路径、有效行数和窗口数。
- `reports/model*_quality_1s.csv`：模型点位质量报告。
- `reports/low_information_inputs_model*_1s.csv`：长期近零/常值输入点位及剔除原因。
- `reports/unusable_outputs_model*_1s.csv`：缺失率过高、无有效值或长期常值的输出点位；这些点位会从训练目标中剔除，避免一个不可用输出让整段窗口全部无效。
- `reports/training_mask_model*_1s.parquet`：逐时间点训练有效性 mask。
- `reports/excluded_time_windows_model*_1s.csv`：原始停炉/检修/集体异常候选窗口。
- `reports/common_low_anomaly_windows_model*_1s.csv`：多变量共识低位长异常窗口。
- `reports/common_low_candidates_model*_1s.csv`：参与共识低位检测的候选变量及自适应低位阈值。
- `reports/common_low_score_model*_1s.parquet`：逐时间点共识低位得分。
- `reports/invalid_training_windows_model*_1s.csv`：应用 buffer 后不可训练窗口。
- `reports/window_index_model*_1s_seq1200_pred300.csv`：滑动窗口索引，包含输入段和预测段的起止行号/时间。
- `reports/valve_response_diagnostics_model*_1s.csv`：控制量和响应量的相关性/单调性诊断。

## 测试结果可视化

运行数据工程脚本后，可以打开 [visualize_test_results.py](visualize_test_results.py)，修改顶部参数后在 VSCode 中直接运行：

```python
INTERACTIVE_MODEL = "子模型2"
INTERACTIVE_FREQ_SECONDS = 1
INTERACTIVE_SEQ_LEN = 1200
INTERACTIVE_PRED_LEN = 300
INTERACTIVE_PREVIEW_COLUMNS = [
    "B3FT80",
    "B3FT00",
    "B3QT04A",
    "B3QT04ABC",
    "B3_Q02A.MV",
]
```

报告默认输出到：

`offline/artifacts/forecast/data_engineering/outputs/visualizations/model2_test_report_1s.html`

报告包含：

- 总行数、训练有效行、有效滑窗数、低信息输入列数量。
- `valid_for_training` 时间线，用于确认测试段是否被异常规则大量排除。
- 关键点位趋势预览，用于快速检查目标和运行指标是否有明显异常。
- 低信息输入点位列表和条形图。
- 质量预警点位表。
- 不可训练窗口和滑动窗口样例。

## 训练/验证/测试划分

分块训练帧生成后，打开 [prepare_dataset_splits.py](prepare_dataset_splits.py)，修改顶部参数后直接运行：

```python
INTERACTIVE_MODEL_SLUG = "model2"
INTERACTIVE_FREQ_SECONDS = 1
INTERACTIVE_SEQ_LEN = 1200
INTERACTIVE_PRED_LEN = 300
INTERACTIVE_TRAIN_RATIO = 0.70
INTERACTIVE_VAL_RATIO = 0.15
INTERACTIVE_TEST_RATIO = 0.15
INTERACTIVE_PURGE_GAP_CHUNKS = 1
```

输出位于 `datasets/chunked/model*_1s_seq1200_pred300/splits`：

- `split_manifest.csv`：所有可用 chunk 的 split 标记。
- `train_chunks.csv`、`val_chunks.csv`、`test_chunks.csv`：训练脚本应读取的分块清单。
- `split_summary.csv`：各集合的 chunk 数、窗口数和时间范围。
- `dataset_config.json`：后续训练脚本可读取的基础配置。

划分按时间顺序完成，并默认在 train/val、val/test 边界各跳过 1 个 chunk，减少滑动窗口跨集合泄漏。

## 环境

读写 parquet 需要 `duckdb`、`pyarrow`、`pandas`、`openpyxl`。依赖由预测模块的数据工程环境提供；先把 `INTERACTIVE_ACTION` 改成 `check-env` 做环境检查，不要把虚拟环境或依赖目录提交到 Git。

可以先把 `INTERACTIVE_ACTION` 改成 `check-env` 做环境检查。
## 对接原项目模型训练

全量数据不要再导出单体 CSV 宽表。使用分块窗口数据直接喂给 Time-Series-Library。模型 1 使用 10 秒、`seq_len=120`、`pred_len=60`；模型 2 使用 1 秒、`seq_len=1200`、`pred_len=600`。

1. 先运行 [chunked_tslib_dataset.py](../../forecast/training/chunked_tslib_dataset.py)，设置 `model1`、`freq_seconds=10`、`seq_len=120`、`label_len=60`、`pred_len=60`。脚本会生成：
   - `offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60/metadata.json`
   - `offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60/scaler.json`
   - `offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60/tslib_chunked_config.yaml`
   - `offline/artifacts/forecast/training_ready/model1_10s_seq120_pred60/loader_smoke_report.json`
2. 再运行 [run_chunked_tslib_training.py](../../forecast/training/run_chunked_tslib_training.py)。默认 `INTERACTIVE_RUN_TRAINING = False`，只生成并打印训练命令。检查无误后改成 `True`，脚本会调用：
   - `backend/wisburn/vendor/tslib/run.py`
   - `--data custom_chunked`
   - `--root_path <训练入口生成的 model1_10s_seq120_pred60 目录>`
   - `--data_path metadata.json`

已在原项目 TSLib 数据入口注册 `custom_chunked`。它读取分块 parquet 和窗口索引，不再创建几十 GB 的 CSV 宽表；训练 batch 返回格式与原 `Dataset_Custom` 一致：

```python
batch_x, batch_y, batch_x_mark, batch_y_mark
```

其中 `batch_x` 长度为 `seq_len=120`，`batch_y` 长度为 `label_len + pred_len=120`。loss 目标列使用 `metadata.json` 里的 `target_indices`，脚本会自动传给 `--loss_target_indices`。
