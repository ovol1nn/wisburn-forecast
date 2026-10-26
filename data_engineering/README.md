# 模型数据工程

本目录把平台数据存储工具生成的 Silver 点位 Parquet 处理成预测模型数据集。当前新训练数据集统一使用 10 秒粒度、`120→60`；模型 1 为14个物理输出目标，模型 2 为5个排放输出目标。该口径只描述新建数据集，不代表当前发布模型 bundle 已更新。目标列和特征数以点位分类表及对应训练生成的 `metadata.json` 为准。

模型 1 的14个物理输出点由原5点和新增9点组成：主蒸汽压力 A/B、主蒸汽温度、炉膛负压 A/B、汽包水位1/2/3、一次风预热器出口温度。压力和负压的 A/B 点分别作为模型输出，控制层再计算均值。模型 2 的5个输出仍为 CO、HCl、NOx、SO₂、烟尘；点位表中标记为模型2输入的工况点仍按输入角色使用。重训时以点位主表和生成的 metadata 为准，不要沿用旧版目标索引。默认输入见 `assets/data/xingrong_3/silver/by_point/point_code=<点位>/data.parquet` 与 `assets/profiles/xingrong_3/points/point_classification.xlsx`；单独运行模块时可用 `WISBURN_FORECAST_POINT_ROOT` 和 `WISBURN_FORECAST_CLASSIFICATION_XLSX` 覆盖。

仓库中已有分块数据集是旧预处理口径生成的历史产物，不能代表本次严格完整窗口策略。使用本次策略后必须用独立输出目录重建数据集，再重训并复验；旧帧、旧 scaler 与旧模型不自动兼容。

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

如果 `INTERACTIVE_START` / `INTERACTIVE_END` 留空，分块输出写入全量目录，例如模型 2 的 `datasets/chunked/model2_10s_seq120_pred60`。如果设置了短时间范围做测试，脚本会自动在目录名后增加时间后缀，避免覆盖全量 manifest。旧的 `pred300`、逐秒模型目录是早期产物，不能用于当前十秒训练口径。

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

脚本按 `strict_raw_complete_windows_v1` 保留实际观测值，不对模型输入/目标做统计去尖峰、平滑、插值或前后填充。坏质量、缺值和疑似尖峰都不被改写成另一种数值；任何模型点位不完整的训练行都会使覆盖该行的窗口失效。停炉和检修候选仍按独立运行状态规则筛掉。

主要步骤：

1. 从配置的点位分类 Excel 中读取 `模型变量明细`；完整工程优先使用 Profile/数据资产提供的点位分类文件。
2. 逐点读取 Silver Parquet，按右端点重采样；10 秒桶只有恰好包含 10 条有限有效的 1 秒源记录时才生成均值，否则该桶保留缺失。写入 `offline/artifacts/forecast/data_engineering/outputs/cache/resampled_<freq>s`，支持断点续跑。
3. 拼接模型宽表，按同一时间行直接计算一次风分段/占比、一次/二次风比例、推料器位置近似、炉排速度组均值等派生特征，不使用前后时刻或跨块状态。
4. 生成点位质量报告，统计缺失率、近零率、常值率。
5. 对长期近零或常值的输入点位，写入低信息点报告，并从训练帧中剔除。
6. 不对输入、目标或派生值做数据驱动的异常裁剪；异常值保留原样，由完整输入窗口规则拒绝，不插补。
7. 生成 `valid_for_training` mask：排除停炉/检修候选，并要求模型所需点位及派生源在每个相关时间行全部有效；任一点缺失即拒绝覆盖它的完整输入/预测训练窗。
8. 用 `valid_for_training` 生成滑动窗口索引，窗口长度由模型参数决定，并要求输入段、预测段有效且时间连续。

此前配置中的全段 IQR、居中邻点尖峰筛查、双向插值和填充规则已停用。旧版本训练集因此不能与当前在线输入口径相比较；当前严格输入缺口会直接让窗口不合格，而不是估算缺失值。

质量报告的缺失率分母是选定时间范围内应有的模型时间格数，而不是该点自己曾出现过的记录数；因此某点长时间没有归档记录会反映为缺失，不会被误报成“几乎无缺失”。`observed_rows` 另列实际出现过时间戳的行数。

## 检修和异常段

停炉/检修候选由两类规则识别。

第一类是已知运行指标规则：

- 集体缺失率超过 `collective_missing_rate_warn`
- 主蒸汽流量 `B3FT80` 持续低于阈值
- 一次风流量 `B3FT00` 持续低于阈值

阈值在 [config/default_config.json](config/default_config.json) 的 `shutdown_indicators` 中调整。当前默认阈值为 `5`，且要求连续低值至少 300 秒；指标缺失本身不会直接判定为停炉。`B3FT07` 保留为模型特征，但不再单独作为停炉判据：在现有 10 秒训练帧中，它低于 5 的采样点有约 57% 同时处于主蒸汽流量和一次风流量均不低于 5 的状态，单点低阈值会误排除大量正常运行数据。

第二类是更泛用的多变量共识低位规则 `long_common_low_anomaly`：

- 每个变量先按自身分布自适应计算低位阈值。
- 长期缺失、长期为 0、长期常值的点位不参与投票。
- 同一时间如果足够多变量同时处于低位，则形成共识低位得分。
- 只保留持续时间超过 `min_duration_seconds` 的长异常段，默认 `21600` 秒，即 6 小时。
- 允许中间短暂恢复并合并，默认 `bridge_gap_seconds = 3600` 秒。

这条规则用于识别你图中那类“一年内多个点位共同出现的长时间异常低位段”，不会把普通短时尖峰、短时下探直接视为异常工况。

触发持续低流量/检修候选后，会按 `training_exclusion_buffer_seconds` 做前后 buffer，默认 `3600` 秒，避免启停过渡段进入滑动窗口。修改筛选规则后，旧 `frames/windows/manifest` 不会自动更新；应使用独立输出目录重新构建并比较有效行、窗口数与排除窗口，再用于正式训练。

可以单独审计若干点位的全年共性异常：

```powershell
python "offline\forecast\data_engineering\build_model_datasets.py" audit-anomalies `
  --codes B3TE07D,B3TE19A,B3ZI86,B3TE05,B3_ST19 `
  --freq-seconds 300
```

审计输出位于 `reports/shared/screening/`。

## 主要输出

默认输出目录为 `offline/artifacts/forecast/data_engineering/outputs`，不进入 Git。

- `cache/resampled_<freq>s/*.parquet`：逐点重采样缓存，可断点续跑。
- `datasets/model*_1s.parquet`：模型宽表，保留原始清洗后的行。
- `datasets/model*_training_frame_1s.parquet`：训练帧，包含 `valid_for_training`，并剔除长期近零/常值输入列。
- `datasets/chunked/model1_10s_seq120_pred60/`、`datasets/chunked/model2_10s_seq120_pred60/` 下的 `frames/*.parquet`：当前窗口口径的分块训练帧。
- 上述目录的 `windows/*_windows.csv`：每个分块对应的滑动窗口索引，窗口起点只落在该块核心时间段内。
- 上述目录的 `manifest.csv`：分块清单，记录每块的核心时间、加载时间、训练帧路径、窗口索引路径、有效行数和窗口数。
- `reports/model1/`、`reports/model2/`：按模型分别存放报告；每个模型下再分 `quality/`（点位质量、不可用输出和低信息输入）、`screening/`（训练 mask、异常候选及剔除窗口）和 `diagnostics/`（窗口索引、控制量响应诊断等）。
- `reports/shared/quality/`：跨模型的点位质量、点位覆盖和变量清单。
- `reports/shared/screening/`：不专属于某一个模型的异常审计报告。

报告目录由数据工程脚本自动创建。脚本新生成的报告会按上述规则落盘；旧的平铺报告只需整理一次，不影响数据集和缓存。

## 测试结果可视化

运行数据工程脚本后，可以打开 [visualize_test_results.py](visualize_test_results.py)，修改顶部参数后在 VSCode 中直接运行。该脚本里的旧示例参数需按本次数据集改为 `10s / 120→60`：

```python
INTERACTIVE_MODEL = "子模型2"
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_PRED_LEN = 60
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
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_PRED_LEN = 60
INTERACTIVE_TRAIN_RATIO = 0.70
INTERACTIVE_VAL_RATIO = 0.15
INTERACTIVE_TEST_RATIO = 0.15
INTERACTIVE_PURGE_GAP_CHUNKS = 1
```

输出位于所选模型和时间参数对应的 `datasets/chunked/<模型>_<频率>s_seq<输入点数>_pred<预测点数>/splits`：

- `split_manifest.csv`：所有可用 chunk 的 split 标记。
- `train_chunks.csv`、`val_chunks.csv`、`test_chunks.csv`：训练脚本应读取的分块清单。
- `split_summary.csv`：各集合的 chunk 数、窗口数和时间范围。
- `dataset_config.json`：后续训练脚本可读取的基础配置。

划分按时间顺序完成，并默认在 train/val、val/test 边界各跳过 1 个 chunk，减少滑动窗口跨集合泄漏。

## 环境

读写 parquet 需要 `duckdb`、`pyarrow`、`pandas`、`openpyxl`。依赖由预测模块的数据工程环境提供；先把 `INTERACTIVE_ACTION` 改成 `check-env` 做环境检查，不要把虚拟环境或依赖目录提交到 Git。

可以先把 `INTERACTIVE_ACTION` 改成 `check-env` 做环境检查。
## 对接原项目模型训练

全量数据不要再导出单体 CSV 宽表。使用分块窗口数据直接喂给 Time-Series-Library。当前模型 1、模型 2 均使用 10 秒、`seq_len=120`、`label_len=60`、`pred_len=60`。

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
