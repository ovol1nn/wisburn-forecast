# 预测置信度校准

本模块不训练、不改写现有 Crossformer 权重，也不直接参与 DCS 或控制决策。它只针对当前发布的预测模型，离线回答两个问题：

1. 在不同目标和不同预测时域上，预测相对“保持当前值不变”是否有稳定收益；
2. 当模型给出一条预测时，可以配套给出多宽的经验预测区间，以及该区间在后续数据上的覆盖是否接近标称水平。

## 当前版本和输入来源

| 运行模型 | 运行版本 | 对应训练测试产物 | 原生步长 |
| --- | --- | --- | ---: |
| model1 | `model1-10s-v20260823-r1` | `model1_xingrong3_model1_10s_trend_endpoint_v1_0` | 10 秒 |
| model2 | `model2-10s-history-only-v1` | `model2_10s_seq120_pred60_history_only` | 10 秒 |

脚本会核对 checkpoint SHA-256、目标顺序和时间契约；任一项不一致即终止，不允许用旧模型的预测结果给当前模型贴标签。

## 评价方法

测试预测按保存顺序分成前 40% 的校准段和后 60% 的评价段。校准段只用于估计各目标、各时域的经验残差区间；评价段只用于核验，因此不会用同一批误差同时“定区间”和“宣布区间可靠”。新版 Model 2 的保持值基线从对应测试窗口的本地历史帧读取预测起点实测值，并核对第一步真实值与已保存数组顺序一致，不用未来真值代替当前值。这是离线时间顺序验证，不是现场闭环验证。

时域固定为 10、30、60、120、300、600 秒；模型 1 和新的 Model 2 均在 10 秒网格上取相应节点。

每个“目标—时域”输出：

- 工程单位 MAE；
- 相对保持不变基线的改善率；
- 有明显变化窗口上的趋势方向一致率；
- 校准段估计的 90% 经验预测区间及其在评价段上的实际覆盖率；
- 由上述证据形成的 `base_confidence`（0–1）及 `low/medium/high` 分级。

保持不变基线采用预测起点最后一个已观测值。Model 2 的混合系数另由`reports/evaluate_model2_no_retrain_correction.py`只在验证期选定，再在后续测试期评价；它与本脚本的经验置信区间用途不同，不能把测试期反复调参后的分数称为独立验证。

`base_confidence` 是“该目标该时域在已保存测试数据上的经验可靠性”，不是预测正确的概率，也不等于控制动作后的响应可信度。`release_confidence` 是更保守的发布值：若保存预测未覆盖原测试集的 95%，最高限制为中等水平（0.65），避免把局部样本的好结果伪装为全工况高置信度。Model 1 保存预测与当前本地窗口修订不一致，报告保留首个未来真值代理基线并明确标记，不能把该模型的基线改善率当作有效比较；本轮可靠的新 Model 2 基线结论只看 Model 2 行。控制场景还必须叠加数据质量、输入分布外推、规则证据和安全约束。

## 输入分布外推（OOD）

当前报告尚未计算输入分布外推（OOD）分数，因此仍将 `input_ood_status` 标为 `unavailable`。后续应按训练集特征范围、稳健 z 分数和缺失率计算该分数，再与 `base_confidence` 共同解释；能读取历史窗口不代表 OOD 评价已经完成。

## 历史报告归档

`offline/artifacts/forecast/confidence/archive/` 下的 `current-models-v1`、`current-models-v2`、`current-models-v3` 保留旧 Model 2（1 秒`pred600-v20260809-r1`）及其当时报告。新版报告输出到`current-models-v4`，其中 Model 2 使用真实预测起点实测值；Model 1因旧保存数组与当前本地窗口版本不一致，只保留旧代理比较并在报告中标明无效。Model 2 的模型—保持值混合专项报告输出到`offline/artifacts/forecast/candidates/model2_history_only_blend_20261028_v3/`。旧报告不覆盖，也不再代表当前 Model 2。

## 执行顺序

```powershell
# 小规模 smoke：验证契约、计算和图表，不用于发布
.\.venv\Scripts\python.exe offline\forecast\confidence\calibrate_forecast_confidence.py --max-windows 2000 --output offline\artifacts\forecast\confidence\smoke-current

# 全部已保存测试窗口：生成当前模型版本的正式离线评估资产
.\.venv\Scripts\python.exe offline\forecast\confidence\calibrate_forecast_confidence.py --full --output offline\artifacts\forecast\confidence\current-models-v4

# Model 2 不重训混合校准：验证期选权重，测试期评价
.\.venv\Scripts\python.exe offline\forecast\reports\evaluate_model2_no_retrain_correction.py --output offline\artifacts\forecast\candidates\model2_history_only_blend_20261028_v3
```

输出目录包含 `manifest.json`、`forecast_confidence.json`、`per_target_horizon_metrics.csv`、`report.md`、`confidence_heatmap.png` 和 `interval_coverage.png`。正式接入后端前，还需人工审阅完整回放结果；本模块不会自动修改后端的 `Forecast.confidence`。
