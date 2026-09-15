# 预测置信度校准

本模块不训练、不改写现有 Crossformer 权重，也不直接参与 DCS 或控制决策。它只针对当前发布的预测模型，离线回答两个问题：

1. 在不同目标和不同预测时域上，预测相对“保持当前值不变”是否有稳定收益；
2. 当模型给出一条预测时，可以配套给出多宽的经验预测区间，以及该区间在后续数据上的覆盖是否接近标称水平。

## 当前版本和输入来源

| 运行模型 | 运行版本 | 对应训练测试产物 | 原生步长 |
| --- | --- | --- | ---: |
| model1 | `model1-10s-v20260823-r1` | `model1_xingrong3_model1_10s_trend_endpoint_v1_0` | 10 秒 |
| model2 | `pred600-v20260809-r1` | `model2_bundle_source` | 1 秒 |

脚本会核对 checkpoint SHA-256、目标顺序和时间契约；任一项不一致即终止，不允许用旧模型的预测结果给当前模型贴标签。

## 评价方法

测试预测按保存顺序分成前 40% 的校准段和后 60% 的评价段。校准段只用于估计各目标、各时域的经验残差区间；评价段只用于核验，因此不会用同一批误差同时“定区间”和“宣布区间可靠”。这是离线近似的时间顺序验证，不是现场闭环验证。

时域固定为 10、30、60、120、300、600 秒；模型 1 在其 10 秒网格上取相应节点，模型 2 在其 1 秒网格上取相应节点。

每个“目标—时域”输出：

- 工程单位 MAE；
- 相对保持不变基线的改善率；
- 有明显变化窗口上的趋势方向一致率；
- 校准段估计的 90% 经验预测区间及其在评价段上的实际覆盖率；
- 由上述证据形成的 `base_confidence`（0–1）及 `low/medium/high` 分级。

保持不变基线采用该预测轨迹的首个真实未来节点作为当前状态代理。这对 10 秒/1 秒最近节点会偏乐观，脚本会在输出中明确标注；等获得保存的输入历史窗口后，应改用真正的输入末点重新计算。

`base_confidence` 是“该目标该时域在已保存测试数据上的经验可靠性”，不是预测正确的概率，也不等于控制动作后的响应可信度。`release_confidence` 是更保守的发布值：若保存预测未覆盖原测试集的 95%，最高限制为中等水平（0.65），避免把局部样本的好结果伪装为全工况高置信度。控制场景还必须叠加数据质量、输入分布外推、规则证据和安全约束。

## 输入分布外推（OOD）

当前训练包保存了 `pred.npy` 和 `true.npy`，没有保存与每个预测一一对应的输入窗口；所以本首版只将 `input_ood_status` 标为 `unavailable`，不会伪造 OOD 分数。后续获得输入窗口后，应按训练集特征的分位数范围、稳健 z 分数和缺失率计算 `input_ood_score`，再将其与 `base_confidence` 合成运行时置信度。

## 执行顺序

```powershell
# 小规模 smoke：验证契约、计算和图表，不用于发布
.\.venv\Scripts\python.exe offline\forecast\confidence\calibrate_forecast_confidence.py --max-windows 2000 --output offline\artifacts\forecast\confidence\smoke-current

# 全部已保存测试窗口：生成当前模型版本的正式离线评估资产
.\.venv\Scripts\python.exe offline\forecast\confidence\calibrate_forecast_confidence.py --full --output offline\artifacts\forecast\confidence\current-models-v1
```

输出目录包含 `manifest.json`、`forecast_confidence.json`、`per_target_horizon_metrics.csv`、`report.md`、`confidence_heatmap.png` 和 `interval_coverage.png`。正式接入后端前，还需人工审阅完整回放结果；本模块不会自动修改后端的 `Forecast.confidence`。
