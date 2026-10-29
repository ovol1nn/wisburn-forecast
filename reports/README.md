# 预测离线报告

本目录脚本读取训练就绪目录或训练输出中的 metadata、预测和真实值，生成评估图表、CSV/HTML 报告或 Word 汇总。它们不会修改训练数据、发布 bundle、Profile 或 DCS 接口；输出目录由脚本参数或环境变量决定，生成物默认不提交。

- `generate_model_forecast_review.py`：两个模型的工程单位误差、趋势与代表窗口图。
- `generate_model1_10s_forecast_charts.py`：模型 1 十秒原生预测图。
- `generate_smoothed_forecast_review.py`：原始预测、十秒聚合和历史 60 秒平滑的离线对比；60 秒平滑不进入控制链。
- `generate_training_summary_doc.py`：训练成果 Word 汇总。
- `diagnose_model1_history_only.py`：读取模型1历史输入版的现有输出，按日期、负荷、趋势与控制变化做诊断，并检查分块波动、输入缺失和清洗前视依赖；只生成研究报告，不训练或发布模型。结果位于模型1训练成果目录的 `evaluation_history_only_20261021/diagnostics_20261021/`。
- `validate_preheater_fixed_blend.py`：先复现已有测试预测，再在原验证期各分块最多16个不重叠历史与预测窗口上，复核固定预热温度组合权重；不拟合参数、不训练、不发布。原验证期曾参与模型选择，因此结果仅是回顾性跨时期复核。
- `build_history_only_candidate.py`：6组预设历史门控短时规则只用较早验证块选型，冻结后复核较晚验证块及全测试期；结果保存在forecast artifacts的`candidates/model1_history_only_guarded_v1_20261022/`。首次执行不带参数进行评价；随后用`--package-only`生成独立调用包，用`--report-only`生成研究报告。不允许覆盖已冻结版本；不会发布到Profile。
- `improve_oxygen_history_only.py`：氧量历史/预测缓存与简单偏移试验；`calibrate_oxygen_state.py`：一次三参数状态校准试验；`report_oxygen_diagnostics.py`：生成分时距、案例、时滞及延续/反转诊断。结果保存在forecast artifacts的`candidates/model1_oxygen_corrected_v2_20261022/`，两阶段均未通过完整替换验收，因此没有v2模型bundle。前两个脚本仅在新实验首次运行，冻结记录不覆盖；报告脚本可按缓存复算。所有数据仍在本地，未重训或发布模型。

当前运行版本的置信度校准不放在本目录，见 `../confidence/`。它按目标与时域输出经验预测区间、相对保持不变的动态收益和基础置信度，且不会修改模型权重或后端运行配置。

- `evaluate_model2_no_retrain_correction.py`：不改权重，按目标和10/30/60/120/300/600秒时域在验证期校准模型与保持当前值的混合比例，并在后续测试期比较原模型、保持值和混合预测；同时记录MAE、RMSE及误差尾部，并说明与MPC滚动残差校正的组合顺序。报告输出到`offline/artifacts/forecast/candidates/model2_history_only_blend_20261028_v3/`，不接入运行配置。
