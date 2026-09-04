# 预测离线报告

本目录脚本读取训练就绪目录或训练输出中的 metadata、预测和真实值，生成评估图表、CSV/HTML 报告或 Word 汇总。它们不会修改训练数据、发布 bundle、Profile 或 DCS 接口；输出目录由脚本参数或环境变量决定，生成物默认不提交。

- `generate_model_forecast_review.py`：两个模型的工程单位误差、趋势与代表窗口图。
- `generate_model1_10s_forecast_charts.py`：模型 1 十秒原生预测图。
- `generate_smoothed_forecast_review.py`：原始预测、十秒聚合和历史 60 秒平滑的离线对比；60 秒平滑不进入控制链。
- `generate_training_summary_doc.py`：训练成果 Word 汇总。
