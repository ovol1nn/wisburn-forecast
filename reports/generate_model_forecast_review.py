"""Create a sampled, engineering-unit forecast review for trained model bundles."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runtime" / "runs" / "model_forecast_review" / "20260817"
SAMPLE_WINDOWS = 10_000
HORIZONS = {"1分钟": 59, "5分钟": 299, "10分钟": 599}
UNITS = {
    "B3_F80.PV": "t/h", "B3QT00": "%", "B3_O2_S1": "%", "B3_TE1119": "°C",
    "B3_GIC01": "原始单位未登记", "B3QT02BBC": "mg/m³", "B3QT02ABC": "mg/m³",
    "B3QT02CBC": "mg/m³", "B3QT02EBC": "mg/m³", "B3QT02FC2": "mg/m³",
}
DISPLAY_NAMES = {
    "B3_F80.PV": "主蒸汽流量", "B3QT00": "省煤器出口氧量", "B3_O2_S1": "氧量",
    "B3_TE1119": "中上部炉膛温度", "B3_GIC01": "垃圾层厚度",
    "B3QT02BBC": "CO", "B3QT02ABC": "SO₂", "B3QT02CBC": "HCl",
    "B3QT02EBC": "NOx", "B3QT02FC2": "烟尘",
}


def fmt(value):
    return f"{value:.3f}" if np.isfinite(value) else "—"


def svg_chart(path, title, seconds, actual, predicted, unit):
    width, height, left, right, top, bottom = 1120, 420, 80, 28, 52, 56
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = np.r_[actual, predicted]
    lo, hi = np.nanpercentile(all_values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(all_values), np.nanmax(all_values)
    padding = max((hi - lo) * 0.08, 1e-6)
    lo, hi = lo - padding, hi + padding

    def xy(values):
        points = []
        for x, value in zip(seconds, values):
            px = left + (x / 600) * plot_w
            py = top + (hi - value) / (hi - lo) * plot_h
            points.append(f"{px:.1f},{py:.1f}")
        return " ".join(points)

    grid = []
    for frac in (0, .25, .5, .75, 1):
        y = top + frac * plot_h
        value = hi - frac * (hi - lo)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/><text x="8" y="{y+4:.1f}" class="axis">{value:.2f}</text>')
    xaxis = []
    for sec in (0, 120, 240, 360, 480, 600):
        x = left + (sec / 600) * plot_w
        xaxis.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/><text x="{x-9:.1f}" y="{height-24}" class="axis">{sec//60}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>.title{{font:600 18px Arial,"Microsoft YaHei",sans-serif}}.axis{{font:12px Arial,"Microsoft YaHei",sans-serif;fill:#4b5563}}.legend{{font:13px Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}}.grid{{stroke:#e5e7eb;stroke-width:1}}.actual{{fill:none;stroke:#2563eb;stroke-width:2}}.pred{{fill:none;stroke:#e4572e;stroke-width:2}}</style>
<rect width="100%" height="100%" fill="white"/><text x="{left}" y="26" class="title">{title}</text><text x="{left}" y="45" class="axis">单位：{unit}；横轴：预测后分钟</text>{''.join(grid)}{''.join(xaxis)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9ca3af"/><polyline points="{xy(actual)}" class="actual"/><polyline points="{xy(predicted)}" class="pred"/>
<line x1="{width-230}" y1="26" x2="{width-208}" y2="26" class="actual"/><text x="{width-203}" y="31" class="legend">实际</text><line x1="{width-145}" y1="26" x2="{width-123}" y2="26" class="pred"/><text x="{width-118}" y="31" class="legend">预测</text>
<text x="{left}" y="{height-7}" class="axis">分钟</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def evaluate(bundle_name, model_label):
    bundle = ROOT / "offline" / "artifacts" / "forecast" / "training_ready" / bundle_name
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((bundle / "scaler.json").read_text(encoding="utf-8"))
    feature_columns, targets = metadata["feature_columns"], metadata["target_columns"]
    target_idx = [feature_columns.index(target) for target in targets]
    mean = np.asarray(scaler["mean"], dtype=np.float64)[target_idx]
    scale = np.asarray(scaler["scale"], dtype=np.float64)[target_idx]
    pred = np.load(bundle / "pred.npy", mmap_mode="r")
    true = np.load(bundle / "true.npy", mmap_mode="r")
    selected = np.linspace(0, pred.shape[0] - 1, min(SAMPLE_WINDOWS, pred.shape[0]), dtype=np.int64)
    sums = {
        target: {"abs": 0.0, "sq": 0.0, "n": 0, "variation_pred": 0.0,
                 "variation_true": 0.0, "curvature_pred": 0.0, "curvature_true": 0.0,
                 "direction_match": 0, "direction_eligible": 0,
                 **{f"h_{h}": [] for h in HORIZONS}}
        for target in targets
    }
    representative = {target: None for target in targets}
    candidates = selected[np.linspace(0, len(selected) - 1, 80, dtype=int)]
    candidate_errors = {target: [] for target in targets}
    for start in range(0, len(selected), 100):
        idx = selected[start:start + 100]
        p = np.asarray(pred[idx, :, :][:, :, target_idx], dtype=np.float64) * scale + mean
        t = np.asarray(true[idx, :, :][:, :, target_idx], dtype=np.float64) * scale + mean
        for col, target in enumerate(targets):
            err = p[:, :, col] - t[:, :, col]
            sums[target]["abs"] += np.abs(err).sum()
            sums[target]["sq"] += np.square(err).sum()
            sums[target]["n"] += err.size
            sums[target]["variation_pred"] += np.abs(np.diff(p[:, :, col], axis=1)).sum()
            sums[target]["variation_true"] += np.abs(np.diff(t[:, :, col], axis=1)).sum()
            sums[target]["curvature_pred"] += np.abs(np.diff(p[:, :, col], n=2, axis=1)).sum()
            sums[target]["curvature_true"] += np.abs(np.diff(t[:, :, col], n=2, axis=1)).sum()
            actual_net = t[:, -1, col] - t[:, 0, col]
            predicted_net = p[:, -1, col] - p[:, 0, col]
            eligible = np.abs(actual_net) >= max(scale[col] * 0.1, 1e-9)
            sums[target]["direction_eligible"] += int(eligible.sum())
            sums[target]["direction_match"] += int((np.sign(actual_net[eligible]) == np.sign(predicted_net[eligible])).sum())
            for horizon, point in HORIZONS.items():
                sums[target][f"h_{horizon}"].append(np.abs(err[:, point]).ravel())
    # A representative median-error window avoids selecting an unusually good or bad case.
    for idx in candidates:
        p = np.asarray(pred[idx, :, :][:, target_idx], dtype=np.float64) * scale + mean
        t = np.asarray(true[idx, :, :][:, target_idx], dtype=np.float64) * scale + mean
        for col, target in enumerate(targets):
            candidate_errors[target].append(float(np.mean(np.abs(p[:, col] - t[:, col]))))
    rows = []
    for col, target in enumerate(targets):
        result = sums[target]
        row = {
            "model": model_label, "point_code": target, "name": DISPLAY_NAMES.get(target, target),
            "unit": UNITS.get(target, "原始单位未登记"), "sample_windows": len(selected),
            "mae_all_10min": result["abs"] / result["n"], "rmse_all_10min": (result["sq"] / result["n"]) ** 0.5,
            "trend_direction_accuracy": result["direction_match"] / result["direction_eligible"] if result["direction_eligible"] else float("nan"),
            "meaningful_trend_windows": result["direction_eligible"],
            "variation_ratio_pred_true": result["variation_pred"] / result["variation_true"] if result["variation_true"] else float("nan"),
            "high_frequency_ratio_pred_true": result["curvature_pred"] / result["curvature_true"] if result["curvature_true"] else float("nan"),
        }
        for horizon in HORIZONS:
            row[f"mae_{horizon}"] = float(np.concatenate(result[f"h_{horizon}"]).mean())
        rows.append(row)
        median_pos = int(np.argsort(candidate_errors[target])[len(candidates)//2])
        idx = int(candidates[median_pos])
        p = np.asarray(pred[idx, :, :][:, target_idx], dtype=np.float64) * scale + mean
        t = np.asarray(true[idx, :, :][:, target_idx], dtype=np.float64) * scale + mean
        seconds = np.arange(1, 601)
        svg_chart(OUT / f"{model_label}_{target}_representative.svg", f"{model_label} · {DISPLAY_NAMES.get(target, target)} ({target})", seconds, t[:, col], p[:, col], row["unit"])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = evaluate("model1_bundle_source", "子模型1") + evaluate("model2_bundle_source", "子模型2")
    fields = list(rows[0])
    with (OUT / "key_target_metrics_sampled.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    diagnostics_fields = ["model", "point_code", "name", "unit", "sample_windows", "meaningful_trend_windows", "trend_direction_accuracy", "variation_ratio_pred_true", "high_frequency_ratio_pred_true"]
    with (OUT / "trend_and_volatility_sampled.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=diagnostics_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 新预测模型关键目标评估（抽样）", "", "- 数据源：两个已训练模型包内保存的 `pred.npy` / `true.npy`。", "- 口径：使用各模型自身的 `scaler.json` 还原工程单位后计算。", f"- 样本：每个模型均匀抽取 {SAMPLE_WINDOWS:,} 个测试窗口；每个窗口为 10 分钟、1 秒粒度。", "- 图：每个目标选取 80 个候选窗口中绝对误差处于中位数的一个代表性窗口，不代表最好或最差工况。", ""]
    lines.extend(["| 模型 | 目标 | 10分钟全程 MAE | 10分钟全程 RMSE | 1分钟 MAE | 5分钟 MAE | 10分钟点 MAE |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        u = row["unit"]
        lines.append(f"| {row['model']} | {row['name']}（{row['point_code']}） | {fmt(row['mae_all_10min'])} {u} | {fmt(row['rmse_all_10min'])} {u} | {fmt(row['mae_1分钟'])} {u} | {fmt(row['mae_5分钟'])} {u} | {fmt(row['mae_10分钟'])} {u} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} target metrics and charts to {OUT}")


if __name__ == "__main__":
    main()
