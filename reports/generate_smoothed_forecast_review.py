"""Compare raw forecasts with 10-second aggregation and 60-second smoothing."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runtime" / "runs" / "model_forecast_review" / "20260817" / "postprocessed"
SAMPLE_WINDOWS = 10_000
WINDOW = 60
DISPLAY = {
    "B3_F80.PV": ("子模型1", "主蒸汽流量", "t/h"),
    "B3_TE1119": ("子模型1", "中上部炉膛温度", "°C"),
    "B3QT02BBC": ("子模型2", "CO", "mg/m³"),
    "B3QT02EBC": ("子模型2", "NOx", "mg/m³"),
    "B3QT02ABC": ("子模型2", "SO₂", "mg/m³"),
}


def centered_mean(values, width=WINDOW):
    left, right = (width - 1) // 2, width // 2
    padded = np.pad(values, ((0, 0), (left, right), (0, 0)), mode="edge")
    cumulative = np.concatenate([np.zeros_like(padded[:, :1]), np.cumsum(padded, axis=1)], axis=1)
    return (cumulative[:, width:] - cumulative[:, :-width]) / width


def polyline(values, width, height, left, top, lo, hi):
    points = []
    for pos, value in enumerate(values):
        x = left + pos / (len(values) - 1) * width
        y = top + (hi - value) / (hi - lo) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def make_chart(path, title, unit, actual, raw, smooth):
    # Curves are 10-second aggregates (60 points) so differences are readable at runtime cadence.
    width, height, left, right, top, bottom = 1120, 420, 80, 28, 52, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    values = np.r_[actual, raw, smooth]
    lo, hi = np.percentile(values, [1, 99])
    margin = max((hi - lo) * .08, 1e-6)
    lo, hi = lo - margin, hi + margin
    grid = []
    for frac in (0, .25, .5, .75, 1):
        y = top + frac * plot_h
        label = hi - frac * (hi - lo)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/><text x="8" y="{y+4:.1f}" class="axis">{label:.2f}</text>')
    ticks = []
    for minute in range(0, 11, 2):
        x = left + minute / 10 * plot_w
        ticks.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" class="grid"/><text x="{x-8:.1f}" y="{height-25}" class="axis">{minute}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>.title{{font:600 18px Arial,"Microsoft YaHei",sans-serif}}.axis{{font:12px Arial,"Microsoft YaHei",sans-serif;fill:#4b5563}}.legend{{font:13px Arial,"Microsoft YaHei",sans-serif;fill:#1f2937}}.grid{{stroke:#e5e7eb;stroke-width:1}}.actual{{fill:none;stroke:#2563eb;stroke-width:2}}.raw{{fill:none;stroke:#e4572e;stroke-width:1.5;opacity:.8}}.smooth{{fill:none;stroke:#059669;stroke-width:2}}</style>
<rect width="100%" height="100%" fill="white"/><text x="{left}" y="26" class="title">{title}</text><text x="{left}" y="45" class="axis">单位：{unit}；三条曲线均按 10 秒聚合</text>{''.join(grid)}{''.join(ticks)}
<polyline points="{polyline(actual, plot_w, plot_h, left, top, lo, hi)}" class="actual"/><polyline points="{polyline(raw, plot_w, plot_h, left, top, lo, hi)}" class="raw"/><polyline points="{polyline(smooth, plot_w, plot_h, left, top, lo, hi)}" class="smooth"/>
<line x1="{width-325}" y1="26" x2="{width-303}" y2="26" class="actual"/><text x="{width-298}" y="31" class="legend">实际</text><line x1="{width-235}" y1="26" x2="{width-213}" y2="26" class="raw"/><text x="{width-208}" y="31" class="legend">原始预测</text><line x1="{width-125}" y1="26" x2="{width-103}" y2="26" class="smooth"/><text x="{width-98}" y="31" class="legend">60秒平滑</text><text x="{left}" y="{height-7}" class="axis">预测后分钟</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def evaluate_bundle(bundle_name):
    bundle = ROOT / "offline" / "artifacts" / "forecast" / "training_ready" / bundle_name
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((bundle / "scaler.json").read_text(encoding="utf-8"))
    columns, targets = metadata["feature_columns"], metadata["target_columns"]
    indices = [columns.index(x) for x in targets]
    mean = np.asarray(scaler["mean"], dtype=float)[indices]
    scale = np.asarray(scaler["scale"], dtype=float)[indices]
    pred, true = np.load(bundle / "pred.npy", mmap_mode="r"), np.load(bundle / "true.npy", mmap_mode="r")
    selected = np.linspace(0, len(pred) - 1, min(SAMPLE_WINDOWS, len(pred)), dtype=int)
    metrics = {x: {"raw_abs": 0., "agg_abs": 0., "smooth_abs": 0., "n": 0, "var_true": 0., "var_raw": 0., "var_smooth": 0.} for x in targets}
    candidates = selected[np.linspace(0, len(selected) - 1, 80, dtype=int)]
    candidate_scores = {x: [] for x in targets}
    for start in range(0, len(selected), 100):
        rows = selected[start:start + 100]
        raw = np.asarray(pred[rows, :, :][:, :, indices], dtype=float) * scale + mean
        actual = np.asarray(true[rows, :, :][:, :, indices], dtype=float) * scale + mean
        smooth = centered_mean(raw)
        raw_10 = raw.reshape(len(rows), 60, 10, len(targets)).mean(axis=2)
        actual_10 = actual.reshape(len(rows), 60, 10, len(targets)).mean(axis=2)
        smooth_10 = smooth.reshape(len(rows), 60, 10, len(targets)).mean(axis=2)
        for col, target in enumerate(targets):
            m = metrics[target]
            m["raw_abs"] += np.abs(raw[:, :, col] - actual[:, :, col]).sum()
            m["agg_abs"] += np.abs(raw_10[:, :, col] - actual_10[:, :, col]).sum()
            m["smooth_abs"] += np.abs(smooth_10[:, :, col] - actual_10[:, :, col]).sum()
            m["n"] += raw[:, :, col].size
            m["var_true"] += np.abs(np.diff(actual_10[:, :, col], axis=1)).sum()
            m["var_raw"] += np.abs(np.diff(raw_10[:, :, col], axis=1)).sum()
            m["var_smooth"] += np.abs(np.diff(smooth_10[:, :, col], axis=1)).sum()
    for row in candidates:
        raw = np.asarray(pred[row, :, :][:, indices], dtype=float) * scale + mean
        actual = np.asarray(true[row, :, :][:, indices], dtype=float) * scale + mean
        for col, target in enumerate(targets):
            candidate_scores[target].append(float(np.abs(raw[:, col] - actual[:, col]).mean()))
    results = []
    for col, target in enumerate(targets):
        if target not in DISPLAY:
            continue
        m = metrics[target]
        model, name, unit = DISPLAY[target]
        result = {"model": model, "point_code": target, "name": name, "unit": unit,
                  "raw_mae_1s": m["raw_abs"] / m["n"], "mae_10s_aggregated": m["agg_abs"] / (m["n"] / 10),
                  "mae_60s_smoothed_10s": m["smooth_abs"] / (m["n"] / 10),
                  "raw_variation_ratio_10s": m["var_raw"] / m["var_true"], "smoothed_variation_ratio_10s": m["var_smooth"] / m["var_true"]}
        results.append(result)
        median_row = int(candidates[np.argsort(candidate_scores[target])[len(candidates) // 2]])
        raw = np.asarray(pred[median_row, :, :][:, indices], dtype=float) * scale + mean
        actual = np.asarray(true[median_row, :, :][:, indices], dtype=float) * scale + mean
        smooth = centered_mean(raw[None, :, :])[0]
        make_chart(OUT / f"{target}_10s_aggregate_vs_60s_smooth.svg", f"{model} · {name} ({target})", unit,
                   actual[:, col].reshape(60, 10).mean(axis=1), raw[:, col].reshape(60, 10).mean(axis=1), smooth[:, col].reshape(60, 10).mean(axis=1))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = evaluate_bundle("model1_bundle_source") + evaluate_bundle("model2_bundle_source")
    fields = list(rows[0])
    with (OUT / "postprocessing_comparison_sampled.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} postprocessing comparisons to {OUT}")


if __name__ == "__main__":
    main()
