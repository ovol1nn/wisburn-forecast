from __future__ import annotations

import csv
import html
import math
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
ARTIFACT_ROOT = Path(os.environ.get(
    "WISBURN_FORECAST_ARTIFACT_ROOT",
    WORKSPACE_ROOT / "offline" / "artifacts" / "forecast",
)) / "data_engineering"

# VSCode interactive defaults. Edit these, then click "Run Python File".
INTERACTIVE_MODEL = "子模型2"
INTERACTIVE_FREQ_SECONDS = 1
INTERACTIVE_SEQ_LEN = 1200
INTERACTIVE_PRED_LEN = 600
INTERACTIVE_OUTPUT_HTML = ""
INTERACTIVE_PREVIEW_COLUMNS = [
    "B3FT80",
    "B3FT00",
    "B3QT04A",
    "B3QT04ABC",
    "B3_Q02A.MV",
]
MAX_LINE_POINTS = 1400


def maybe_add_project_site_packages() -> None:
    if sys.version_info[:2] != (3, 13):
        return
    site_packages = WORKSPACE_ROOT / ".venv_data_store" / "Lib" / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def require_pandas():
    maybe_add_project_site_packages()
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(
            "缺少 pandas/pyarrow 环境，建议用本机 Python 3.13 运行本脚本。"
        ) from exc
    return pd


def model_slug(model: str) -> str:
    return model.replace("子模型", "model")


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(value: Any, digits: int = 2) -> str:
    try:
        f = float(value)
    except Exception:
        return ""
    if math.isnan(f):
        return ""
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{digits}f}"


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return ""


def html_table(rows: list[dict[str, Any]], columns: list[str], limit: int = 20) -> str:
    if not rows:
        return "<p class='muted'>无数据</p>"
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = []
    for row in rows[:limit]:
        cells = "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)
        body.append(f"<tr>{cells}</tr>")
    more = f"<p class='muted'>仅展示前 {limit} 行，共 {len(rows)} 行。</p>" if len(rows) > limit else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}"


def aggregate_mask(pd: Any, mask: Any, max_bins: int = 240) -> Any:
    if mask.empty:
        return mask
    frame = mask[["timestamp", "valid_for_training"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if len(frame) <= max_bins:
        frame["valid_rate"] = frame["valid_for_training"].astype(float)
        return frame[["timestamp", "valid_rate"]]
    bins = min(max_bins, max(1, len(frame) // 30))
    frame["_bin"] = pd.cut(range(len(frame)), bins=bins, labels=False)
    out = frame.groupby("_bin", observed=True).agg(
        timestamp=("timestamp", "min"),
        valid_rate=("valid_for_training", "mean"),
    )
    return out.reset_index(drop=True)


def svg_valid_timeline(pd: Any, mask: Any, width: int = 980, height: int = 72) -> str:
    data = aggregate_mask(pd, mask)
    if data.empty:
        return "<p class='muted'>无 mask 数据</p>"
    n = len(data)
    bar_w = max(1, width / n)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'>"]
    for i, row in data.iterrows():
        rate = float(row["valid_rate"])
        color = "#129575" if rate >= 0.999 else ("#f5a524" if rate > 0 else "#d54b4b")
        x = i * bar_w
        parts.append(f"<rect x='{x:.2f}' y='16' width='{bar_w + 0.5:.2f}' height='28' fill='{color}'/>")
    start = str(data["timestamp"].iloc[0])
    end = str(data["timestamp"].iloc[-1])
    parts.append(f"<text x='0' y='64' class='axis'>{html.escape(start)}</text>")
    parts.append(f"<text x='{width}' y='64' class='axis end'>{html.escape(end)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_bar(values: list[tuple[str, float]], width: int = 980, row_h: int = 26) -> str:
    if not values:
        return "<p class='muted'>无数据</p>"
    height = max(80, 28 + row_h * len(values))
    max_v = max(v for _, v in values) or 1.0
    label_w = 180
    plot_w = width - label_w - 80
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'>"]
    for i, (label, value) in enumerate(values):
        y = 24 + i * row_h
        w = plot_w * value / max_v
        parts.append(f"<text x='0' y='{y + 14}' class='label'>{html.escape(label)}</text>")
        parts.append(f"<rect x='{label_w}' y='{y}' width='{w:.2f}' height='16' rx='2' fill='#3b82f6'/>")
        parts.append(f"<text x='{label_w + w + 8:.2f}' y='{y + 13}' class='value'>{value:.1f}%</text>")
    parts.append("</svg>")
    return "".join(parts)


def downsample_frame(df: Any, max_points: int) -> Any:
    if len(df) <= max_points:
        return df
    step = max(1, math.ceil(len(df) / max_points))
    return df.iloc[::step].copy()


def svg_line_chart(pd: Any, df: Any, columns: list[str], width: int = 980, height: int = 260) -> str:
    columns = [c for c in columns if c in df.columns]
    if not columns or df.empty:
        return "<p class='muted'>无可绘制点位</p>"
    plot = downsample_frame(df[["timestamp"] + columns], MAX_LINE_POINTS)
    colors = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#8b5cf6"]
    left, right, top, bottom = 54, 16, 20, 44
    inner_w = width - left - right
    inner_h = height - top - bottom
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img'>"]
    parts.append(f"<rect x='{left}' y='{top}' width='{inner_w}' height='{inner_h}' fill='#fbfdff' stroke='#d9e2ec'/>")
    for idx, col in enumerate(columns[: len(colors)]):
        s = plot[col]
        valid = s.dropna()
        if valid.empty:
            continue
        lo = float(valid.quantile(0.01))
        hi = float(valid.quantile(0.99))
        if hi <= lo:
            hi = lo + 1.0
        points = []
        for i, value in enumerate(s):
            if pd.isna(value):
                continue
            x = left + inner_w * i / max(len(plot) - 1, 1)
            y = top + inner_h * (1 - (float(value) - lo) / (hi - lo))
            y = max(top, min(top + inner_h, y))
            points.append(f"{x:.2f},{y:.2f}")
        if len(points) >= 2:
            color = colors[idx]
            parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='1.7' points='{' '.join(points)}'/>")
            parts.append(f"<text x='{left + 8}' y='{top + 18 + idx * 18}' fill='{color}' class='legend'>{html.escape(col)}</text>")
    parts.append(f"<text x='{left}' y='{height - 14}' class='axis'>{html.escape(str(plot['timestamp'].iloc[0]))}</text>")
    parts.append(f"<text x='{width - right}' y='{height - 14}' class='axis end'>{html.escape(str(plot['timestamp'].iloc[-1]))}</text>")
    parts.append("</svg>")
    return "".join(parts)


def build_report() -> Path:
    pd = require_pandas()
    slug = model_slug(INTERACTIVE_MODEL)
    out_dir = ARTIFACT_ROOT / "outputs"
    dataset_dir = out_dir / "datasets"
    report_dir = out_dir / "reports"
    visual_dir = out_dir / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)

    training_frame_path = dataset_dir / f"{slug}_training_frame_{INTERACTIVE_FREQ_SECONDS}s.parquet"
    mask_path = report_dir / f"training_mask_{slug}_{INTERACTIVE_FREQ_SECONDS}s.parquet"
    window_path = report_dir / f"window_index_{slug}_{INTERACTIVE_FREQ_SECONDS}s_seq{INTERACTIVE_SEQ_LEN}_pred{INTERACTIVE_PRED_LEN}.csv"
    quality_path = report_dir / f"{slug}_quality_{INTERACTIVE_FREQ_SECONDS}s.csv"
    low_info_path = report_dir / f"low_information_inputs_{slug}_{INTERACTIVE_FREQ_SECONDS}s.csv"
    invalid_path = report_dir / f"invalid_training_windows_{slug}_{INTERACTIVE_FREQ_SECONDS}s.csv"

    missing = [p for p in (training_frame_path, mask_path, window_path) if not p.exists()]
    if missing:
        raise SystemExit("缺少可视化输入，请先运行 build：\n" + "\n".join(rel_path(p) for p in missing))

    frame = pd.read_parquet(training_frame_path)
    mask = pd.read_parquet(mask_path)
    windows = read_csv_rows(window_path)
    quality = read_csv_rows(quality_path)
    low_info = read_csv_rows(low_info_path)
    invalid = read_csv_rows(invalid_path)

    valid_rows = int(mask["valid_for_training"].sum()) if "valid_for_training" in mask else 0
    total_rows = len(mask)
    window_count = len(windows)
    low_info_count = len(low_info)

    warn_quality = [
        r for r in quality
        if str(r.get("warn_missing", "")).lower() == "true"
        or str(r.get("warn_zero", "")).lower() == "true"
        or str(r.get("warn_constant", "")).lower() == "true"
    ]

    low_info_values = []
    for row in low_info[:12]:
        value = max(float(row.get("zero_ratio") or 0), float(row.get("constant_ratio") or 0)) * 100
        low_info_values.append((row.get("point_code", ""), value))

    report_path = Path(INTERACTIVE_OUTPUT_HTML) if INTERACTIVE_OUTPUT_HTML else visual_dir / f"{slug}_test_report_{INTERACTIVE_FREQ_SECONDS}s.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#1f2937;background:#f7f9fc}
    h1{font-size:24px;margin:0 0 6px} h2{font-size:18px;margin:28px 0 12px}
    .muted{color:#6b7280}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}
    .card{background:white;border:1px solid #dbe4ef;border-radius:8px;padding:14px}
    .k{font-size:13px;color:#64748b}.v{font-size:26px;font-weight:700;margin-top:6px}
    table{border-collapse:collapse;width:100%;background:white;border:1px solid #dbe4ef}
    th,td{border-bottom:1px solid #e7eef6;padding:7px 8px;text-align:left;font-size:13px}
    th{background:#eef4fb;color:#334155}.panel{background:white;border:1px solid #dbe4ef;border-radius:8px;padding:14px;margin-bottom:14px}
    .axis{font-size:11px;fill:#64748b}.axis.end{text-anchor:end}.label{font-size:12px;fill:#334155}.value{font-size:12px;fill:#334155}.legend{font-size:12px;font-weight:600}
    """
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(INTERACTIVE_MODEL)} 测试结果验证</title><style>{css}</style></head>
<body>
<h1>{html.escape(INTERACTIVE_MODEL)} 数据处理测试结果验证</h1>
<p class="muted">数据源：{html.escape(rel_path(training_frame_path))}</p>
<section class="grid">
  <div class="card"><div class="k">总行数</div><div class="v">{total_rows:,}</div></div>
  <div class="card"><div class="k">训练有效行</div><div class="v">{valid_rows:,}</div></div>
  <div class="card"><div class="k">有效窗口</div><div class="v">{window_count:,}</div></div>
  <div class="card"><div class="k">低信息输入列</div><div class="v">{low_info_count:,}</div></div>
</section>
<h2>训练有效性时间线</h2>
<div class="panel">{svg_valid_timeline(pd, mask)}</div>
<h2>关键点位趋势预览</h2>
<div class="panel">{svg_line_chart(pd, frame, INTERACTIVE_PREVIEW_COLUMNS)}</div>
<h2>低信息输入点位</h2>
<div class="panel">{svg_bar(low_info_values)}</div>
{html_table(low_info, ["point_code", "zero_ratio", "constant_ratio", "action", "reason"], limit=20)}
<h2>质量预警点位</h2>
{html_table(warn_quality, ["point_code", "missing_rate", "zero_ratio", "constant_ratio", "warn_missing", "warn_zero", "warn_constant"], limit=30)}
<h2>不可训练窗口</h2>
{html_table(invalid, ["start_time", "end_time", "rows", "reason"], limit=30)}
<h2>滑动窗口索引样例</h2>
{html_table(windows, ["input_start_time", "input_end_time", "target_start_time", "target_end_time"], limit=12)}
</body></html>"""
    report_path.write_text(content, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    output = build_report()
    print(f"visual report: {rel_path(output)}")
