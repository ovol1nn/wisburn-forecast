"""Generate transparent test-set forecast charts for the 10-second model1 bundle."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[3]
BUNDLE = next((PROJECT / "offline" / "artifacts" / "forecast" / "training_ready").glob("model1_xingrong3_model1_10s_trend_endpoint_v1_*"))
OUT = PROJECT / "runtime" / "runs" / "model1_10s_forecast_review" / "20260823"
TARGETS = ["B3_F80.PV", "B3QT00", "B3_O2_S1", "B3_TE1119", "B3_GIC01"]
NAMES = {
    "B3_F80.PV": "Main steam flow",
    "B3QT00": "Economizer outlet O2",
    "B3_O2_S1": "Furnace O2",
    "B3_TE1119": "Furnace temperature",
    "B3_GIC01": "Calorific-value index",
}
UNITS = {"B3_F80.PV": "t/h", "B3QT00": "%", "B3_O2_S1": "%", "B3_TE1119": "°C", "B3_GIC01": ""}
BLUE = "#0077BB"
ORANGE = "#EE7733"
GRID = "#D8DEE9"
TEXT = "#1F2937"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size=size)


def line_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], truth: np.ndarray, pred: np.ndarray, title: str, unit: str, note: str) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 75, top + 65, right - 30, bottom - 70
    all_values = np.r_[truth, pred]
    pad = max(float(np.ptp(all_values)) * 0.12, 0.04 if unit == "%" else 0.2)
    y_min, y_max = float(np.min(all_values) - pad), float(np.max(all_values) + pad)
    if y_max <= y_min:
        y_max = y_min + 1.0

    draw.text((left + 14, top + 12), title, fill=TEXT, font=font(25, True))
    draw.text((left + 14, top + 42), note, fill="#4B5563", font=font(17))
    for frac in np.linspace(0, 1, 5):
        y = plot_bottom - frac * (plot_bottom - plot_top)
        value = y_min + frac * (y_max - y_min)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((left + 4, y - 10), f"{value:.2f}", fill="#4B5563", font=font(15))
    for minute in (0, 2, 4, 6, 8, 10):
        x = plot_left + minute / 10 * (plot_right - plot_left)
        draw.line((x, plot_top, x, plot_bottom), fill="#F1F5F9", width=1)
        draw.text((x - 9, plot_bottom + 10), str(minute), fill="#4B5563", font=font(15))
    draw.text((plot_left, plot_bottom + 35), "Forecast horizon (minutes)", fill="#4B5563", font=font(16))
    draw.text((plot_left, plot_top - 24), unit, fill="#4B5563", font=font(16))

    def points(values: np.ndarray) -> list[tuple[float, float]]:
        result = []
        for i, value in enumerate(values):
            x = plot_left + i / max(len(values) - 1, 1) * (plot_right - plot_left)
            y = plot_bottom - (float(value) - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            result.append((x, y))
        return result

    draw.line(points(truth), fill=BLUE, width=4, joint="curve")
    draw.line(points(pred), fill=ORANGE, width=4, joint="curve")
    draw.rectangle((plot_right - 205, plot_top + 10, plot_right - 12, plot_top + 62), fill="white", outline=GRID)
    draw.line((plot_right - 193, plot_top + 27, plot_right - 160, plot_top + 27), fill=BLUE, width=4)
    draw.text((plot_right - 152, plot_top + 15), "Actual", fill=TEXT, font=font(15))
    draw.line((plot_right - 193, plot_top + 49, plot_right - 160, plot_top + 49), fill=ORANGE, width=4)
    draw.text((plot_right - 152, plot_top + 37), "Predicted", fill=TEXT, font=font(15))


def pick_median_index(values: np.ndarray, eligible: np.ndarray) -> int:
    candidate = np.flatnonzero(eligible)
    median = float(np.median(values[candidate]))
    return int(candidate[np.argmin(np.abs(values[candidate] - median))])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BUNDLE / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((BUNDLE / "scaler.json").read_text(encoding="utf-8"))
    indices = [metadata["feature_columns"].index(code) for code in TARGETS]
    mean = np.asarray(scaler["mean"], dtype=np.float64)[indices]
    scale = np.asarray(scaler["scale"], dtype=np.float64)[indices]
    pred_file = np.load(BUNDLE / "pred.npy", mmap_mode="r")
    true_file = np.load(BUNDLE / "true.npy", mmap_mode="r")

    sample_ids = np.linspace(0, len(pred_file) - 1, 5000, dtype=np.int64)
    pred_parts, true_parts = [], []
    for start in range(0, len(sample_ids), 100):
        rows = sample_ids[start:start + 100]
        pred_parts.append(pred_file[rows, :, :][:, :, indices].astype(np.float64) * scale + mean)
        true_parts.append(true_file[rows, :, :][:, :, indices].astype(np.float64) * scale + mean)
    pred = np.concatenate(pred_parts, axis=0)
    truth = np.concatenate(true_parts, axis=0)
    mae = np.mean(np.abs(pred - truth), axis=1)

    steam_delta = truth[:, -1, 0] - truth[:, 0, 0]
    abs_delta = np.abs(steam_delta)
    steam_mae = mae[:, 0]
    q20, q80 = np.quantile(steam_delta, [.2, .8])
    stable_limit = np.quantile(abs_delta, .2)
    cases = [
        ("Typical decline", steam_delta <= q20),
        ("Typical stable period", abs_delta <= stable_limit),
        ("Typical rise", steam_delta >= q80),
        ("Challenging case (high error)", steam_mae >= np.quantile(steam_mae, .98)),
    ]
    selected: list[tuple[str, int]] = []
    for label, mask in cases[:3]:
        selected.append((label, pick_median_index(steam_mae, mask)))
    selected.append((cases[3][0], int(np.flatnonzero(cases[3][1])[len(np.flatnonzero(cases[3][1])) // 2])))

    canvas = Image.new("RGB", (1900, 1280), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 20), "Model1 10-second forecast: main steam flow test cases", fill=TEXT, font=font(35, True))
    draw.text((45, 65), "Four cases selected from 5,000 evenly spaced test windows; the first three have median error within their trend group.", fill="#4B5563", font=font(18))
    records = []
    for slot, (label, sample_pos) in enumerate(selected):
        row, col = divmod(slot, 2)
        box = (35 + col * 935, 115 + row * 575, 930 + col * 935, 665 + row * 575)
        actual, forecast = truth[sample_pos, :, 0], pred[sample_pos, :, 0]
        note = f"test window #{int(sample_ids[sample_pos])}; actual change {actual[-1] - actual[0]:+.2f} {UNITS[TARGETS[0]]}; MAE {steam_mae[sample_pos]:.3f}"
        line_chart(draw, box, actual, forecast, label, UNITS[TARGETS[0]], note)
        records.append({"chart": label, "target": TARGETS[0], "test_window_index": int(sample_ids[sample_pos]), "actual_change": float(actual[-1] - actual[0]), "predicted_change": float(forecast[-1] - forecast[0]), "mae": float(steam_mae[sample_pos]), "endpoint_error": float(abs(forecast[-1] - actual[-1]))})
    canvas.save(OUT / "main_steam_test_cases.png")

    canvas = Image.new("RGB", (1900, 1520), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 20), "Model1 10-second forecast: representative test windows for five targets", fill=TEXT, font=font(34, True))
    draw.text((45, 65), "Each panel uses the window whose MAE is closest to the median of 5,000 evenly spaced test samples.", fill="#4B5563", font=font(18))
    for target_pos, code in enumerate(TARGETS):
        sample_pos = pick_median_index(mae[:, target_pos], np.ones(len(sample_ids), dtype=bool))
        row, col = divmod(target_pos, 2)
        box = (35 + col * 935, 115 + row * 455, 930 + col * 935, 545 + row * 455)
        actual, forecast = truth[sample_pos, :, target_pos], pred[sample_pos, :, target_pos]
        note = f"test window #{int(sample_ids[sample_pos])}; MAE {mae[sample_pos, target_pos]:.3f}; endpoint error {abs(forecast[-1] - actual[-1]):.3f}"
        line_chart(draw, box, actual, forecast, NAMES[code], UNITS[code], note)
        records.append({"chart": "median_MAE_case", "target": code, "test_window_index": int(sample_ids[sample_pos]), "actual_change": float(actual[-1] - actual[0]), "predicted_change": float(forecast[-1] - forecast[0]), "mae": float(mae[sample_pos, target_pos]), "endpoint_error": float(abs(forecast[-1] - actual[-1]))})
    canvas.save(OUT / "five_targets_test_cases.png")

    with (OUT / "case_selection.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT / "README.md").write_text(
        "Charts use 5,000 evenly spaced windows from the complete test prediction arrays. "
        "Main-steam rise/decline/stable cases use the median-MAE case within their actual-trend group; "
        "the fourth main-steam panel intentionally shows a high-error case.\n",
        encoding="utf-8",
    )
    print(f"charts written to {OUT}")


if __name__ == "__main__":
    main()
