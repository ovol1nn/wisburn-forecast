"""Calibrate per-target, per-horizon forecast confidence for current runtime bundles."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[3]
TRAINING_READY = ROOT / "offline" / "artifacts" / "forecast" / "training_ready"
RUNTIME_MODELS = ROOT / "assets" / "models" / "xingrong_3"
HORIZONS_SECONDS = (10, 30, 60, 120, 300, 600)
CALIBRATION_FRACTION = 0.40
INTERVAL_LEVEL = 0.90

DISPLAY = {
    "B3_F80.PV": ("主蒸汽流量", "t/h"), "B3QT00": ("省煤器出口氧量", "%"),
    "B3_O2_S1": ("炉膛氧量", "%"), "B3_TE1119": ("中上部炉膛温度", "℃"),
    "B3_GIC01": ("垃圾层厚度", "工程单位"), "B3QT02BBC": ("CO", "mg/Nm³"),
    "B3QT02CBC": ("HCl", "mg/Nm³"), "B3QT02EBC": ("NOx", "mg/Nm³"),
    "B3QT02ABC": ("SO2", "mg/Nm³"), "B3QT02FC2": ("烟尘", "mg/Nm³"),
}


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    runtime_version: str
    runtime_dir: Path
    source_dir: Path


SPECS = (
    ModelSpec("model1", "model1-10s-v20260823-r1",
              RUNTIME_MODELS / "model1-10s-v20260823-r1" / "model1",
              TRAINING_READY / "model1_xingrong3_model1_10s_trend_endpoint_v1_0"),
    ModelSpec("model2", "pred600-v20260809-r1",
              RUNTIME_MODELS / "pred600-v20260809-r1" / "model2",
              TRAINING_READY / "model2_bundle_source"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confidence_score(improvement: float, direction_accuracy: float, coverage: float) -> float:
    """Evidence score, intentionally not a probability of being correct."""
    skill = float(np.clip(improvement / 0.50, 0.0, 1.0))
    direction = float(np.clip((direction_accuracy - 0.50) / 0.40, 0.0, 1.0))
    calibration = float(np.clip(1.0 - abs(coverage - INTERVAL_LEVEL) / INTERVAL_LEVEL, 0.0, 1.0))
    return round(0.45 * skill + 0.30 * direction + 0.25 * calibration, 4)


def confidence_level(score: float, improvement: float, direction_accuracy: float) -> str:
    if improvement <= 0 or direction_accuracy < 0.60:
        return "low"
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def validate_spec(spec: ModelSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime = load_json(spec.runtime_dir / "runtime_bundle.json")
    runtime_meta = load_json(spec.runtime_dir / "metadata.json")
    source_meta = load_json(spec.source_dir / "metadata.json")
    if runtime.get("runtime_bundle_version") != spec.runtime_version:
        raise ValueError(f"{spec.slug}: runtime version mismatch")
    expected = runtime["contract"]
    for key in ("freq_seconds", "seq_len", "label_len", "pred_len", "target_columns"):
        if runtime_meta.get(key) != expected.get(key) or source_meta.get(key) != expected.get(key):
            raise ValueError(f"{spec.slug}: {key} differs between runtime and training source")
    source_sha = sha256(spec.source_dir / "checkpoint.pth")
    if source_sha != runtime.get("checkpoint_sha256"):
        raise ValueError(f"{spec.slug}: source checkpoint SHA-256 does not match runtime bundle")
    return runtime, runtime_meta, source_meta


def selected_indices(total: int, limit: int | None) -> np.ndarray:
    if limit is None or limit >= total:
        return np.arange(total, dtype=np.int64)
    return np.linspace(0, total - 1, limit, dtype=np.int64)


def metadata_test_windows(metadata: dict[str, Any]) -> int | None:
    for item in metadata.get("split_summary", []):
        if item.get("split") == "test":
            return int(item["windows"])
    return None


def read_targets(array: np.ndarray, rows: np.ndarray, horizons: list[int], target_indices: list[int], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # Slice rows first to keep memory use bounded; source arrays are memory mapped.
    values = np.asarray(array[rows, :, :][:, horizons, :][:, :, target_indices], dtype=np.float64)
    return values * scale[None, None, :] + mean[None, None, :]


def calculate(spec: ModelSpec, max_windows: int | None, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime, runtime_meta, source_meta = validate_spec(spec)
    scaler = load_json(spec.source_dir / "scaler.json")
    pred = np.load(spec.source_dir / "pred.npy", mmap_mode="r")
    truth = np.load(spec.source_dir / "true.npy", mmap_mode="r")
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError(f"{spec.slug}: pred/true arrays are incompatible: {pred.shape}, {truth.shape}")
    if pred.shape[1] != int(source_meta["pred_len"]):
        raise ValueError(f"{spec.slug}: saved prediction length differs from metadata")

    step = int(source_meta["freq_seconds"])
    horizons = [seconds // step - 1 for seconds in HORIZONS_SECONDS]
    if min(horizons) < 0 or max(horizons) >= pred.shape[1]:
        raise ValueError(f"{spec.slug}: saved arrays do not cover all requested horizons")
    targets = list(source_meta["target_columns"])
    target_indices = list(source_meta["target_indices"])
    feature_columns = list(source_meta["feature_columns"])
    if [feature_columns[i] for i in target_indices] != targets:
        raise ValueError(f"{spec.slug}: target order no longer matches feature columns")
    mean = np.asarray(scaler["mean"], dtype=np.float64)[target_indices]
    scale = np.asarray(scaler["scale"], dtype=np.float64)[target_indices]
    rows = selected_indices(pred.shape[0], max_windows)
    split = max(1, min(len(rows) - 1, int(len(rows) * CALIBRATION_FRACTION)))
    calibration_rows, evaluation_rows = rows[:split], rows[split:]
    test_windows = metadata_test_windows(source_meta)
    saved_coverage = float(pred.shape[0] / test_windows) if test_windows else None
    release_cap = 0.65 if saved_coverage is not None and saved_coverage < 0.95 else 1.0

    residuals: list[list[list[np.ndarray]]] = [[[] for _ in horizons] for _ in targets]
    for begin in range(0, len(calibration_rows), batch_size):
        batch = calibration_rows[begin:begin + batch_size]
        p = read_targets(pred, batch, horizons, target_indices, mean, scale)
        y = read_targets(truth, batch, horizons, target_indices, mean, scale)
        for target_i in range(len(targets)):
            for horizon_i in range(len(horizons)):
                residuals[target_i][horizon_i].append(y[:, horizon_i, target_i] - p[:, horizon_i, target_i])
    quantiles = [[np.quantile(np.concatenate(residuals[t][h]), [(1 - INTERVAL_LEVEL) / 2, 1 - (1 - INTERVAL_LEVEL) / 2])
                  for h in range(len(horizons))] for t in range(len(targets))]

    sums = np.zeros((len(targets), len(horizons), 5), dtype=np.float64)
    # columns: abs model error, abs persistence error, covered count, direction correct, direction eligible
    for begin in range(0, len(evaluation_rows), batch_size):
        batch = evaluation_rows[begin:begin + batch_size]
        p = read_targets(pred, batch, horizons, target_indices, mean, scale)
        y = read_targets(truth, batch, horizons, target_indices, mean, scale)
        current = read_targets(truth, batch, [0], target_indices, mean, scale)[:, 0, :]
        for target_i in range(len(targets)):
            for horizon_i in range(len(horizons)):
                error = y[:, horizon_i, target_i] - p[:, horizon_i, target_i]
                actual_change = y[:, horizon_i, target_i] - current[:, target_i]
                predicted_change = p[:, horizon_i, target_i] - current[:, target_i]
                threshold = max(float(np.quantile(np.abs(actual_change), 0.10)), 1e-9)
                eligible = np.abs(actual_change) >= threshold
                lower, upper = quantiles[target_i][horizon_i]
                sums[target_i, horizon_i, 0] += np.abs(error).sum()
                sums[target_i, horizon_i, 1] += np.abs(actual_change).sum()
                sums[target_i, horizon_i, 2] += ((error >= lower) & (error <= upper)).sum()
                sums[target_i, horizon_i, 3] += (np.sign(actual_change[eligible]) == np.sign(predicted_change[eligible])).sum()
                sums[target_i, horizon_i, 4] += eligible.sum()

    records: list[dict[str, Any]] = []
    for target_i, code in enumerate(targets):
        name, unit = DISPLAY.get(code, (code, "工程单位"))
        for horizon_i, seconds in enumerate(HORIZONS_SECONDS):
            model_mae, persistence_mae, covered, correct, eligible = sums[target_i, horizon_i]
            model_mae /= len(evaluation_rows)
            persistence_mae /= len(evaluation_rows)
            coverage = covered / len(evaluation_rows)
            direction = correct / eligible if eligible else 0.5
            improvement = 1 - model_mae / persistence_mae if persistence_mae > 1e-12 else 0.0
            score = confidence_score(improvement, direction, coverage)
            release_score = min(score, release_cap)
            lower, upper = quantiles[target_i][horizon_i]
            records.append({
                "model_slug": spec.slug, "runtime_version": spec.runtime_version, "point_code": code,
                "target_name": name, "unit": unit, "horizon_seconds": seconds,
                "evaluation_windows": len(evaluation_rows), "calibration_windows": len(calibration_rows),
                "model_mae": round(float(model_mae), 6), "persistence_proxy_mae": round(float(persistence_mae), 6),
                "improvement_vs_persistence_proxy": round(float(improvement), 6),
                "trend_direction_accuracy": round(float(direction), 6), "direction_eligible_windows": int(eligible),
                "interval_level": INTERVAL_LEVEL, "interval_residual_lower": round(float(lower), 6),
                "interval_residual_upper": round(float(upper), 6), "interval_width": round(float(upper - lower), 6),
                "interval_coverage": round(float(coverage), 6), "base_confidence": score,
                "confidence_level": confidence_level(score, improvement, direction),
                "release_confidence": round(float(release_score), 4),
                "release_confidence_level": confidence_level(release_score, improvement, direction),
                "saved_test_coverage": round(saved_coverage, 6) if saved_coverage is not None else None,
                "release_confidence_cap": release_cap,
                "input_ood_status": "unavailable", "input_ood_score": None,
                "limitations": "persistence uses first true future node as current-state proxy; saved input windows unavailable",
            })
    manifest = {
        "model_slug": spec.slug, "runtime_version": spec.runtime_version,
        "checkpoint_sha256": runtime["checkpoint_sha256"], "source_training_package": str(spec.source_dir),
        "saved_test_windows": int(pred.shape[0]), "selected_windows": int(len(rows)),
        "metadata_test_windows": test_windows, "saved_test_coverage": saved_coverage,
        "release_confidence_cap": release_cap,
        "calibration_windows": int(len(calibration_rows)), "evaluation_windows": int(len(evaluation_rows)),
        "contract": runtime["contract"], "input_ood_status": "unavailable",
    }
    return records, manifest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    grouped = {slug: [row for row in rows if row["model_slug"] == slug] for slug in ("model1", "model2")}

    def matrix(metric: str, title: str, filename: str, cmap: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
        image = None
        for ax, (slug, items) in zip(axes, grouped.items()):
            target_codes = list(dict.fromkeys(row["point_code"] for row in items))
            horizon_values = list(HORIZONS_SECONDS)
            values = np.asarray([[next(row[metric] for row in items if row["point_code"] == code and row["horizon_seconds"] == horizon)
                                  for horizon in horizon_values] for code in target_codes], dtype=float)
            image = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set_title(f"{slug} · {items[0]['runtime_version']}")
            ax.set_xticks(range(len(horizon_values)), [f"{value}s" for value in horizon_values])
            ax.set_yticks(range(len(target_codes)), [DISPLAY.get(code, (code, ""))[0] for code in target_codes])
            for row_i in range(values.shape[0]):
                for col_i in range(values.shape[1]):
                    color = "white" if values[row_i, col_i] < .35 or values[row_i, col_i] > .72 else "black"
                    ax.text(col_i, row_i, f"{values[row_i, col_i]:.2f}", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=axes, shrink=.9, label="比例")
        fig.suptitle(title, fontsize=15)
        fig.savefig(output / filename, dpi=180); plt.close(fig)

    matrix("release_confidence", "当前版本：各目标、各时域发布可用置信度（非概率）", "confidence_heatmap.png", "RdYlGn")
    matrix("interval_coverage", "90% 经验预测区间：评价段实际覆盖率", "interval_coverage.png", "Blues")


def write_report(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = ["# 当前预测模型置信度校准", "", "## 版本与口径", "",
             f"- 运行模型版本：`{', '.join(f'{x['model_slug']}={x['runtime_version']}' for x in manifest['models'])}`。",
             f"- 选取测试窗口：{manifest['selected_windows']:,}；校准/评价：{manifest['calibration_windows']:,} / {manifest['evaluation_windows']:,}。",
             "- 置信度是已保存测试数据上的经验可靠性，不是预测正确概率，不用于证明控制因果效应。",
             "- 输入窗口未随预测数组保存，OOD 分数当前不可评价；不得将本结果表述为现场实时置信度。",
             "- 若保存预测未覆盖原测试集 95%，发布可用置信度上限为 0.65，避免将局部保存样本的好结果误标为高置信度。", "",
             "## 各目标、各时域结果", "",
             "| 模型 | 目标 | 时域 | MAE | 相对保持不变改善 | 方向一致率 | 90%区间覆盖 | 基础/发布可用置信度 | 发布分级 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['model_slug']} | {row['target_name']} | {row['horizon_seconds']}s | {row['model_mae']:.4f} {row['unit']} | {row['improvement_vs_persistence_proxy']:.1%} | {row['trend_direction_accuracy']:.1%} | {row['interval_coverage']:.1%} | {row['base_confidence']:.3f}/{row['release_confidence']:.3f} | {row['release_confidence_level']} |")
    lines += ["", "## 使用边界", "", "- `low` 表示在相对保持不变的动态收益、方向判断或区间可校准性中至少一项不足；不是目标值一定预测错误。",
              "- 近端时域的保持不变基线使用首个未来真值代理，须在获得输入历史窗口后复核。",
              "- 后端接入前需要对完整测试回放和典型异常工况进行人工审核；本次输出不自动改变 `Forecast.confidence`。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Current-version per-target forecast confidence calibration")
    parser.add_argument("--max-windows", type=int, default=10_000, help="evenly sampled saved test windows per model")
    parser.add_argument("--full", action="store_true", help="use every saved test window")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for spec in SPECS:
        records, item = calculate(spec, None if args.full else args.max_windows, args.batch_size)
        rows.extend(records); manifests.append(item)
        print(f"calibrated {spec.slug}: {len(records)} target-horizon cells", flush=True)
    manifest = {"schema_version": "1.0", "method": "chronological_saved_test_calibration", "interval_level": INTERVAL_LEVEL,
                "selected_windows": sum(x["selected_windows"] for x in manifests),
                "calibration_windows": sum(x["calibration_windows"] for x in manifests),
                "evaluation_windows": sum(x["evaluation_windows"] for x in manifests), "models": manifests,
                "input_ood_status": "unavailable"}
    write_csv(output / "per_target_horizon_metrics.csv", rows)
    (output / "forecast_confidence.json").write_text(json.dumps({"schema_version": "1.0", "models": manifests, "cells": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output / "report.md", rows, manifest)
    plot(rows, output)
    print(f"wrote {len(rows)} cells to {output}", flush=True)


if __name__ == "__main__":
    main()
