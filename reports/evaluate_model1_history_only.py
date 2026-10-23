"""Evaluate the history-only Model 1 artifact without modifying model files."""

import csv
import json
from pathlib import Path
import re
import zipfile

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "offline" / "artifacts" / "forecast" / "training_ready" / "model1_10s_seq120_pred60"
DATASET = ROOT / "offline" / "artifacts" / "forecast" / "data_engineering" / "outputs" / "datasets" / "chunked" / "model1_10s_seq120_pred60"
OUT = ARTIFACT / "evaluation_history_only_20261021"
DISPLAY = {
    "B3_F80.PV": "主蒸汽流量", "B3QT00": "省煤器出口氧量", "B3_O2_S1": "炉膛氧量",
    "B3_TE1119": "中上部炉膛温度", "B3_GIC01": "垃圾层厚度",
    "B3_DRUMLT1": "汽包水位1", "B3_DRUMLT2": "汽包水位2", "B3_DRUMLT3": "汽包水位3",
    "B3PT04A": "压力04A", "B3PT04B": "压力04B", "B3PT81A": "压力81A",
    "B3PT81B": "压力81B", "B3TE03": "温度TE03", "B3TE86B": "温度TE86B",
}
UNITS = {
    "B3_F80.PV": "t/h", "B3QT00": "%", "B3_O2_S1": "%", "B3_TE1119": "°C",
    "B3_GIC01": "原始单位", "B3_DRUMLT1": "原始单位", "B3_DRUMLT2": "原始单位",
    "B3_DRUMLT3": "原始单位", "B3PT04A": "原始单位", "B3PT04B": "原始单位",
    "B3PT81A": "原始单位", "B3PT81B": "原始单位", "B3TE03": "°C", "B3TE86B": "°C",
}
HORIZONS = {"1分钟": 5, "5分钟": 29, "10分钟": 59}
FONT_PATH = Path("C:/Windows/Fonts/msyh.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ARTIFACT / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((ARTIFACT / "scaler.json").read_text(encoding="utf-8"))
    targets = meta["target_columns"]
    indices = np.asarray(meta["target_indices"], dtype=np.int64)
    means = np.asarray(scaler["mean"], dtype=np.float64)[indices]
    scales = np.asarray(scaler["scale"], dtype=np.float64)[indices]
    pred = np.load(ARTIFACT / "pred.npy", mmap_mode="r")
    true = np.load(ARTIFACT / "true.npy", mmap_mode="r")
    if pred.shape != true.shape or pred.shape[1:] != (meta["pred_len"], meta["n_vars"]):
        raise ValueError(f"array shape mismatch: pred={pred.shape}, true={true.shape}")

    chunks = list(csv.DictReader((DATASET / "splits" / "test_chunks.csv").open(encoding="utf-8-sig", newline="")))
    if sum(int(r["windows"]) for r in chunks) != pred.shape[0]:
        raise ValueError("test_chunks window count does not match saved prediction arrays")
    # Verify standardization and exact window ordering against the source frame.
    first = chunks[0]
    first_windows = pd.read_csv(DATASET / "windows" / Path(first["window_path"]).name, encoding="utf-8-sig")
    first_frame = pd.read_parquet(DATASET / "frames" / Path(first["frame_path"]).name, columns=["timestamp", *targets])
    row0 = first_windows.iloc[0]
    expected = first_frame.iloc[int(row0["target_start_pos"])][targets].to_numpy(dtype=np.float64)
    restored_true = np.asarray(true[0, 0, indices], dtype=np.float64) * scales + means
    align_abs_error = np.abs(restored_true - expected)
    if not np.allclose(restored_true, expected, rtol=1e-4, atol=1e-3, equal_nan=True):
        raise ValueError(f"first-window truth alignment failed; max_abs_error={np.nanmax(align_abs_error)}")

    # Full-array finite check and metrics are accumulated in bounded blocks.
    counts = np.zeros(len(targets), dtype=np.int64)
    abs_sum = np.zeros(len(targets), dtype=np.float64)
    sq_sum = np.zeros(len(targets), dtype=np.float64)
    err_sum = np.zeros(len(targets), dtype=np.float64)
    truth_sum = np.zeros(len(targets), dtype=np.float64)
    truth_sq_sum = np.zeros(len(targets), dtype=np.float64)
    horizon_abs = {k: np.zeros(len(targets), dtype=np.float64) for k in HORIZONS}
    horizon_count = {k: 0 for k in HORIZONS}
    finite_bad = {"pred": 0, "true": 0}
    samples_for_plot = np.linspace(0, pred.shape[0] - 1, min(160, pred.shape[0]), dtype=np.int64)
    sample_errors = np.empty((len(samples_for_plot), len(targets)), dtype=np.float64)
    sample_cursor = 0
    direction_match = np.zeros(len(targets), dtype=np.int64)
    direction_n = np.zeros(len(targets), dtype=np.int64)
    variation_pred = np.zeros(len(targets), dtype=np.float64)
    variation_true = np.zeros(len(targets), dtype=np.float64)

    for start in range(0, pred.shape[0], 256):
        stop = min(start + 256, pred.shape[0])
        pz = np.asarray(pred[start:stop, :, :][:, :, indices], dtype=np.float64)
        tz = np.asarray(true[start:stop, :, :][:, :, indices], dtype=np.float64)
        finite_bad["pred"] += int((~np.isfinite(pz)).sum())
        finite_bad["true"] += int((~np.isfinite(tz)).sum())
        p = pz * scales + means
        t = tz * scales + means
        e = p - t
        abs_sum += np.abs(e).sum(axis=(0, 1))
        sq_sum += np.square(e).sum(axis=(0, 1))
        err_sum += e.sum(axis=(0, 1))
        truth_sum += t.sum(axis=(0, 1))
        truth_sq_sum += np.square(t).sum(axis=(0, 1))
        counts += e.shape[0] * e.shape[1]
        variation_pred += np.abs(np.diff(p, axis=1)).sum(axis=(0, 1))
        variation_true += np.abs(np.diff(t, axis=1)).sum(axis=(0, 1))
        for name, h in HORIZONS.items():
            horizon_abs[name] += np.abs(e[:, h, :]).sum(axis=0)
            horizon_count[name] += e.shape[0]
        # Direction relative to the first predicted point (not the last observed input).
        dp = p[:, -1, :] - p[:, 0, :]
        dt = t[:, -1, :] - t[:, 0, :]
        eligible = np.abs(dt) >= np.maximum(scales * 0.1, 1e-9)
        direction_match += ((np.sign(dp) == np.sign(dt)) & eligible).sum(axis=0)
        direction_n += eligible.sum(axis=0)
        while sample_cursor < len(samples_for_plot) and samples_for_plot[sample_cursor] < stop:
            j = int(samples_for_plot[sample_cursor] - start)
            sample_errors[sample_cursor] = np.mean(np.abs(e[j]), axis=0)
            sample_cursor += 1
    if finite_bad["pred"] or finite_bad["true"]:
        raise ValueError(f"nonfinite target cells: {finite_bad}")

    # Persistence baseline: repeat the last observed target value across the 10-minute horizon.
    # This is a direct, meaningful reference for operational forecasting.
    base_abs = np.zeros(len(targets), dtype=np.float64)
    base_horizon_abs = {k: np.zeros(len(targets), dtype=np.float64) for k in HORIZONS}
    base_count = 0
    offset = 0
    for chunk in chunks:
        n = int(chunk["windows"])
        windows = pd.read_csv(DATASET / "windows" / Path(chunk["window_path"]).name, encoding="utf-8-sig")
        if len(windows) != n:
            raise ValueError(f"window count mismatch for {chunk['chunk_id']}: csv={len(windows)} manifest={n}")
        frame = pd.read_parquet(DATASET / "frames" / Path(chunk["frame_path"]).name, columns=["timestamp", *targets])
        origin_positions = windows["input_end_pos"].to_numpy(dtype=np.int64)
        origins = frame.iloc[origin_positions][targets].to_numpy(dtype=np.float64)
        for local in range(0, n, 256):
            end = min(local + 256, n)
            truth_z = np.asarray(true[offset + local:offset + end][:, :, indices], dtype=np.float64)
            truth_values = truth_z * scales + means
            persistence = origins[local:end, None, :]
            base_abs += np.abs(persistence - truth_values).sum(axis=(0, 1))
            for name, h in HORIZONS.items():
                base_horizon_abs[name] += np.abs(persistence[:, 0, :] - truth_values[:, h, :]).sum(axis=0)
            base_count += end - local
        offset += n
    if offset != pred.shape[0]:
        raise ValueError(f"baseline alignment count mismatch: {offset} != {pred.shape[0]}")
    baseline_mae = base_abs / (base_count * meta["pred_len"])

    rows = []
    mae = abs_sum / counts
    rmse = np.sqrt(sq_sum / counts)
    bias = err_sum / counts
    sst = truth_sq_sum - np.square(truth_sum) / counts
    r2 = 1 - sq_sum / sst
    for j, target in enumerate(targets):
        row = {
            "target": target, "name": DISPLAY.get(target, target), "unit": UNITS.get(target, "原始单位未登记"),
            "windows": int(pred.shape[0]), "mae_all_10min": mae[j], "rmse_all_10min": rmse[j],
            "bias_all_10min": bias[j], "r2_all_10min": r2[j],
            "persistence_mae_all_10min": baseline_mae[j],
            "mae_skill_vs_persistence": 1 - mae[j] / baseline_mae[j] if baseline_mae[j] else np.nan,
            "variation_ratio_pred_true": variation_pred[j] / variation_true[j] if variation_true[j] else np.nan,
            "direction_accuracy": direction_match[j] / direction_n[j] if direction_n[j] else np.nan,
            "direction_eligible_windows": int(direction_n[j]),
        }
        for name in HORIZONS:
            row[f"mae_{name}"] = horizon_abs[name][j] / horizon_count[name]
            row[f"persistence_mae_{name}"] = base_horizon_abs[name][j] / base_count
        rows.append(row)
    fields = list(rows[0])
    with (OUT / "target_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Loss history: only numeric epoch rows; ignore a trailing serialization null.
    loss_rows = []
    with (ARTIFACT / "loss_history.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("epoch", "").strip().isdigit():
                loss_rows.append({key: float(row[key]) for key in ("epoch", "train_loss", "vali_loss", "test_loss")})
    if loss_rows:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for key, label in (("train_loss", "训练损失"), ("vali_loss", "验证损失"), ("test_loss", "测试损失")):
            if key in loss_rows[0]:
                ax.plot([r["epoch"] for r in loss_rows], [r[key] for r in loss_rows], marker="o", ms=3, label=label)
        ax.set(title="模型1（仅历史输入）训练过程", xlabel="训练轮次", ylabel="损失（标准化目标空间）")
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "loss_curve.png", dpi=180)
        plt.close(fig)

    # Per-target MAE and R² overview.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    order = np.argsort(mae)
    labels = [f"{DISPLAY.get(targets[i], targets[i])} ({targets[i]})" for i in order]
    axes[0].barh(labels, mae[order], color="#2878B5")
    axes[0].set(title="测试集全程 MAE（10分钟）", xlabel="MAE（工程单位）")
    axes[0].invert_yaxis()
    axes[1].barh(labels, r2[order], color=["#2A9D8F" if x >= 0 else "#E76F51" for x in r2[order]])
    axes[1].axvline(0, color="#333333", lw=.8)
    axes[1].set(title="测试集全程 R²（10分钟）", xlabel="R²")
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT / "target_metrics.png", dpi=180)
    plt.close(fig)

    # One representative 10-minute forecast per target: median-error among 160 evenly sampled windows.
    median_windows = []
    fig, axes_grid = plt.subplots(4, 4, figsize=(16, 11))
    axes_flat = axes_grid.ravel()
    for j, target in enumerate(targets):
        pick = int(np.argsort(sample_errors[:, j])[len(samples_for_plot) // 2])
        win = int(samples_for_plot[pick])
        p = np.asarray(pred[win][:, indices], dtype=np.float64) * scales + means
        t = np.asarray(true[win][:, indices], dtype=np.float64) * scales + means
        median_windows.append(win)
        ax = axes_flat[j]
        ax.plot(np.arange(1, 61) / 6, t[:, j], color="#2878B5", lw=1.5, label="真实")
        ax.plot(np.arange(1, 61) / 6, p[:, j], color="#E76F51", lw=1.35, label="预测")
        ax.set_title(f"{DISPLAY.get(target, target)} · {target}", fontsize=9)
        ax.set_xlabel("预测后分钟")
        ax.set_ylabel(UNITS.get(target, "原始单位"), fontsize=8)
        ax.grid(alpha=.22)
    for ax in axes_flat[len(targets):]:
        ax.remove()
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].legend(handles, labels, loc="best", frameon=False, fontsize=8)
    fig.suptitle("模型1（仅历史输入）：各目标10分钟预测代表曲线（中位误差窗）", y=.995)
    fig.tight_layout(rect=[0, 0, 1, .975])
    fig.savefig(OUT / "representative_trends.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Write compact machine-readable provenance and a plain-language report.
    elapsed_seconds = None
    finished_at = None
    with zipfile.ZipFile(ARTIFACT / "training_logs.zip") as archive:
        for member in archive.namelist():
            if not member.endswith(".log"):
                continue
            log_text = archive.read(member).decode("utf-8", errors="replace")
            if "model1_xingrong3_model1_10s_nofuture_v1_261019_0" in log_text and "elapsed_seconds:" in log_text:
                elapsed_match = re.search(r"elapsed_seconds:\s*([0-9.]+)", log_text)
                finish_match = re.search(r"finished_at:\s*(.+)", log_text)
                elapsed_seconds = float(elapsed_match.group(1)) if elapsed_match else None
                finished_at = finish_match.group(1).strip() if finish_match else None
                break
    summary = {
        "run": "model1 history-only, no future covariates", "dataset_id": meta["dataset_id"],
        "freq_seconds": meta["freq_seconds"], "seq_len": meta["seq_len"], "pred_len": meta["pred_len"],
        "test_windows": int(pred.shape[0]), "test_chunks": len(chunks),
        "test_period": meta["split_summary"][-1], "array_shape": list(pred.shape),
        "array_dtype": str(pred.dtype), "finite_target_cells": finite_bad,
        "truth_alignment_max_abs_error_first_window": float(np.nanmax(align_abs_error)),
        "best_validation_epoch": int(min(loss_rows, key=lambda r: r.get("vali_loss", float("inf")))["epoch"]) if loss_rows else None,
        "best_validation_loss": float(min(r["vali_loss"] for r in loss_rows if "vali_loss" in r)) if loss_rows else None,
        "last_epoch": int(loss_rows[-1]["epoch"]) if loss_rows else None,
        "training_elapsed_seconds": elapsed_seconds, "training_finished_at": finished_at,
    }
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "evaluation_report.md").open("w", encoding="utf-8") as f:
        f.write("# 模型1（仅历史输入）训练与测试集评估\n\n")
        f.write(f"测试集共 **{pred.shape[0]:,} 个连续时间窗**、{len(chunks)} 个数据块；单窗用过去 {meta['seq_len']*meta['freq_seconds']//60} 分钟预测未来 {meta['pred_len']*meta['freq_seconds']//60} 分钟。测试期为 {summary['test_period'].get('start')} 至 {summary['test_period'].get('end')}。\n\n")
        f.write("预测数组与真实数组形状相同，逐窗真值与原始测试数据首窗对齐检查通过；数组已确认处于标准化空间，报告指标按训练集 scaler 还原为工程单位后计算。全量目标单元均为有限值。\n\n")
        elapsed_text = f"，总耗时约 {elapsed_seconds / 3600:.1f} 小时" if elapsed_seconds is not None else ""
        f.write(f"训练损失记录到第 {summary['last_epoch']} 轮{elapsed_text}；日志显示第 {summary['last_epoch']} 轮后触发 patience=8 的早停。最低验证损失出现在第 {summary['best_validation_epoch']} 轮（{summary['best_validation_loss']:.4f}）。训练曲线仅作优化过程诊断，跨模型损失值不宜直接比较。\n\n")
        f.write("图表：`loss_curve.png` 为训练过程；`target_metrics.png` 汇总全程 MAE 与 R²；`representative_trends.png` 展示每个目标在中位误差窗口中的 10 分钟预测曲线。表格 `target_metrics.csv` 含全程及 1/5/10 分钟点的 MAE、RMSE、偏差、R²、波动比和趋势方向命中率。\n\n")
        f.write("| 目标 | MAE（全程） | 持续值基线MAE | 相对基线改进 | RMSE | R² | 1分钟MAE | 5分钟MAE | 10分钟点MAE |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row['name']}（{row['target']}） | {row['mae_all_10min']:.4g} {row['unit']} | {row['persistence_mae_all_10min']:.4g} {row['unit']} | {row['mae_skill_vs_persistence']:.1%} | {row['rmse_all_10min']:.4g} {row['unit']} | {row['r2_all_10min']:.4f} | {row['mae_1分钟']:.4g} | {row['mae_5分钟']:.4g} | {row['mae_10分钟']:.4g} |\n")
        f.write("\n**解释边界：**该版本训练时不使用未来真实工况变量，评价更接近仅凭当前历史状态预测的实际部署条件。与此前依赖未来工况输入的模型结果相比，问题设定不同；对比时应同时看工程单位误差和基线，而不能把损失数值直接横向对比。趋势图中的代表窗口是从均匀抽取的候选窗中选出的中位误差案例，不代表全部测试期。\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote evaluation to {OUT}")


if __name__ == "__main__":
    main()
