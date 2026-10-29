"""Calibrate a history-only Model 2 forecast blend without changing weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "offline/artifacts/forecast/training_ready/model2_10s_seq120_pred60_history_only"
BUNDLE = ROOT / "offline/artifacts/forecast/runtime_candidates/model2_10s_history_only_20261028"
DATA = ROOT / "offline/artifacts/forecast/data_engineering/outputs/datasets/chunked/model2_10s_seq120_pred60"
OUT = ROOT / "offline/artifacts/forecast/candidates/model2_history_only_blend_20261028_v1"
HORIZONS_SECONDS = (10, 30, 60, 120, 300, 600)
NONOVERLAP_STRIDE = 180  # 120 history + 60 forecast samples, at 10 seconds each.
ALPHA_GRID = np.linspace(0.0, 1.0, 21)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runtime():
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    scaler = json.loads((SOURCE / "scaler.json").read_text(encoding="utf-8"))
    bundle = json.loads((BUNDLE / "runtime_bundle.json").read_text(encoding="utf-8"))
    command = (SOURCE / "last_train_command.txt").read_text(encoding="utf-8")
    if not bundle.get("history_only") or not bundle.get("no_future_covariates") or "--no_future_covariates" not in command:
        raise ValueError("Model 2 package does not prove history-only training")
    if bundle.get("checkpoint_sha256") != sha256(SOURCE / "checkpoint.pth"):
        raise ValueError("Candidate checkpoint differs from the training source")
    if metadata["feature_columns"] != scaler["columns"]:
        raise ValueError("Scaler order differs from model input order")
    if metadata["target_columns"] != [metadata["feature_columns"][i] for i in metadata["target_indices"]]:
        raise ValueError("Target order differs from target indices")
    if len(metadata["target_columns"]) != 5 or metadata["freq_seconds"] != 10:
        raise ValueError("Unexpected Model 2 contract")

    model_config = {
        "task_name": "long_term_forecast", "enc_in": metadata["n_vars"],
        "seq_len": metadata["seq_len"], "pred_len": metadata["pred_len"],
        "patch_len": 16, "d_model": 128, "n_heads": 4, "e_layers": 2,
        "d_ff": 256, "factor": 3, "dropout": 0.1, "activation": "gelu",
        "use_dec_mask_embedding": True, "num_class": 2,
    }
    for key, value in (("patch_len", 16), ("d_model", 128), ("n_heads", 4),
                       ("e_layers", 2), ("d_ff", 256), ("factor", 3)):
        if f"--{key} {value}" not in command:
            raise ValueError(f"Training command does not match expected {key}={value}")

    tslib = ROOT / "offline/forecast/tslib"
    training = ROOT / "offline/forecast/training"
    sys.path.insert(0, str(tslib))
    sys.path.insert(0, str(training))
    from models.Crossformer import Model
    import chunked_tslib_dataset as adapter

    torch.set_num_threads(4)
    model = Model(SimpleNamespace(**model_config))
    state = torch.load(SOURCE / "checkpoint.pth", map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()}, strict=True)
    model.eval()
    scaler_mean = np.asarray(scaler["mean"], dtype=np.float64)
    scaler_scale = np.asarray(scaler["scale"], dtype=np.float64)
    return metadata, bundle, model, adapter, scaler_mean, scaler_scale


def make_dataset(adapter, metadata, mean, scale, split: str):
    scaler = adapter.StandardScaler(
        list(metadata["feature_columns"]), mean.astype("float32"), scale.astype("float32")
    )
    return adapter.ChunkedWindowDataset(
        DATA, split, feature_columns=list(metadata["feature_columns"]),
        target_columns=list(metadata["target_columns"]), scaler=scaler,
        seq_len=int(metadata["seq_len"]), label_len=int(metadata["label_len"]),
        pred_len=int(metadata["pred_len"]), cache_size=4,
    )


def infer_validation(model, metadata, mean, scale, dataset, selected_indices, max_batches: int | None = None):
    from torch.utils.data import DataLoader, Subset

    if max_batches is not None:
        selected_indices = selected_indices[:max_batches * 64]
    loader = DataLoader(Subset(dataset, selected_indices.tolist()), batch_size=64, shuffle=False, num_workers=0)
    inds = np.asarray(metadata["target_indices"], dtype=np.int64)
    tgt_mean, tgt_scale = mean[inds], scale[inds]
    horizons = np.asarray(HORIZONS_SECONDS) // int(metadata["freq_seconds"]) - 1
    predictions, truths, origins = [], [], []
    with torch.inference_mode():
        for batch_index, (x, y, x_mark, y_mark) in enumerate(loader):
            x, y = x.float(), y.float()
            decoder = y.clone()
            decoder[:, -metadata["pred_len"]:, :] = x[:, -1:, :].expand(-1, metadata["pred_len"], -1)
            target_tensor = torch.as_tensor(inds, dtype=torch.long)
            historical_mean = decoder[:, :metadata["label_len"], :].index_select(-1, target_tensor).mean(1, keepdim=True)
            decoder[:, -metadata["pred_len"]:, target_tensor] = historical_mean.expand(-1, metadata["pred_len"], -1)
            mask = torch.zeros((len(x), metadata["pred_len"], metadata["n_vars"]), dtype=torch.bool)
            mask[:, :, target_tensor] = True
            output = model(x, x_mark.float(), decoder, y_mark.float(), dec_mask=mask)
            p = output[:, horizons, :].index_select(-1, target_tensor).cpu().numpy()
            truth = y[:, -metadata["pred_len"]:, :].index_select(-1, target_tensor)[:, horizons, :].cpu().numpy()
            origin = x[:, -1, :].index_select(-1, target_tensor).cpu().numpy()
            predictions.append(p.astype("float64") * tgt_scale + tgt_mean)
            truths.append(truth.astype("float64") * tgt_scale + tgt_mean)
            origins.append(origin.astype("float64") * tgt_scale + tgt_mean)
            if (batch_index + 1) % 100 == 0:
                print(f"validation inference: {batch_index + 1} batches", flush=True)
    return tuple(np.concatenate(items, axis=0) for items in (predictions, truths, origins)), selected_indices


def read_test_arrays(metadata, mean, scale, adapter):
    source_pred = np.load(SOURCE / "pred.npy", mmap_mode="r")
    source_true = np.load(SOURCE / "true.npy", mmap_mode="r")
    if source_pred.shape != source_true.shape or source_pred.ndim != 3:
        raise ValueError("Saved prediction and truth arrays are incompatible")
    if source_pred.shape[1] != metadata["pred_len"] or source_pred.shape[2] != metadata["n_vars"]:
        raise ValueError("Saved arrays differ from the current model contract")
    dataset = make_dataset(adapter, metadata, mean, scale, "test")
    if len(dataset) != source_pred.shape[0]:
        raise ValueError(f"Test array windows {source_pred.shape[0]} != dataset windows {len(dataset)}")

    inds = np.asarray(metadata["target_indices"], dtype=np.int64)
    horizons = np.asarray(HORIZONS_SECONDS) // int(metadata["freq_seconds"]) - 1
    tgt_mean, tgt_scale = mean[inds], scale[inds]
    pred = np.empty((len(dataset), len(horizons), len(inds)), dtype="float64")
    truth = np.empty_like(pred)
    for start in range(0, len(dataset), 1024):
        end = min(len(dataset), start + 1024)
        pred[start:end] = np.asarray(source_pred[start:end, :, :][:, horizons, :][:, :, inds], dtype="float64") * tgt_scale + tgt_mean
        truth[start:end] = np.asarray(source_true[start:end, :, :][:, horizons, :][:, :, inds], dtype="float64") * tgt_scale + tgt_mean

    origin_parts = []
    import pandas as pd
    for chunk in dataset.chunks:
        windows = pd.read_csv(chunk.window_path, encoding="utf-8-sig")
        frame = pd.read_parquet(chunk.frame_path, columns=["timestamp", *metadata["target_columns"]])
        positions = windows["input_end_pos"].to_numpy(dtype=np.int64)
        origin_parts.append(frame.iloc[positions][metadata["target_columns"]].to_numpy(dtype="float64"))
        first_future = frame.iloc[windows["target_start_pos"].to_numpy(dtype=np.int64)][metadata["target_columns"]].to_numpy(dtype="float64")
        saved_first_future = truth[chunk.start:chunk.end, 0, :]
        if not np.allclose(first_future, saved_first_future, rtol=1e-4, atol=1e-5):
            raise ValueError(f"Saved test arrays do not align with local window order in {chunk.chunk_id}")
    origin = np.concatenate(origin_parts, axis=0)
    if len(origin) != len(dataset):
        raise ValueError("Could not align test prediction origins with saved windows")
    return pred, truth, origin, dataset


def independent_indices(dataset, stride: int = NONOVERLAP_STRIDE) -> np.ndarray:
    selected = []
    for chunk in dataset.chunks:
        selected.extend(range(chunk.start, chunk.end, stride))
    return np.asarray(selected, dtype=np.int64)


def fit_weights(pred, truth, origin, sample_indices):
    weights = np.empty((len(HORIZONS_SECONDS), pred.shape[2]), dtype="float64")
    for h in range(len(HORIZONS_SECONDS)):
        for target in range(pred.shape[2]):
            delta = pred[sample_indices, h, target] - origin[sample_indices, target]
            actual = truth[sample_indices, h, target] - origin[sample_indices, target]
            errors = np.abs(actual[:, None] - ALPHA_GRID[None, :] * delta[:, None]).mean(axis=0)
            minimum = errors.min()
            # When weights tie, prefer the more conservative (lower model share).
            weights[h, target] = ALPHA_GRID[np.flatnonzero(np.isclose(errors, minimum, atol=1e-12))[0]]
    return weights


def metrics(pred, truth, origin):
    error = pred - truth
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "persistence_mae": float(np.abs(origin - truth).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--max-val-batches", type=int, default=None, help="Small smoke run only; not an acceptance report")
    parser.add_argument("--calibration-stride", type=int, default=NONOVERLAP_STRIDE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {args.output}")
    if args.calibration_stride < NONOVERLAP_STRIDE:
        raise ValueError(f"calibration-stride must be at least {NONOVERLAP_STRIDE} samples")

    metadata, bundle, model, adapter, mean, scale = load_runtime()
    val_dataset = make_dataset(adapter, metadata, mean, scale, "val")
    calibration_ids = independent_indices(val_dataset, args.calibration_stride)
    smoke = args.max_val_batches is not None
    (val_pred, val_true, val_origin), calibration_ids = infer_validation(
        model, metadata, mean, scale, val_dataset, calibration_ids, args.max_val_batches
    )
    if len(calibration_ids) < 30:
        raise ValueError(f"Only {len(calibration_ids)} calibration windows; need at least 30")
    weights = fit_weights(val_pred, val_true, val_origin, np.arange(len(val_pred), dtype=np.int64))

    test_pred, test_true, test_origin, test_dataset = read_test_arrays(metadata, mean, scale, adapter)
    # Convert the six calibrated horizon weights into a fixed 10-second model-share curve.
    model_weights = np.vstack([
        np.interp(np.arange(1, metadata["pred_len"] + 1),
                  np.asarray(HORIZONS_SECONDS) / metadata["freq_seconds"], weights[:, target])
        for target in range(len(metadata["target_columns"]))
    ]).T
    eval_steps = np.asarray(HORIZONS_SECONDS) // metadata["freq_seconds"] - 1
    test_blend = test_origin[:, None, :] + model_weights[None, :, :] * (
        np.asarray(np.load(SOURCE / "pred.npy", mmap_mode="r")[:, :, metadata["target_indices"]], dtype="float64")
        * scale[np.asarray(metadata["target_indices"])][None, None, :]
        + mean[np.asarray(metadata["target_indices"])][None, None, :]
        - test_origin[:, None, :]
    )
    test_raw_at_horizons = test_pred
    test_blend_at_horizons = test_blend[:, eval_steps, :]

    full_indices = np.arange(len(test_pred), dtype=np.int64)
    independent = independent_indices(test_dataset, args.calibration_stride)
    records = []
    for h_i, horizon in enumerate(HORIZONS_SECONDS):
        for j, code in enumerate(metadata["target_columns"]):
            for split_name, indices in (("overlapping_all", full_indices), ("nonoverlap_30min", independent)):
                row = {"target": code, "horizon_seconds": horizon, "sample_set": split_name,
                       "windows": int(len(indices)), "model_weight": round(float(weights[h_i, j]), 4)}
                for name, values in (("raw_model", test_raw_at_horizons[indices][:, h_i, j]),
                                     ("persistence", test_origin[indices, j]),
                                     ("calibrated_blend", test_blend_at_horizons[indices][:, h_i, j])):
                    result = metrics(values, test_true[indices][:, h_i, j], test_origin[indices, j])
                    for metric_name, metric_value in result.items():
                        row[f"{name}_{metric_name}"] = round(metric_value, 6)
                records.append(row)

    args.output.mkdir(parents=True)
    csv_path = args.output / "per_target_horizon_metrics.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "schema_version": "1.0", "method": "validation_calibrated_model_persistence_blend",
        "model_slug": "model2", "checkpoint_sha256": bundle["checkpoint_sha256"],
        "source_training_package": str(SOURCE), "runtime_candidate": str(BUNDLE),
        "model_unchanged": True, "fit_split": "validation", "evaluation_split": "test",
        "fit_windows_total": int(len(val_dataset)), "fit_windows_used": int(len(calibration_ids)),
        "fit_sampling": (f"every {args.calibration_stride} samples within each validation chunk"
                         if not smoke else "smoke subset; not for conclusions"),
        "test_windows": int(len(test_pred)), "test_nonoverlap_windows": int(len(independent)),
        "horizons_seconds": list(HORIZONS_SECONDS), "weight_meaning": "prediction = current_observed + weight * (model - current_observed)",
        "weights_by_horizon_and_target": {
            str(h): {code: float(weights[i, j]) for j, code in enumerate(metadata["target_columns"])}
            for i, h in enumerate(HORIZONS_SECONDS)
        },
        "limitations": [
            "校准窗口至少相隔30分钟以减少重叠，但仍来自同一段按时间排列的验证期。",
            "测试期重叠窗口高度相关；30分钟间隔结果用于敏感性对照，不等于完全独立样本。",
            "本次回顾性分析不能证明现场精度或控制安全。",
            "混合权重是预测后处理策略，不改变checkpoint权重。",
        ],
        "smoke_only": smoke,
    }
    (args.output / "calibration.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(args.output / "report.md", payload, records)
    print(json.dumps({"output": str(args.output), "fit_windows": len(calibration_ids),
                      "test_windows": len(test_pred), "nonoverlap_test_windows": len(independent),
                      "smoke_only": smoke}, ensure_ascii=False, indent=2), flush=True)


def render_report(path: Path, payload: dict, rows: list[dict]):
    lines = ["# Model 2 不重训预测校正回放", "",
             f"- Checkpoint SHA-256：`{payload['checkpoint_sha256']}`",
             f"- 方案：验证期校准模型权重，测试期只评价一次；权重含义：`当前实测 + w × (模型预测 - 当前实测)`。",
             f"- 校准样本：{payload['fit_windows_used']:,} 个，间隔至少30分钟；测试样本：{payload['test_windows']:,} 个重叠窗口、{payload['test_nonoverlap_windows']:,} 个30分钟间隔窗口。",
             "- 不改模型权重；模型预测、保持当前值、混合预测均在工程单位评价。", "",
             "## 评价口径", "",
             "主表含MAE、RMSE、偏差、绝对误差95/99分位及最大误差。`model_weight=0`表示完全保持当前值，`1`表示完全采用模型。30分钟间隔样本用于降低窗口重叠影响；全量重叠窗口用于与常规报告对照。", "",
             "|目标|时域|样本集|窗口数|模型权重|模型MAE|保持MAE|混合MAE|模型RMSE|混合RMSE|模型P99|混合P99|",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["sample_set"] != "nonoverlap_30min":
            continue
        lines.append("|{target}|{horizon_seconds}s|{sample_set}|{windows}|{model_weight:.2f}|{raw_model_mae:.4f}|{persistence_mae:.4f}|{calibrated_blend_mae:.4f}|{raw_model_rmse:.4f}|{calibrated_blend_rmse:.4f}|{raw_model_p99_abs_error:.4f}|{calibrated_blend_p99_abs_error:.4f}|".format(**row))
    lines += ["", "## MPC校正口径对齐", "",
              "先将Model 2原始预测按目标/时域混合为基础轨迹：`y_base(k+h)=y_current(k)+w_target(h)·(y_model(k+h)-y_current(k))`。随后由MPC现有滚动残差校正计算`d_k=y_current(k)-y_base(k|k-1)`，以EWMA得到`d̄_k`，再输出`y_MPC(k+h)=y_base(k+h)+α(h)·d̄_k`。其中`w`是验证期冻结的静态模型占比，`α`是现有在线反馈残差的时域衰减系数；两者含义不同。残差必须针对上一周期同一条`y_base`轨迹计算，避免原始/混合预测混用导致重复校正。", "",
              "当前后端的`ResidualCorrector`按传入的`Forecast`保存上一周期轨迹并校正。若未来接入混合策略，应先混合`Forecast.targets`再交给该校正器；本报告只离线验证混合参数，没有改动后端运行配置。", "",
              "## 限制", "",
              *[f"- {item}" for item in payload["limitations"]]]
    if payload["smoke_only"]:
        lines.insert(1, "\n**注意：这是限量 smoke 结果，不可用于判断改善效果。**\n")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
