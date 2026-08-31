#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dl_tslib：Time-Series-Library 训练/评估封装脚本（Crossformer / iTransformer / TimeXer 等）

主要功能：
- 读取宽表 CSV（time_col + 多变量列），必要时按时间断点重排分段
- 过滤输入序列，生成 Dataset_Custom 所需 CSV（首列 date，目标列置于最后）
- 通过 subprocess 调用 third_party/Time-Series-Library/run.py 训练与测试
- 支持 features=MS 多目标逐一训练；支持 auto_search 随机搜索超参
- auto_search 完成后默认使用最优超参做一次 train+val 复训（test 仅用于最终评估）
- 统一输出产物：cv_predictions.csv / metrics_by_target.csv / run_meta.json / config.yaml
  并保存 raw_pred.npy/raw_true.npy/raw_metrics.npy 与 model_checkpoint.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def _resolve_path_like_config(cfg_path: Path, maybe_path: str) -> Path:
    p = Path(maybe_path).expanduser()
    return p if p.is_absolute() else (cfg_path.parent / p).resolve()

def load_wide_data(
    csv_path: Path,
    time_col: str,
    series_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    logger.info(f"正在加载数据：{csv_path}")
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"时间列 '{time_col}' 不在 CSV 中：{csv_path}")
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    if series_cols is None:
        series_cols = [c for c in df.columns if c != time_col]
    else:
        missing = sorted(list(set(series_cols) - set(df.columns)))
        if missing:
            raise ValueError(f"指定的序列列不存在：{missing}")

    logger.info(f"数据形状：{df.shape}，时间范围：{df[time_col].min()} ~ {df[time_col].max()}")
    logger.info(f"使用 {len(series_cols)} 个序列列进行建模")
    return df, series_cols


def _split_segments_by_time_gap(
    df: pd.DataFrame,
    time_col: str,
    *,
    gap: pd.Timedelta,
) -> List[pd.DataFrame]:
    if time_col not in df.columns:
        raise ValueError(f"时间列 '{time_col}' 不在 DataFrame 中")
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    diffs = df_sorted[time_col].diff()
    split_points = diffs[diffs > gap].index.tolist()
    if not split_points:
        return [df_sorted]
    segments = []
    start = 0
    for idx in split_points:
        segments.append(df_sorted.iloc[start:idx].reset_index(drop=True))
        start = idx
    segments.append(df_sorted.iloc[start:].reset_index(drop=True))
    return segments


def _prepare_non_contiguous_split(
    df: pd.DataFrame,
    time_col: str,
    *,
    gap: pd.Timedelta,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, Optional[Dict]]:
    segments = _split_segments_by_time_gap(df, time_col, gap=gap)
    if len(segments) <= 1:
        return df, None

    n_total = sum(len(seg) for seg in segments)
    if n_total == 0:
        return df, None

    num_train = int(n_total * train_ratio)
    num_test = int(n_total * test_ratio)
    num_val = n_total - num_train - num_test

    last_seg = segments[-1]
    last_len = len(last_seg)
    needed_tail = num_val + num_test
    if last_len < needed_tail:
        raise ValueError(
            f"非连续分段切分失败：最后一段长度={last_len}，不足以容纳 val+test={needed_tail}。"
            f"请调整分段阈值或数据比例。"
        )

    last_train_len = last_len - needed_tail
    last_train = last_seg.iloc[:last_train_len]
    last_val = last_seg.iloc[last_train_len:last_train_len + num_val]
    last_test = last_seg.iloc[last_train_len + num_val:last_train_len + num_val + num_test]

    df_reordered = pd.concat(segments[:-1] + [last_train, last_val, last_test], ignore_index=True)
    split_meta = {
        "n_total": int(n_total),
        "num_train": int(num_train),
        "num_val": int(num_val),
        "num_test": int(num_test),
        "test_start_idx": int(n_total - num_test),
        "test_end_idx": int(n_total),
    }
    logger.info(
        "检测到非连续数据（segments=%d, gap>%s）。按 70/10/20 从最后一段切分："
        "train=%d, val=%d, test=%d",
        len(segments),
        str(gap),
        num_train,
        num_val,
        num_test,
    )
    return df_reordered, split_meta


def filter_input_series(
    series_cols: List[str],
    *,
    target_series: List[str],
    non_target_input_series: Optional[List[str]],
) -> List[str]:
    """
    训练/推理时的输入序列选择：
    - target_series 始终会被输入（不受 non_target_input_series 影响）
    - non_target_input_series 仅对“非目标序列”生效：
        - None：表示所有非目标序列都作为输入
        - List[str]：表示只输入该名单里的非目标序列
    """
    if not isinstance(target_series, list) or any(not isinstance(x, str) for x in target_series) or len(target_series) == 0:
        raise ValueError("target_series 必须是非空字符串列表")

    series_set = set(series_cols)
    missing_targets = sorted(list(set(target_series) - series_set))
    if missing_targets:
        raise ValueError(f"target_series 中存在不在数据列里的目标：{missing_targets}")

    if non_target_input_series is None:
        allowed_non_targets = [c for c in series_cols if c not in set(target_series)]
    else:
        if not isinstance(non_target_input_series, list) or any(not isinstance(x, str) for x in non_target_input_series):
            raise ValueError("data.non_target_input_series 必须是 null 或字符串列表")
        missing = sorted(list(set(non_target_input_series) - series_set))
        if missing:
            raise ValueError(f"data.non_target_input_series 中存在不在数据列里的序列：{missing}")
        allowed_non_targets = [c for c in non_target_input_series if c not in set(target_series)]

    allowed_set = set(target_series) | set(allowed_non_targets)
    filtered = [c for c in series_cols if c in allowed_set]
    n_non_target_before = sum(1 for c in series_cols if c not in set(target_series))
    n_non_target_after = sum(1 for c in filtered if c not in set(target_series))
    logger.info(
        "输入序列过滤：保留 %d/%d 个非目标序列 + %d 个目标序列（总计 %d 维）",
        n_non_target_after,
        n_non_target_before,
        len(set(target_series)),
        len(filtered),
    )
    return filtered


def build_tslib_custom_csv(
    df_wide: pd.DataFrame,
    time_col: str,
    series_cols: List[str],
    *,
    target: str,
    out_csv_path: Path,
) -> List[str]:
    """
    生成 Time-Series-Library Dataset_Custom 所需 CSV：
    - 第一列名必须是 `date`
    - Dataset_Custom 内部会将 target 置于最后一列；这里提前按同样逻辑重排，便于我们建立变量维度映射

    Returns:
        var_names: 变量列顺序（不含 date），与 pred.npy/true.npy 的最后一维一致
    """
    if target not in series_cols:
        raise ValueError(f"target={target} 不在 series_cols 中")
    df = df_wide[[time_col] + series_cols].copy()
    df = df.rename(columns={time_col: "date"})
    cols = ["date"] + [c for c in series_cols if c != target] + [target]
    df = df[cols]
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    logger.info(f"已生成 Dataset_Custom CSV：{out_csv_path}（n_vars={len(cols)-1}）")
    return cols[1:]


def _list_subdirs(p: Path) -> List[Path]:
    if not p.exists():
        return []
    return [x for x in p.iterdir() if x.is_dir()]


def _pick_latest_dir(dirs: List[Path]) -> Optional[Path]:
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)

def _loguniform(rng: np.random.RandomState, low: float, high: float) -> float:
    if low <= 0 or high <= 0 or high < low:
        raise ValueError(f"loguniform 需要正数且 high>=low，收到 low={low}, high={high}")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_from_space(rng: np.random.RandomState, space: Dict, default_val):
    """
    space 支持：
    - list: 从列表中均匀采样
    - dict: {min,max}（连续均匀）或 {min,max,log:true}（loguniform）
    - 其它：直接返回（视为固定值）
    """
    if space is None:
        return default_val
    if isinstance(space, list):
        if len(space) == 0:
            return default_val
        return space[int(rng.randint(0, len(space)))]
    if isinstance(space, dict):
        low = space.get("min")
        high = space.get("max")
        if low is None or high is None:
            return default_val
        if bool(space.get("log", False)):
            return _loguniform(rng, float(low), float(high))
        return float(rng.uniform(float(low), float(high)))
    return space


def _format_trial_tag(i: int, params: Dict) -> str:
    # 既可读又不太长
    parts = [f"t{i:03d}", f"dm{params.get('d_model')}", f"nh{params.get('n_heads')}", f"el{params.get('e_layers')}"]
    parts.append(f"lr{params.get('learning_rate'):.2e}")
    return "_".join(str(x) for x in parts)


def _objective_from_metrics(avg_metrics: Dict, objective: str) -> float:
    # objective 支持：MAE_avg/RMSE_avg/MAPE_avg/SMAPE_avg（越小越好），R2_avg（越大越好）
    if objective == "R2_avg":
        v = avg_metrics.get("R2_avg")
        if v is None:
            return float("inf")
        return float(-v)  # 转成最小化
    v = avg_metrics.get(objective)
    if v is None:
        return float("inf")
    return float(v)


def run_tslib_crossformer(
    *,
    tslib_root: Path,
    dataset_root: Path,
    dataset_csv_name: str,
    dataset_target: str,
    model_name: str,
    loss_target_indices: Optional[List[int]] = None,
    use_dec_mask_embedding: bool = True,
    features: str = "M",
    freq: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    n_vars: int,
    train_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    grad_clip_norm: float,
    use_gpu: bool,
    gpu: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    dropout: float,
    factor: int,
    patch_len: Optional[int] = None,
    use_amp: bool,
    inverse: bool,
    num_workers: int,
    seed: int,
    des: str,
    loss_time_weighted: bool = False,
    loss_time_weight_decay: float = 0.0,
    loss_time_weight_min: float = 0.0,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
) -> Path:
    """
    调用 Time-Series-Library/run.py，返回 results/<setting> 目录路径。

    备注：函数名历史原因保留（最初仅用于 Crossformer），但实际支持 Time-Series-Library 的任意 --model（如 TimeXer）。
    """
    results_dir = tslib_root / "results"

    # 标准模式：encoder 仅历史 seq_len，不再区分 hist_len。
    # 额外将变量维度编码到 model_id，避免不同 n_vars 复用同一 setting，
    # 从而误加载旧 checkpoint（典型报错：enc/dec_pos_embedding size mismatch）。
    model_id = f"custom_{model_name}_sl{seq_len}_ll{label_len}_pl{pred_len}_nv{int(n_vars)}"
    cmd = [
        sys.executable,
        "-u",
        str(tslib_root / "run.py"),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--model_id",
        model_id,
        "--model",
        str(model_name),
        "--data",
        "custom",
        "--root_path",
        str(dataset_root),
        "--data_path",
        str(dataset_csv_name),
        "--features",
        str(features),
        "--target",
        str(dataset_target),
        "--freq",
        str(freq),
        "--seq_len",
        str(int(seq_len)),
        "--label_len",
        str(int(label_len)),
        "--pred_len",
        str(int(pred_len)),
        "--enc_in",
        str(int(n_vars)),
        "--dec_in",
        str(int(n_vars)),
        "--c_out",
        str(int(n_vars)),
        "--d_model",
        str(int(d_model)),
        "--n_heads",
        str(int(n_heads)),
        "--e_layers",
        str(int(e_layers)),
        "--d_layers",
        str(int(d_layers)),
        "--d_ff",
        str(int(d_ff)),
        "--dropout",
        str(float(dropout)),
        "--factor",
        str(int(factor)),
        "--train_epochs",
        str(int(train_epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--learning_rate",
        str(float(learning_rate)),
        "--patience",
        str(int(patience)),
        "--grad_clip_norm",
        str(float(grad_clip_norm)),
        "--num_workers",
        str(int(num_workers)),
        "--seed",
        str(int(seed)),
        "--des",
        str(des),
        "--itr",
        "1",
        "--gpu",
        str(int(gpu)),
        "--gpu_type",
        "cuda",
    ]
    # 标准模式（encoder 仅历史 seq_len）下，不传 --hist_len（默认 0），让 Dataset_Custom 走标准切片逻辑。
    # 若未来需要启用“已知未来协变量”模式，则再显式传 --hist_len 且要求 hist_len + pred_len == seq_len。
    if loss_target_indices:
        cmd.extend(["--loss_target_indices", ",".join(str(int(i)) for i in loss_target_indices)])
    if loss_time_weighted:
        cmd.append("--loss_time_weighted")
        cmd.extend(["--loss_time_weight_decay", str(float(loss_time_weight_decay))])
        cmd.extend(["--loss_time_weight_min", str(float(loss_time_weight_min))])
    if train_ratio is not None and val_ratio is not None and test_ratio is not None:
        cmd.extend(["--train_ratio", str(float(train_ratio))])
        cmd.extend(["--val_ratio", str(float(val_ratio))])
        cmd.extend(["--test_ratio", str(float(test_ratio))])
    # TimeXer 等模型会用到 patch_len（run.py 全局支持该参数；其它模型传入也不会影响）
    if patch_len is not None:
        cmd.extend(["--patch_len", str(int(patch_len))])
    if inverse:
        cmd.append("--inverse")
    if use_amp:
        cmd.append("--use_amp")
    # Crossformer decoder mask embedding toggle (default enabled in TSLib).
    if not bool(use_dec_mask_embedding):
        cmd.append("--no_dec_mask_embedding")

    logger.info("开始调用 Time-Series-Library/run.py 训练+测试（model=%s）...", model_name)
    logger.info(" ".join(cmd))
    # 重要：Time-Series-Library/run.py 的 --use_gpu 使用 argparse type=bool，
    # 传入字符串 'False' 也会被解析成 True（bool('False') == True）。
    # 这里通过 CUDA_VISIBLE_DEVICES “隐藏 GPU” 的方式实现 CPU 模式。
    env = os.environ.copy()
    if not use_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # CUDA 预热（经验问题）：
        # 在部分环境/驱动组合下，第一次触发 torch.cuda.is_available() 可能出现瞬时的 driver init 失败，
        # 导致 Time-Series-Library/run.py 误判为 CPU。这里在同一 cwd+env 下提前触发一次 CUDA 初始化，
        # 让后续 run.py 的首个 is_available 判断更稳定。
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; "
                    "print('cuda_available', torch.cuda.is_available()); "
                    "print('cuda_device_count', torch.cuda.device_count());",
                ],
                cwd=str(tslib_root),
                env=env,
                check=False,
            )
        except Exception as e:
            logger.warning("CUDA 预热失败（将继续运行 TSLib）：%s", str(e))
    subprocess.run(cmd, cwd=str(tslib_root), check=True, env=env)

    # Time-Series-Library 的 setting 命名规则来自 run.py（ii=0）
    # 注意：我们未显式传的参数取 run.py argparse 的默认值：
    # expand=2, d_conv=4, embed=timeF, distil=True
    setting = (
        f"long_term_forecast_{model_id}_{model_name}_custom_"
        f"ft{features}_sl{seq_len}_ll{label_len}_pl{pred_len}_"
        f"dm{d_model}_nh{n_heads}_el{e_layers}_dl{d_layers}_df{d_ff}_"
        f"expand2_dc4_fc{factor}_ebtimeF_dtTrue_{des}_0"
    )
    expected = results_dir / setting
    if expected.exists():
        picked = expected
    else:
        # 兜底：若上游命名规则变化或参数默认值不同，则回退到“最新目录”
        after_dirs = _list_subdirs(results_dir)
        picked = _pick_latest_dir(after_dirs)
        logger.warning("未找到期望的 results 目录：%s，将回退读取最新目录：%s", str(expected), str(picked) if picked else "None")
        if picked is None:
            raise RuntimeError("未找到 Time-Series-Library 生成的 results/<setting> 目录")

    for fn in ["pred.npy", "true.npy", "metrics.npy"]:
        if not (picked / fn).exists():
            raise RuntimeError(f"results 目录缺少 {fn}：{picked}")
    logger.info(f"Time-Series-Library 结果目录：{picked}")
    return picked


def build_cv_predictions_from_npy(
    *,
    prepared_csv: Path,
    var_names: List[str],
    target_series: List[str],
    features: str,
    seq_len: int,
    pred_len: int,
    preds: np.ndarray,
    trues: np.ndarray,
    pred_col: str = "Crossformer",
    split_meta: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    将 pred/true（形状 [N, pred_len, D]）展开为 long 表，列对齐 dl_control_model 的 cv_predictions.csv：
    - unique_id: 变量名（只输出 target_series）
    - ds: 预测时间戳
    - y: 真值
    - {pred_col}: 预测值
    - cutoff: 输入窗口截止时间（最后一个历史点）
    """
    df_raw = pd.read_csv(prepared_csv)
    if "date" not in df_raw.columns:
        raise ValueError(f"prepared_csv 缺少 date 列：{prepared_csv}")
    dates = pd.to_datetime(df_raw["date"]).to_numpy()
    n_total = len(dates)

    # Dataset_Custom 固定切分：70% train, 20% test, rest val
    if split_meta and "test_start_idx" in split_meta and "test_end_idx" in split_meta:
        test_start = int(split_meta["test_start_idx"])
        test_end = int(split_meta["test_end_idx"])
    else:
        num_test = int(n_total * 0.2)
        test_start = n_total - num_test
        test_end = n_total
    border1_test = test_start - seq_len
    border2_test = test_end
    test_dates = dates[border1_test:border2_test]

    # 标准模式下：__len__ = len(data_x) - seq_len - pred_len + 1
    # 其中 len(data_x) == len(test_dates)
    n_samples_expected = len(test_dates) - seq_len - pred_len + 1
    if preds.shape[0] != n_samples_expected:
        raise ValueError(
            f"preds 样本数不匹配：preds.shape[0]={preds.shape[0]} vs expected={n_samples_expected}"
        )
    if preds.shape != trues.shape:
        raise ValueError(f"preds/true shape 不一致：{preds.shape} vs {trues.shape}")
    if preds.shape[1] != pred_len:
        raise ValueError(f"pred_len 不一致：preds.shape[1]={preds.shape[1]} vs pred_len={pred_len}")
    # Time-Series-Library 在 features=MS 时会只保留最后一维（target）作为输出，因此 pred/true 的最后一维通常为 1
    expected_d = 1 if str(features).upper() == "MS" else len(var_names)
    if preds.shape[2] != expected_d:
        raise ValueError(f"变量维度不一致：preds.shape[2]={preds.shape[2]} vs expected_d={expected_d} (features={features})")

    target_indices: Dict[str, int] = {}
    if str(features).upper() == "MS":
        # MS 模式：只预测一个 target（最后一维），因此要求 target_series 只有一个
        if len(target_series) != 1:
            raise ValueError(f"features=MS 时只支持单目标；收到 target_series={target_series}")
        target_indices[target_series[0]] = 0
    else:
        for t in target_series:
            if t not in var_names:
                raise ValueError(f"target_series 中存在不在数据列里的目标：{t}")
            target_indices[t] = var_names.index(t)

    rows: List[Dict] = []
    for s_begin in range(preds.shape[0]):
        cutoff = pd.Timestamp(test_dates[s_begin + seq_len - 1])
        ds_slice = test_dates[s_begin + seq_len : s_begin + seq_len + pred_len]
        for t in target_series:
            idx = target_indices[t]
            y_true = trues[s_begin, :, idx].astype(float)
            y_pred = preds[s_begin, :, idx].astype(float)
            for h in range(pred_len):
                rows.append(
                    {
                        "unique_id": t,
                        "ds": pd.Timestamp(ds_slice[h]),
                        "y": float(y_true[h]),
                        pred_col: float(y_pred[h]),
                        "cutoff": cutoff,
                        "horizon": int(h + 1),
                        "sample_idx": int(s_begin),
                    }
                )
    out = pd.DataFrame(rows)
    out = out.sort_values(["unique_id", "ds", "cutoff", "horizon"]).reset_index(drop=True)
    return out


def evaluate_predictions(
    cv_df: pd.DataFrame,
    *,
    target_series: List[str],
    pred_col: str = "Crossformer",
    eps: float = 1e-8,
) -> Tuple[pd.DataFrame, Dict]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    metrics_list = []
    for t in target_series:
        df_s = cv_df[cv_df["unique_id"] == t].copy()
        if df_s.empty:
            logger.warning(f"目标序列 {t} 在预测输出中为空，跳过")
            continue
        y_true = df_s["y"].to_numpy(dtype=float)
        y_pred = df_s[pred_col].to_numpy(dtype=float)

        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.all(finite_mask):
            dropped = int(np.size(finite_mask) - np.count_nonzero(finite_mask))
            logger.warning(f"{t}: 发现 {dropped} 条非有限值样本，已在评估中忽略")
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        if y_true.size == 0:
            logger.warning(f"{t}: 全部样本为 NaN/Inf，跳过评估")
            continue

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
        smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0)

        metrics_list.append(
            {
                "unique_id": t,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE": mape,
                "SMAPE": smape,
                "n_samples": int(len(df_s)),
            }
        )
        logger.info(
            f"{t}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%, SMAPE={smape:.2f}% (n={len(df_s)})"
        )

    metrics_df = pd.DataFrame(metrics_list)
    avg_metrics = {
        "MAE_avg": float(metrics_df["MAE"].mean()) if not metrics_df.empty else None,
        "RMSE_avg": float(metrics_df["RMSE"].mean()) if not metrics_df.empty else None,
        "R2_avg": float(metrics_df["R2"].mean()) if not metrics_df.empty else None,
        "MAPE_avg": float(metrics_df["MAPE"].mean()) if not metrics_df.empty else None,
        "SMAPE_avg": float(metrics_df["SMAPE"].mean()) if not metrics_df.empty else None,
    }
    return metrics_df, avg_metrics


def save_outputs(
    *,
    out_dir: Path,
    cv_predictions: pd.DataFrame,
    metrics_df: pd.DataFrame,
    avg_metrics: Dict,
    config: Dict,
    config_path: Path,
    pred_col: str,
    model_name: str,
    tslib_results_dir: Path,
):
    def _copy_tslib_checkpoint(out_dir: Path, tslib_results_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Time-Series-Library 会将最优模型保存到：
          <tslib_root>/checkpoints/<setting>/checkpoint.pth
        其中 <setting> == results/<setting> 的目录名。
        """
        try:
            tslib_root = tslib_results_dir.parent.parent
            setting = tslib_results_dir.name
            ckpt_src = tslib_root / "checkpoints" / setting / "checkpoint.pth"
            if not ckpt_src.exists():
                logger.warning("未找到 Time-Series-Library checkpoint：%s", str(ckpt_src))
                return None, ckpt_src
            ckpt_dst = out_dir / "model_checkpoint.pth"
            shutil.copy2(ckpt_src, ckpt_dst)
            logger.info("已保存模型 checkpoint 到：%s", str(ckpt_dst))
            return ckpt_dst, ckpt_src
        except Exception as e:
            logger.warning("复制 Time-Series-Library checkpoint 失败：%s", str(e))
            return None, None

    def _copy_loss_history(out_dir: Path, tslib_results_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        复制 Time-Series-Library 训练过程输出的收敛原始数据：
          <tslib_root>/results/<setting>/loss_history.csv
        """
        history_src = tslib_results_dir / "loss_history.csv"
        if not history_src.exists():
            logger.warning("未找到收敛原始数据文件：%s", str(history_src))
            return None, history_src
        history_dst = out_dir / "loss_history.csv"
        try:
            shutil.copy2(history_src, history_dst)
            logger.info("已保存收敛原始数据到：%s", str(history_dst))
            return history_dst, history_src
        except Exception as e:
            logger.warning("复制收敛原始数据失败：%s", str(e))
            return None, history_src

    out_dir.mkdir(parents=True, exist_ok=True)

    cv_path = out_dir / "cv_predictions.csv"
    cv_predictions.to_csv(cv_path, index=False)

    m_path = out_dir / "metrics_by_target.csv"
    metrics_df.to_csv(m_path, index=False)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "model_name": str(model_name),
        "pred_col": pred_col,
        "freq": (config.get("data") or {}).get("freq"),
        "seq_len": (config.get("model") or {}).get("seq_len"),
        "label_len": (config.get("model") or {}).get("label_len"),
        "pred_len": (config.get("model") or {}).get("pred_len"),
        "target_series": (config.get("data") or {}).get("target_series"),
        "avg_metrics": avg_metrics,
        "tslib_results_dir": str(tslib_results_dir),
    }
    # 同步 Time-Series-Library 的最佳 checkpoint 到实验目录
    ckpt_dst, ckpt_src = _copy_tslib_checkpoint(out_dir=out_dir, tslib_results_dir=tslib_results_dir)
    history_dst, history_src = _copy_loss_history(out_dir=out_dir, tslib_results_dir=tslib_results_dir)
    meta["tslib_setting"] = str(tslib_results_dir.name)
    meta["tslib_checkpoint_src"] = str(ckpt_src) if ckpt_src is not None else None
    meta["model_checkpoint"] = str(ckpt_dst) if ckpt_dst is not None else None
    meta["tslib_loss_history_src"] = str(history_src) if history_src is not None else None
    meta["loss_history_csv"] = str(history_dst) if history_dst is not None else None
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    shutil.copy2(config_path, out_dir / "config.yaml")

    for fn, new_name in [("pred.npy", "raw_pred.npy"), ("true.npy", "raw_true.npy"), ("metrics.npy", "raw_metrics.npy")]:
        shutil.copy2(tslib_results_dir / fn, out_dir / new_name)

    logger.info(f"输出已保存到：{out_dir}")


def run_ms_loop_targets(
    *,
    config: Dict,
    config_path: Path,
    target_series: List[str],
    model_name: str,
    csv_path: Path,
    out_root: Path,
    time_col: str,
    freq: str,
    series_cols: Optional[List[str]],
    non_target_input_series: Optional[List[str]],
    model_cfg: Dict,
    train_cfg: Dict,
    loss_cfg: Dict,
    use_gpu_effective: bool,
    gpu_effective: int,
    tslib_root: Path,
    eps: float,
) -> int:
    """
    MS 模式下遍历每个目标分别训练，结果汇总到同一目录。

    当 features=MS 且 target_series 包含多个目标时调用此函数。
    对每个目标：
    1. 构建 TSL 所需的 CSV（target 放到最后一列）
    2. 调用 Time-Series-Library 训练
    3. 收集预测结果和指标
    最后将所有目标的结果汇总输出。
    """
    seq_len = int(model_cfg.get("seq_len", model_cfg.get("input_size", 60)))
    pred_len = int(model_cfg.get("pred_len", model_cfg.get("h", 15)))
    label_len = int(model_cfg.get("label_len", max(1, seq_len // 2)))
    if label_len > seq_len:
        label_len = int(seq_len)
    inverse = bool(model_cfg.get("inverse", True))
    use_dec_mask_embedding = bool(model_cfg.get("use_dec_mask_embedding", True))
    time_weight_cfg = loss_cfg.get("time_weighting") or {}
    loss_time_weighted = bool(time_weight_cfg.get("enabled", False))
    loss_time_weight_decay = float(time_weight_cfg.get("decay", 0.0))
    loss_time_weight_min = float(time_weight_cfg.get("min_weight", 0.0))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = out_root / f"{timestamp}_{model_name}"
    dataset_dir = run_out_dir / "_tslib_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始数据（只加载一次）
    df_wide, series_cols_actual = load_wide_data(csv_path=csv_path, time_col=time_col, series_cols=series_cols)
    df_wide, split_meta = _prepare_non_contiguous_split(
        df_wide,
        time_col,
        gap=pd.Timedelta(minutes=1),
    )

    all_cv_predictions: List[pd.DataFrame] = []
    all_metrics: List[pd.DataFrame] = []
    last_tslib_results_dir: Optional[Path] = None

    logger.info("=" * 80)
    logger.info(f"MS 模式 + 多目标：将对 {len(target_series)} 个目标分别训练和评估")
    logger.info("=" * 80)

    for target_idx, target in enumerate(target_series):
        logger.info("=" * 80)
        logger.info(f"[{target_idx + 1}/{len(target_series)}] 训练目标: {target}")
        logger.info("=" * 80)

        # 每个目标需要单独过滤输入序列（target_series 只包含当前目标）
        single_target_series = [target]
        series_cols_filtered = filter_input_series(
            series_cols_actual,
            target_series=single_target_series,
            non_target_input_series=non_target_input_series,
        )

        # 构建 TSL CSV（target 放到最后一列）
        dataset_csv = dataset_dir / f"custom_{target_idx}.csv"
        var_names = build_tslib_custom_csv(
            df_wide=df_wide,
            time_col=time_col,
            series_cols=series_cols_filtered,
            target=target,
            out_csv_path=dataset_csv,
        )
        n_vars = len(var_names)

        # MS 模式下 loss_target_indices 不需要传（只有一个 target，且 TSL 默认只对最后一维计算 loss）
        loss_target_indices_to_pass: Optional[List[int]] = None

        # 调用 Time-Series-Library 训练
        results_setting_dir = run_tslib_crossformer(
            tslib_root=tslib_root,
            dataset_root=dataset_dir,
            dataset_csv_name=dataset_csv.name,
            dataset_target=target,
            model_name=model_name,
            loss_target_indices=loss_target_indices_to_pass,
            use_dec_mask_embedding=use_dec_mask_embedding,
            features="MS",
            freq=freq,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            n_vars=n_vars,
            train_epochs=int(train_cfg.get("epochs", 10)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            patience=int(train_cfg.get("patience", 3)),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
            use_gpu=use_gpu_effective,
            gpu=gpu_effective,
            d_model=int(model_cfg.get("d_model", 256)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            e_layers=int(model_cfg.get("e_layers", 2)),
            d_layers=int(model_cfg.get("d_layers", 1)),
            d_ff=int(model_cfg.get("d_ff", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            factor=int(model_cfg.get("factor", 3)),
            patch_len=(int(model_cfg.get("patch_len")) if model_cfg.get("patch_len", None) is not None else None),
            use_amp=bool(train_cfg.get("use_amp", False)),
            inverse=inverse,
            num_workers=int(train_cfg.get("num_workers", 4)),
            seed=int(train_cfg.get("seed", 2021)),
            des=str(model_cfg.get("des", "Exp")),
            loss_time_weighted=loss_time_weighted,
            loss_time_weight_decay=loss_time_weight_decay,
            loss_time_weight_min=loss_time_weight_min,
        )
        last_tslib_results_dir = results_setting_dir

        # 读取预测结果
        preds = np.load(results_setting_dir / "pred.npy")
        trues = np.load(results_setting_dir / "true.npy")
        pred_col = str(model_name)

        # 构建预测 DataFrame
        cv_predictions = build_cv_predictions_from_npy(
            prepared_csv=dataset_csv,
            var_names=var_names,
            target_series=single_target_series,
            features="MS",
            seq_len=seq_len,
            pred_len=pred_len,
            preds=preds,
            trues=trues,
            pred_col=pred_col,
            split_meta=split_meta,
        )
        all_cv_predictions.append(cv_predictions)

        # 评估当前目标
        metrics_df, avg_metrics = evaluate_predictions(
            cv_df=cv_predictions,
            target_series=single_target_series,
            pred_col=pred_col,
            eps=eps,
        )
        all_metrics.append(metrics_df)

    # 汇总所有目标的结果
    combined_cv = pd.concat(all_cv_predictions, ignore_index=True)
    combined_cv = combined_cv.sort_values(["unique_id", "ds", "cutoff", "horizon"]).reset_index(drop=True)

    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # 计算整体平均指标
    avg_metrics_combined = {
        "MAE_avg": float(combined_metrics["MAE"].mean()) if not combined_metrics.empty else None,
        "RMSE_avg": float(combined_metrics["RMSE"].mean()) if not combined_metrics.empty else None,
        "R2_avg": float(combined_metrics["R2"].mean()) if not combined_metrics.empty else None,
        "MAPE_avg": float(combined_metrics["MAPE"].mean()) if not combined_metrics.empty else None,
        "SMAPE_avg": float(combined_metrics["SMAPE"].mean()) if not combined_metrics.empty else None,
    }

    # 保存输出
    save_outputs(
        out_dir=run_out_dir,
        cv_predictions=combined_cv,
        metrics_df=combined_metrics,
        avg_metrics=avg_metrics_combined,
        config=config,
        config_path=config_path,
        pred_col=str(model_name),
        model_name=model_name,
        tslib_results_dir=last_tslib_results_dir if last_tslib_results_dir else dataset_dir,
    )

    logger.info("=" * 80)
    logger.info("MS 多目标训练与评估完成！")
    logger.info(f"输出目录：{run_out_dir}")
    logger.info("=" * 80)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="dl_tslib：训练与评估（Time-Series-Library）")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径（YAML）")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖配置文件中的 model.name（Time-Series-Library 的 --model）。例如：Crossformer / iTransformer / TimeXer",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=(
            "指定使用的 GPU（优先级高于 YAML 的 train.gpu）。"
            "支持物理 GPU 编号或 CUDA_VISIBLE_DEVICES 格式的数字列表："
            "--gpu 0 / --gpu 2 / --gpu 0,1（将使用列表中的第一个 GPU）。"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"--config 指定的文件不存在：{config_path}")
    if config_path.suffix.lower() not in [".yaml", ".yml"]:
        logger.warning(
            "--config 看起来不是 YAML（后缀=%s）。你是不是误传了脚本/其它文件？建议使用：%s 或 %s",
            config_path.suffix,
            str((Path(__file__).resolve().parent / "config_smoke.yaml")),
            str((Path(__file__).resolve().parent / "config.yaml")),
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(
            f"配置文件解析失败：{config_path}\n"
            f"请确认 --config 指向 YAML（例如 config_smoke.yaml / config.yaml），而不是 .py。\n"
            f"示例：\n"
            f"  python {Path(__file__).resolve()} --config {Path(__file__).resolve().parent / 'config_smoke.yaml'}\n"
            f"  python {Path(__file__).resolve()} --config {Path(__file__).resolve().parent / 'config.yaml'}\n"
            f"原始错误：{e}"
        ) from e

    csv_path = _resolve_path_like_config(config_path, (config.get("data") or {}).get("csv_path"))
    out_root = _resolve_path_like_config(config_path, (config.get("output") or {}).get("dir"))

    data_cfg = config.get("data") or {}
    model_cfg = config.get("model") or {}
    train_cfg = config.get("train") or {}
    auto_cfg = config.get("auto_search") or {}

    # 命令行优先级高于 config.yaml
    model_name = str(args.model) if args.model is not None and str(args.model).strip() != "" else str(model_cfg.get("name", "Crossformer"))

    # ========== GPU 选择（命令行优先级高于 YAML）==========
    # 说明：
    # - 本脚本通过 subprocess 调用 Time-Series-Library/run.py
    # - 若命令行传入 --gpu，则通过设置 CUDA_VISIBLE_DEVICES 来“只暴露”指定物理卡给下游
    # - 同时将传给 Time-Series-Library 的 --gpu 固定为 0（表示可见 GPU 列表的第一个）
    use_gpu_effective = bool(train_cfg.get("use_gpu", True))
    gpu_effective = int(train_cfg.get("gpu", 0))
    if args.gpu is not None and str(args.gpu).strip() != "":
        gpu_str = str(args.gpu).strip()
        parts = [p.strip() for p in gpu_str.split(",")]
        if any(p == "" for p in parts) or not all(p.isdigit() for p in parts):
            raise ValueError(f"--gpu 只支持数字或逗号分隔的数字列表（例如 0 / 2 / 0,1），收到：{args.gpu!r}")
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        use_gpu_effective = True
        gpu_effective = 0
        logger.info("命令行指定 GPU：CUDA_VISIBLE_DEVICES=%s（传给 Time-Series-Library: --gpu=%d）", gpu_str, gpu_effective)

    # 仅做轻量校验：Time-Series-Library 支持的模型很多，但这里优先保证常用模型
    if model_name not in {"Crossformer", "iTransformer", "TimeXer", "TemporalFusionTransformer"}:
        logger.warning(
            "model.name=%s 不在推荐列表 {Crossformer,iTransformer,TimeXer,TemporalFusionTransformer} 中；将仍尝试按 Time-Series-Library 的 --model 直接运行。",
            model_name,
        )

    time_col = str(data_cfg.get("time_col", "time"))
    freq = str(data_cfg.get("freq", "min"))
    series_cols = data_cfg.get("series_cols")
    target_series = data_cfg.get("target_series") or []
    if not isinstance(target_series, list) or any(not isinstance(x, str) for x in target_series) or len(target_series) == 0:
        raise ValueError("data.target_series 必须是非空字符串列表")

    seq_len = int(model_cfg.get("seq_len", model_cfg.get("input_size", 60)))
    pred_len = int(model_cfg.get("pred_len", model_cfg.get("h", 15)))
    label_len = int(model_cfg.get("label_len", max(1, seq_len // 2)))
    if label_len > seq_len:
        logger.warning("label_len(%s) > seq_len(%s)，将自动截断为 seq_len", label_len, seq_len)
        label_len = int(seq_len)
    inverse = bool(model_cfg.get("inverse", True))
    use_dec_mask_embedding = bool(model_cfg.get("use_dec_mask_embedding", True))

    # ========== features 选择（Time-Series-Library 的 --features）==========
    # - TimeXer 默认使用 MS（区分“外生变量(协变量)” vs “目标(内生)”：默认把最后一列作为 target）
    # - 其它模型默认使用 M
    # - 可通过 model.features 显式覆盖（例如：M / MS / S）
    features = str(model_cfg.get("features")).strip() if model_cfg.get("features", None) is not None else ""
    if features == "":
        features = "MS" if model_name == "TimeXer" else "M"
    features = features.upper()
    if features not in {"M", "MS", "S"}:
        raise ValueError(f"model.features 只能是 M/MS/S，收到：{features}")

    # MS 模式在 TSL 中通常表示“多变量输入 + 单目标输出（最后一维）”，因此强制单目标
    # tslib_root 和 eps 提前定义（MS 多目标遍历模式需要用到）
    tslib_root = Path(__file__).resolve().parent / "third_party" / "Time-Series-Library"
    if not tslib_root.exists():
        raise FileNotFoundError(f"未找到 third_party/Time-Series-Library：{tslib_root}")
    loss_cfg = config.get("loss") or {}
    eps = float(loss_cfg.get("eps", 1e-8))
    time_weight_cfg = loss_cfg.get("time_weighting") or {}
    loss_time_weighted = bool(time_weight_cfg.get("enabled", False))
    loss_time_weight_decay = float(time_weight_cfg.get("decay", 0.0))
    loss_time_weight_min = float(time_weight_cfg.get("min_weight", 0.0))

    # MS 模式 + 多目标：自动遍历每个目标分别训练
    if features == "MS" and len(target_series) > 1:
        return run_ms_loop_targets(
            config=config,
            config_path=config_path,
            target_series=target_series,
            model_name=model_name,
            csv_path=csv_path,
            out_root=out_root,
            time_col=time_col,
            freq=freq,
            series_cols=series_cols,
            non_target_input_series=data_cfg.get("non_target_input_series"),
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            loss_cfg=loss_cfg,
            use_gpu_effective=use_gpu_effective,
            gpu_effective=gpu_effective,
            tslib_root=tslib_root,
            eps=eps,
        )

    df_wide, series_cols_actual = load_wide_data(csv_path=csv_path, time_col=time_col, series_cols=series_cols)
    df_wide, split_meta = _prepare_non_contiguous_split(
        df_wide,
        time_col,
        gap=pd.Timedelta(minutes=1),
    )
    series_cols_actual = filter_input_series(
        series_cols_actual,
        target_series=target_series,
        non_target_input_series=data_cfg.get("non_target_input_series"),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = out_root / f"{timestamp}_{model_name}"
    dataset_dir = run_out_dir / "_tslib_dataset"
    dataset_csv = dataset_dir / "custom.csv"

    dataset_target = target_series[0]
    var_names = build_tslib_custom_csv(
        df_wide=df_wide,
        time_col=time_col,
        series_cols=series_cols_actual,
        target=dataset_target,
        out_csv_path=dataset_csv,
    )
    n_vars = len(var_names)
    # 训练 loss 只对 target_series 这些变量维度计算：把变量名映射为 TSV-Lib 的列索引（对应 pred/true 的最后一维）
    loss_target_indices: List[int] = []
    _seen = set()
    for t in target_series:
        if t not in var_names:
            raise ValueError(f"target_series 中存在不在 var_names 里的目标：{t}")
        idx = int(var_names.index(t))
        if idx not in _seen:
            _seen.add(idx)
            loss_target_indices.append(idx)

    # 是否将 loss_target_indices 传给 Time-Series-Library：
    # - iTransformer：我们的 encoder 输入扩展逻辑需要依赖该索引来区分 target vs non-target，因此必须传
    # - Crossformer：我们的 decoder 输入特殊逻辑（仅对 Crossformer 生效）也依赖该索引，因此必须传
    # - TemporalFusionTransformer：同样支持只对 target_series 计算 loss
    # - TimeXer（以及其它模型）：默认不传，保持 Time-Series-Library 原始“对全维度计算 loss”的行为，避免你之前的改动影响其训练目标
    #   如需让 TimeXer 也只对 target_series 计算 loss，可在 YAML 里设置：loss.use_target_indices_for_all_models: true
    use_target_indices_for_all_models = bool(loss_cfg.get("use_target_indices_for_all_models", False))
    loss_target_indices_to_pass: Optional[List[int]] = None
    if model_name in {"iTransformer", "Crossformer", "TemporalFusionTransformer"}:
        loss_target_indices_to_pass = loss_target_indices
    elif use_target_indices_for_all_models:
        loss_target_indices_to_pass = loss_target_indices

    enabled = bool(auto_cfg.get("enabled", False))
    if not enabled:
        results_setting_dir = run_tslib_crossformer(
            tslib_root=tslib_root,
            dataset_root=dataset_dir,
            dataset_csv_name=dataset_csv.name,
            dataset_target=dataset_target,
            model_name=model_name,
            loss_target_indices=loss_target_indices_to_pass,
            use_dec_mask_embedding=use_dec_mask_embedding,
            features=features,
            freq=freq,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            n_vars=n_vars,
            train_epochs=int(train_cfg.get("epochs", 10)),
            batch_size=int(train_cfg.get("batch_size", 32)),
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            patience=int(train_cfg.get("patience", 3)),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
            use_gpu=use_gpu_effective,
            gpu=gpu_effective,
            d_model=int(model_cfg.get("d_model", 256)),
            n_heads=int(model_cfg.get("n_heads", 8)),
            e_layers=int(model_cfg.get("e_layers", 2)),
            d_layers=int(model_cfg.get("d_layers", 1)),
            d_ff=int(model_cfg.get("d_ff", 512)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            factor=int(model_cfg.get("factor", 3)),
            patch_len=(int(model_cfg.get("patch_len")) if model_cfg.get("patch_len", None) is not None else None),
            use_amp=bool(train_cfg.get("use_amp", False)),
            inverse=inverse,
            num_workers=int(train_cfg.get("num_workers", 4)),
            seed=int(train_cfg.get("seed", 2021)),
            des=str(model_cfg.get("des", "Exp")),
            loss_time_weighted=loss_time_weighted,
            loss_time_weight_decay=loss_time_weight_decay,
            loss_time_weight_min=loss_time_weight_min,
        )

        preds = np.load(results_setting_dir / "pred.npy")
        trues = np.load(results_setting_dir / "true.npy")
        pred_col = str(model_name)
        cv_predictions = build_cv_predictions_from_npy(
            prepared_csv=dataset_csv,
            var_names=var_names,
            target_series=target_series,
            features=features,
            seq_len=seq_len,
            pred_len=pred_len,
            preds=preds,
            trues=trues,
            pred_col=pred_col,
            split_meta=split_meta,
        )
        metrics_df, avg_metrics = evaluate_predictions(
            cv_df=cv_predictions,
            target_series=target_series,
            pred_col=pred_col,
            eps=eps,
        )
        save_outputs(
            out_dir=run_out_dir,
            cv_predictions=cv_predictions,
            metrics_df=metrics_df,
            avg_metrics=avg_metrics,
            config=config,
            config_path=config_path,
            pred_col=pred_col,
            model_name=model_name,
            tslib_results_dir=results_setting_dir,
        )
    else:
        # ========== 超参搜索（外部编排：多次调用 Time-Series-Library/run.py） ==========
        num_samples = int(auto_cfg.get("num_samples", 10))
        search_seed = int(auto_cfg.get("seed", 2021))
        objective = str(auto_cfg.get("objective", "MAE_avg"))
        space = auto_cfg.get("space") or {}
        if not isinstance(space, dict):
            raise ValueError("auto_search.space 必须是 dict")
        trials_dir = run_out_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)

        rng = np.random.RandomState(search_seed)
        trial_rows: List[Dict] = []
        pred_col = str(model_name)
        best_score = float("inf")
        best_params: Optional[Dict] = None
        best_trial_dir: Optional[Path] = None
        best_results_dir: Optional[Path] = None
        best_avg: Optional[Dict] = None
        best_metrics_df: Optional[pd.DataFrame] = None
        best_cv: Optional[pd.DataFrame] = None

        for i in range(num_samples):
            # 以 config 的默认值为 base，再按 space 覆盖
            params = {
                "optimizer": str(train_cfg.get("optimizer", "adam")),
                "d_model": int(_sample_from_space(rng, space.get("d_model"), int(model_cfg.get("d_model", 256)))),
                "n_heads": int(_sample_from_space(rng, space.get("n_heads"), int(model_cfg.get("n_heads", 8)))),
                "e_layers": int(_sample_from_space(rng, space.get("e_layers"), int(model_cfg.get("e_layers", 2)))),
                "d_layers": int(_sample_from_space(rng, space.get("d_layers"), int(model_cfg.get("d_layers", 1)))),
                "d_ff": int(_sample_from_space(rng, space.get("d_ff"), int(model_cfg.get("d_ff", 512)))),
                "dropout": float(_sample_from_space(rng, space.get("dropout"), float(model_cfg.get("dropout", 0.1)))),
                "factor": int(_sample_from_space(rng, space.get("factor"), int(model_cfg.get("factor", 3)))),
                "batch_size": int(_sample_from_space(rng, space.get("batch_size"), int(train_cfg.get("batch_size", 32)))),
                "learning_rate": float(
                    _sample_from_space(
                        rng,
                        space.get("learning_rate", {"min": float(train_cfg.get("learning_rate", 1e-4)), "max": float(train_cfg.get("learning_rate", 1e-4)), "log": True}),
                        float(train_cfg.get("learning_rate", 1e-4)),
                    )
                ),
                "train_epochs": int(_sample_from_space(rng, space.get("epochs"), int(train_cfg.get("epochs", 10)))),
                "patience": int(_sample_from_space(rng, space.get("patience"), int(train_cfg.get("patience", 3)))),
            }
            trial_tag = _format_trial_tag(i, params)
            trial_out_dir = trials_dir / trial_tag
            logger.info("=" * 80)
            logger.info(f"AutoSearch trial {i+1}/{num_samples}: {trial_tag}")
            logger.info("=" * 80)

            results_setting_dir = run_tslib_crossformer(
                tslib_root=tslib_root,
                dataset_root=dataset_dir,
                dataset_csv_name=dataset_csv.name,
                dataset_target=dataset_target,
                model_name=model_name,
                loss_target_indices=loss_target_indices_to_pass,
                use_dec_mask_embedding=use_dec_mask_embedding,
                features=features,
                freq=freq,
                seq_len=seq_len,
                label_len=label_len,
                pred_len=pred_len,
                n_vars=n_vars,
                train_epochs=int(params["train_epochs"]),
                batch_size=int(params["batch_size"]),
                learning_rate=float(params["learning_rate"]),
                patience=int(params["patience"]),
                grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
                use_gpu=use_gpu_effective,
                gpu=gpu_effective,
                d_model=int(params["d_model"]),
                n_heads=int(params["n_heads"]),
                e_layers=int(params["e_layers"]),
                d_layers=int(params["d_layers"]),
                d_ff=int(params["d_ff"]),
                dropout=float(params["dropout"]),
                factor=int(params["factor"]),
                patch_len=(int(model_cfg.get("patch_len")) if model_cfg.get("patch_len", None) is not None else None),
                use_amp=bool(train_cfg.get("use_amp", False)),
                inverse=inverse,
                num_workers=int(train_cfg.get("num_workers", 4)),
                seed=int(train_cfg.get("seed", 2021)),
                des=str(model_cfg.get("des", "Exp")),
                loss_time_weighted=loss_time_weighted,
                loss_time_weight_decay=loss_time_weight_decay,
                loss_time_weight_min=loss_time_weight_min,
            )

            preds = np.load(results_setting_dir / "pred.npy")
            trues = np.load(results_setting_dir / "true.npy")
            cv_predictions = build_cv_predictions_from_npy(
                prepared_csv=dataset_csv,
                var_names=var_names,
                target_series=target_series,
                features=features,
                seq_len=seq_len,
                pred_len=pred_len,
                preds=preds,
                trues=trues,
                pred_col=pred_col,
                split_meta=split_meta,
            )
            metrics_df, avg_metrics = evaluate_predictions(
                cv_df=cv_predictions,
                target_series=target_series,
                pred_col=pred_col,
                eps=eps,
            )

            # 保存 trial 输出
            # config_used：在原 config 基础上记录本次 trial 的参数
            cfg_used = json.loads(json.dumps(config, ensure_ascii=False))
            cfg_used.setdefault("model", {})
            cfg_used.setdefault("train", {})
            cfg_used["model"].update(
                {
                    "d_model": int(params["d_model"]),
                    "n_heads": int(params["n_heads"]),
                    "e_layers": int(params["e_layers"]),
                    "d_layers": int(params["d_layers"]),
                    "d_ff": int(params["d_ff"]),
                    "dropout": float(params["dropout"]),
                    "factor": int(params["factor"]),
                }
            )
            cfg_used["train"].update(
                {
                    "optimizer": str(params["optimizer"]),
                    "epochs": int(params["train_epochs"]),
                    "batch_size": int(params["batch_size"]),
                    "learning_rate": float(params["learning_rate"]),
                    "patience": int(params["patience"]),
                }
            )
            trial_out_dir.mkdir(parents=True, exist_ok=True)
            with open(trial_out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_used, f, allow_unicode=True, sort_keys=False)
            save_outputs(
                out_dir=trial_out_dir,
                cv_predictions=cv_predictions,
                metrics_df=metrics_df,
                avg_metrics=avg_metrics,
                config=cfg_used,
                config_path=config_path,  # 仍拷贝原 config.yaml（便于回溯）
                pred_col=pred_col,
                model_name=model_name,
                tslib_results_dir=results_setting_dir,
            )

            score = _objective_from_metrics(avg_metrics, objective=objective)
            row = {"trial": trial_tag, "objective": objective, "score": score, **params, **avg_metrics, "trial_dir": str(trial_out_dir)}
            trial_rows.append(row)
            if score < best_score:
                best_score = score
                best_params = dict(params)
                best_trial_dir = trial_out_dir
                best_results_dir = results_setting_dir
                best_avg = avg_metrics
                best_metrics_df = metrics_df
                best_cv = cv_predictions

        summary_df = pd.DataFrame(trial_rows).sort_values("score").reset_index(drop=True)
        summary_df.to_csv(run_out_dir / "auto_search_summary.csv", index=False)

        if (
            best_trial_dir is None
            or best_cv is None
            or best_metrics_df is None
            or best_avg is None
            or best_results_dir is None
            or best_params is None
        ):
            raise RuntimeError("auto_search 未产生有效 trial")

        # 搜索结束后默认做一次 train+val 复训（test 仅用于最终评估）
        retrain_split = {"train_ratio": 0.8, "val_ratio": 0.0, "test_ratio": 0.2}
        logger.info("=" * 80)
        logger.info("AutoSearch 结束：使用最优超参进行 train+val 复训（split=%s）", retrain_split)
        logger.info("best_trial=%s, score=%.6f, params=%s", str(best_trial_dir), best_score, best_params)
        logger.info("=" * 80)

        retrain_results_dir = run_tslib_crossformer(
            tslib_root=tslib_root,
            dataset_root=dataset_dir,
            dataset_csv_name=dataset_csv.name,
            dataset_target=dataset_target,
            model_name=model_name,
            loss_target_indices=loss_target_indices_to_pass,
            use_dec_mask_embedding=use_dec_mask_embedding,
            features=features,
            freq=freq,
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            n_vars=n_vars,
            train_epochs=int(best_params["train_epochs"]),
            batch_size=int(best_params["batch_size"]),
            learning_rate=float(best_params["learning_rate"]),
            patience=int(best_params["patience"]),
            grad_clip_norm=float(train_cfg.get("grad_clip_norm", 0.0)),
            use_gpu=use_gpu_effective,
            gpu=gpu_effective,
            d_model=int(best_params["d_model"]),
            n_heads=int(best_params["n_heads"]),
            e_layers=int(best_params["e_layers"]),
            d_layers=int(best_params["d_layers"]),
            d_ff=int(best_params["d_ff"]),
            dropout=float(best_params["dropout"]),
            factor=int(best_params["factor"]),
            patch_len=(int(model_cfg.get("patch_len")) if model_cfg.get("patch_len", None) is not None else None),
            use_amp=bool(train_cfg.get("use_amp", False)),
            inverse=inverse,
            num_workers=int(train_cfg.get("num_workers", 4)),
            seed=int(train_cfg.get("seed", 2021)),
            des=str(model_cfg.get("des", "Exp")),
            loss_time_weighted=loss_time_weighted,
            loss_time_weight_decay=loss_time_weight_decay,
            loss_time_weight_min=loss_time_weight_min,
            train_ratio=float(retrain_split["train_ratio"]),
            val_ratio=float(retrain_split["val_ratio"]),
            test_ratio=float(retrain_split["test_ratio"]),
        )
        retrain_preds = np.load(retrain_results_dir / "pred.npy")
        retrain_trues = np.load(retrain_results_dir / "true.npy")
        retrain_cv = build_cv_predictions_from_npy(
            prepared_csv=dataset_csv,
            var_names=var_names,
            target_series=target_series,
            features=features,
            seq_len=seq_len,
            pred_len=pred_len,
            preds=retrain_preds,
            trues=retrain_trues,
            pred_col=pred_col,
            split_meta=split_meta,
        )
        retrain_metrics_df, retrain_avg = evaluate_predictions(
            cv_df=retrain_cv,
            target_series=target_series,
            pred_col=pred_col,
            eps=eps,
        )

        cfg_retrain = json.loads(json.dumps(config, ensure_ascii=False))
        cfg_retrain.setdefault("model", {})
        cfg_retrain.setdefault("train", {})
        cfg_retrain.setdefault("auto_search", {})
        cfg_retrain["model"].update(
            {
                "d_model": int(best_params["d_model"]),
                "n_heads": int(best_params["n_heads"]),
                "e_layers": int(best_params["e_layers"]),
                "d_layers": int(best_params["d_layers"]),
                "d_ff": int(best_params["d_ff"]),
                "dropout": float(best_params["dropout"]),
                "factor": int(best_params["factor"]),
            }
        )
        cfg_retrain["train"].update(
            {
                "optimizer": str(best_params["optimizer"]),
                "epochs": int(best_params["train_epochs"]),
                "batch_size": int(best_params["batch_size"]),
                "learning_rate": float(best_params["learning_rate"]),
                "patience": int(best_params["patience"]),
            }
        )
        cfg_retrain["auto_search"]["retrain_data_split"] = retrain_split

        # run 根目录最终输出以“复训结果”为准（replace_root）
        save_outputs(
            out_dir=run_out_dir,
            cv_predictions=retrain_cv,
            metrics_df=retrain_metrics_df,
            avg_metrics=retrain_avg,
            config=cfg_retrain,
            config_path=config_path,
            pred_col=pred_col,
            model_name=model_name,
            tslib_results_dir=retrain_results_dir,
        )
        with open(run_out_dir / "auto_search_best.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_trial_dir": str(best_trial_dir),
                    "best_trial_results_dir": str(best_results_dir),
                    "best_score": best_score,
                    "objective": objective,
                    "best_params": best_params,
                    "retrain_split": retrain_split,
                    "retrain_results_dir": str(retrain_results_dir),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    logger.info("=" * 80)
    logger.info("训练与评估完成！")
    logger.info(f"输出目录：{run_out_dir}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


