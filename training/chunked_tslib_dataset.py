from __future__ import annotations

import bisect
import csv
import json
import math
import os
import random
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[2]
DATA_ENGINEERING_ROOT_OVERRIDE = os.environ.get("WISBURN_DATA_ENGINEERING_ROOT")
DATA_ENGINEERING_CANDIDATES = [
    WORKSPACE_ROOT / "offline" / "forecast" / "data_engineering",
    WORKSPACE_ROOT / "backend" / "pipeline" / "model_data_engineering",
]
if DATA_ENGINEERING_ROOT_OVERRIDE:
    DATA_ENGINEERING_DIR = Path(DATA_ENGINEERING_ROOT_OVERRIDE)
else:
    DATA_ENGINEERING_DIR = next(
        (path for path in DATA_ENGINEERING_CANDIDATES if (path / "config" / "default_config.json").exists()),
        DATA_ENGINEERING_CANDIDATES[0],
    )
ARTIFACT_ROOT = Path(os.environ.get(
    "WISBURN_FORECAST_ARTIFACT_ROOT",
    WORKSPACE_ROOT / "offline" / "artifacts" / "forecast",
))
DATASET_OUTPUT_ROOT_OVERRIDE = os.environ.get("WISBURN_FORECAST_DATASET_OUTPUT_ROOT")
DEFAULT_DATASET_OUTPUT_ROOTS = [
    ARTIFACT_ROOT / "data_engineering" / "outputs",
    DATA_ENGINEERING_DIR / "outputs",
    WORKSPACE_ROOT / "backend" / "pipeline" / "model_data_engineering" / "outputs",
]
TRAINING_READY_ROOT_OVERRIDE = os.environ.get("WISBURN_FORECAST_TRAINING_READY_ROOT")
if TRAINING_READY_ROOT_OVERRIDE:
    TRAINING_READY_ROOT = Path(TRAINING_READY_ROOT_OVERRIDE)
elif SCRIPT_DIR.name == "model_training":
    TRAINING_READY_ROOT = SCRIPT_DIR / "outputs" / "training_ready"
else:
    TRAINING_READY_ROOT = ARTIFACT_ROOT / "training_ready"
DEFAULT_CONFIG = DATA_ENGINEERING_DIR / "config" / "default_config.json"
if str(DATA_ENGINEERING_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_ENGINEERING_DIR))

# VSCode interactive defaults.
# Edit these values, then click "Run Python File" in VSCode.
INTERACTIVE_MODEL_SLUG = "model1"  # model1 | model2
INTERACTIVE_FREQ_SECONDS = 10
INTERACTIVE_SEQ_LEN = 120
INTERACTIVE_LABEL_LEN = 60
INTERACTIVE_PRED_LEN = 60
INTERACTIVE_BATCH_SIZE = 16
INTERACTIVE_NUM_WORKERS = 0
INTERACTIVE_CACHE_SIZE = 4
INTERACTIVE_MAX_SMOKE_WINDOWS = 512
INTERACTIVE_FORCE_REFIT_SCALER = True


def dataset_output_root_for(dataset_id: str) -> Path:
    if DATASET_OUTPUT_ROOT_OVERRIDE:
        return Path(DATASET_OUTPUT_ROOT_OVERRIDE)
    dataset_dir = Path("datasets") / "chunked" / dataset_id
    for output_root in DEFAULT_DATASET_OUTPUT_ROOTS:
        if (output_root / dataset_dir).exists():
            return output_root
    return DEFAULT_DATASET_OUTPUT_ROOTS[0]


def maybe_add_project_site_packages() -> None:
    site_packages = WORKSPACE_ROOT / ".venv_data_store" / "Lib" / "site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


try:
    import numpy as np
    import pandas as pd
except Exception:
    maybe_add_project_site_packages()
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:  # pragma: no cover - import guard for clearer user error.
        raise SystemExit(
            "Missing data dependencies. Run with the project Python environment, "
            "or install numpy pandas pyarrow."
        ) from exc

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
except Exception:
    torch = None
    DataLoader = None
    Dataset = object
    Sampler = object


MODEL_SLUG_TO_INDEX = {"model1": 0, "model2": 1, "model3": 2}
CONTROL_COLUMNS = {"timestamp", "valid_for_training"}
# fit_scaler floors a zero variance at 1e-6.  A non-target input at that
# floor is constant throughout the usable training split and must not be
# allowed to amplify a small later-period drift into a million-scale z-score.
NEAR_CONSTANT_TRAIN_SCALE = 1e-5


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    frame_path: Path
    window_path: Path
    windows: int
    start: int
    end: int


@dataclass(frozen=True)
class StandardScaler:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        out = (values.astype("float32", copy=False) - self.mean) / self.scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def model_dataset_id(model_slug: str, freq_seconds: int, seq_len: int, pred_len: int) -> str:
    return f"{model_slug}_{freq_seconds}s_seq{seq_len}_pred{pred_len}"


def chunk_root_for(model_slug: str, freq_seconds: int, seq_len: int, pred_len: int) -> Path:
    dataset_id = model_dataset_id(model_slug, freq_seconds, seq_len, pred_len)
    return dataset_output_root_for(dataset_id) / "datasets" / "chunked" / dataset_id


def training_ready_dir_for(model_slug: str, freq_seconds: int, seq_len: int, pred_len: int) -> Path:
    return TRAINING_READY_ROOT / model_dataset_id(model_slug, freq_seconds, seq_len, pred_len)


def stored_paths_for(dataset_id: str) -> dict[str, str]:
    return {
        "chunk_root": str(dataset_output_root_for(dataset_id) / "datasets" / "chunked" / dataset_id),
        "adapter_py": str(SCRIPT_DIR / "chunked_tslib_dataset.py"),
        "metadata_json": "metadata.json",
        "scaler_json": "scaler.json",
    }


def metadata_file_path(metadata: dict[str, Any]) -> Path:
    raw = Path(metadata["metadata_json"])
    if raw.is_absolute():
        return raw
    if metadata.get("training_ready_dir"):
        return Path(metadata["training_ready_dir"]) / raw
    return training_ready_dir_for(
        metadata["model_slug"],
        int(metadata["freq_seconds"]),
        int(metadata["seq_len"]),
        int(metadata["pred_len"]),
    ) / raw


def resolve_from_metadata(metadata_path: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (Path(metadata_path).parent / path).resolve()


def load_build_specs(model_slug: str) -> tuple[list[str], dict[str, Any]]:
    import build_model_datasets as bmd

    cfg = bmd.load_config(DEFAULT_CONFIG)
    specs = bmd.load_variable_specs(cfg)
    model_idx = MODEL_SLUG_TO_INDEX.get(model_slug)
    if model_idx is None or model_idx >= len(cfg.get("models", [])):
        raise ValueError(f"Unknown model_slug: {model_slug}")
    model_name = cfg["models"][model_idx]
    target_codes = [
        s.point_code
        for s in specs
        if s.model == model_name and bmd.is_output_role(s.io_type)
    ]
    return target_codes, cfg


def unusable_output_codes(model_slug: str, freq_seconds: int) -> set[str]:
    reports_root = dataset_output_root_for(model_dataset_id(model_slug, freq_seconds, 120, 60)) / "reports"
    report = reports_root / model_slug / "quality" / f"unusable_outputs_{model_slug}_{freq_seconds}s.csv"
    if not report.exists():
        report = reports_root / f"unusable_outputs_{model_slug}_{freq_seconds}s.csv"
    if not report.exists():
        return set()
    rows = read_csv_rows(report)
    return {r.get("point_code", "") for r in rows if r.get("point_code", "")}


def resolve_chunk_artifact(chunk_root: Path, raw_path: str, directory: str) -> Path:
    """Resolve a manifest path after the dataset is copied to another machine."""
    path = Path(raw_path)
    if path.exists():
        return path
    filename = Path(raw_path.replace("\\", "/")).name
    fallback = Path(chunk_root) / directory / filename
    if fallback.exists():
        return fallback
    return path


def infer_columns(chunk_root: Path, model_slug: str, freq_seconds: int) -> tuple[list[str], list[str]]:
    manifest = read_csv_rows(chunk_root / "manifest.csv")
    first_frame = None
    for row in manifest:
        frame_path = resolve_chunk_artifact(chunk_root, row["frame_path"], "frames")
        if frame_path.exists():
            first_frame = frame_path
            break
    if first_frame is None:
        raise SystemExit(f"No frame parquet found under {chunk_root}")

    df_head = pd.read_parquet(first_frame)
    feature_columns = [c for c in df_head.columns if c not in CONTROL_COLUMNS]

    target_candidates, _ = load_build_specs(model_slug)
    bad_targets = unusable_output_codes(model_slug, freq_seconds)
    contract = bmd.load_model_contract(model_slug)
    target_columns = list(contract["target_columns"])
    missing_targets = [c for c in target_columns if c not in feature_columns]
    invalid_roles = [c for c in target_columns if c not in target_candidates]
    excluded_targets = [c for c in target_columns if c in bad_targets]
    if missing_targets or invalid_roles or excluded_targets:
        raise SystemExit(
            f"{model_slug} 输出列不符合统一契约：missing={missing_targets}, "
            f"wrong_role={invalid_roles}, unusable={excluded_targets}"
        )
    if len(feature_columns) != int(contract["feature_count"]):
        raise SystemExit(
            f"{model_slug} 输入列数 {len(feature_columns)} 与统一契约 {contract['feature_count']} 不一致"
        )
    optional = list(contract["missing_input_policy"].get("columns") or [])
    missing_optional = [c for c in optional if c not in feature_columns]
    if missing_optional:
        raise SystemExit(f"{model_slug} 缺失策略点不在训练特征中: {missing_optional}")
    if not target_columns:
        raise SystemExit(f"No usable target columns found for {model_slug}.")
    return feature_columns, target_columns


def validate_current_splits(chunk_root: Path) -> None:
    manifest_rows = read_csv_rows(chunk_root / "manifest.csv")
    split_rows = read_csv_rows(chunk_root / "splits" / "split_manifest.csv")
    current = {row["chunk_id"]: row for row in manifest_rows}
    included: set[str] = set()
    mismatches: list[str] = []
    for row in split_rows:
        chunk_id = row["chunk_id"]
        source = current.get(chunk_id)
        if source is None:
            mismatches.append(f"unknown chunk {chunk_id}")
        elif row.get("param_id") != source.get("param_id") or row.get("windows") != source.get("windows"):
            mismatches.append(f"outdated chunk {chunk_id}")
        included.add(chunk_id)

    expected = {row["chunk_id"] for row in manifest_rows if int(float(row.get("windows") or 0)) > 0}
    missing = expected - included
    if missing:
        mismatches.append(f"{len(missing)} current non-empty chunks are missing")

    expected_windows = sum(int(float(row.get("windows") or 0)) for row in manifest_rows)
    split_windows = sum(int(float(row.get("windows") or 0)) for row in split_rows)
    if split_windows != expected_windows:
        mismatches.append(f"window total {split_windows} != current manifest {expected_windows}")

    for split in ("train", "val", "test"):
        path = chunk_root / "splits" / f"{split}_chunks.csv"
        if not path.exists() or not read_csv_rows(path):
            mismatches.append(f"missing or empty {split}_chunks.csv")

    if mismatches:
        details = "; ".join(mismatches[:5])
        raise SystemExit(
            f"Stale or incomplete split files under {chunk_root / 'splits'}: {details}. "
            "Regenerate splits from the current manifest before preparing training."
        )


def split_rows(chunk_root: Path, split: str) -> list[dict[str, str]]:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split}")
    return read_csv_rows(chunk_root / "splits" / f"{split}_chunks.csv")


def build_chunks(
    chunk_root: Path,
    rows: list[dict[str, str]],
    max_windows: int | None = None,
) -> list[ChunkInfo]:
    chunks: list[ChunkInfo] = []
    cursor = 0
    for row in rows:
        n = int(float(row.get("windows") or 0))
        if n <= 0:
            continue
        if max_windows is not None:
            remaining = max_windows - cursor
            if remaining <= 0:
                break
            n = min(n, remaining)
        start = cursor
        end = cursor + n
        chunks.append(
            ChunkInfo(
                chunk_id=row["chunk_id"],
                frame_path=resolve_chunk_artifact(chunk_root, row["frame_path"], "frames"),
                window_path=resolve_chunk_artifact(chunk_root, row["window_path"], "windows"),
                windows=n,
                start=start,
                end=end,
            )
        )
        cursor = end
    return chunks


def save_scaler(path: Path, scaler: StandardScaler) -> None:
    write_json(
        path,
        {
            "columns": scaler.columns,
            "mean": scaler.mean.astype(float).tolist(),
            "scale": scaler.scale.astype(float).tolist(),
        },
    )


def load_scaler(path: Path) -> StandardScaler:
    payload = load_json(path)
    return StandardScaler(
        columns=list(payload["columns"]),
        mean=np.asarray(payload["mean"], dtype="float32"),
        scale=np.asarray(payload["scale"], dtype="float32"),
    )


def drop_near_constant_train_inputs(
    feature_columns: list[str],
    target_columns: list[str],
    scaler: StandardScaler,
    preserve_columns: set[str] | None = None,
) -> tuple[list[str], StandardScaler, list[str]]:
    target_set = set(target_columns)
    preserve = set(preserve_columns or ())
    kept_indices: list[int] = []
    dropped: list[str] = []
    for idx, column in enumerate(feature_columns):
        if (column not in target_set and column not in preserve
                and float(scaler.scale[idx]) <= NEAR_CONSTANT_TRAIN_SCALE):
            dropped.append(column)
        else:
            kept_indices.append(idx)
    if not dropped:
        return feature_columns, scaler, dropped
    print(
        "dropping near-constant train inputs: " + ", ".join(dropped),
        flush=True,
    )
    return (
        [feature_columns[idx] for idx in kept_indices],
        StandardScaler(
            [feature_columns[idx] for idx in kept_indices],
            scaler.mean[kept_indices],
            scaler.scale[kept_indices],
        ),
        dropped,
    )


def fit_scaler(
    chunk_root: Path,
    feature_columns: list[str],
    *,
    max_chunks: int | None = None,
) -> StandardScaler:
    rows = split_rows(chunk_root, "train")
    if max_chunks is not None:
        rows = rows[:max_chunks]
    n_features = len(feature_columns)
    count = np.zeros(n_features, dtype="float64")
    total = np.zeros(n_features, dtype="float64")
    total_sq = np.zeros(n_features, dtype="float64")

    read_columns = ["valid_for_training", *feature_columns]
    used_frames = 0
    for idx, row in enumerate(rows, 1):
        if int(float(row.get("windows") or 0)) <= 0:
            continue
        frame_path = resolve_chunk_artifact(chunk_root, row["frame_path"], "frames")
        if not frame_path.exists():
            raise FileNotFoundError(frame_path)
        frame = pd.read_parquet(frame_path, columns=read_columns)
        if "valid_for_training" in frame.columns:
            frame = frame[frame["valid_for_training"].fillna(False).astype(bool)]
        values = frame.reindex(columns=feature_columns).to_numpy(dtype="float64", copy=False)
        finite = np.isfinite(values)
        count += finite.sum(axis=0)
        clean = np.where(finite, values, 0.0)
        total += clean.sum(axis=0)
        total_sq += (clean * clean).sum(axis=0)
        used_frames += 1
        if idx % 25 == 0:
            print(f"fit scaler: {idx}/{len(rows)} train chunks", flush=True)

    safe_count = np.maximum(count, 1.0)
    mean = total / safe_count
    variance = total_sq / safe_count - mean * mean
    scale = np.sqrt(np.maximum(variance, 1e-12))
    scale[count <= 1] = 1.0
    print(f"fit scaler done: frames={used_frames} features={n_features}", flush=True)
    return StandardScaler(feature_columns, mean.astype("float32"), scale.astype("float32"))


def time_features(timestamps: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(timestamps)
    data = np.column_stack(
        [
            dt.dt.month.to_numpy(dtype="float32") / 12.0,
            dt.dt.day.to_numpy(dtype="float32") / 31.0,
            dt.dt.dayofweek.to_numpy(dtype="float32") / 6.0,
            dt.dt.hour.to_numpy(dtype="float32") / 23.0,
            dt.dt.minute.to_numpy(dtype="float32") / 59.0,
            dt.dt.second.to_numpy(dtype="float32") / 59.0,
        ]
    )
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)


class ChunkedWindowDataset(Dataset):
    """Time-Series-Library style dataset backed by chunked parquet + window CSV."""

    def __init__(
        self,
        chunk_root: Path,
        split: str,
        *,
        feature_columns: list[str],
        target_columns: list[str],
        scaler: StandardScaler,
        seq_len: int,
        label_len: int,
        pred_len: int,
        missing_input_columns: list[str] | None = None,
        max_windows: int | None = None,
        cache_size: int = 4,
    ) -> None:
        if label_len > seq_len:
            raise ValueError("label_len must be <= seq_len")
        self.chunk_root = Path(chunk_root)
        self.split = split
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.target_indices = [self.feature_columns.index(c) for c in self.target_columns]
        self.scaler = scaler
        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.missing_input_columns = list(missing_input_columns or [])
        unknown_missing = sorted(set(self.missing_input_columns) - set(self.feature_columns))
        if unknown_missing:
            raise ValueError(f"Missing-input policy columns are absent from features: {unknown_missing}")
        self.missing_input_indices = [self.feature_columns.index(c) for c in self.missing_input_columns]
        self.cache_size = max(1, int(cache_size))
        self.chunks = build_chunks(
            self.chunk_root,
            split_rows(self.chunk_root, split),
            max_windows=max_windows,
        )
        self.ends = [c.end for c in self.chunks]
        self._frame_cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self._window_cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def chunk_index_for(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self.ends, index)

    def _get_cached_frame(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if path in self._frame_cache:
            self._frame_cache.move_to_end(path)
            return self._frame_cache[path]
        columns = ["timestamp", *self.feature_columns]
        frame = pd.read_parquet(path, columns=columns)
        self._frame_cache[path] = frame
        while len(self._frame_cache) > self.cache_size:
            self._frame_cache.popitem(last=False)
        return frame

    def _get_cached_windows(self, path: Path) -> pd.DataFrame:
        path = Path(path)
        if path in self._window_cache:
            self._window_cache.move_to_end(path)
            return self._window_cache[path]
        windows = pd.read_csv(path, encoding="utf-8-sig")
        self._window_cache[path] = windows
        while len(self._window_cache) > self.cache_size:
            self._window_cache.popitem(last=False)
        return windows

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
        chunk_idx = self.chunk_index_for(index)
        chunk = self.chunks[chunk_idx]
        local_idx = index - chunk.start
        windows = self._get_cached_windows(chunk.window_path)
        row = windows.iloc[local_idx]
        frame = self._get_cached_frame(chunk.frame_path)

        input_start = int(row["input_start_pos"])
        input_end = int(row["input_end_pos"])
        target_end = int(row["target_end_pos"])
        label_start = input_end - self.label_len + 1
        y_end_exclusive = target_end + 1

        seq_x_frame = frame.iloc[input_start : input_end + 1]
        seq_y_frame = frame.iloc[label_start:y_end_exclusive]

        seq_x = seq_x_frame[self.feature_columns].to_numpy(dtype="float32", copy=False)
        seq_y = seq_y_frame[self.feature_columns].to_numpy(dtype="float32", copy=False)
        if self.missing_input_indices:
            seq_x = seq_x.copy()
            seq_y = seq_y.copy()
            means = self.scaler.mean[self.missing_input_indices]
            seq_x[:, self.missing_input_indices] = means
            seq_y[:, self.missing_input_indices] = means
        seq_x = self.scaler.transform(seq_x)
        seq_y = self.scaler.transform(seq_y)
        seq_x_mark = time_features(seq_x_frame["timestamp"])
        seq_y_mark = time_features(seq_y_frame["timestamp"])

        if torch is not None:
            return (
                torch.from_numpy(seq_x),
                torch.from_numpy(seq_y),
                torch.from_numpy(seq_x_mark),
                torch.from_numpy(seq_y_mark),
            )
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class ChunkBatchSampler(Sampler):
    """Yield batches grouped by chunk to reduce parquet/cache thrashing."""

    def __init__(
        self,
        dataset: ChunkedWindowDataset,
        batch_size: int,
        *,
        shuffle_chunks: bool = False,
        shuffle_within_chunk: bool = False,
        seed: int = 2026,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle_chunks = bool(shuffle_chunks)
        self.shuffle_within_chunk = bool(shuffle_within_chunk)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        chunk_indices = list(range(len(self.dataset.chunks)))
        if self.shuffle_chunks:
            rng.shuffle(chunk_indices)
        for chunk_idx in chunk_indices:
            chunk = self.dataset.chunks[chunk_idx]
            indices = list(range(chunk.start, chunk.end))
            if self.shuffle_within_chunk:
                rng.shuffle(indices)
            for pos in range(0, len(indices), self.batch_size):
                batch = indices[pos : pos + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch
        self.epoch += 1

    def __len__(self) -> int:
        if self.drop_last:
            return sum(c.windows // self.batch_size for c in self.dataset.chunks)
        return sum(math.ceil(c.windows / self.batch_size) for c in self.dataset.chunks)


def build_dataloader(
    dataset: ChunkedWindowDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    train: bool = False,
) -> Any:
    if DataLoader is None:
        raise RuntimeError("torch is not installed; DataLoader is unavailable.")
    sampler = ChunkBatchSampler(
        dataset,
        batch_size,
        shuffle_chunks=train,
        shuffle_within_chunk=train,
        drop_last=train,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)


def write_bridge_yaml(path: Path, metadata: dict[str, Any]) -> None:
    def q(value: str) -> str:
        return "'" + value.replace("'", "''").replace("\\", "/") + "'"

    lines = [
        "data:",
        "  adapter: chunked_window",
        f"  dataset_root: {q(metadata['chunk_root'])}",
        f"  metadata_json: {q(metadata['metadata_json'])}",
        f"  scaler_json: {q(metadata['scaler_json'])}",
        "  target_series:",
    ]
    lines += [f"    - {name}" for name in metadata["target_columns"]]
    lines += [
        "  target_indices:",
        *[f"    - {idx}" for idx in metadata["target_indices"]],
        "model:",
        f"  seq_len: {metadata['seq_len']}",
        f"  label_len: {metadata['label_len']}",
        f"  pred_len: {metadata['pred_len']}",
        f"  enc_in: {metadata['n_vars']}",
        f"  dec_in: {metadata['n_vars']}",
        f"  c_out: {metadata['n_vars']}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_bridge(
    model_slug: str,
    *,
    freq_seconds: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    force_refit_scaler: bool = False,
    output_root: Path | None = None,
) -> dict[str, Any]:
    chunk_root = chunk_root_for(model_slug, freq_seconds, seq_len, pred_len)
    if not chunk_root.exists():
        raise SystemExit(
            f"Missing chunked dataset: {chunk_root}. Set WISBURN_FORECAST_DATASET_OUTPUT_ROOT "
            "to the directory containing datasets/chunked, if the dataset is stored elsewhere."
        )
    validate_current_splits(chunk_root)
    ready_dir = (Path(output_root) if output_root is not None else TRAINING_READY_ROOT) / model_dataset_id(
        model_slug, freq_seconds, seq_len, pred_len
    )
    scaler_path = ready_dir / "scaler.json"
    metadata_path = ready_dir / "metadata.json"
    yaml_path = ready_dir / "tslib_chunked_config.yaml"

    candidate_columns, target_columns = infer_columns(chunk_root, model_slug, freq_seconds)
    model_contract = bmd.load_model_contract(model_slug)
    missing_input_columns = list(model_contract["missing_input_policy"].get("columns") or [])
    preserve_columns = set(missing_input_columns)
    expected_targets = list(model_contract["target_columns"])
    if target_columns != expected_targets:
        raise SystemExit(
            f"{model_slug} target_columns does not match the registered output order. "
            "Check the point configuration and unusable-output report before training."
        )
    feature_columns = list(candidate_columns)
    dropped_near_constant_inputs: list[str] = []
    if scaler_path.exists() and not force_refit_scaler:
        scaler = load_scaler(scaler_path)
        if scaler.columns == feature_columns:
            feature_columns, scaler, dropped_near_constant_inputs = drop_near_constant_train_inputs(
                feature_columns, target_columns, scaler, preserve_columns=preserve_columns
            )
            if dropped_near_constant_inputs:
                save_scaler(scaler_path, scaler)
        elif set(scaler.columns).issubset(candidate_columns) and set(target_columns).issubset(scaler.columns):
            feature_columns = list(scaler.columns)
            dropped_near_constant_inputs = [
                column for column in candidate_columns if column not in feature_columns
            ]
            missing_policy_dropped = sorted(preserve_columns - set(feature_columns))
            if missing_policy_dropped:
                scaler = fit_scaler(chunk_root, candidate_columns)
                feature_columns = list(candidate_columns)
                feature_columns, scaler, dropped_near_constant_inputs = drop_near_constant_train_inputs(
                    feature_columns, target_columns, scaler, preserve_columns=preserve_columns
                )
                save_scaler(scaler_path, scaler)
        else:
            print("scaler columns changed; refitting scaler", flush=True)
            scaler = fit_scaler(chunk_root, feature_columns)
            feature_columns, scaler, dropped_near_constant_inputs = drop_near_constant_train_inputs(
                feature_columns, target_columns, scaler, preserve_columns=preserve_columns
            )
            save_scaler(scaler_path, scaler)
    else:
        scaler = fit_scaler(chunk_root, feature_columns)
        feature_columns, scaler, dropped_near_constant_inputs = drop_near_constant_train_inputs(
            feature_columns, target_columns, scaler, preserve_columns=preserve_columns
        )
        save_scaler(scaler_path, scaler)

    if len(feature_columns) != int(model_contract["feature_count"]):
        raise SystemExit(
            f"{model_slug} 输入列数 {len(feature_columns)} 与统一契约 "
            f"{model_contract['feature_count']} 不一致；请重建训练帧/冻结 scaler。"
        )
    missing_policy_dropped = sorted(preserve_columns - set(feature_columns))
    if missing_policy_dropped:
        raise SystemExit(f"{model_slug} 缺失策略输入未保留: {missing_policy_dropped}")

    target_indices = [feature_columns.index(c) for c in target_columns]
    split_summary_path = chunk_root / "splits" / "split_summary.csv"
    split_summary = read_csv_rows(split_summary_path) if split_summary_path.exists() else []
    dataset_id = model_dataset_id(model_slug, freq_seconds, seq_len, pred_len)
    stored_paths = stored_paths_for(dataset_id)
    metadata = {
        "model_slug": model_slug,
        "dataset_id": dataset_id,
        "training_ready_dir": str(ready_dir.resolve()),
        "freq_seconds": freq_seconds,
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "chunk_root": stored_paths["chunk_root"],
        "adapter_py": stored_paths["adapter_py"],
        "metadata_json": stored_paths["metadata_json"],
        "scaler_json": stored_paths["scaler_json"],
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_indices": target_indices,
        "n_vars": len(feature_columns),
        "history_only": True,
        "no_future_covariates": True,
        "missing_input_policy": dict(model_contract["missing_input_policy"]),
        "preprocessing_policy": "strict_raw_complete_windows_v1",
        "time_feature_dim": 6,
        "split_summary": split_summary,
        "dropped_near_constant_train_inputs": dropped_near_constant_inputs,
        "notes": [
            "Use ChunkedWindowDataset/get_dataloaders instead of Dataset_Custom CSV for full-year 1s chunked data.",
            "Returned batch tuple is (batch_x, batch_y, batch_x_mark, batch_y_mark), matching Time-Series-Library shape conventions.",
            "batch_y length is label_len + pred_len; compute loss on batch_y[:, -pred_len:, target_indices].",
        ],
    }
    write_json(metadata_path, metadata)
    write_bridge_yaml(yaml_path, metadata)
    print(f"metadata: {rel_path(metadata_path)}", flush=True)
    print(f"scaler: {rel_path(scaler_path)}", flush=True)
    print(f"bridge yaml: {rel_path(yaml_path)}", flush=True)
    return metadata


def get_datasets(
    metadata_path: str | Path,
    *,
    cache_size: int = 4,
    max_windows: int | None = None,
) -> dict[str, ChunkedWindowDataset]:
    metadata_path = Path(metadata_path)
    metadata = load_json(metadata_path)
    scaler = load_scaler(resolve_from_metadata(metadata_path, metadata["scaler_json"]))
    chunk_root = resolve_from_metadata(metadata_path, metadata["chunk_root"])
    return {
        split: ChunkedWindowDataset(
            chunk_root,
            split,
            feature_columns=metadata["feature_columns"],
            target_columns=metadata["target_columns"],
            scaler=scaler,
            seq_len=int(metadata["seq_len"]),
            label_len=int(metadata["label_len"]),
            pred_len=int(metadata["pred_len"]),
            missing_input_columns=list((metadata.get("missing_input_policy") or {}).get("columns") or []),
            max_windows=max_windows,
            cache_size=cache_size,
        )
        for split in ("train", "val", "test")
    }


def get_dataloaders(
    metadata_path: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    cache_size: int = 4,
    max_windows: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = load_json(Path(metadata_path))
    datasets = get_datasets(metadata_path, cache_size=cache_size, max_windows=max_windows)
    loaders = {
        "train": build_dataloader(datasets["train"], batch_size=batch_size, num_workers=num_workers, train=True),
        "val": build_dataloader(datasets["val"], batch_size=batch_size, num_workers=num_workers, train=False),
        "test": build_dataloader(datasets["test"], batch_size=batch_size, num_workers=num_workers, train=False),
    }
    return loaders, metadata


def smoke_test(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_path = metadata_file_path(metadata)
    datasets = get_datasets(
        metadata_path,
        cache_size=INTERACTIVE_CACHE_SIZE,
        max_windows=INTERACTIVE_MAX_SMOKE_WINDOWS,
    )
    report: dict[str, Any] = {
        "model_slug": metadata["model_slug"],
        "n_vars": metadata["n_vars"],
        "target_columns": metadata["target_columns"],
        "target_indices": metadata["target_indices"],
        "dataset_lengths": {k: len(v) for k, v in datasets.items()},
    }

    first = datasets["train"][0]
    shapes = [list(x.shape) for x in first]
    report["single_item_shapes"] = {
        "seq_x": shapes[0],
        "seq_y": shapes[1],
        "seq_x_mark": shapes[2],
        "seq_y_mark": shapes[3],
    }

    if DataLoader is not None:
        loaders, _ = get_dataloaders(
            metadata_path,
            batch_size=INTERACTIVE_BATCH_SIZE,
            num_workers=INTERACTIVE_NUM_WORKERS,
            cache_size=INTERACTIVE_CACHE_SIZE,
            max_windows=INTERACTIVE_MAX_SMOKE_WINDOWS,
        )
        batch = next(iter(loaders["train"]))
        batch_shapes = [list(x.shape) for x in batch]
        report["batch_shapes"] = {
            "batch_x": batch_shapes[0],
            "batch_y": batch_shapes[1],
            "batch_x_mark": batch_shapes[2],
            "batch_y_mark": batch_shapes[3],
        }
    else:
        report["torch_available"] = False

    path = metadata_path.parent / "loader_smoke_report.json"
    write_json(path, report)
    print(f"smoke report: {rel_path(path)}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> int:
    metadata = prepare_bridge(
        INTERACTIVE_MODEL_SLUG,
        freq_seconds=INTERACTIVE_FREQ_SECONDS,
        seq_len=INTERACTIVE_SEQ_LEN,
        label_len=INTERACTIVE_LABEL_LEN,
        pred_len=INTERACTIVE_PRED_LEN,
        force_refit_scaler=INTERACTIVE_FORCE_REFIT_SCALER,
    )
    smoke_test(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
