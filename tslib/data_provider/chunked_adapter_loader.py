from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def load_adapter(metadata_path: Path):
    import json

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    adapter_path = Path(metadata["adapter_py"])
    if not adapter_path.is_absolute():
        adapter_path = (metadata_path.parent / adapter_path).resolve()
    spec = importlib.util.spec_from_file_location("model_data_engineering_chunked_tslib_dataset", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load chunked adapter: {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, metadata


def resolve_from_metadata(metadata_path: Path, raw):
    path = Path(raw)
    if path.is_absolute():
        return path
    return (metadata_path.parent / path).resolve()


class Dataset_Chunked(Dataset):
    """TSLib-compatible wrapper for model_data_engineering chunked windows."""

    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="metadata.json",
        target="OT",
        scale=True,
        timeenc=1,
        freq="s",
        seasonal_patterns=None,
    ):
        if size is None:
            size = [96, 48, 96]
        self.args = args
        self.root_path = Path(root_path)
        self.data_path = data_path
        self.metadata_path = self.root_path / self.data_path
        self.flag = flag
        self.seq_len, self.label_len, self.pred_len = [int(x) for x in size]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.adapter, self.metadata = load_adapter(self.metadata_path)
        self.scaler = self.adapter.load_scaler(resolve_from_metadata(self.metadata_path, self.metadata["scaler_json"]))
        max_windows_arg = (
            getattr(args, "chunk_max_train_windows", 0)
            if flag == "train"
            else getattr(args, "chunk_max_eval_windows", 0)
        )
        max_windows = int(max_windows_arg or 0) or None
        self.inner = self.adapter.ChunkedWindowDataset(
            resolve_from_metadata(self.metadata_path, self.metadata["chunk_root"]),
            flag,
            feature_columns=self.metadata["feature_columns"],
            target_columns=self.metadata["target_columns"],
            scaler=self.scaler,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            max_windows=max_windows,
            cache_size=int(getattr(args, "chunk_cache_size", 4)),
        )
        self.chunk_batch_sampler_cls = self.adapter.ChunkBatchSampler

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        return self.inner[index]

    def make_batch_sampler(self, batch_size, train=False, drop_last=False):
        return self.chunk_batch_sampler_cls(
            self.inner,
            batch_size,
            shuffle_chunks=bool(train),
            shuffle_within_chunk=bool(train),
            seed=int(getattr(self.args, "seed", 2021)),
            drop_last=bool(drop_last),
        )

    def inverse_transform(self, data):
        mean = self.scaler.mean.reshape((1,) * (data.ndim - 1) + (-1,))
        scale = self.scaler.scale.reshape((1,) * (data.ndim - 1) + (-1,))
        return np.asarray(data) * scale + mean
