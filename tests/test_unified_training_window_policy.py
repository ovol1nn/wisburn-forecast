from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "data_engineering"))
sys.path.insert(0, str(MODULE_ROOT / "training"))
import build_model_datasets as bmd
import chunked_tslib_dataset as bridge


class UnifiedTrainingWindowPolicyTests(unittest.TestCase):
    def test_registered_sparse_inputs_do_not_invalidate_training_rows(self):
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="10s"),
            "required": [1.0, 2.0, 3.0],
            "sparse": [np.nan, np.nan, np.nan],
            "target": [4.0, 5.0, 6.0],
        })

        mask = bmd.build_training_mask(
            pd, frame, {}, ["required", "sparse", "target"], ["target"],
            freq_seconds=10, model="子模型2", optional_missing_input_codes=["sparse"],
        )

        self.assertEqual([True, True, True], mask["valid_for_training"].tolist())
        frame.loc[1, "required"] = np.nan
        mask = bmd.build_training_mask(
            pd, frame, {}, ["required", "sparse", "target"], ["target"],
            freq_seconds=10, model="子模型2", optional_missing_input_codes=["sparse"],
        )
        self.assertEqual([True, False, True], mask["valid_for_training"].tolist())

    def test_chunk_dataset_fills_registered_columns_with_frozen_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frames").mkdir()
            (root / "windows").mkdir()
            (root / "splits").mkdir()
            frame = pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=180, freq="10s"),
                "target": np.arange(180, dtype="float32"),
                "sparse": np.linspace(-500, 500, 180, dtype="float32"),
            })
            frame.loc[15, "sparse"] = np.nan
            frame.to_parquet(root / "frames" / "frame.parquet", index=False)
            pd.DataFrame([{
                "input_start_pos": 0,
                "input_end_pos": 119,
                "target_end_pos": 179,
            }]).to_csv(root / "windows" / "window.csv", index=False)
            with (root / "splits" / "train_chunks.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "chunk_id", "frame_path", "window_path", "windows",
                ])
                writer.writeheader()
                writer.writerow({
                    "chunk_id": "test",
                    "frame_path": "frames/frame.parquet",
                    "window_path": "windows/window.csv",
                    "windows": 1,
                })
            scaler = bridge.StandardScaler(
                columns=["target", "sparse"],
                mean=np.array([0.0, 7.0], dtype="float32"),
                scale=np.array([1.0, 2.0], dtype="float32"),
            )
            dataset = bridge.ChunkedWindowDataset(
                root, "train", feature_columns=["target", "sparse"],
                target_columns=["target"], scaler=scaler, seq_len=120,
                label_len=60, pred_len=60, missing_input_columns=["sparse"],
            )

            seq_x, seq_y, _x_mark, _y_mark = dataset[0]
            if hasattr(seq_x, "detach"):
                seq_x, seq_y = seq_x.detach().cpu().numpy(), seq_y.detach().cpu().numpy()

            self.assertTrue(np.allclose(seq_x[:, 1], 0.0))
            self.assertTrue(np.allclose(seq_y[:, 1], 0.0))
            self.assertEqual((120, 2), seq_x.shape)
            self.assertEqual((120, 2), seq_y.shape)

    def test_training_profiles_for_both_models_are_history_only(self):
        runner = (MODULE_ROOT / "training" / "run_chunked_tslib_training.py").read_text(encoding="utf-8")
        for slug in ("model1_10s", "model2_10s"):
            profile = runner.split(f'"{slug}": {{', 1)[1].split("\n    },", 1)[0]
            self.assertIn('"no_future_covariates": True', profile)
        self.assertIn('cmd.append("--no_future_covariates")', runner)


if __name__ == "__main__":
    unittest.main()
