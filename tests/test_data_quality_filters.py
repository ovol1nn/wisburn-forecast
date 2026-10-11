from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE = Path(__file__).resolve().parents[1] / "data_engineering" / "build_model_datasets.py"
TRAINING_DIR = MODULE.parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
SPEC = importlib.util.spec_from_file_location("forecast_data_quality", MODULE)
assert SPEC and SPEC.loader
data_quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = data_quality
SPEC.loader.exec_module(data_quality)


class QuantileOutlierFilterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = {
            "enabled": True,
            "lower_quantile": 0.25,
            "upper_quantile": 0.75,
            "iqr_multiplier": 3.0,
            "min_non_null_rows": 100,
        }
        self.baseline = [10 + (i % 5) * 0.1 for i in range(200)]

    def test_removes_only_isolated_outer_fence_spike(self) -> None:
        values = pd.Series(self.baseline)
        values.iloc[100] = 100

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        self.assertTrue(pd.isna(cleaned.iloc[100]))
        self.assertEqual(audit["isolated_outliers_removed"], 1)
        self.assertEqual(audit["status"], "applied")

    def test_keeps_contiguous_tail_values(self) -> None:
        values = pd.Series(self.baseline)
        values.iloc[100:106] = 100

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        self.assertTrue(cleaned.iloc[100:106].notna().all())
        self.assertEqual(audit["isolated_outliers_removed"], 0)

    def test_skips_when_sample_count_is_too_small(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        self.assertTrue(cleaned.equals(values))
        self.assertEqual(audit["status"], "insufficient_samples")

    def test_masks_single_spike_on_zero_iqr_input(self) -> None:
        values = pd.Series([0.0] * 50 + [2.0] + [0.0] * 50)
        settings = {
            "enabled": True,
            "window_points": 15,
            "min_periods": 5,
            "spike_sigma": 8.0,
            "neighbor_sigma": 3.0,
            "absolute_floor": 1e-6,
        }

        cleaned, audit = data_quality.isolated_input_spike_clean_series(pd, values, settings)

        self.assertTrue(pd.isna(cleaned.iloc[50]))
        self.assertEqual(audit["isolated_spikes_masked"], 1)
        self.assertEqual(audit["status"], "applied")

    def test_keeps_sustained_level_change(self) -> None:
        values = pd.Series([0.0] * 50 + [2.0] * 20 + [0.0] * 50)
        settings = {
            "enabled": True,
            "window_points": 15,
            "min_periods": 5,
            "spike_sigma": 8.0,
            "neighbor_sigma": 3.0,
            "absolute_floor": 1e-6,
        }

        cleaned, audit = data_quality.isolated_input_spike_clean_series(pd, values, settings)

        self.assertTrue(cleaned.iloc[50:70].notna().all())
        self.assertEqual(audit["isolated_spikes_masked"], 0)

    def test_ignores_quantized_steps_but_masks_large_spike(self) -> None:
        levels = [0.000635, 0.000684, 0.000733, 0.000782]
        values = pd.Series((levels * 25) + [0.036] + (levels * 25))
        settings = {
            "enabled": True,
            "window_points": 15,
            "min_periods": 5,
            "spike_sigma": 8.0,
            "neighbor_sigma": 3.0,
            "min_step_count": 20,
            "minimum_step_multiple": 4.0,
            "neighbor_step_multiple": 2.0,
            "absolute_floor": 1e-6,
        }

        cleaned, audit = data_quality.isolated_input_spike_clean_series(pd, values, settings)

        self.assertTrue(pd.isna(cleaned.iloc[100]))
        self.assertEqual(audit["isolated_spikes_masked"], 1)

    def test_training_bridge_reads_artifacts_dataset_root(self) -> None:
        import chunked_tslib_dataset as bridge

        workspace = Path(__file__).resolve().parents[3]
        expected = (
            workspace
            / "offline"
            / "artifacts"
            / "forecast"
            / "data_engineering"
            / "outputs"
            / "datasets"
            / "chunked"
            / "model1_10s_seq120_pred60"
        )
        self.assertEqual(bridge.chunk_root_for("model1", 10, 120, 60), expected)


if __name__ == "__main__":
    unittest.main()
