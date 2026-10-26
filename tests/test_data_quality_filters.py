from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
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

    def test_strict_raw_policy_keeps_quantile_samples_unchanged(self) -> None:
        values = pd.Series(self.baseline)
        values.iloc[100] = 100

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        pd.testing.assert_series_equal(cleaned, values)
        self.assertEqual(audit["status"], "not_applied_strict_raw_window")

    def test_strict_raw_policy_keeps_sustained_level_change(self) -> None:
        values = pd.Series(self.baseline)
        values.iloc[100:106] = 100

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        pd.testing.assert_series_equal(cleaned, values)
        self.assertEqual(audit["status"], "not_applied_strict_raw_window")

    def test_skips_when_sample_count_is_too_small(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])

        cleaned, audit = data_quality.quantile_iqr_clean_series(pd, values, self.settings)

        self.assertTrue(cleaned.equals(values))
        self.assertEqual(audit["status"], "not_applied_strict_raw_window")

    def test_strict_raw_policy_keeps_single_spike_for_window_quality_gate(self) -> None:
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

        pd.testing.assert_series_equal(cleaned, values)
        self.assertEqual(audit["isolated_spikes_masked"], 0)
        self.assertEqual(audit["status"], "not_applied_strict_raw_window")

    def test_strict_raw_policy_keeps_sustained_level_change(self) -> None:
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

        pd.testing.assert_series_equal(cleaned, values)
        self.assertEqual(audit["isolated_spikes_masked"], 0)

    def test_strict_raw_policy_keeps_quantized_values_unchanged(self) -> None:
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

        pd.testing.assert_series_equal(cleaned, values)
        self.assertEqual(audit["isolated_spikes_masked"], 0)

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

    def test_clean_model_frame_does_not_fill_or_smooth_missing_source_values(self) -> None:
        frame = pd.DataFrame({"timestamp": pd.date_range("2026-10-24", periods=3, freq="10s"),
                              "X": [1.0, None, 3.0]})
        audit = []
        result = data_quality.clean_model_frame(
            pd, frame, [], {"derived_features": {
                "primary_air_sections": [{"name": "derived", "sources": ["A", "B"]}],
            }}, model="子模型1", input_spike_audit=audit,
        )
        pd.testing.assert_frame_equal(result, frame)
        self.assertTrue(audit)

    def test_training_mask_rejects_any_missing_model_input(self) -> None:
        frame = pd.DataFrame({
            "timestamp": pd.date_range("2026-10-24", periods=3, freq="10s"),
            "X": [1.0, None, 3.0], "Y": [2.0, 2.5, 3.0],
        })
        mask = data_quality.build_training_mask(
            pd, frame, {"training_exclusion_buffer_seconds": 0},
            ["X", "Y"], ["Y"], freq_seconds=10, model="子模型1",
        )
        self.assertEqual([True, False, True], mask["valid_for_training"].tolist())

    def test_derived_features_do_not_change_when_future_rows_are_added(self) -> None:
        config = {"derived_features": {
            "primary_air_sections": [{"name": "section", "sources": ["A", "B"]}],
        }}
        prefix = pd.DataFrame({"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]})
        extended = pd.DataFrame({"A": [1000.0, 1000.0, 9000.0], "B": [1000.0, 1000.0, 9000.0]})
        before, _ = data_quality.add_derived_features(pd, prefix, config)
        after, _ = data_quality.add_derived_features(pd, extended, config)
        pd.testing.assert_series_equal(before["section"], after["section"].iloc[:2].reset_index(drop=True))

    def test_resampling_rejects_incomplete_ten_second_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.parquet", root / "cache.parquet"
            pd.DataFrame({
                "timestamp": pd.date_range("1970-01-01T00:00:01", periods=19, freq="1s"),
                "value": list(range(1, 20)),
            }).to_parquet(source, index=False)
            connection = duckdb.connect(":memory:")
            data_quality.resample_point(
                connection, pd, source, "X", output,
                freq_seconds=10, start=None, end=None,
            )
            result = pd.read_parquet(output)
            self.assertEqual([10, 20], result["timestamp"].dt.second.tolist())
            self.assertEqual(5.5, result["X"].iloc[0])
            self.assertTrue(pd.isna(result["X"].iloc[1]))
            self.assertEqual([10, 9], result["raw_count"].tolist())
            connection.close()

    def test_quality_missing_rate_uses_expected_time_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({
                "timestamp": pd.to_datetime(["2026-10-24T00:00:10", "2026-10-24T00:00:30"]),
                "X": [1.0, 3.0],
            }).to_parquet(root / "X.parquet", index=False)
            quality = data_quality.global_quality_from_caches(
                pd, {"cache": root}, ["X"], {}, freq_seconds=10,
                start=pd.Timestamp("2026-10-24T00:00:10"),
                end=pd.Timestamp("2026-10-24T00:00:40"),
            ).iloc[0]
            self.assertEqual(3, quality["rows"])
            self.assertEqual(2, quality["observed_rows"])
            self.assertAlmostEqual(1 / 3, quality["missing_rate"])


if __name__ == "__main__":
    unittest.main()
