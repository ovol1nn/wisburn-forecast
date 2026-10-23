import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

INFERENCE = Path(__file__).resolve().parents[1] / 'inference'
sys.path.insert(0, str(INFERENCE))
spec = importlib.util.spec_from_file_location('short_horizon_policy', INFERENCE / 'short_horizon_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
from predict import HistoryOnlyPredictor, validate_history


class ShortHorizonPolicyTests(unittest.TestCase):
    def test_flat_history_shrinks_short_horizon_only(self):
        history = np.full((1, 120), 100.)
        pred = np.full((1, 60), 102.)
        result, gate = policy.apply_policy(history, pred, dict(enabled=True, threshold_c=1., end_seconds=180))
        self.assertEqual(gate[0], 1.)
        self.assertLess(result[0, 0], 100.02)
        np.testing.assert_array_equal(result[:, 17:], pred[:, 17:])
        np.testing.assert_array_equal(pred, np.full((1, 60), 102.))

    def test_moving_or_missing_history_disables_shrinkage(self):
        history = np.stack([np.linspace(90., 100., 120), np.full(120, 100.)])
        history[1, -10] = np.nan
        pred = np.full((2, 60), 102.)
        result, gate = policy.apply_policy(history, pred, dict(enabled=True, threshold_c=1., end_seconds=120))
        np.testing.assert_array_equal(gate, [0., 0.])
        np.testing.assert_array_equal(result, pred)

    def test_disabled_policy_is_identity_even_if_current_missing(self):
        history = np.full((1, 120), np.nan)
        pred = np.full((1, 60), 102.)
        result, gate = policy.apply_policy(history, pred, dict(enabled=False))
        np.testing.assert_array_equal(result, pred)
        np.testing.assert_array_equal(gate, [0.])

    def test_gate_uses_history_not_prediction(self):
        history = np.linspace(100., 100.3, 120)[None]
        _, a = policy.apply_policy(history, np.zeros((1, 60)), dict(enabled=True, threshold_c=1., end_seconds=120))
        _, b = policy.apply_policy(history, np.ones((1, 60)), dict(enabled=True, threshold_c=1., end_seconds=120))
        np.testing.assert_array_equal(a, b)

    def test_invalid_settings_or_shapes_raise(self):
        for config in [dict(enabled=True, threshold_c=0., end_seconds=120), dict(enabled=True, threshold_c=1., end_seconds=999)]:
            with self.assertRaises(ValueError):
                policy.apply_policy(np.zeros((1, 120)), np.zeros((1, 60)), config)
        with self.assertRaises(ValueError):
            policy.apply_policy(np.zeros((1, 119)), np.zeros((1, 60)), dict(enabled=False))


class HistoryInputTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'timestamp': pd.date_range('2026-01-01', periods=120, freq='10s'), 'a': 1.})
        self.origin = self.frame.timestamp.iloc[-1]

    def test_auxiliary_missing_column_is_reportable_not_future_filled(self):
        values, _ = validate_history(self.frame, ['a', 'b'], self.origin)
        self.assertTrue(np.isnan(values[:, 1]).all())

    def test_extra_future_row_is_rejected(self):
        f = pd.concat([self.frame, self.frame.iloc[-1:]], ignore_index=True)
        with self.assertRaises(ValueError):
            validate_history(f, ['a'], self.origin)
        with self.assertRaises(ValueError):
            validate_history(self.frame, ['a'], self.origin - pd.Timedelta(seconds=10))

    def test_duplicate_columns_irregular_grid_or_unexpected_column_rejected(self):
        frames = [pd.concat([self.frame, self.frame[['a']]], axis=1), self.frame.assign(future_true=1)]
        wrong_time = self.frame.copy()
        wrong_time.loc[10, 'timestamp'] += pd.Timedelta(seconds=1)
        frames.append(wrong_time)
        for frame in frames:
            with self.assertRaises(ValueError):
                validate_history(frame, ['a'], self.origin)

    def test_bundle_hash_mismatch_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'policy.json').write_text('{}', encoding='utf-8')
            files = {n: '0' * 64 for n in ['policy.json', 'checkpoint.pth', 'metadata.json', 'scaler.json', 'model_config.json']}
            (root / 'manifest.json').write_text(json.dumps(dict(history_only=True, usage='offline_shadow_candidate', files=files)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                HistoryOnlyPredictor(root)


class PredictionQualityTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.predictor = HistoryOnlyPredictor.__new__(HistoryOnlyPredictor)
        self.predictor.torch = torch
        self.predictor.features = [f'v{i}' for i in range(304)]
        self.predictor.features[12] = 'B3TE03'
        self.predictor.targets = self.predictor.features[:14]
        self.predictor.inds = list(range(14))
        self.predictor.mean = np.zeros(304)
        self.predictor.scale = np.ones(304)
        self.predictor.device = torch.device('cpu')
        self.predictor.policy = dict(enabled=True, threshold_c=2., end_seconds=180)
        self.predictor.manifest = dict(version='unit-test')
        self.frame = pd.DataFrame(np.ones((120, 304)), columns=self.predictor.features)
        self.frame.insert(0, 'timestamp', pd.date_range('2026-01-01', periods=120, freq='10s'))
        self.origin = self.frame.timestamp.iloc[-1]
        self.captured = []
        def fake_model(x, xm, dec, dm, dec_mask):
            self.captured.append((x.clone(), dec.clone(), dec_mask.clone()))
            return torch.zeros((1, 60, 304))
        self.predictor.model = fake_model

    def run_prediction(self):
        result = self.predictor.predict(self.frame, as_of=self.origin)
        json.dumps(result, allow_nan=False)
        self.assertFalse(result['usable_for_control'])
        return result

    def test_other_targets_preserved_and_decoder_uses_only_history(self):
        self.frame.loc[:, 'v15'] = np.arange(120)
        result = self.run_prediction()
        p, raw = np.array(result['prediction']), np.array(result['raw_prediction'])
        np.testing.assert_array_equal(p[:, list(range(12)) + [13]], raw[:, list(range(12)) + [13]])
        np.testing.assert_array_equal(p[17:], raw[17:])
        x, dec, mask = self.captured[0]
        np.testing.assert_array_equal(dec[:, :60], x[:, -60:])
        np.testing.assert_array_equal(dec[:, -60:, 15], np.full((1, 60), 119))
        self.assertTrue(mask[:, :, :14].all())
        self.assertFalse(mask[:, :, 14:].any())

    def test_missing_current_target_returns_unavailable_not_fake_value(self):
        self.frame.loc[119, 'B3TE03'] = np.nan
        result = self.run_prediction()
        self.assertEqual(result['target_status']['B3TE03'], 'unavailable')
        self.assertTrue(all(r[12] is None for r in result['prediction']))

    def test_bad_target_history_degrades_to_current_value(self):
        self.frame.loc[:40, 'v0'] = np.nan
        result = self.run_prediction()
        self.assertEqual(result['target_status']['v0'], 'degraded_persistence')
        self.assertTrue(all(r[0] == 1. for r in result['prediction']))

    def test_bad_auxiliary_column_does_not_drop_all_targets(self):
        self.frame.loc[:, 'v15'] = np.nan
        result = self.run_prediction()
        self.assertEqual(result['target_status']['v0'], 'model')
        self.assertEqual(result['quality']['missing_input_fraction']['v15'], 1.)

    def test_broad_missing_data_skips_model_and_reports_degradation(self):
        self.frame.loc[:, self.predictor.features[14:]] = np.nan
        result = self.run_prediction()
        self.assertEqual(len(self.captured), 0)
        self.assertTrue(all(s == 'degraded_persistence' for s in result['target_status'].values()))

    def test_nonfinite_model_output_degrades_explicitly(self):
        self.predictor.model = lambda *args, **kwargs: self.torch.full((1, 60, 304), float('nan'))
        result = self.run_prediction()
        self.assertTrue(all(s == 'degraded_persistence' for s in result['target_status'].values()))


if __name__ == '__main__':
    unittest.main()
